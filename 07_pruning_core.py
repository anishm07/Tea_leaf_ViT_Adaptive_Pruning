"""
Stage 3a: Zeta-gated attention head pruning - core module
=============================================================

Implements Algorithm 1 from the paper:
  1. Wrap every attention head in the pretrained ViT-B/16 with a learnable
     gate parameter (zeta), one per head per layer (12 layers x 12 heads
     = 144 gates total).
  2. Importance Learning (Search Phase): freeze the backbone, train ONLY
     the zeta gates using a combined loss:
       - Lce:  cross-entropy against true labels (using gated student output)
       - Lkd:  soft-target KD loss against the frozen, fully-pretrained
               (unpruned, fine-tuned) teacher
       - Lreg: L1 sparsity penalty on the gates (sum |zeta|), pushing
               unimportant heads toward zero
  3. Extract learned importance scores (sigmoid(zeta) per head).
  4. Physically prune the lowest-importance heads per the chosen ratio
     scheme (adaptive layer-wise OR uniform -- both provided here for
     the ablation comparison), rebuilding smaller qkv/proj weight
     matrices so FLOPs are genuinely reduced (not just masked to zero).
  5. Fine-tune the pruned model to recover any accuracy loss.

This module is imported by 07_run_pruning_pipeline.py, which runs the
full pipeline for both the adaptive and uniform variants.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Gated attention wrapper
# ---------------------------------------------------------------------------

class GatedAttention(nn.Module):
    """
    Drop-in replacement for timm's ViT Attention module that adds a
    learnable per-head gate (zeta) applied multiplicatively to each head's
    output before the final projection. Weights are copied from the
    original module, so behavior is identical when all gates are open
    (sigmoid(zeta) ~ 1).
    """

    def __init__(self, orig_attn: nn.Module):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim if hasattr(orig_attn, "head_dim") \
            else orig_attn.qkv.in_features // orig_attn.num_heads
        self.scale = getattr(orig_attn, "scale", self.head_dim ** -0.5)
        dim = orig_attn.qkv.in_features

        self.qkv = nn.Linear(dim, dim * 3, bias=orig_attn.qkv.bias is not None)
        self.qkv.weight.data = orig_attn.qkv.weight.data.clone()
        if orig_attn.qkv.bias is not None:
            self.qkv.bias.data = orig_attn.qkv.bias.data.clone()

        self.proj = nn.Linear(dim, dim, bias=orig_attn.proj.bias is not None)
        self.proj.weight.data = orig_attn.proj.weight.data.clone()
        if orig_attn.proj.bias is not None:
            self.proj.bias.data = orig_attn.proj.bias.data.clone()

        self.attn_drop = getattr(orig_attn, "attn_drop", nn.Dropout(0.0))
        self.proj_drop = getattr(orig_attn, "proj_drop", nn.Dropout(0.0))

        # Learnable per-head gate, initialized near 1 after sigmoid
        # (zeta=2.0 -> sigmoid ~0.88, gently open so training starts close
        # to the unpruned model's behavior)
        self.zeta = nn.Parameter(torch.full((self.num_heads,), 2.0))
        self.gates_frozen_mask = None  # set after physical pruning, if reused

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # B, heads, N, head_dim

        gate = torch.sigmoid(self.zeta).view(1, self.num_heads, 1, 1)
        out = out * gate

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def wrap_model_with_gates(model: nn.Module) -> nn.Module:
    """Replace every block's attention module with a GatedAttention wrapper."""
    for block in model.blocks:
        block.attn = GatedAttention(block.attn)
    return model


def get_all_zeta_params(model: nn.Module):
    return [block.attn.zeta for block in model.blocks]


def freeze_backbone_except_zeta(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = "zeta" in name


# ---------------------------------------------------------------------------
# Step 2: Importance learning (search phase) loss
# ---------------------------------------------------------------------------

def kd_loss(student_logits, teacher_logits, temperature: float = 4.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def importance_learning_step(student, teacher, images, labels, optimizer,
                              reg_weight: float = 1e-4, kd_weight: float = 1.0,
                              temperature: float = 4.0):
    student.train()
    teacher.eval()

    optimizer.zero_grad()

    student_logits = student(images)
    with torch.no_grad():
        teacher_logits = teacher(images)

    l_ce = F.cross_entropy(student_logits, labels)
    l_kd = kd_loss(student_logits, teacher_logits, temperature)

    zetas = torch.cat([block.attn.zeta for block in student.blocks])
    l_reg = torch.sum(torch.abs(zetas))

    loss = l_ce + kd_weight * l_kd + reg_weight * l_reg
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "l_ce": l_ce.item(),
            "l_kd": l_kd.item(), "l_reg": l_reg.item()}


def extract_importance_scores(model: nn.Module):
    """Returns a (num_layers, num_heads) tensor of learned head importances."""
    scores = []
    for block in model.blocks:
        scores.append(torch.sigmoid(block.attn.zeta.detach()).cpu())
    return torch.stack(scores)  # shape: [num_layers, num_heads]


# ---------------------------------------------------------------------------
# Step 3: Pruning ratio schemes
# ---------------------------------------------------------------------------

def adaptive_layerwise_ratios(num_layers: int = 12):
    """
    Matches Algorithm 1: Layers 1-4: 25%, Layers 5-8: 35%, Layers 9-12: 30%.
    Generalizes proportionally if num_layers != 12.
    """
    ratios = []
    for i in range(num_layers):
        frac = i / num_layers
        if frac < 1 / 3:
            ratios.append(0.25)
        elif frac < 2 / 3:
            ratios.append(0.35)
        else:
            ratios.append(0.30)
    return ratios


def adaptive_ratios_at_sparsity(target_sparsity: float, num_layers: int = 12):
    """
    Scales the base adaptive scheme (25/35/30, averaging 30%) to a different
    overall target sparsity, while preserving the relative depth-wise shape
    (shallow < deep < middle). Used for the sparsity sweep ablation.
    Clips to a safe [0.05, 0.85] range per layer to avoid pruning away an
    entire layer's heads at extreme targets.
    """
    base = adaptive_layerwise_ratios(num_layers)
    base_avg = sum(base) / len(base)  # 0.30
    scale = target_sparsity / base_avg
    return [min(max(r * scale, 0.05), 0.85) for r in base]


def uniform_ratios(num_layers: int = 12, target_overall: float = None,
                    importance_scores=None):
    """
    Baseline: same pruning ratio applied to every layer. If target_overall
    is not given, it's computed to match the OVERALL sparsity that the
    adaptive scheme would produce, so the two variants are directly
    comparable at equal total sparsity (a fair ablation).
    """
    if target_overall is None:
        adaptive = adaptive_layerwise_ratios(num_layers)
        target_overall = sum(adaptive) / len(adaptive)
    return [target_overall] * num_layers


# ---------------------------------------------------------------------------
# Step 4: Physical structural pruning
# ---------------------------------------------------------------------------

def select_heads_to_keep(importance_scores: torch.Tensor, ratios: list):
    """
    For each layer, determine which head indices to KEEP based on learned
    importance and the layer's target pruning ratio (heads with the
    LOWEST importance are removed).
    Returns: list of lists, kept_heads[layer_idx] = [head indices to keep]
    """
    num_layers, num_heads = importance_scores.shape
    kept_heads = []
    for l in range(num_layers):
        ratio = ratios[l]
        n_to_prune = round(ratio * num_heads)
        n_to_prune = min(n_to_prune, num_heads - 1)  # never prune all heads in a layer
        layer_scores = importance_scores[l]
        sorted_idx = torch.argsort(layer_scores, descending=True)
        keep_idx = sorted(sorted_idx[: num_heads - n_to_prune].tolist())
        kept_heads.append(keep_idx)
    return kept_heads


def build_pruned_attention(gated_attn: GatedAttention, keep_idx: list) -> nn.Module:
    """
    Rebuilds a smaller Attention module containing only the kept heads.
    Slices the qkv weight (organized as [3, num_heads, head_dim, dim] when
    reshaped) and the proj weight (organized as [dim_out, num_heads*head_dim]).
    """
    head_dim = gated_attn.head_dim
    dim = gated_attn.qkv.in_features
    n_keep = len(keep_idx)

    # qkv.weight shape: (3*dim, dim). Reshape to (3, num_heads, head_dim, dim)
    qkv_w = gated_attn.qkv.weight.data.view(3, gated_attn.num_heads, head_dim, dim)
    qkv_w_pruned = qkv_w[:, keep_idx, :, :].reshape(3 * n_keep * head_dim, dim)

    qkv_b_pruned = None
    if gated_attn.qkv.bias is not None:
        qkv_b = gated_attn.qkv.bias.data.view(3, gated_attn.num_heads, head_dim)
        qkv_b_pruned = qkv_b[:, keep_idx, :].reshape(3 * n_keep * head_dim)

    # proj.weight shape: (dim, dim) = (dim_out, num_heads*head_dim)
    proj_w = gated_attn.proj.weight.data.view(dim, gated_attn.num_heads, head_dim)
    proj_w_pruned = proj_w[:, keep_idx, :].reshape(dim, n_keep * head_dim)
    proj_b_pruned = gated_attn.proj.bias.data.clone() if gated_attn.proj.bias is not None else None

    pruned = PrunedAttention(n_keep, head_dim, dim, gated_attn.scale)
    pruned.qkv.weight.data = qkv_w_pruned
    if qkv_b_pruned is not None:
        pruned.qkv.bias.data = qkv_b_pruned
    pruned.proj.weight.data = proj_w_pruned
    if proj_b_pruned is not None:
        pruned.proj.bias.data = proj_b_pruned

    return pruned


class PrunedAttention(nn.Module):
    """Structurally smaller attention module with a reduced head count."""

    def __init__(self, num_heads: int, head_dim: int, dim_out: int, scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        qkv_dim = num_heads * head_dim
        self.qkv = nn.Linear(dim_out, qkv_dim * 3)
        self.proj = nn.Linear(qkv_dim, dim_out)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv_dim = self.num_heads * self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, qkv_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def apply_structural_pruning(model: nn.Module, kept_heads: list) -> nn.Module:
    """Replaces each block's GatedAttention with a smaller PrunedAttention."""
    for l, block in enumerate(model.blocks):
        block.attn = build_pruned_attention(block.attn, kept_heads[l])
    return model


def count_total_and_pruned_heads(kept_heads: list, num_heads_per_layer: int = 12):
    total = len(kept_heads) * num_heads_per_layer
    kept = sum(len(k) for k in kept_heads)
    pruned = total - kept
    return pruned, total
