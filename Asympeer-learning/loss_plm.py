# loss_plm.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

@torch.no_grad()
def _topk_small_indices(loss_vec: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of k smallest losses from a 1-D tensor."""
    if loss_vec.dim() != 1:
        loss_vec = loss_vec.reshape(-1)
    k = int(max(0, min(k, loss_vec.numel())))
    if k == 0:
        return loss_vec.new_zeros((0,), dtype=torch.long)
    # torch.topk with largest=False selects smallest values
    _, idx = torch.topk(loss_vec, k=k, largest=False, sorted=False)
    return idx

def _per_sample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    balanced_softmax_loss=None,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Return per-sample loss vector (no reduction).
    Keep this grad-enabled by default; callers decide whether to use no_grad().
    """
    if balanced_softmax_loss is not None:
        return balanced_softmax_loss(
            logits, labels, reduction='none', label_smoothing=label_smoothing
        )
    else:
        return F.cross_entropy(
            logits, labels, reduction='none', label_smoothing=label_smoothing
        )

def _scalar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    balanced_softmax_loss=None,
    label_smoothing: float = 0.0,
    tau_eff: Optional[float] = None
) -> torch.Tensor:
    """
    Return scalar loss (mean). If BalancedSoftmaxLoss is provided, use it
    (optionally overriding its tau for this call only).
    """
    if balanced_softmax_loss is None:
        return F.cross_entropy(logits, labels, reduction='mean', label_smoothing=label_smoothing)

    # Temporarily override tau (for annealing), then restore to avoid side effects.
    old_tau = getattr(balanced_softmax_loss, 'tau', 1.0)
    if tau_eff is not None:
        setattr(balanced_softmax_loss, 'tau', float(tau_eff))
    try:
        loss = balanced_softmax_loss(
            logits, labels, reduction='mean', label_smoothing=label_smoothing
        )
    finally:
        setattr(balanced_softmax_loss, 'tau', old_tau)
    return loss

def _split_agreement(
    logits1: torch.Tensor, logits2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute predictions and return (agree_idx, disagree_idx, agree_mask).
    Runs with no grad; only indices are needed.
    """
    with torch.no_grad():
        pred1 = logits1.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)
        agree_mask = pred1.eq(pred2)
        agree_idx = torch.nonzero(agree_mask, as_tuple=False).squeeze(1)
        disagree_idx = torch.nonzero(~agree_mask, as_tuple=False).squeeze(1)
    return agree_idx, disagree_idx, agree_mask

def peer_learning_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    labels: torch.Tensor,
    rate: float,
    *,
    balanced_softmax_loss=None,        # BalancedSoftmaxLoss instance or None
    label_smoothing: float = 0.0,
    # τ annealing setup (kept for compatibility with your main.py)
    bs_tau_min: float = 0.7,
    bs_tau_max: float = 1.0,
    anneal_with_rate: bool = True,
    max_rate_hint: float = 0.25
):
    """
    Two-network peer learning with small-loss selection on the agreement set.

    Steps:
      1) Split batch into agreement/disagreement by argmax predictions (no grad).
      2) On the agreement subset, compute per-sample losses **only for ranking**
         and pick k = round((1-rate) * |A|) smallest per model (no grad).
      3) Keep all disagreement samples (as in the original).
      4) Compute final scalar losses on the selected indices **with grad** using
         BalancedSoftmaxLoss (if provided) or standard CE.
      5) Optionally anneal BalancedSoftmax tau using current rate.

    Memory/compute notes:
      - All selection logic (argmax/topk/per-sample ranking) is wrapped in
        torch.no_grad() to avoid building unnecessary autograd graphs.
      - Final losses remain grad-enabled for backprop.
    """
    assert logits1.shape == logits2.shape and logits1.shape[0] == labels.shape[0], \
        "logits1/logits2 must have same shape and batch size must match labels"
    N = labels.shape[0]
    device = logits1.device

    # --- Selection path: no grads needed ---
    agree_idx, disagree_idx, _ = _split_agreement(logits1, logits2)

    with torch.no_grad():
        if agree_idx.numel() > 0:
            # Per-sample losses used solely for *ranking*; no graph needed.
            l1_agree = _per_sample_loss(
                logits1[agree_idx], labels[agree_idx],
                balanced_softmax_loss, label_smoothing
            )
            l2_agree = _per_sample_loss(
                logits2[agree_idx], labels[agree_idx],
                balanced_softmax_loss, label_smoothing
            )

            remember_ratio = float(max(0.0, min(1.0, 1.0 - rate)))
            k = int(round(remember_ratio * agree_idx.numel()))
            idx_small_1 = _topk_small_indices(l1_agree, k)
            idx_small_2 = _topk_small_indices(l2_agree, k)

            keep_idx_1 = agree_idx[idx_small_1]
            keep_idx_2 = agree_idx[idx_small_2]
        else:
            keep_idx_1 = torch.zeros((0,), dtype=torch.long, device=device)
            keep_idx_2 = torch.zeros((0,), dtype=torch.long, device=device)

        # Disagreement samples: keep all (indices only).
        if disagree_idx.numel() > 0:
            final_idx_1 = torch.unique(torch.cat([keep_idx_1, disagree_idx], dim=0))
            final_idx_2 = torch.unique(torch.cat([keep_idx_2, disagree_idx], dim=0))
        else:
            final_idx_1 = keep_idx_1
            final_idx_2 = keep_idx_2

        # Safety: if selection becomes empty due to extreme settings, fall back to full batch.
        if final_idx_1.numel() == 0:
            final_idx_1 = torch.arange(N, device=device)
        if final_idx_2.numel() == 0:
            final_idx_2 = torch.arange(N, device=device)

    # --- Trainable path: compute scalar losses with grads ---
    tau_eff = None
    if balanced_softmax_loss is not None and anneal_with_rate:
        # Linearly anneal tau from max->min as rate grows, clamped by max_rate_hint.
        progress = float(min(1.0, max(0.0, rate / max(1e-8, max_rate_hint))))
        tau_eff = bs_tau_max - (bs_tau_max - bs_tau_min) * progress

    loss_1_update = _scalar_loss(
        logits1[final_idx_1], labels[final_idx_1],
        balanced_softmax_loss=balanced_softmax_loss,
        label_smoothing=label_smoothing,
        tau_eff=tau_eff
    )
    loss_2_update = _scalar_loss(
        logits2[final_idx_2], labels[final_idx_2],
        balanced_softmax_loss=balanced_softmax_loss,
        label_smoothing=label_smoothing,
        tau_eff=tau_eff
    )

    return loss_1_update, loss_2_update
