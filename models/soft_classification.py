import torch
import torch.nn.functional as F

def gaussian_soft_labels(y_idx, K, sigma=1.0):
    # y_idx: integer bin or fractional bin in [0, K-1], shape (B, T)
    classes = torch.arange(K, device=y_idx.device).view(1,1,K).float()
    d2 = (classes - y_idx.unsqueeze(-1))**2
    P = torch.exp(-0.5 * d2 / (sigma**2))
    P = P / P.sum(dim=-1, keepdim=True)
    return P  # (B, T, K)

def masked_soft_ce(logits, soft_targets, lengths):
    logp = F.log_softmax(logits, dim=-1)              # (B, T, K)
    ce = -(soft_targets * logp).sum(dim=-1)           # (B, T)
    B, T = ce.shape
    mask = torch.arange(T, device=ce.device)[None, :] < lengths[:, None]
    return (ce * mask).sum() / mask.sum().clamp_min(1)

def unimodal_penalty_from_logits(logits: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-8):
    """
    logits: (B, T, K)
    lengths: (B,)
    returns scalar penalty
    """
    probs = F.softmax(logits, dim=-1)         # (B, T, K)
    # 2nd finite difference along class axis
    d1 = probs[..., 1:] - probs[..., :-1]     # (B, T, K-1)
    d2 = d1[..., 1:] - d1[..., :-1]           # (B, T, K-2)
    pen = d2.abs()                            # L1 curvature

    # mask padded timesteps
    B, T, _ = probs.shape
    mask = (torch.arange(T, device=logits.device)[None, :] < lengths[:, None]).float()
    mask = mask.unsqueeze(-1)                 # (B, T, 1)
    return (pen * mask).sum() / (mask.sum() * pen.shape[-1] + eps)