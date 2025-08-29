import torch
import numpy as np
from collections import deque

class ConfidenceSmoother:
    def __init__(self, num_classes, window_size=25, beta=2.0, eps=1e-6, low_conf_th=0.2):
        """
        Args:
            num_classes (int): number of stage classes
            window_size (int): sliding window length
            beta (float): exponent to emphasize high-confidence values
            eps (float): small constant to avoid division by zero
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.beta = beta
        self.eps = eps
        self.low_conf_th = low_conf_th
        self.hist_vals = deque(maxlen=window_size)
        self.hist_confs = deque(maxlen=window_size)

    def update(self, stage_prob: torch.Tensor, batch_idx: int = 0, t_idx: int = 0):
        C = self.num_classes
        classes = torch.arange(C, device=stage_prob.device).float()

        # soft prediction & confidence
        expect_stage = (stage_prob * classes).sum(dim=-1) / (C - 1)   # (B,T)
        stage_pred = stage_prob.argmax(dim=-1)
        stage_conf = stage_prob.gather(-1, stage_pred.unsqueeze(-1)).squeeze(-1)  # (B,T)

        val_t  = expect_stage[batch_idx, t_idx].item()
        conf_t = stage_conf[batch_idx, t_idx].item()

        # --- baseline from history ONLY (no current point) ---
        if len(self.hist_vals) > 0:
            vals_hist = np.array(self.hist_vals, dtype=np.float32)
            w_hist    = np.array(self.hist_confs, dtype=np.float32)
            w_hist    = np.clip(w_hist, self.eps, 1.0) ** self.beta
            baseline = float((w_hist * vals_hist).sum() / (w_hist.sum() + self.eps))
        else:
            baseline = val_t  # first step fallback

        # --- low-confidence handling BEFORE updating history ---
        if conf_t < self.low_conf_th:
            return (self.last_smoothed if hasattr(self, "last_smoothed") else baseline), conf_t
            # smoothed_item = baseline
            # return smoothed_item, conf_t

        # otherwise, incorporate current point and compute smoothed with it
        self.hist_vals.append(val_t)
        self.hist_confs.append(conf_t)

        vals = np.array(self.hist_vals, dtype=np.float32)
        w    = np.array(self.hist_confs, dtype=np.float32)
        w    = np.clip(w, self.eps, 1.0) ** self.beta
        smoothed_item = float((w * vals).sum() / (w.sum() + self.eps))
        self.last_smoothed = smoothed_item  # store for next low-conf case

        return smoothed_item, conf_t
