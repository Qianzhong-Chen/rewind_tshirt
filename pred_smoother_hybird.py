from collections import deque
from typing import Optional, Tuple

class RegressionConfidenceSmoother:
    def __init__(
        self,
        window_size: int = 30, # 15 for demo, 30 for rollout
        beta: float = 5.0,
        eps: float = 1e-6,
        low_conf_th: float = 0.95,
        value_range: Optional[Tuple[float, float]] = None,  # e.g., (min_val, max_val)
    ):
        """
        value: float regression output
        confidence: float in [0,1]
        """
        self.window_size = window_size
        self.beta = beta
        self.eps = eps
        self.low_conf_th = low_conf_th
        self.value_range = value_range

        self.hist_vals = deque(maxlen=window_size)
        self.hist_confs = deque(maxlen=window_size)
        self.last_smoothed: Optional[float] = None

    def reset(self):
        self.hist_vals.clear()
        self.hist_confs.clear()
        self.last_smoothed = None

    def _normalize_val(self, v: float) -> float:
        if self.value_range is None:
            return float(v)
        vmin, vmax = self.value_range
        if vmax <= vmin:
            return float(v)
        return (float(v) - vmin) / (vmax - vmin)

    def update(self, value: float, confidence: float):
        # sanitize inputs
        val_t = self._normalize_val(value)
        conf_t = max(0.0, min(1.0, float(confidence)))  # clamp to [0,1]

        # baseline from history only
        if self.hist_vals:
            # weights = conf^beta (clipped away from 0)
            weights = [max(self.eps, c) ** self.beta for c in self.hist_confs]
            wsum = sum(weights) + self.eps
            baseline = sum(w * v for w, v in zip(weights, self.hist_vals)) / wsum
        else:
            baseline = val_t

        if self.last_smoothed is None:
            self.last_smoothed = baseline

        # low confidence: skip update, return last smoothed
        if conf_t < self.low_conf_th:
            self.hist_vals.append(self.last_smoothed)
            self.hist_confs.append(self.low_conf_th)
            return self.last_smoothed

        # accept point, update history
        self.hist_vals.append(val_t)
        self.hist_confs.append(conf_t)

        # recompute smoothed with current point included
        weights = [max(self.eps, c) ** self.beta for c in self.hist_confs]
        wsum = sum(weights) + self.eps
        smoothed_item = sum(w * v for w, v in zip(weights, self.hist_vals)) / wsum

        self.last_smoothed = smoothed_item
        return smoothed_item
