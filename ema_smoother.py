import torch

class EMASmoother:
    def __init__(self, num_classes, alpha: float = 0.2, conf_exponent: float = 2.0, low_conf_th: float = 0.2):
        """
        Args:
            alpha (float): base smoothing factor (0 < alpha <= 1).
                           Larger alpha = more responsive, less smooth.
            conf_exponent (float): exponent to amplify effect of confidence on alpha.
            low_conf_th (float): if confidence below this, freeze update.
        """
        self.num_classes = num_classes
        self.alpha = alpha
        self.conf_exponent = conf_exponent
        self.low_conf_th = low_conf_th
        self.last_smoothed = None

    def update(self, stage_prob: torch.Tensor, batch_idx: int = 0, t_idx: int = 0):
        """
        stage_prob: (B,T,C) tensor of probabilities
        """
        C = self.num_classes
        classes = torch.arange(C, device=stage_prob.device).float()

        # soft expectation in [0,1]
        expect_stage = (stage_prob * classes).sum(dim=-1) / (C - 1)   # (B,T)
        stage_pred = stage_prob.argmax(dim=-1)
        stage_conf = stage_prob.gather(-1, stage_pred.unsqueeze(-1)).squeeze(-1)  # (B,T)

        val_t  = expect_stage[batch_idx, t_idx].item()
        conf_t = stage_conf[batch_idx, t_idx].item()

        # low confidence -> freeze (reuse last smoothed)
        if conf_t < self.low_conf_th and self.last_smoothed is not None:
            return self.last_smoothed, conf_t

        # adjust alpha by confidence
        adj_alpha = self.alpha * (conf_t ** self.conf_exponent)

        # update EMA
        if self.last_smoothed is None:
            self.last_smoothed = val_t
        else:
            self.last_smoothed = (1 - adj_alpha) * self.last_smoothed + adj_alpha * val_t

        return self.last_smoothed, conf_t
