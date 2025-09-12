import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 text_emb_dim: int = 512,
                 state_dim: int = 7,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 128,
                 num_cameras: int = 1,
                 dense_annotation: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.dense_annotation = dense_annotation

        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.lang_proj   = nn.Linear(text_emb_dim, d_model)
        self.state_proj  = nn.Linear(state_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # cams (N) + lang (1) + state (1) + stage (1) => N + 3 tracks
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(d_model * (num_cameras + 3)),
            nn.Linear(d_model * (num_cameras + 3), d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _broadcast_lang(lang_emb: torch.Tensor, B: int, T: int, d_model: int,
                        proj: nn.Linear, dense_annotation: bool) -> torch.Tensor:
        if dense_annotation:
            if lang_emb.dim() != 3 or lang_emb.shape[1] != T:
                raise ValueError(f"dense_annotation=True expects lang_emb (B,T,Dt), got {tuple(lang_emb.shape)}")
            lang_proj = proj(lang_emb).unsqueeze(1)  # (B,1,T,d)
        else:
            if lang_emb.dim() != 2:
                raise ValueError(f"dense_annotation=False expects lang_emb (B,Dt), got {tuple(lang_emb.shape)}")
            lang_proj = proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, d_model)  # (B,1,T,d)
        return lang_proj

    def _stage_to_dmodel(self, stage_onehot: torch.Tensor) -> torch.Tensor:
        """
        Deterministic projection of one-hot to d_model by pad/truncate.
        stage_onehot: (B,1,T,C) -> (B,1,T,d_model)
        """
        B, one, T, C = stage_onehot.shape
        D = self.d_model
        if D == C:
            return stage_onehot
        elif D > C:
            pad = torch.zeros(B, one, T, D - C, device=stage_onehot.device, dtype=stage_onehot.dtype)
            return torch.cat([stage_onehot, pad], dim=-1)
        else:
            return stage_onehot[..., :D]

    def forward(self,
                img_seq: torch.Tensor,     # (B,N,T,vis_emb_dim)
                lang_emb: torch.Tensor,    # (B,Dt) or (B,T,Dt) if dense
                state: torch.Tensor,       # (B,1,T,state_dim)
                lengths: torch.Tensor,     # (B,)
                stage_onehot: torch.Tensor # (B,1,T,C) one-hot (from gen_stage_emb)
                ) -> torch.Tensor:
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        vis_proj   = self.visual_proj(img_seq)                          # (B,N,T,D)
        state_proj = self.state_proj(state).unsqueeze(1)                # (B,1,T,D)
        lang_proj  = self._broadcast_lang(lang_emb, B, T, D, self.lang_proj, self.dense_annotation)  # (B,1,T,D)

        stage_emb  = self._stage_to_dmodel(stage_onehot)                # (B,1,T,D)

        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)  # (B,N+3,T,D)
        x[:, :N, 0, :] += self.first_pos

        x = x.view(B, (N + 3) * T, D)

        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)

        h = self.transformer(x, src_key_padding_mask=mask)              # (B,(N+3)*T,D)
        h = h.view(B, N + 3, T, D)
        h_flat = h.view(B, T, (N + 3) * D)
        r = self.fusion_net(h_flat).squeeze(-1)                         # (B,T)
        return torch.sigmoid(r)
