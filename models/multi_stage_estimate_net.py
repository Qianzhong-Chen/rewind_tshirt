import torch
import torch.nn as nn

class StageTransformer(nn.Module):
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
                 num_classes: int = 4,
                 dense_annotation: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.max_seq_len = max_seq_len
        self.dense_annotation = dense_annotation

        # Projection layers
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # Positional bias only for first visual frame (to avoid leaking absolute time)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # Fusion MLP
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(d_model * (num_cameras + 2)),
            nn.Linear(d_model * (num_cameras + 2), d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self,
                img_seq: torch.Tensor,     # (B, N, T, vis_emb_dim)
                lang_emb: torch.Tensor,    # (B, text_emb_dim)
                state: torch.Tensor,       # (B, T, state_dim)
                lengths: torch.Tensor      # (B,)
                ) -> torch.Tensor:
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        # Project vision
        vis_proj = self.visual_proj(img_seq)                      # (B, N, T, d_model)

        # Project state
        state_proj = self.state_proj(state).unsqueeze(1)               # (B, 1, T, d_model)

        # Project language and expand
        if self.dense_annotation:
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1)  # (B, 1, T, d_model)
        else:
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d_model)
            lang_proj = lang_proj.expand(B, 1, T, D)                        # (B, 1, T, d_model)
        # Concatenate along time dimension
        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)              # (B, N+2, T, d_model)

        # Add first_pos to camera’s first frame
        x[:, :N, 0, :] += self.first_pos                          # (B, N+2, T, d_model)

        # Reshape for transformer: (B, N*(T+1), d_model)
        x = x.view(B, (N + 2) * T, D)

        # Build transformer mask
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        mask = base_mask.unsqueeze(1).expand(B, N + 2, T).reshape(B, (N + 2) * T)  # (B, (N+2)*T)

        # Apply transformer
        h = self.transformer(x, src_key_padding_mask=mask)        # (B, (N+2)*T, d_model)

        # Reshape to (B, N, T+1, d_model)
        h = h.view(B, N + 2, T, D)

        flatterned_h = h.view(B, T, -1)  # (B, T, (N+2)*d_model)
        logits = self.fusion_net(flatterned_h)                   # (B, T, num_classes)
        return logits  # raw logits for cross-entropy
