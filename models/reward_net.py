import torch
import torch.nn as nn

class RewardTransformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 vis_emb_dim: int,
                 text_emb_dim: int,
                 n_layers: int,
                 n_heads: int,
                 dropout: float,
                 max_seq_len: int,
                 num_cameras: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.max_seq_len = max_seq_len

        # Projection layers
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # Positional bias only for first visual frame (to avoid leaking absolute time)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # Fusion MLP
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(d_model * num_cameras),
            nn.Linear(d_model * num_cameras, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self,
                img_seq: torch.Tensor,     # (B, N, T, vis_emb_dim)
                lang_emb: torch.Tensor,    # (B, text_emb_dim)
                lengths: torch.Tensor      # (B,)
                ) -> torch.Tensor:
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        # Project vision
        vis_proj = self.visual_proj(img_seq)                      # (B, N, T, d_model)

        # Project language and expand
        lang_proj = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d_model)
        lang_proj = lang_proj.expand(B, N, 1, D)                        # (B, N, 1, d_model)

        # Concatenate along time dimension
        x = torch.cat([vis_proj, lang_proj], dim=2)              # (B, N, T+1, d_model)

        # Add first_pos to camera’s first frame
        x[:, :, 0, :] += self.first_pos                          # (B, N, T+1, d_model)

        # Reshape for transformer: (B, N*(T+1), d_model)
        x = x.view(B, N * (T + 1), D)

        # Build transformer mask
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        vis_mask = base_mask.unsqueeze(1).expand(B, N, T).reshape(B, N * T)              # (B, N*T)
        lang_mask = torch.zeros((B, N), device=device, dtype=torch.bool)                # (B, N)
        mask = torch.cat([vis_mask, lang_mask], dim=1)                                  # (B, N*(T+1))

        # Apply transformer
        h = self.transformer(x, src_key_padding_mask=mask)        # (B, N*(T+1), d_model)

        # Reshape to (B, N, T+1, d_model)
        h = h.view(B, N, T + 1, D)

        # Remove the language token (last timestep) → (B, N, T, d_model)
        h = h[:, :, :T, :]   

        # Transpose for fusion: (B, T, N, d_model)
        h = h.permute(0, 2, 1, 3).contiguous()

        # Fuse camera tokens at each timestep
        h_fused = h.view(B, T, N * D)                      # (B, T, N*d_model)
        r = self.fusion_net(h_fused).squeeze(-1)      # (B, T, 1)
        r = torch.sigmoid(r)
        return r
