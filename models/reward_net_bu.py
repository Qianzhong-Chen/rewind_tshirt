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
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)  # Assuming RGB images with 3 channels

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))


        pos = torch.arange(max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("sinusoid_pe", pe, persistent=False)

        self.total_tokens = num_cameras + 1  # +1 for language token
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(d_model * self.total_tokens),
            nn.Linear(d_model * self.total_tokens, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )


    def forward(self, img_seq: torch.Tensor,
                lang_emb: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        B, N, T, d = img_seq.shape # self.total_tokens = N + 1
        lang_emb = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d_model) 
        lang_emb = lang_emb.expand(B, 1, T, -1) 
        vis_emb = self.visual_proj(img_seq)
        x = torch.cat([lang_emb, vis_emb], 1).view(B, (N+1)*T, -1)

        # add positional embedding to each camer's first frame
        for i in range(N):  
            idx = 1 + i * T
            x[:, idx, :] += self.first_pos  

        # mask out void timesteps
        base_mask = torch.arange(T, device=img_seq.device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 1, T).reshape(B, (N + 1) * T)  # (B, (N+1)*T)
        h = self.transformer(x, src_key_padding_mask=mask)
        h = h.view(B, N + 1, T, -1).transpose(1, 2).contiguous()  # (B, T, N+1, d_model)
        flatterned_h = h.view(B, T, -1)  # (B, T, (N+1)*d_model)
        r = self.fusion_net(flatterned_h).squeeze(-1)  # (B, T, 1)
        r = torch.sigmoid(r)  # Ensure output is in [0, 1] range
        return r
