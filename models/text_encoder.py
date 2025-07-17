import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict


class FrozenTextEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str] length B
        returns: (B, text_emb_dim)   (pooled CLS hidden state)
        """
        with torch.no_grad():
            toks = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=64, return_tensors="pt").to(self.model.device)
            out = self.model(**toks)
            emb = out.last_hidden_state[:, 0]   # CLS
        return emb
