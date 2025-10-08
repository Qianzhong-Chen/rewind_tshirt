import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from transformers import AutoFeatureExtractor, AutoModel



class FrozenVisionEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        """
        Loads DINOv2 from Hugging Face.
        ckpt: e.g., 'facebook/dinov2-base'
        """
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(ckpt)
        # self.processor = AutoFeatureExtractor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 3, 224, 224), float32 in [0, 1]
        returns: (B, vis_emb_dim)
        """
        # Convert to list of PIL images
        imgs = imgs.cpu()
        imgs = [T.ToPILImage()(img) for img in imgs]

        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            return out.last_hidden_state[:, 0]  # CLS token

