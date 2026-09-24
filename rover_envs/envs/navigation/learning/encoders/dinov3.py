"""RGB encoder built on a frozen DINOv3 backbone."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ViTModel


class PerceiverBlock(nn.Module):
    """Learned queries cross-attend to the image tokens, followed by an MLP."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        queries = queries + self.attn(self.attn_norm(queries), tokens, tokens, need_weights=False)[0]
        return queries + self.mlp(self.mlp_norm(queries))


class DinoV3Encoder(nn.Module):
    """Encode RGB frames [B, H, W, 3] with values in 0-255 into a [B, 512] embedding.

    The frames are resized to ``image_size`` and passed through the frozen DINOv3 ViT-S/16. Its CLS and patch
    tokens (the register tokens are dropped) are projected 384 -> 192 -> 256 and pooled by 8 learned queries over
    2 cross-attention blocks. The mean and max over the queries and the CLS token are fused: 768 -> 1024 -> 512.
    """

    out_dim = 512

    def __init__(self, model_path: str, image_size: tuple[int, int] = (288, 512)):
        super().__init__()
        self.backbone = DINOv3ViTModel.from_pretrained(os.path.expanduser(model_path)).requires_grad_(False)
        config = self.backbone.config
        self.image_size = tuple(image_size)
        self.num_registers = config.num_register_tokens
        num_tokens = 1 + (image_size[0] // config.patch_size) * (image_size[1] // config.patch_size)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.token_proj = nn.Linear(config.hidden_size, 192)
        self.patch_mlp = nn.Sequential(nn.Linear(192, 256), nn.GELU(), nn.Linear(256, 256))
        self.cls_proj = nn.Linear(192, 256)
        # Cross-attention is permutation invariant, so the tokens need their image position.
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, num_tokens, 256))
        self.token_norm = nn.LayerNorm(256)
        self.queries = nn.Parameter(0.02 * torch.randn(1, 8, 256))
        self.blocks = nn.ModuleList(PerceiverBlock(256, num_heads=8) for _ in range(2))
        self.head = nn.Sequential(
            nn.Linear(3 * 256, 1024), nn.GELU(), nn.Linear(1024, self.out_dim), nn.LayerNorm(self.out_dim)
        )

    def train(self, mode: bool = True):
        # DINOv3 randomly rescales its RoPE coordinates in train mode, so the frozen backbone always stays in eval.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=self.image_size, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            tokens = self.backbone(pixel_values=x).last_hidden_state
        # [CLS, registers, patches] -> [CLS, patches]
        tokens = self.token_proj(torch.cat((tokens[:, :1], tokens[:, 1 + self.num_registers :]), dim=1))
        cls = self.cls_proj(tokens[:, 0])
        tokens = torch.cat((cls.unsqueeze(1), self.patch_mlp(tokens[:, 1:])), dim=1)
        tokens = self.token_norm(tokens + self.pos_embed)
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, tokens)
        return self.head(torch.cat((queries.mean(dim=1), queries.amax(dim=1), cls), dim=-1))
