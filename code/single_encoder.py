"""
-----------Single-Scale Embedder Architecture for model-variant--------------

Description:
    1. This module implements the SingleScaleEmbedder, a single-encoder transformer architecture for unsupervised tract-level embedding.
    Use this in place of the DualScaleEmbedder to assess the effect of removing the dual-encoder structure.

Workflow:
- Replace DualScaleEmbedder with SingleScaleEmbedder in your pipeline to run the architecture ablation.
- All other pipeline code and training methods remain unchanged.

Note: See dual_scale_encoder.py for the full dual-encoder version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleScaleEmbedder(nn.Module):
    def __init__(self, d_building=75, d_hidden=64, d_feedforward=512,
                 building_head=4, building_layers=3):
        super().__init__()

        # Project input features to hidden dimension
        self.building_projector = nn.Linear(d_building, d_hidden)

        # Single transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_hidden,
                nhead=building_head,
                dim_feedforward=d_feedforward,
                batch_first=False
            ),
            num_layers=building_layers
        )

        self.proj_head = nn.Sequential(
            nn.Linear(d_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, building_feature, building_mask):
        """
        Args:
            building_feature: Tensor, shape [batch_size, seq_len, d_building]
            building_mask: Tensor, shape [batch_size, seq_len] (True for padding)
        Returns:
            Tuple: (embedding, contrastive_embedding)
        """
        # Project and prepare for transformer
        x = self.building_projector(building_feature)  # [bs, seq_len, d_hidden]
        x = x.transpose(0, 1)  # [seq_len, bs, d_hidden]
        padding_mask = building_mask.bool()  # [bs, seq_len]

        # Single encoder
        encoded_feats = self.encoder(x, src_key_padding_mask=padding_mask)

        # Masked mean pooling
        def pool(feats, mask):
            feats = feats.transpose(0, 1)  # [bs, seq_len, d_hidden]
            feats = feats * (~mask).unsqueeze(-1)
            return feats.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1e-5)

        pooled = pool(encoded_feats, padding_mask)

        return self.proj_head(pooled), self.contrastive_proj(pooled)

