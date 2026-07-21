from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ansi_scaler.refiner.config import RefinerConfig


@dataclass
class RefinerOutput:
    glyph_logits: torch.Tensor
    foreground: torch.Tensor
    background: torch.Tensor
    background_logits: torch.Tensor


class LocalAnsiRefiner(nn.Module):
    def __init__(self, vocabulary_size: int, config: RefinerConfig) -> None:
        super().__init__()
        width = config.d_model
        self.glyph_embedding = nn.Embedding(vocabulary_size, width // 2)
        self.cell_projection = nn.Linear(width // 2 + 7, width)
        self.context_positions = nn.Parameter(torch.randn(32, width) * 0.02)
        context_layer = nn.TransformerEncoderLayer(
            width, config.heads, width * 4, config.dropout, batch_first=True, norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(context_layer, config.context_layers, enable_nested_tensor=False)
        self.metadata = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.output_queries = nn.Parameter(torch.randn(18, width) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            width, config.heads, width * 4, config.dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.glyph_head = nn.Linear(width, vocabulary_size)
        self.foreground_head = nn.Linear(width, 3)
        self.background_head = nn.Linear(width, 3)
        self.background_present_head = nn.Linear(width, 1)

    def forward(
        self,
        glyphs: torch.Tensor,
        foreground: torch.Tensor,
        background: torch.Tensor,
        background_present: torch.Tensor,
        metadata: torch.Tensor,
    ) -> RefinerOutput:
        batch = glyphs.shape[0]
        cells = torch.cat(
            [self.glyph_embedding(glyphs), foreground, background, background_present.unsqueeze(-1)], dim=-1
        )
        context = self.context_encoder(self.cell_projection(cells) + self.context_positions.unsqueeze(0))
        meta = self.metadata(metadata).unsqueeze(1)
        memory = torch.cat([context, meta], dim=1)
        queries = self.output_queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return RefinerOutput(
            glyph_logits=self.glyph_head(decoded),
            foreground=torch.sigmoid(self.foreground_head(decoded)),
            background=torch.sigmoid(self.background_head(decoded)),
            background_logits=self.background_present_head(decoded).squeeze(-1),
        )


def refiner_loss(
    output: RefinerOutput,
    target_glyphs: torch.Tensor,
    target_foreground: torch.Tensor,
    target_background: torch.Tensor,
    target_background_present: torch.Tensor,
    space_id: int,
    config: RefinerConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    glyph = F.cross_entropy(output.glyph_logits.transpose(1, 2), target_glyphs)
    present = F.binary_cross_entropy_with_logits(output.background_logits, target_background_present)
    foreground_mask = target_glyphs != space_id
    foreground = F.smooth_l1_loss(
        output.foreground[foreground_mask], target_foreground[foreground_mask]
    ) if foreground_mask.any() else output.foreground.sum() * 0
    background_mask = target_background_present.bool()
    background = F.smooth_l1_loss(
        output.background[background_mask], target_background[background_mask]
    ) if background_mask.any() else output.background.sum() * 0
    total = (
        glyph * config.glyph_loss_weight
        + foreground * config.foreground_loss_weight
        + background * config.background_loss_weight
        + present * config.presence_loss_weight
    )
    return total, {"loss": total, "glyph": glyph, "foreground": foreground, "background": background, "presence": present}
