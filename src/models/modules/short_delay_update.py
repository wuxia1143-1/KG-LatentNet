from __future__ import annotations

import torch
from torch import nn


class ShortDelayUpdate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, sequence: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(sequence)
        lengths = sequence_mask.long().sum(dim=1)
        safe_lengths = torch.clamp(lengths, min=1)
        gather_idx = (safe_lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        last = output.gather(1, gather_idx).squeeze(1)
        return torch.where(lengths.view(-1, 1) > 0, last, torch.zeros_like(last))
