"""Custom implementation of a lightweight autoregressive Transformer
decoder-only network.  This module purposefully avoids using the high-level
``torch.nn.TransformerDecoder`` convenience wrapper so that we have full
control over the architecture – useful for experimenting with alternative
building blocks such as the ones used in *Qwen3* – while still keeping the
external interface identical to :class:`VanillaTransformerDecoderAR` that
existing code relies on.

Only the pieces that are required by our unit-tests have been implemented:

* token & positional embeddings (learnable)
* stack of *N* decoder layers consisting of
  - causal self-attention
  - optional encoder–decoder (cross) attention if *memory* is supplied
  - feed-forward network (GeLU)
  - residual connections & layer normalisation

The forward signature matches that of ``torch.nn.TransformerDecoder`` so that
this implementation can be dropped in without touching any other code.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "CustomTransformerDecoderLayer",
    "CustomTransformerDecoder",
]


class CustomTransformerDecoderLayer(nn.Module):
    """A single decoder layer consisting of self-attention, optional cross
    attention and a feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Modules -----------------------------------------------------------
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = nn.GELU()

    # ---------------------------------------------------------------------
    def _sa_block(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention block with residual & norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout(attn_output)
        return self.norm1(x)

    def _ca_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Cross-attention block with residual & norm
        attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(attn_output)
        return self.norm2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # Feed-forward network with residual & norm
        y = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        x = x + self.dropout(y)
        return self.norm3(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Decoder input ``(B, T, d_model)``.
            memory: Encoder outputs ``(B, S, d_model)``.  If *None* a zero
                vector of length 1 is used so that the dimensionalities are
                always well-defined while effectively disabling cross-attention.
            tgt_mask: Causal mask for self-attention (True denotes *masked*).
            memory_mask: Optional mask for cross-attention.
        """

        x = self._sa_block(x, tgt_mask)

        if memory is None:
            # (B, 1, d_model)
            memory = torch.zeros(x.size(0), 1, x.size(-1), device=x.device, dtype=x.dtype)

        x = self._ca_block(x, memory, memory_mask)
        x = self._ff_block(x)
        return x


class CustomTransformerDecoder(nn.Module):
    """Stack of :class:`CustomTransformerDecoderLayer`.  The API mirrors
    ``torch.nn.TransformerDecoder`` but only implements the subset required by
    our codebase.
    """

    def __init__(
        self,
        decoder_layer: CustomTransformerDecoderLayer,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [decoder_layer if i == 0 else self._clone_layer(decoder_layer) for i in range(num_layers)]
        )

    @staticmethod
    def _clone_layer(layer: CustomTransformerDecoderLayer) -> CustomTransformerDecoderLayer:
        # A tiny utility to deep-copy a prototype layer without the heavy
        # `copy.deepcopy` to avoid re-initialising weights.
        import copy

        return copy.deepcopy(layer)

    # ------------------------------------------------------------------
    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output
