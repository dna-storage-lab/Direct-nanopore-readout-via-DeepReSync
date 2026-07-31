"""DeepResync neural posterior model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.coding import BLOCK_LENGTH, SPARSE_BITS_PER_SYMBOL, SPARSE_SYMBOL_COUNT
from ..utils.data import PAD_TOKEN_ID


MODEL_NAME = "deepresync"


class DeepResync(nn.Module):
    """Transformer model that predicts the direct 423-bit information posterior."""

    def __init__(
        self,
        block_len: int = BLOCK_LENGTH,
        max_read_len: int = 256,
        d_model: int = 128,
        nhead: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        symbol_hidden_size: int | None = None,
        symbol_layers: int = 1,
    ) -> None:
        super().__init__()
        if block_len != BLOCK_LENGTH:
            raise ValueError(f"Expected block_len={BLOCK_LENGTH}, got {block_len}.")
        if block_len % SPARSE_BITS_PER_SYMBOL != 0:
            raise ValueError("block_len must be divisible by SPARSE_BITS_PER_SYMBOL.")

        self.block_len = block_len
        self.max_read_len = max_read_len
        self.d_model = d_model

        self.read_embedding = nn.Embedding(5, d_model, padding_idx=PAD_TOKEN_ID)
        self.read_pos_embedding = nn.Embedding(max_read_len, d_model)
        self.query_pos_embedding = nn.Embedding(block_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(d_model, 4)

        symbol_hidden = d_model if symbol_hidden_size is None else int(symbol_hidden_size)
        self.symbol_group_projection = nn.Sequential(
            nn.LayerNorm(SPARSE_BITS_PER_SYMBOL * d_model),
            nn.Linear(SPARSE_BITS_PER_SYMBOL * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.symbol_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=symbol_hidden,
            num_layers=int(symbol_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if int(symbol_layers) > 1 else 0.0,
        )
        self.symbol_head = nn.Sequential(
            nn.LayerNorm(2 * symbol_hidden),
            nn.Linear(2 * symbol_hidden, 16),
        )
        self.lower_head = nn.Linear(d_model, 1)

    def decode_hidden(
        self,
        read_tokens: torch.Tensor,
        watermark_bits: torch.Tensor,
        read_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del watermark_bits
        batch_size, read_len = read_tokens.shape
        if read_len > self.max_read_len:
            raise ValueError(f"Read length {read_len} exceeds configured max_read_len {self.max_read_len}.")

        device = read_tokens.device
        if read_padding_mask is None:
            read_padding_mask = read_tokens.eq(PAD_TOKEN_ID)

        read_pos = torch.arange(read_len, device=device).unsqueeze(0).expand(batch_size, read_len)
        memory = self.read_embedding(read_tokens) + self.read_pos_embedding(read_pos)
        memory = self.encoder(memory, src_key_padding_mask=read_padding_mask)

        query_pos = torch.arange(self.block_len, device=device).unsqueeze(0).expand(batch_size, self.block_len)
        query = self.query_pos_embedding(query_pos)
        return self.decoder(query, memory, memory_key_padding_mask=read_padding_mask)

    def forward(
        self,
        read_tokens: torch.Tensor,
        watermark_bits: torch.Tensor,
        read_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        decoded = self.decode_hidden(read_tokens, watermark_bits, read_padding_mask)

        batch_size = decoded.shape[0]
        grouped = decoded.reshape(batch_size, SPARSE_SYMBOL_COUNT, SPARSE_BITS_PER_SYMBOL * self.d_model)
        symbol_features = self.symbol_group_projection(grouped)
        symbol_context, _ = self.symbol_lstm(symbol_features)

        return {
            "base_logits": self.output(decoded),
            "symbol_logits": self.symbol_head(symbol_context),
            "lower_bit_logits": self.lower_head(decoded).squeeze(-1),
        }
