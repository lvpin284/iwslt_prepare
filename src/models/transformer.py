"""Transformer model for machine translation.

Implements the standard Transformer architecture from
"Attention Is All You Need" (Vaswani et al., 2017),
adapted for IWSLT machine translation tasks.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Standard Transformer encoder-decoder for sequence-to-sequence tasks.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Embedding / hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: Feed-forward inner dimension.
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encoding.
        pad_idx: Padding token index.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Xavier uniform initialization for all parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate a causal mask for the decoder."""
        return torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)

    def _make_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create a boolean padding mask (True = padded position)."""
        return tokens == self.pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: Source token ids, shape (batch, src_len).
            tgt: Target token ids, shape (batch, tgt_len).

        Returns:
            Logits of shape (batch, tgt_len, tgt_vocab_size).
        """
        # Masks
        src_key_padding_mask = self._make_pad_mask(src)
        tgt_key_padding_mask = self._make_pad_mask(tgt)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # Embeddings + positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer forward
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.output_projection(output)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """Greedy auto-regressive decoding.

        Args:
            src: Source token ids, shape (batch, src_len).
            bos_idx: Beginning-of-sentence token index.
            eos_idx: End-of-sentence token index.
            max_len: Maximum decoding length.

        Returns:
            Decoded token ids, shape (batch, decoded_len).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        src_key_padding_mask = self._make_pad_mask(src)
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Start with BOS
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1), device)
            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.output_projection(output[:, -1:, :])
            next_token = logits.argmax(dim=-1)  # (batch, 1)
            ys = torch.cat([ys, next_token], dim=1)

            finished |= next_token.squeeze(-1) == eos_idx
            if finished.all():
                break

        return ys
