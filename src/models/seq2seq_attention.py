"""Seq2Seq model with Bahdanau (additive) attention.

A simpler baseline compared to the Transformer, useful for
quick experimentation and comparison on IWSLT tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Bidirectional GRU encoder."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source sequence.

        Args:
            src: (batch, src_len) token ids.

        Returns:
            encoder_outputs: (batch, src_len, hidden_dim * 2)
            hidden: (num_layers, batch, hidden_dim) — last hidden state
                    projected from bidirectional to unidirectional.
        """
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden: (num_layers * 2, batch, hidden_dim) -> merge directions
        # Take the last layer's forward and backward hidden states
        num_layers = hidden.size(0) // 2
        hidden_fwd = hidden[2 * torch.arange(num_layers)]  # (num_layers, batch, hidden_dim)
        hidden_bwd = hidden[2 * torch.arange(num_layers) + 1]
        hidden = torch.tanh(
            self.fc_hidden(torch.cat([hidden_fwd, hidden_bwd], dim=-1))
        )
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector.

        Args:
            decoder_hidden: (batch, decoder_dim)
            encoder_outputs: (batch, src_len, encoder_dim)

        Returns:
            context: (batch, encoder_dim)
            attn_weights: (batch, src_len)
        """
        # (batch, 1, attention_dim) + (batch, src_len, attention_dim)
        score = self.v(
            torch.tanh(
                self.W_dec(decoder_hidden).unsqueeze(1) + self.W_enc(encoder_outputs)
            )
        ).squeeze(-1)  # (batch, src_len)

        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """GRU decoder with Bahdanau attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_dim: int,
        attention_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        self.rnn = nn.GRU(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a single time-step.

        Args:
            tgt_token: (batch, 1) current target token.
            hidden: (num_layers, batch, hidden_dim) decoder hidden state.
            encoder_outputs: (batch, src_len, encoder_dim)

        Returns:
            logits: (batch, vocab_size)
            hidden: updated hidden state.
            attn_weights: (batch, src_len)
        """
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed_dim)
        # Use the top-layer hidden state for attention
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)

        logits = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        )
        return logits, hidden, attn_weights


class Seq2SeqAttention(nn.Module):
    """Complete Seq2Seq model with attention for IWSLT MT tasks.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension for encoder/decoder.
        attention_dim: Attention projection dimension.
        num_layers: Number of RNN layers.
        dropout: Dropout probability.
        pad_idx: Padding token index.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        encoder_dim = hidden_dim * 2  # bidirectional
        self.encoder = Encoder(
            src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            embed_dim,
            hidden_dim,
            encoder_dim,
            attention_dim,
            num_layers,
            dropout,
            pad_idx,
        )
        self.pad_idx = pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Args:
            src: (batch, src_len) source token ids.
            tgt: (batch, tgt_len) target token ids.
            teacher_forcing_ratio: Probability of using ground-truth token
                as next decoder input during training.

        Returns:
            outputs: (batch, tgt_len - 1, tgt_vocab_size) logits.
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len - 1, tgt_vocab_size, device=src.device)
        encoder_outputs, hidden = self.encoder(src)

        # First input is BOS token
        decoder_input = tgt[:, 0:1]

        for t in range(tgt_len - 1):
            logits, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = logits

            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t + 1 : t + 2]
            else:
                decoder_input = logits.argmax(dim=-1, keepdim=True)

        return outputs

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
            src: (batch, src_len) source token ids.
            bos_idx: Beginning-of-sentence token index.
            eos_idx: End-of-sentence token index.
            max_len: Maximum decoding length.

        Returns:
            Decoded token ids, shape (batch, decoded_len).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        encoder_outputs, hidden = self.encoder(src)
        decoder_input = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        all_tokens = [decoder_input]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            next_token = logits.argmax(dim=-1, keepdim=True)
            all_tokens.append(next_token)
            decoder_input = next_token

            finished |= next_token.squeeze(-1) == eos_idx
            if finished.all():
                break

        return torch.cat(all_tokens, dim=1)
