import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int = 512,
        dropout_p: float = 0.1,
        learned: bool = False,
        device=None,  # Added device
        dtype=None,  # Added dtype
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout_p
        self.learned = learned
        self.dropout = nn.Dropout(p=dropout_p)

        factory_kwargs = {"device": device, "dtype": dtype}

        if self.learned:
            self.pos_embedding = nn.Embedding(max_seq_len, hidden_size, **factory_kwargs)
            # Optional: Initialize weights, though default nn.Embedding initialization is often fine.
            # self.pos_embedding.weight.data.normal_(mean=0.0, std=0.02)
        else:
            # Create a buffer for sinusoidal positional encodings
            # pe shape: (1, max_seq_len, hidden_size)
            pe = torch.zeros(1, max_seq_len, hidden_size, **factory_kwargs)

            # position shape: (max_seq_len, 1)
            # Calculations for sinusoidal encoding should ideally be done in float32 for precision,
            # then cast to the target dtype if needed. However, for simplicity and directness,
            # we can try to use the target device and a float dtype for calculation tensors.
            # If factory_kwargs['dtype'] is a float type, use it, else default to torch.float for calculations.
            calc_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float

            position = torch.arange(0, max_seq_len, device=device, dtype=calc_dtype).unsqueeze(1)

            # div_term shape: (hidden_size / 2)
            div_term_base = torch.arange(0, hidden_size, 2, device=device, dtype=calc_dtype)
            div_term = torch.exp(div_term_base * (-math.log(10000.0) / hidden_size))

            # Apply sin to even indices in the hidden_size dimension
            pe[0, :, 0::2] = torch.sin(position * div_term)
            # Apply cos to odd indices in the hidden_size dimension
            pe[0, :, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, hidden_size]
            start_pos: Initial position index for the sequence.
        """
        seq_len = x.size(1)
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence endpoint {start_pos + seq_len} exceeds maximum sequence length {self.max_seq_len}"
            )

        if self.learned:
            # Create position IDs [start_pos, ..., start_pos + seq_len - 1]
            pos_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_enc = self.pos_embedding(pos_ids)
            x = x + pos_enc
        else:
            # self.pe is [1, max_seq_len, hidden_size]
            x = x + self.pe[:, start_pos : start_pos + seq_len, :]

        return self.dropout(x)
