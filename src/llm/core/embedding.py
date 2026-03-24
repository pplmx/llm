import math

import torch
import torch.nn as nn

from llm.core.positional_encoding import PositionalEncoding
from llm.utils.common import make_factory_kwargs


class EmbeddingLayer(nn.Module):
    """
    Combines token embeddings with positional encodings.

    The layer first embeds input token IDs into dense vectors, then scales these
    embeddings by the square root of the hidden size, and finally adds
    positional encodings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int = 512,
        pos_encoding_learned: bool = False,
        dropout_p: float = 0.1,
        padding_idx: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initializes the EmbeddingLayer.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The embedding dimension.
            max_seq_len (int, default=512): Maximum sequence length for positional encoding.
            pos_encoding_learned (bool, default=False): If True, use learned positional embeddings.
                                                       If False, use sinusoidal.
            dropout_p (float, default=0.1): Dropout probability for positional encoding.
            padding_idx (int, optional, default=None): If specified, the entries at `padding_idx`
                                                       in `token_embeddings` do not contribute to
                                                       the gradient; furthermore, the embedding vector
                                                       for `padding_idx` is initialized to all zeros.
            device (torch.device | str | None, default=None): Target device for the layers.
            dtype (torch.dtype | None, default=None): Target data type for the layers.
        """
        factory_kwargs = make_factory_kwargs(device, dtype)
        super().__init__()

        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=padding_idx, **factory_kwargs
        )

        # PositionalEncoding should be initialized with factory_kwargs
        # This assumes PositionalEncoding's __init__ is modified to accept and use **factory_kwargs
        # for its internal nn.Embedding if learned=True.
        self.positional_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout_p=dropout_p,
            learned=pos_encoding_learned,
            **factory_kwargs,  # Pass factory_kwargs here
        )
        # No explicit .to(device, dtype) for self.positional_encoding is needed here
        # if PositionalEncoding correctly uses factory_kwargs for its parameters (learned case)
        # and its buffers are handled by the parent module's .to() method (sinusoidal case).

    def forward(
        self, input_ids: torch.Tensor, start_pos: int = 0, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the EmbeddingLayer.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape (batch_size, seq_len).
            start_pos (int): Initial position index for the sequence.
            position_ids (torch.Tensor, optional): Explicit position IDs.

        Returns:
            torch.Tensor: Tensor of embeddings with positional encodings,
                          of shape (batch_size, seq_len, hidden_size).
        """
        token_embs = self.token_embeddings(input_ids)
        scaled_embs = token_embs * math.sqrt(self.hidden_size)
        output_embs = self.positional_encoding(scaled_embs, start_pos=start_pos, position_ids=position_ids)
        return output_embs
