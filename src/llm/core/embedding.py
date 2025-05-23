import torch
import torch.nn as nn
import math

try:
    # Assuming 'llm' is in PYTHONPATH or installed
    from llm.core.positional_encoding import PositionalEncoding
except ModuleNotFoundError:
    # Fallback for local/CI execution where 'src' might be the root for Python
    try:
        from src.llm.core.positional_encoding import PositionalEncoding
    except ModuleNotFoundError:
        # Direct relative import if 'embedding.py' and 'positional_encoding.py' are siblings
        # This is less robust but can work in some structures.
        from .positional_encoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    """
    Combines token embeddings with positional encodings.

    The layer first embeds input token IDs into dense vectors, then scales these
    embeddings by the square root of the hidden size, and finally adds
    positional encodings.
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 max_seq_len: int = 512,
                 pos_encoding_learned: bool = False,
                 dropout_p: float = 0.1,
                 padding_idx: int | None = None,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None):
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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
            **factory_kwargs
        )

        # PositionalEncoding should be initialized with factory_kwargs
        # This assumes PositionalEncoding's __init__ is modified to accept and use **factory_kwargs
        # for its internal nn.Embedding if learned=True.
        self.positional_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout_p=dropout_p,
            learned=pos_encoding_learned,
            **factory_kwargs # Pass factory_kwargs here
        )
        # No explicit .to(device, dtype) for self.positional_encoding is needed here
        # if PositionalEncoding correctly uses factory_kwargs for its parameters (learned case)
        # and its buffers are handled by the parent module's .to() method (sinusoidal case).

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EmbeddingLayer.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Tensor of embeddings with positional encodings,
                          of shape (batch_size, seq_len, hidden_size).
        """
        token_embs = self.token_embeddings(input_ids)
        scaled_embs = token_embs * math.sqrt(self.hidden_size)
        output_embs = self.positional_encoding(scaled_embs)
        return output_embs

if __name__ == '__main__':
    # Example Usage
    # This requires PositionalEncoding to be updated to handle factory_kwargs in its __init__
    # specifically for its nn.Embedding when learned=True.
    # If PositionalEncoding's __init__ is:
    # def __init__(self, hidden_size, max_seq_len=512, dropout_p=0.1, learned=False, **kwargs_ignored_for_now):
    # Then the **factory_kwargs passed above will be ignored by PE for its own submodules.
    # It should be:
    # def __init__(self, ..., **factory_kwargs):
    #   if learned: self.pos_embedding = nn.Embedding(..., **factory_kwargs)

    # Assuming PositionalEncoding is correctly modified:
    vocab_size_ex = 1000
    hidden_size_ex = 512
    max_seq_len_ex = 100
    batch_size_ex = 4
    seq_len_ex = 50
    device_ex = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_ex = torch.float32

    print(f"Using device: {device_ex}, dtype: {dtype_ex}")
    print("Note: This example assumes PositionalEncoding's __init__ correctly handles factory_kwargs for its own submodules if any (e.g., learned nn.Embedding).")

    # Test with sinusoidal positional encoding
    print("\nTesting with Sinusoidal Positional Encoding...")
    embedding_layer_sin = EmbeddingLayer(
        vocab_size=vocab_size_ex, hidden_size=hidden_size_ex, max_seq_len=max_seq_len_ex,
        pos_encoding_learned=False, dropout_p=0.1, padding_idx=0,
        device=device_ex, dtype=dtype_ex
    )
    print(f"  Token embeddings weight device: {embedding_layer_sin.token_embeddings.weight.device}, dtype: {embedding_layer_sin.token_embeddings.weight.dtype}")
    # For sinusoidal PE, 'pe' is a buffer. Buffers are moved with the module.
    # The factory_kwargs passed to PE's init are not directly used for buffer creation,
    # but the module itself (PE) being on device/dtype means its buffers will be too.
    if hasattr(embedding_layer_sin.positional_encoding, 'pe'):
        print(f"  Positional encoding (sinusoidal) buffer 'pe' device: {embedding_layer_sin.positional_encoding.pe.device}, dtype: {embedding_layer_sin.positional_encoding.pe.dtype}")


    dummy_input_ids = torch.randint(0, vocab_size_ex, (batch_size_ex, seq_len_ex), device=device_ex, dtype=torch.long)
    dummy_input_ids[0, 0] = 0

    output_sin = embedding_layer_sin(dummy_input_ids)
    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    print(f"  Output tensor shape: {output_sin.shape}, device: {output_sin.device}, dtype: {output_sin.dtype}")
    assert output_sin.shape == (batch_size_ex, seq_len_ex, hidden_size_ex)
    assert str(output_sin.device) == str(device_ex) # Compare string representations for robustness
    assert output_sin.dtype == dtype_ex


    # Test with learned positional encoding
    print("\nTesting with Learned Positional Encoding...")
    embedding_layer_lrn = EmbeddingLayer(
        vocab_size=vocab_size_ex, hidden_size=hidden_size_ex, max_seq_len=max_seq_len_ex,
        pos_encoding_learned=True, dropout_p=0.1, padding_idx=0,
        device=device_ex, dtype=dtype_ex
    )
    print(f"  Token embeddings weight device: {embedding_layer_lrn.token_embeddings.weight.device}, dtype: {embedding_layer_lrn.token_embeddings.weight.dtype}")
    # For learned PE, 'pos_embedding' is an nn.Embedding. It should get factory_kwargs.
    if hasattr(embedding_layer_lrn.positional_encoding, 'pos_embedding'):
         print(f"  Positional encoding (learned) embedding weight device: {embedding_layer_lrn.positional_encoding.pos_embedding.weight.device}, dtype: {embedding_layer_lrn.positional_encoding.pos_embedding.weight.dtype}")


    output_lrn = embedding_layer_lrn(dummy_input_ids)
    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    print(f"  Output tensor shape: {output_lrn.shape}, device: {output_lrn.device}, dtype: {output_lrn.dtype}")
    assert output_lrn.shape == (batch_size_ex, seq_len_ex, hidden_size_ex)
    assert str(output_lrn.device) == str(device_ex)
    assert output_lrn.dtype == dtype_ex

    print("\nBasic __main__ tests passed (assuming PositionalEncoding handles factory_kwargs).")
