import torch
from torch import nn

from llm.utils.common import get_activation_layer


class MLP(nn.Module):
    """
    Pre-LN Multi-Layer Perceptron (MLP) implementation, commonly used in
    the Feed-Forward Network (FFN) part of Transformer architectures.

    Applies Layer Normalization before the MLP layers (Pre-LN).

    Args:
        hidden_size: Input and output hidden dimension size.
        intermediate_size: Dimension size of the intermediate layer. Defaults to 4 * hidden_size.
        activation: Activation function. Can be "gelu", "relu", or any nn.Module activation layer.
        dropout_p: Dropout probability. Defaults to 0.1.
        layer_norm_eps: Epsilon value for Layer Normalization. Defaults to 1e-5.
        bias: Whether to use bias in the linear layers. Defaults to True.
        use_layer_norm: Whether to use Layer Normalization at the beginning. Defaults to True.
        device: Device for the model.
        dtype: Data type for the model parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout_p: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        use_layer_norm: bool = True,  # This now controls the Pre-LN
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.use_layer_norm = use_layer_norm

        # Layer Normalization (applied first in forward pass if use_layer_norm is True)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # First linear layer: hidden_size -> intermediate_size
        self.fc1 = nn.Linear(hidden_size, self.intermediate_size, bias=bias, device=device, dtype=dtype)

        # Activation function
        if isinstance(activation, str):
            self.activation = get_activation_layer(activation)()
        else:
            self.activation = activation

        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)

        # Second linear layer: intermediate_size -> hidden_size
        self.fc2 = nn.Linear(self.intermediate_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the MLP."""
        # Use a slightly more modern default initialization if possible, but Xavier is fine.
        # Consider Kaiming He initialization if using ReLU variants.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Pre-Layer Normalization.

        Args:
            hidden_states: Input tensor, shape [..., hidden_size].

        Returns:
            Output tensor with the same shape as the input.
        """
        residual = hidden_states

        # 1. Apply Layer Normalization (Pre-LN) if enabled
        # The normalization happens *before* the main MLP transformation.
        if self.use_layer_norm:
            normalized_states = self.layer_norm(hidden_states)
        else:
            # If LayerNorm is disabled, the MLP still processes the original input.
            # The residual connection remains unchanged.
            normalized_states = hidden_states

        # 2. MLP Core Transformation
        intermediate_states = self.fc1(normalized_states)
        intermediate_states = self.activation(intermediate_states)
        intermediate_states = self.dropout(intermediate_states)
        output_states = self.fc2(intermediate_states)

        # 3. Add residual connection
        # The output of the MLP block is added back to the *original* input.
        output = residual + output_states

        return output


# Potential Future Enhancements:
# - Consider adding support for GLU variants (SwiGLU, GeGLU) via configuration.
# - Explore torch.compile for potential performance gains at the model level.
# - Evaluate different weight initialization strategies if needed.
