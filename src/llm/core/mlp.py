import math

import torch
from torch import nn

from llm.utils.common import get_activation_layer


class MLP(nn.Module):
    """
    MLP block for Transformer-style Feed-Forward Networks with flexible normalization.

    Args:
        hidden_size (int): Dimensionality of inputs and outputs.
        intermediate_size (int, optional): Dimensionality of the inner layer. Defaults to 4 * hidden_size.
        activation (str or nn.Module): Activation name or module. Defaults to "gelu".
        dropout_p (float): Dropout probability. Defaults to 0.1.
        norm_first (bool): Whether to apply normalization before or after (pre-LN vs post-LN). Defaults to True.
        norm_type (Type[nn.Module] or nn.Module): Normalization layer type or instance. Defaults to nn.LayerNorm.
        norm_eps (float): Epsilon for normalization layers. Defaults to 1e-5.
        bias (bool): Whether to include bias terms in Linear layers. Defaults to True.
        device (torch.device, optional): Device for parameters. Defaults to None.
        dtype (torch.dtype, optional): Dtype for parameters. Defaults to None.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout_p: float = 0.1,
        norm_first: bool = True,
        norm_type: type[nn.Module] | nn.Module = nn.LayerNorm,
        norm_eps: float = 1e-5,
        bias: bool = True,
        use_glu: bool = False,  # New parameter for SwiGLU/GLU support
        include_norm_residual: bool = True,  # New parameter
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        self.norm_first = norm_first
        self.use_glu = use_glu
        self.include_norm_residual = include_norm_residual

        self.norm = None
        if self.include_norm_residual:
            # Create normalization layer based on provided type or instance
            if isinstance(norm_type, type):
                # Specific handling for known norm types with eps
                if norm_type == nn.LayerNorm or norm_type == nn.RMSNorm:
                    self.norm = norm_type(hidden_size, eps=norm_eps, device=device, dtype=dtype)
                else:
                    # For other norm types that might have different init parameters
                    # This might need adjustment if they don't follow `norm(hidden_size, **kwargs)`
                    try:
                        self.norm = norm_type(hidden_size, device=device, dtype=dtype)
                    except TypeError:  # Fallback if eps is not accepted but common for other norms
                        self.norm = norm_type(hidden_size, eps=norm_eps, device=device, dtype=dtype)

            else:
                # If an instance is provided, use it directly and ensure it's on the
                # requested device/dtype (if provided). This prevents mismatch when
                # callers pass a pre-created normalization layer that defaults to CPU.
                self.norm = norm_type
                # Move provided norm instance to the same device/dtype as other params
                if device is not None or dtype is not None:
                    # .to() is in-place for nn.Module, but assign back in case it returns a new object
                    try:
                        moved_norm = self.norm.to(device=device, dtype=dtype)
                        # If .to() returned a new module, keep that reference
                        self.norm = moved_norm
                    except Exception:
                        # If moving fails for some custom module, ignore and let the caller
                        # be responsible for device/dtype placement.
                        pass

        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        if self.use_glu:
            self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(self.intermediate_size, hidden_size, bias=bias, **factory_kwargs)

        # Determine activation module and name
        if isinstance(activation, str):
            self.activation_name = activation.lower()
            self.activation = get_activation_layer(self.activation_name)()
        else:
            self.activation = activation
            self.activation_name = activation.__class__.__name__.lower()

        self.dropout = nn.Dropout(dropout_p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the MLP based on the activation."""
        act = self.activation_name
        neg_slope = getattr(self.activation, "negative_slope", 0.0)

        # Dynamic weight initialization
        if act in ("relu", "leaky_relu"):
            # He initialization for ReLU variants
            nn.init.kaiming_uniform_(self.fc1.weight, a=neg_slope, nonlinearity=act)
            if self.use_glu:
                nn.init.kaiming_uniform_(self.gate_proj.weight, a=neg_slope, nonlinearity=act)
            nn.init.kaiming_uniform_(self.fc2.weight, a=neg_slope, nonlinearity=act)
        elif act in ("gelu", "silu", "swish"):
            # Truncated normal for smoother activations
            std1 = 1.0 / math.sqrt(self.hidden_size)
            std2 = 1.0 / math.sqrt(self.intermediate_size)
            try:
                nn.init.trunc_normal_(self.fc1.weight, std=std1)
                if self.use_glu:
                    nn.init.trunc_normal_(self.gate_proj.weight, std=std1)
                nn.init.trunc_normal_(self.fc2.weight, std=std2)
            except AttributeError:
                nn.init.normal_(self.fc1.weight, mean=0.0, std=std1)
                if self.use_glu:
                    nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std1)
                nn.init.normal_(self.fc2.weight, mean=0.0, std=std2)
        else:
            # Default Xavier/Glorot
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

        # Zero out biases for stable training
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.use_glu and self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional pre-LayerNorm and residual connection.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [..., hidden_size].

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        if self.include_norm_residual:
            residual = hidden_states

            # Apply normalization first if using pre-norm
            # Ensure self.norm exists before calling it
            x = self.norm(hidden_states) if self.norm_first and self.norm else hidden_states

            # MLP computation (common for both pre-norm and post-norm)
            if self.use_glu:
                # GLU logic
                x_fc1 = self.fc1(x)
                x_fc1 = self.activation(x_fc1)
                x_gate = self.gate_proj(x)
                x_mlp = x_fc1 * x_gate
            else:
                x_mlp = self.fc1(x)
                x_mlp = self.activation(x_mlp)
            x_mlp = self.dropout(x_mlp)
            x_mlp = self.fc2(x_mlp)

            # Add residual connection
            x_mlp = residual + x_mlp

            # Apply normalization after if using post-norm
            # Ensure self.norm exists before calling it
            output = x_mlp if self.norm_first or not self.norm else self.norm(x_mlp)
            return output
        else:
            # No internal norm or residual connection
            x = hidden_states  # Direct input to MLP
            if self.use_glu:
                # GLU logic: (x * activation(gate(x)))
                # For SwiGLU, activation is SiLU
                x_fc1 = self.fc1(x)
                x_fc1 = self.activation(x_fc1)
                x_gate = self.gate_proj(x)
                x = x_fc1 * x_gate
            else:
                x = self.fc1(x)
                x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
