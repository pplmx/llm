import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.core.mlp import MLP  # Reusing the MLP for experts


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) layer.

    Args:
        hidden_size (int): The dimensionality of the input and output.
        num_experts (int): The total number of experts.
        top_k (int): The number of top experts to select for each token.
        intermediate_size (int, optional): The intermediate size for each expert's MLP.
                                           Defaults to 4 * hidden_size.
        activation (str or nn.Module): Activation function for experts. Defaults to "gelu".
        dropout_p (float): Dropout probability for experts. Defaults to 0.1.
        norm_type (Type[nn.Module] or nn.Module): Normalization layer type for experts. Defaults to nn.LayerNorm.
        norm_eps (float): Epsilon for normalization layers in experts. Defaults to 1e-5.
        bias (bool): Whether to include bias terms in Linear layers of experts. Defaults to True.
        device (torch.device, optional): Device for parameters. Defaults to None.
        dtype (torch.dtype, optional): Dtype for parameters. Defaults to None.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout_p: float = 0.1,
        norm_type: type[nn.Module] | nn.Module = nn.LayerNorm,
        norm_eps: float = 1e-5,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})")
        if top_k == 0:
            raise ValueError("top_k must be at least 1")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        factory_kwargs = {"device": device, "dtype": dtype}

        # Gating network (router)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, **factory_kwargs)

        # Experts (always using MLP)
        self.experts = nn.ModuleList(
            [
                MLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    activation=activation,
                    dropout_p=dropout_p,
                    norm_first=False,  # MoE typically handles norm/residual externally
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    bias=bias,
                    include_norm_residual=False,  # Experts are simple MLPs, norm/residual handled by TransformerBlock
                    **factory_kwargs,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [..., hidden_size].

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        original_shape = x.shape
        x = x.view(-1, self.hidden_size)  # Flatten to [batch_size * seq_len, hidden_size]

        # 1. Gating network to get expert scores
        gate_logits = self.gate(x)  # [batch_size * seq_len, num_experts]

        # 2. Select top-k experts
        # top_k_logits: [batch_size * seq_len, top_k]
        # top_k_indices: [batch_size * seq_len, top_k]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # 3. Apply softmax to get weights for selected experts
        # expert_weights: [batch_size * seq_len, top_k]
        expert_weights = F.softmax(top_k_logits, dim=-1, dtype=x.dtype)

        # Initialize output tensor
        output = torch.zeros_like(x)  # [batch_size * seq_len, hidden_size]

        # Create a list of lists, where each inner list contains the indices
        # of tokens routed to that expert.
        expert_inputs = [[] for _ in range(self.num_experts)]
        expert_weights_per_token = [[] for _ in range(self.num_experts)]
        expert_original_indices = [[] for _ in range(self.num_experts)]

        for i in range(x.size(0)):  # Iterate over each token
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[i, k_idx].item()
                expert_inputs[expert_idx].append(x[i])
                expert_weights_per_token[expert_idx].append(expert_weights[i, k_idx])
                expert_original_indices[expert_idx].append(i)

        # Process each expert
        for i, expert in enumerate(self.experts):
            if expert_inputs[i]:
                expert_input_batch = torch.stack(expert_inputs[i])
                expert_output_batch = expert(expert_input_batch)
                expert_weights_batch = torch.stack(expert_weights_per_token[i]).unsqueeze(-1)
                # Weighted sum and scatter
                weighted_expert_output = expert_output_batch * expert_weights_batch
                output.index_add_(0, torch.tensor(expert_original_indices[i], device=x.device), weighted_expert_output)

        return output.view(original_shape)


if __name__ == "__main__":
    # Example Usage
    device_ex = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_ex = torch.float32

    batch_size_ex = 2
    seq_len_ex = 10
    hidden_size_ex = 64
    num_experts_ex = 4
    top_k_ex = 2

    print(f"MoE Example running on device: {device_ex}, dtype: {dtype_ex}")

    # Dummy input
    dummy_input = torch.randn(batch_size_ex, seq_len_ex, hidden_size_ex, device=device_ex, dtype=dtype_ex)

    # Instantiate MoE layer
    moe_layer = MoE(
        hidden_size=hidden_size_ex,
        num_experts=num_experts_ex,
        top_k=top_k_ex,
        device=device_ex,
        dtype=dtype_ex,
    )
    moe_layer.eval()  # Set to eval mode for consistent dropout behavior if any

    # Forward pass
    output = moe_layer(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape mismatch!"

    print("\nAll basic __main__ tests passed for MoE.")
