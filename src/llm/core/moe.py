import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    混合专家模型层 (Mixture of Experts Layer)

    该层将输入路由到多个“专家”网络（通常是FFN）中的一部分，并对它们的输出进行加权组合。
    这种方法可以在增加模型参数容量的同时，保持每个输入的计算成本相对较低。

    参数:
        hidden_size (int): 输入和输出特征的维度
        num_experts (int): 专家网络的总数量。
        ffn_hidden_size (int, 可选): 每个专家网络中前馈网络 (FFN) 的中间层维度。默认为 hidden_size 的 4 倍
        top_k (int, 可选): 每个 token 被路由到的专家数量。默认为 1。
        router_jitter_noise (float, 可选): 添加到路由器 logits 的抖动噪声的标准差。
                                         仅在训练时使用，有助于负载均衡和防止专家特化过度。默认为 0.01。
        router_bias (bool, 可选): 路由器线性层是否使用偏置。默认为 False。
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        ffn_hidden_size: int | None = None,
        top_k: int = 1,
        router_jitter_noise: float = 0.01,
        router_bias: bool = False,
    ):
        super().__init__()

        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) 不能大于 num_experts ({num_experts})")
        if hidden_size <= 0 or num_experts * top_k <= 0:
            raise ValueError("hidden_size, num_experts, 和 top_k 必须是正整数")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ffn_hidden_size = ffn_hidden_size if ffn_hidden_size is not None else 4 * hidden_size
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise

        # 1. 路由器 (Gating Network)
        # 路由器的作用是为每个输入token计算分配给各个专家的权重（或logits）
        # 输入: token的隐藏状态 (hidden_size)
        # 输出: 每个专家的logit分数 (num_experts)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=router_bias)

        # 2. 专家网络 (Expert Networks)
        # 每个专家通常是一个标准的前馈网络 (FFN)
        # FFN: Linear -> Activation -> Linear
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.ffn_hidden_size),
                    nn.GELU(),  # GELU 是现代 Transformer 中常用的激活函数
                    nn.Linear(self.ffn_hidden_size, self.hidden_size),
                )
                for _ in range(self.num_experts)
            ]
        )

    def _add_router_jitter(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        在训练期间向路由器 logits 添加抖动噪声。
        这有助于专家之间的负载均衡，并可以提高模型的泛化能力。
        参考: "Designing Effective Sparse Expert Models" (https://arxiv.org/abs/2401.00954)
              "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (https://arxiv.org/abs/2101.03961)
        """
        if self.router_jitter_noise > 0.0 and self.training:
            # torch.rand_like 生成 [0, 1) 的均匀分布噪声
            # 乘以 2.0 再减去 1.0，将其转换为 [-1, 1) 的均匀分布噪声 (近似)
            # 或者直接使用标准正态分布噪声 torch.randn_like
            # 这里我们使用均匀噪声乘以 router_jitter_noise
            noise = torch.rand_like(router_logits) * self.router_jitter_noise
            # 为了使其均值为0，可以乘以 2 再减 1，然后乘以 noise_std
            # noise = (torch.rand_like(router_logits) * 2.0 - 1.0) * self.router_jitter_noise
            # 原始代码是直接加正向噪声，这里保持一致性，但通常会用均值为0的噪声
            router_logits = router_logits + noise
        return router_logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE层的前向传播。

        参数:
            hidden_states (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_size]

        返回:
            output (torch.Tensor): 输出张量，形状与输入相同 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 为了方便处理，将输入reshape为 [batch_size * seq_len, hidden_size]
        # 这样每个token都可以独立地进行路由
        flat_hidden_states = hidden_states.reshape(-1, self.hidden_size)  # Shape: [N, H] where N = B * S

        # 1. 通过路由器计算每个token到各个专家的logits
        # router_logits: [N, num_experts]
        router_logits = self.router(flat_hidden_states)
        router_logits = self._add_router_jitter(router_logits)

        # 2. 为每个token选择top-k个专家，并计算它们的权重
        # torch.topk 返回 (values, indices)
        # top_k_logits: [N, top_k], 选出的top-k个专家的logits
        # top_k_indices: [N, top_k], 选出的top-k个专家的索引
        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)

        # 将top-k logits通过softmax转换为权重（概率分布）
        # routing_weights: [N, top_k]
        routing_weights = F.softmax(top_k_logits, dim=-1)

        # 3. 将token分发给选中的专家并计算输出
        # 初始化最终输出张量
        output = torch.zeros_like(flat_hidden_states)  # Shape: [N, H]

        # 遍历top_k中的每一个选择（例如，第1选择的专家，第2选择的专家...）
        for k_idx in range(self.top_k):
            expert_indices_kth_choice = top_k_indices[:, k_idx]  # Shape: [N], 第k个选择的专家索引
            weights_kth_choice = routing_weights[:, k_idx]  # Shape: [N], 第k个选择的路由权重

            # 遍历所有专家
            for expert_id in range(self.num_experts):
                # 找到当前遍历到的expert_id被选为第k选择的所有token
                # token_mask: [N], bool张量，标记哪些token的第k选择是当前expert_id
                token_mask_for_expert = expert_indices_kth_choice == expert_id

                # 获取这些token的实际索引
                # tokens_to_process_indices: [num_tokens_for_this_expert_at_kth_choice]
                tokens_to_process_indices = torch.where(token_mask_for_expert)[0]

                if tokens_to_process_indices.numel() == 0:
                    # 如果没有token路由到这个专家（作为其第k选择），则跳过
                    continue

                # 从flat_hidden_states中选出这些token
                # tokens_for_expert: [num_tokens_for_this_expert_at_kth_choice, hidden_size]
                tokens_for_expert = flat_hidden_states[tokens_to_process_indices]

                # 获取这些token对应的路由权重
                # weights_for_tokens: [num_tokens_for_this_expert_at_kth_choice]
                weights_for_tokens = weights_kth_choice[tokens_to_process_indices]

                # 通过专家网络处理
                expert_output = self.experts[expert_id](tokens_for_expert)
                # expert_output: [num_tokens_for_this_expert_at_kth_choice, hidden_size]

                # 将专家输出乘以对应的权重
                # weights_for_tokens.unsqueeze(1) -> [num_tokens_for_this_expert_at_kth_choice, 1]
                # 以便与expert_output进行广播乘法
                weighted_expert_output = expert_output * weights_for_tokens.unsqueeze(1)

                # 使用 index_add_ 将加权输出累加到 output 的对应位置
                # index_add_ (dim, index, tensor) 会将 tensor 中的值加到 self 张量在 dim 维度上由 index 指定的位置
                # 这里 dim=0 表示在第0维（token维度）上操作
                output.index_add_(0, tokens_to_process_indices, weighted_expert_output)

        # 将输出reshape回原始形状 [batch_size, seq_len, hidden_size]
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        return output


def moe_example():
    """
    MoE层使用示例
    """
    # 设置参数
    batch_size = 2
    seq_len = 10  # 序列长度
    hidden_size = 32  # 模型/隐藏层维度
    num_experts = 4  # 专家数量
    ffn_hidden_size = hidden_size * 4  # FFN中间层维度
    top_k = 2  # 每个token路由到2个专家

    print("--- MoE Layer Demo ---")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Hidden size: {hidden_size}")
    print(f"Number of experts: {num_experts}, FFN hidden size: {ffn_hidden_size}, Top-K: {top_k}")

    # 创建MoE层实例
    moe_layer = MoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        ffn_hidden_size=ffn_hidden_size,
        top_k=top_k,
        router_jitter_noise=0.01,  # 在训练时使用
    )
    # print("\nMoE Layer Structure:")
    # print(moe_layer)

    # 创建模拟输入数据
    # hidden_states: [batch_size, seq_len, hidden_size]
    input_tensor = torch.rand(batch_size, seq_len, hidden_size)
    print(f"\nInput tensor shape: {input_tensor.shape}")

    # 模拟训练模式
    moe_layer.train()
    output_train = moe_layer(input_tensor)
    print(f"Output tensor shape (train mode): {output_train.shape}")

    # 模拟评估模式 (无抖动噪声)
    moe_layer.eval()
    output_eval = moe_layer(input_tensor)
    print(f"Output tensor shape (eval mode): {output_eval.shape}")

    # 检查输出维度是否正确
    assert output_train.shape == input_tensor.shape, "训练模式下输出形状不匹配"
    assert output_eval.shape == input_tensor.shape, "评估模式下输出形状不匹配"

    print("\n--- MoE Layer Demo End ---")
    return output_train, output_eval


if __name__ == "__main__":
    moe_example()
