import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFN(nn.Module):
    """
    一个标准的 Feed-Forward Network (FFN)，也称为多层感知机 (MLP)。
    这是 Transformer 模块中的一个基本构建块。
    结构: Linear -> Activation -> Linear
    """

    def __init__(self, hidden_size):
        super().__init__()
        ffn_hidden_size = 4 * hidden_size  # 这是一个常见的扩展因子
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        # 注释: 现代LLM中，ReLU常被GELU或SwiGLU/SiLU等更复杂的激活函数替代，以获得更好的性能。
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        print("\n" + "=" * 20 + " SimpleFFN forward " + "=" * 20)
        print(f"Input x shape: {x.shape}")
        print("-" * 60)

        x = self.fc1(x)
        x = self.activation(x)
        print(f"After fc1 + Activation shape: {x.shape}")

        x = self.fc2(x)
        print(f"After fc2 shape: {x.shape}")
        print("=" * 64)
        return x


class SimpleMoE(nn.Module):
    """
    Mixture-of-Experts (MoE) 模块。
    它用多个稀疏激活的 "Expert" FFNs 来替代单个密集的 FFN。
    核心思想: 增加模型参数量，但不成比例地增加计算量。
    """

    def __init__(self, hidden_size, num_experts=4, k=2):
        super().__init__()
        # 防御性编程: 确保 k (选择的 expert 数量) 不超过总 expert 数量
        assert k <= num_experts, "k must be less than or equal to num_experts"

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k

        ffn_hidden_size = 4 * hidden_size

        # Experts 是 num_experts 个独立的 FFN
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, ffn_hidden_size),
                    nn.ReLU(),  # 每个 expert 内部使用 ReLU 激活
                    nn.Linear(ffn_hidden_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network: 一个简单的线性层，用于决定每个 token 该由哪些 experts 处理
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        batch_size, hidden_dim = x.shape

        print("\n" + "=" * 20 + " MoE Forward Pass Breakdown " + "=" * 20)
        print(f"Input x shape: {x.shape}")
        print("-" * 60)

        # --- 1. Gating: 为每个 Token 选择最合适的 Experts ---
        print("--- 1. Gating: Selecting Experts ---")
        gate_scores = self.gate(x)
        # 注释: 在混合精度训练中，门控分数和后续的softmax通常会转换到float32以保证数值稳定性。

        # 找到分数最高的 top-k 个 experts 及其索引
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)

        # 对 top-k 的分数应用 softmax，得到权重
        # 注释: 这是现代MoE（如Mixtral）的常见做法。它比对所有expert分数做softmax更高效。
        topk_weights = F.softmax(topk_scores, dim=-1)

        print("Gating decision for each token:")
        for i in range(batch_size):
            formatted_weights = [f"{w:.2f}" for w in topk_weights[i].tolist()]
            print(f"  Token {i} -> Chosen Experts: {topk_indices[i].tolist()}, Weights: {formatted_weights}")
        print("-" * 60)

        # --- 2. Routing & Computation: 将 Token 分发给 Expert 并行计算 ---
        print("--- 2. Routing & Computation (Scatter-Gather) ---")
        # 创建一个扁平化的张量，用于收集所有 expert 的输出
        # 形状为 [batch_size * k, hidden_size]，为每个 token-expert 对准备了位置
        dispatched_x = torch.zeros(batch_size * self.k, hidden_dim, device=x.device, dtype=x.dtype)

        # 将 top-k 索引扁平化，得到一个长列表，指明每个计算任务应由哪个 expert 处理
        flat_topk_indices = topk_indices.view(-1)

        # 将输入 token 复制 k 次，因为每个 token 都需要被 k 个 expert 处理
        repeated_x = x.repeat_interleave(self.k, dim=0)

        # 以 Expert 为中心进行计算 (Gather & Compute)
        for i in range(self.num_experts):
            # 找到所有需要由当前 expert i 处理的 token 的位置
            mask = flat_topk_indices == i
            if not mask.any():
                continue

            # 获取这些 token 的输入
            tokens_for_expert = repeated_x[mask]
            print(f"Expert #{i} processing {tokens_for_expert.shape[0]} token(s).")

            # 批量计算
            expert_output = self.experts[i](tokens_for_expert)

            # (Scatter back) 将计算结果放回 dispatched_x 的相应位置
            dispatched_x = dispatched_x.masked_scatter(mask.unsqueeze(-1), expert_output)

        print("-" * 60)

        # --- 3. Aggregation: 加权合并 Expert 的输出 ---
        print("--- 3. Aggregation: Combining Outputs ---")

        # 1. 重塑 expert 输出，使其与 token 和 k 值对应
        #    [batch_size * k, hidden_size] -> [batch_size, k, hidden_size]
        outputs_per_expert = dispatched_x.view(batch_size, self.k, hidden_dim)

        # 2. 扩展权重张量以便进行广播乘法
        #    [batch_size, k] -> [batch_size, k, 1]
        weights_expanded = topk_weights.unsqueeze(-1)

        # 3. 对每个 expert 的输出进行加权
        weighted_outputs = outputs_per_expert * weights_expanded

        # 4. 将 k 个加权后的 expert 输出相加，得到最终结果
        final_output = weighted_outputs.sum(dim=1)

        print("\n" + "=" * 20 + " MoE Final Result " + "=" * 20)
        print(f"Final output shape: {final_output.shape}")
        print("=" * 64)

        return final_output


if __name__ == "__main__":
    # --- 配置 ---
    # 使用小的、固定的参数，让输出清晰且可复现
    batch_size_ = 2
    hidden_size_ = 8
    num_experts_ = 4
    k_ = 2

    # 设置随机种子以确保每次运行结果一致，便于分析
    torch.manual_seed(42)

    # --- 数据准备 ---
    input_tensor = torch.randn(batch_size_, hidden_size_)
    print("--- Input Data ---")
    print(f"Input shape: {input_tensor.shape}")
    print("-" * 20)

    # --- 模型实例化 ---
    ffn = SimpleFFN(hidden_size_)
    moe = SimpleMoE(hidden_size_, num_experts=num_experts_, k=k_)

    # --- FFN 前向传播 ---
    out_ffn = ffn(input_tensor)

    # --- MoE 前向传播 ---
    out_moe = moe(input_tensor)

    # --- 核心对比: 参数量 vs 计算 ---
    # 这是理解 MoE 价值的关键部分
    print("\n\n" + "=" * 20 + " FFN vs. MoE: Core Comparison " + "=" * 20)

    # 计算参数量
    ffn_params = sum(p.numel() for p in ffn.parameters() if p.requires_grad)
    moe_params = sum(p.numel() for p in moe.parameters() if p.requires_grad)

    print(f"Total parameters in dense FFN: {ffn_params}")
    print(f"Total parameters in MoE:       {moe_params}")
    # MoE的参数量主要由 experts 贡献，外加一个小的 gate 网络。
    # expert部分的参数量约等于 FFN 的 num_experts 倍。
    print(f"-> MoE has approximately {moe_params / ffn_params:.2f}x more parameters than FFN.")

    print("\nKey Insight:")
    print(
        f"Despite having far more parameters, the MoE model only activates {k_} out of {num_experts_} experts per token."
    )
    print("This means: Model 'knows' more (more parameters), but 'thinks' less (less computation) for each token.")
    print("This is the trade-off that allows scaling models to trillions of parameters.")
    print("=" * 64)
