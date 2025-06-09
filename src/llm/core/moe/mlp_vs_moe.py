import torch
import torch.nn as nn
import torch.nn.functional as F

# --- SimpleFFN (无变化, 保持原样) ---
class SimpleFFN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        ffn_hidden_size = 4 * hidden_size
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        print("=== SimpleFFN forward ===")
        print(f"Input x shape: {x.shape}")
        # print(f"Input x sample: {x[0, :5].tolist()}")

        x1 = F.relu(self.fc1(x))
        print(f"After fc1 + ReLU shape: {x1.shape}")
        # print(f"After fc1 + ReLU sample: {x1[0, :5].tolist()}")

        x2 = self.fc2(x1)
        print(f"After fc2 shape: {x2.shape}")
        # print(f"After fc2 sample: {x2[0, :5].tolist()}")
        return x2


# --- 原始 MoE 实现 (重命名以作区分) ---
class SimpleMoE_Loop(nn.Module):
    def __init__(self, hidden_size, num_experts=4, k=2):
        super().__init__()
        ffn_hidden_size = 4 * hidden_size
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_hidden_size),
                nn.ReLU(),
                nn.Linear(ffn_hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # 这一部分与原版完全相同
        print("=== SimpleMoE_Loop forward (直观但低效的循环实现) ===")
        batch_size, hidden_dim = x.shape
        gate_scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=1)
        topk_weights = F.softmax(topk_scores, dim=1)

        # 核心区别在这里：使用双重循环处理
        final_output = torch.zeros_like(x)
        for i in range(batch_size):  # 遍历 batch 中的每个 token
            for j in range(self.k):   # 遍历该 token 的 top-k experts
                expert_idx = topk_indices[i, j].item()
                expert = self.experts[expert_idx]
                weight = topk_weights[i, j]

                # 将单个 token 输入 expert
                expert_output = expert(x[i].unsqueeze(0))

                # 累加加权后的输出
                final_output[i] += expert_output.squeeze(0) * weight

        print(f"MoE final output shape: {final_output.shape}")
        print(f"MoE final output sample: {final_output[0, :5].tolist()}")
        return final_output


# --- 优化后的 MoE 实现 (向量化) ---
class SimpleMoE_Optimized(nn.Module):
    def __init__(self, hidden_size, num_experts=4, k=2):
        super().__init__()
        ffn_hidden_size = 4 * hidden_size
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_hidden_size),
                nn.ReLU(),
                nn.Linear(ffn_hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        batch_size, hidden_dim = x.shape

        print("\n" + "="*20 + " MoE Forward Pass Breakdown " + "="*20)
        # 为了打印清晰，我们只展示第一个 token 的部分样本数据
        print(f"Input x shape: {x.shape}")
        print(f"Input x (Token 0, first 5 vals): {x[0, :5].tolist()}")
        print("-" * 60)

        # --- 1. Gating: 为每个 Token 选择最合适的 Experts ---
        print("--- 1. Gating: Selecting Experts ---")
        gate_scores = self.gate(x)
        print(f"Gate scores shape (batch_size, num_experts): {gate_scores.shape}")
        # 打印每个 token 的原始门控分数
        for i in range(batch_size):
            # 使用列表推导格式化浮点数，提高可读性
            formatted_scores = [f"{s:.2f}" for s in gate_scores[i].tolist()]
            print(f"  Token {i} raw gate scores: {formatted_scores}")

        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        print("\nAfter Top-K and Softmax:")
        # 这是 MoE 的核心决策，为每个 token 打印出最终选择和权重
        for i in range(batch_size):
            formatted_weights = [f"{w:.2f}" for w in topk_weights[i].tolist()]
            print(f"  Token {i} -> Chosen Experts: {topk_indices[i].tolist()}, Weights: {formatted_weights}")
        print("-" * 60)

        # --- 2. Routing & Computation: 将 Token 分发给 Expert 并行计算 ---
        print("--- 2. Routing & Computation (Scatter-Gather) ---")
        final_output = torch.zeros_like(x)

        # 将 top-k 索引和输入 tensor 扁平化，为 expert 的批量计算做准备
        flat_topk_indices = topk_indices.view(-1)
        repeated_x = x.repeat_interleave(self.k, dim=0)

        # 使用一个与 repeated_x 同样大小的零张量来收集所有 expert 的输出
        dispatched_x = torch.zeros_like(repeated_x)

        # 以 Expert 为中心进行计算 (Gather & Compute)
        for i in range(self.num_experts):
            # 找到所有需要由当前 expert i 处理的 token 的位置
            mask = (flat_topk_indices == i)
            if not mask.any(): # 如果没有 token 分配给这个 expert，则跳过
                print(f"Expert #{i} received 0 tokens. Skipping.")
                continue

            # 获取这些 token 的输入
            tokens_for_expert = repeated_x[mask]
            print(f"Expert #{i} received {tokens_for_expert.shape[0]} token(s) to process.")

            # 批量计算
            expert_output = self.experts[i](tokens_for_expert)

            # (Scatter back) 将计算结果放回 dispatched_x 的相应位置
            dispatched_x = dispatched_x.masked_scatter(mask.unsqueeze(-1), expert_output)

        # dispatched_x 现在包含了所有 token 在其各自选择的 expert 上的原始输出
        # 它的形状是 [batch_size * k, hidden_size]
        print("\nAfter all experts finished computation:")
        print(f"Dispatched output shape (batch*k, hidden): {dispatched_x.shape}")
        print("-" * 60)

        # --- 3. Aggregation: 加权合并 Expert 的输出 ---
        print("--- 3. Aggregation: Combining Outputs ---")

        # 将 expert 输出的形状变回 (batch_size, k, hidden_size) 以便加权
        dispatched_x = dispatched_x.view(batch_size, self.k, hidden_dim)

        # 将权重张量扩展维度 (batch_size, k) -> (batch_size, k, 1) 以便广播
        topk_weights_expanded = topk_weights.unsqueeze(-1)

        # 核心加权步骤
        weighted_output = dispatched_x * topk_weights_expanded

        # 打印加权过程的细节
        for i in range(batch_size):
            print(f"  Aggregating for Token {i}:")
            for j in range(self.k):
                expert_idx = topk_indices[i, j].item()
                weight = topk_weights[i, j].item()
                # 打印来自单个 expert 的加权贡献
                print(f"    - From Expert #{expert_idx} (weight={weight:.2f}), weighted output (first 5 vals): {weighted_output[i, j, :5].tolist()}")

        # 将 k 个 expert 的加权输出相加
        final_output = weighted_output.sum(dim=1)

        print("\n" + "="*20 + " MoE Final Result " + "="*20)
        print(f"Final output shape: {final_output.shape}")
        print(f"Final output (Token 0, first 5 vals): {final_output[0, :5].tolist()}")
        print("=" * 64)

        return final_output

if __name__ == "__main__":
    # 使用小的、固定的参数，让输出清晰且可复现
    batch_size_ = 2
    hidden_size_ = 8
    num_experts_ = 4
    k_ = 2

    # 设置随机种子以确保每次运行结果一致，便于分析
    torch.manual_seed(42)
    x = torch.randn(batch_size_, hidden_size_)

    print("--- Input Data ---")
    print(f"Input shape: {x.shape}")
    for i in range(batch_size_):
        print(f"Input (Token {i}): {x[i].tolist()}")
    print("-" * 20)

    # 运行标准 FFN (保持简单)
    ffn = SimpleFFN(hidden_size_)
    out_ffn = ffn(x)

    # 运行优化后的 MoE
    moe_opt = SimpleMoE_Optimized(hidden_size_, num_experts=num_experts_, k=k_)
    out_moe_opt = moe_opt(x)
