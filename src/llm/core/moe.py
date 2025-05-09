import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    混合专家模型层 (Mixture of Experts Layer)

    参数:
        hidden_size (int): 隐藏层维度，同时作为输入和输出维度
        num_experts (int): 专家网络的数量
        ffn_hidden_size (int, 可选): 专家FFN中间层的大小，默认为 4 * hidden_size
        num_experts_per_tok (int, 可选): 每个token要路由到的专家数量，默认为1
        router_jitter_noise (float, 可选): 路由器抖动噪声大小，默认为0.0（不使用噪声）
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        ffn_hidden_size: int | None = None,
        num_experts_per_tok: int = 1,
        router_jitter_noise: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_jitter_noise = router_jitter_noise

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size  # 默认使用4倍hidden_size，与Transformer一致

        # 初始化路由器网络
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # 初始化专家网络 - 使用标准FFN结构（与Transformer中的FFN一致）
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, ffn_hidden_size),
                    nn.GELU(),  # 使用GELU激活函数，与现代Transformer一致
                    nn.Linear(ffn_hidden_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

    def _compute_router_probabilities(self, hidden_states):
        """
        计算路由概率

        参数:
            hidden_states (torch.Tensor): 形状为[batch_size, seq_len, hidden_size]的输入张量

        返回:
            router_probs (torch.Tensor): 归一化后的路由概率
            router_indices (torch.Tensor): top-k专家的索引
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 重塑张量以便于路由计算
        hidden_states_reshaped = hidden_states.view(-1, self.hidden_size)  # [batch_size * seq_len, hidden_size]

        # 计算路由logits
        router_logits = self.router(hidden_states_reshaped)  # [batch_size * seq_len, num_experts]

        if self.router_jitter_noise > 0.0 and self.training:
            # 添加噪声以增强训练稳定性和多样性
            router_logits += torch.rand_like(router_logits) * self.router_jitter_noise

        # 获取每个token的top-k专家
        router_probs, router_indices = torch.topk(
            router_logits, k=self.num_experts_per_tok, dim=-1
        )  # [batch_size * seq_len, num_experts_per_tok]

        # 对top-k概率进行softmax归一化
        router_probs = F.softmax(router_probs, dim=-1)

        return router_probs, router_indices

    def forward(self, hidden_states):
        """
        前向传播

        参数:
            hidden_states (torch.Tensor): 形状为[batch_size, seq_len, hidden_size]的输入张量

        返回:
            output (torch.Tensor): 形状为[batch_size, seq_len, hidden_size]的输出张量
        """
        batch_size, seq_len, _ = hidden_states.shape
        flat_size = batch_size * seq_len

        # 计算路由概率和索引
        router_probs, router_indices = self._compute_router_probabilities(hidden_states)

        # 初始化输出
        output = torch.zeros(flat_size, self.hidden_size, device=hidden_states.device)

        # 重塑输入便于处理
        hidden_states = hidden_states.reshape(flat_size, self.hidden_size)

        # 处理每个专家
        for expert_idx in range(self.num_experts):
            # 找出应该被路由到当前专家的token
            # 检查router_indices中是否包含当前专家的索引
            expert_mask = router_indices == expert_idx  # [batch_size * seq_len, num_experts_per_tok]

            # 获取被路由到当前专家的token索引
            token_indices = torch.nonzero(expert_mask.any(dim=-1), as_tuple=True)[0]

            if token_indices.shape[0] == 0:
                # 如果没有token被路由到当前专家，则跳过
                continue

            # 获取这些token及其对应的权重
            expert_inputs = hidden_states[token_indices]  # [num_tokens, hidden_size]

            # 找出每个token中当前专家在top-k中的位置
            expert_positions = torch.nonzero(expert_mask, as_tuple=True)
            token_positions, k_positions = expert_positions[0], expert_positions[1]

            # 只保留与token_indices匹配的位置
            mask = torch.zeros(token_positions.shape[0], dtype=torch.bool, device=token_positions.device)
            for i, idx in enumerate(token_positions):
                mask[i] = idx in token_indices

            token_positions = token_positions[mask]
            k_positions = k_positions[mask]

            # 获取对应的权重
            expert_weights = router_probs[token_positions, k_positions]  # [num_tokens]

            # 通过专家网络处理输入
            expert_outputs = self.experts[expert_idx](expert_inputs)  # [num_tokens, hidden_size]

            # 将输出乘以对应的权重
            weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)  # [num_tokens, hidden_size]

            # 累加到最终输出
            output.index_add_(0, token_indices, weighted_outputs)

        # 重塑回原始形状
        output = output.reshape(batch_size, seq_len, self.hidden_size)

        return output


class SimpleMoELayer(nn.Module):
    """
    简化版混合专家模型层

    这个版本使用更简洁的实现，专注于MoE的核心概念。

    参数:
        hidden_size (int): 隐藏层维度，同时作为输入和输出维度
        num_experts (int): 专家网络的数量
        ffn_hidden_size (int, 可选): 专家FFN中间层的大小，默认为4倍hidden_size
        top_k (int, 可选): 每个token要路由到的专家数量，默认为1
    """

    def __init__(self, hidden_size: int, num_experts: int, ffn_hidden_size: int = None, top_k: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        # 路由器
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # 专家网络 - 使用简单FFN
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, ffn_hidden_size), nn.GELU(), nn.Linear(ffn_hidden_size, hidden_size)
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 形状为[batch_size, seq_len, hidden_size]的输入张量

        返回:
            output (torch.Tensor): 形状为[batch_size, seq_len, hidden_size]的输出张量
        """
        batch_size, seq_len, _ = x.shape

        # 获取路由分数
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # 选择top-k专家
        router_probs, indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        router_probs = F.softmax(router_probs, dim=-1)

        # 初始化输出
        final_output = torch.zeros_like(x)

        # 对每个batch和序列位置处理
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.top_k):
                    # 获取专家索引和权重
                    expert_idx = indices[b, s, k].item()
                    weight = router_probs[b, s, k].item()

                    # 处理输入
                    token = x[b, s].unsqueeze(0)  # 保持维度: [1, hidden_size]
                    output = self.experts[expert_idx](token)

                    # 加权累加
                    final_output[b, s] += output.squeeze(0) * weight

        return final_output


def moe_example():
    """
    MoE层使用示例
    """
    # 设置参数
    hidden_size = 64
    num_experts = 4
    batch_size = 2
    seq_len = 3

    # 创建输入
    x = torch.rand(batch_size, seq_len, hidden_size)

    # 测试MoELayer
    moe = MoELayer(hidden_size=hidden_size, num_experts=num_experts)
    output = moe(x)

    print(f"MoE Input shape: {x.shape}")
    print(f"MoE Output shape: {output.shape}")

    simple_moe = SimpleMoELayer(hidden_size=hidden_size, num_experts=num_experts)
    output = simple_moe(x)
    print(f"Simple MoE Input shape: {x.shape}")
    print(f"Simple MoE Output shape: {output.shape}")

    return output


if __name__ == "__main__":
    moe_example()
