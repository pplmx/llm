import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    混合专家模型层 (Mixture of Experts Layer)

    这是一种条件计算架构，将输入动态路由到多个"专家"网络中，并对其输出进行加权组合。
    MoE 可以显著增加模型参数量而不增加推理时的计算量，是大规模语言模型中常用的扩展技术。

    工作原理:
    1. 路由器网络为每个输入 token 决定应该使用哪些专家
    2. 仅激活被选中的专家子网络进行计算
    3. 专家输出根据路由权重进行组合，形成最终输出

    参数:
        hidden_size: 输入和输出特征的维度
        num_experts: 专家网络的总数量
        ffn_hidden_size: 每个专家FFN的中间层维度，默认为 hidden_size * 4
        top_k: 每个 token 被路由到的专家数量，默认为 1
        router_jitter_noise: 训练时添加到路由器 logits 的噪声标准差，促进负载均衡
        router_bias: 路由器是否使用偏置项
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
        if hidden_size <= 0 or num_experts <= 0 or top_k <= 0:
            raise ValueError("hidden_size, num_experts, 和 top_k 必须是正整数")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ffn_hidden_size = ffn_hidden_size if ffn_hidden_size is not None else 4 * hidden_size
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise

        # 路由器网络: 决定每个token使用哪些专家
        # 输入张量形状: [batch_size * seq_len, hidden_size]
        # 输出张量形状: [batch_size * seq_len, num_experts]
        self.router = nn.Linear(hidden_size, num_experts, bias=router_bias)

        # 专家网络: 每个专家是一个标准FFN (Feed-Forward Network)
        # FFN结构: Linear -> Activation -> Linear
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.ffn_hidden_size),
                    nn.GELU(),
                    nn.Linear(self.ffn_hidden_size, self.hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

    def _compute_router_probabilities(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算路由概率和专家选择

        参数:
            hidden_states: 输入张量，形状为 [batch_size * seq_len, hidden_size]

        返回:
            router_probs: top_k专家的路由概率，形状为 [batch_size * seq_len, top_k]
            expert_indices: 选择的专家索引，形状为 [batch_size * seq_len, top_k]
            router_logits: 路由器原始logits，形状为 [batch_size * seq_len, num_experts]
        """
        # 计算路由logits
        router_logits = self.router(hidden_states)  # [batch_size * seq_len, num_experts]

        # 添加路由抖动噪声 (训练时)
        if self.router_jitter_noise > 0.0 and self.training:
            # 生成均值为0的均匀噪声 [-noise_value, +noise_value]
            noise = (torch.rand_like(router_logits) * 2.0 - 1.0) * self.router_jitter_noise
            router_logits = router_logits + noise

        # 选择top_k个专家
        router_probs, expert_indices = torch.topk(F.softmax(router_logits, dim=-1), k=self.top_k, dim=-1)

        return router_probs, expert_indices, router_logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE层的前向传播

        参数:
            hidden_states: 输入张量，形状为 [batch_size, seq_len, hidden_size]

        返回:
            output: 输出张量，形状与输入相同 [batch_size, seq_len, hidden_size]
        """
        # 保存原始形状信息
        batch_size, seq_len, _ = hidden_states.shape

        # 重塑输入为二维张量以便处理
        # 形状: [batch_size * seq_len, hidden_size]
        flat_hidden_states = hidden_states.reshape(-1, self.hidden_size)

        # 1. 计算路由概率和专家选择
        router_probs, expert_indices, _ = self._compute_router_probabilities(flat_hidden_states)

        # 2. 分派和组合专家输出 (优化的实现)
        output = torch.zeros_like(flat_hidden_states)

        # 对每个专家单独处理，收集需要它处理的所有token
        for expert_idx in range(self.num_experts):
            # 在所有top_k选择中找出需要当前专家处理的token位置
            expert_mask = expert_indices == expert_idx  # [batch_size * seq_len, top_k]
            # 如果没有token路由到这个专家，跳过
            if not expert_mask.any():
                continue

            # 找出路由到当前专家的所有(token位置, top_k位置)对
            token_indices, k_indices = torch.where(expert_mask)

            # 获取输入token
            expert_inputs = flat_hidden_states[token_indices]  # [num_selected, hidden_size]

            # 获取路由权重
            expert_weights = router_probs[token_indices, k_indices].unsqueeze(-1)  # [num_selected, 1]

            # 通过专家处理，得到加权输出
            expert_outputs = self.experts[expert_idx](expert_inputs) * expert_weights  # [num_selected, hidden_size]

            # 更新输出张量
            # 由于token_indices可能有重复，我们使用index_add_来正确累加结果
            output.index_add_(0, token_indices, expert_outputs)

        # 重塑回原始形状
        output = output.reshape(batch_size, seq_len, self.hidden_size)

        return output


def moe_example():
    """
    MoE层使用示例及可视化
    """
    # 设置参数
    batch_size = 2
    seq_len = 10
    hidden_size = 32
    num_experts = 4
    top_k = 2

    print("--- 混合专家模型 (MoE) 演示 ---")
    print(f"批量大小: {batch_size}, 序列长度: {seq_len}, 隐藏层维度: {hidden_size}")
    print(f"专家数量: {num_experts}, Top-K: {top_k}")

    # 创建MoE层
    moe_layer = MoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        router_jitter_noise=0.01,
    )

    # 创建示例输入
    input_tensor = torch.rand(batch_size, seq_len, hidden_size)
    print(f"\n输入张量形状: {input_tensor.shape}")

    # 训练模式 (应用噪声)
    moe_layer.train()
    output_train = moe_layer(input_tensor)
    print(f"训练模式输出形状: {output_train.shape}")

    # 评估模式 (无噪声)
    moe_layer.eval()
    output_eval = moe_layer(input_tensor)
    print(f"评估模式输出形状: {output_eval.shape}")

    # 验证输出尺寸
    assert output_train.shape == input_tensor.shape
    assert output_eval.shape == input_tensor.shape

    # 查看路由决策
    with torch.no_grad():
        # 扁平化输入
        flat_input = input_tensor.reshape(-1, hidden_size)

        # 获取路由logits (使用未优化的方法以便检查)
        router_logits = moe_layer.router(flat_input)

        # 应用softmax获取路由概率
        router_probs = F.softmax(router_logits, dim=-1)

        # 分析路由分布
        expert_assignment = router_probs.argmax(dim=-1)
        expert_counts = torch.bincount(expert_assignment, minlength=num_experts)

        # 打印每个专家被分配的token数
        print("\n专家负载分布:")
        for i, count in enumerate(expert_counts):
            print(f"专家 {i}: {count.item()} tokens ({count.item() / (batch_size * seq_len) * 100:.1f}%)")

    print("\n--- MoE 演示结束 ---")
    return output_train, output_eval


if __name__ == "__main__":
    moe_example()
