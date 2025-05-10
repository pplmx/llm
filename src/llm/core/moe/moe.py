import numpy as np
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


def track_token_flow(input_tensor, output_tensor, moe_layer, sample_token_idx=0):
    """
    跟踪单个token通过MoE层的处理流程

    参数:
        input_tensor: 输入张量
        output_tensor: 输出张量
        moe_layer: MoE层
        sample_token_idx: 样本token索引
    """
    with torch.no_grad():
        batch_size, seq_len, hidden_size = input_tensor.shape

        # 选择第一个批次的指定token
        input_token = input_tensor[0, sample_token_idx].cpu().numpy()
        output_token = output_tensor[0, sample_token_idx].cpu().numpy()

        # 扁平化输入以通过路由器
        flat_input = input_tensor.reshape(-1, hidden_size)
        sample_idx = sample_token_idx  # 第一个批次的指定token索引

        # 获取路由决策
        router_logits = moe_layer.router(flat_input)
        router_probs = F.softmax(router_logits, dim=-1)

        token_router_probs = router_probs[sample_idx].cpu().numpy()
        top_k_probs, top_k_indices = torch.topk(router_probs[sample_idx], k=moe_layer.top_k)
        top_k_probs = top_k_probs.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        # 计算每个专家的输出
        expert_outputs = []
        for expert_idx in range(moe_layer.num_experts):
            expert_output = moe_layer.experts[expert_idx](flat_input[sample_idx : sample_idx + 1])
            expert_outputs.append(expert_output.squeeze(0).cpu().numpy())

        # 打印跟踪信息
        print(f"\n--- 跟踪 Token {sample_token_idx + 1} 的处理流程 ---")
        print(f"输入向量: 均值 = {input_token.mean():.4f}, 标准差 = {input_token.std():.4f}")

        print("\n路由决策:")
        for expert_idx in range(moe_layer.num_experts):
            prob = token_router_probs[expert_idx]
            is_selected = expert_idx in top_k_indices
            print(f"  专家 {expert_idx}: 概率 = {prob:.4f} ({'✓选中' if is_selected else '✗未选'})")

        print("\n专家输出统计:")
        weighted_sum = np.zeros_like(expert_outputs[0])
        for k, expert_idx in enumerate(top_k_indices):
            expert_output = expert_outputs[expert_idx]
            weight = top_k_probs[k]
            weighted_output = expert_output * weight
            weighted_sum += weighted_output

            print(f"  专家 {expert_idx} (权重 = {weight:.4f}):")
            print(f"    输出: 均值 = {expert_output.mean():.4f}, 标准差 = {expert_output.std():.4f}")
            print(f"    加权: 均值 = {weighted_output.mean():.4f}, 标准差 = {weighted_output.std():.4f}")

        print("\n最终输出:")
        print(f"  组合后: 均值 = {weighted_sum.mean():.4f}, 标准差 = {weighted_sum.std():.4f}")
        print(f"  实际输出: 均值 = {output_token.mean():.4f}, 标准差 = {output_token.std():.4f}")

        # 比较组合输出和实际输出
        error = np.abs(weighted_sum - output_token).mean()
        print(f"  平均误差: {error:.6f} (应接近于0)")


def moe_example():
    # 设置参数
    batch_size = 2
    seq_len = 10
    hidden_size = 32
    num_experts = 4
    top_k = 2

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("       混合专家模型 (Mixture of Experts, MoE) 演示       ")
    print("=" * 60)
    print("\n模型配置:")
    print(f"• 批量大小: {batch_size} (每次处理的样本数)")
    print(f"• 序列长度: {seq_len} (每个样本的token数)")
    print(f"• 隐藏层维度: {hidden_size} (每个token的向量维度)")
    print(f"• 专家数量: {num_experts} (并行的FFN专家网络)")
    print(f"• Top-K: {top_k} (每个token选择的专家数)")
    print("• 路由抖动: 0.01 (训练时添加的噪声程度)")

    # 1. 创建MoE层
    print("\n第1步: 创建MoE层")
    moe_layer = MoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        router_jitter_noise=0.01,
    )

    # 计算总参数量
    total_params = sum(p.numel() for p in moe_layer.parameters())
    router_params = sum(p.numel() for p in moe_layer.router.parameters())
    experts_params = sum(p.numel() for p in moe_layer.experts.parameters())

    print(f"• 路由器参数量: {router_params:,} ({router_params / total_params * 100:.1f}%)")
    print(f"• 专家网络参数量: {experts_params:,} ({experts_params / total_params * 100:.1f}%)")
    print(f"• 总参数量: {total_params:,}")

    # 每个专家的参数量
    expert_param_counts = [sum(p.numel() for p in expert.parameters()) for expert in moe_layer.experts]
    print(f"• 每个专家参数量: {expert_param_counts[0]:,}")
    print(f"• 等效于 {num_experts} 个独立FFN的总参数量: {sum(expert_param_counts):,}")

    # 2. 创建示例输入
    print("\n第2步: 创建示例输入")
    input_tensor = torch.rand(batch_size, seq_len, hidden_size)
    print(f"• 输入张量形状: {input_tensor.shape}")
    print(f"• 输入张量总元素数: {input_tensor.numel():,}")
    # 简单统计
    input_mean = input_tensor.mean().item()
    input_std = input_tensor.std().item()
    print(f"• 输入张量统计: 均值 = {input_mean:.4f}, 标准差 = {input_std:.4f}")

    # 3. 训练模式前向传播 (应用路由噪声)
    print("\n第3步: 训练模式前向传播 (应用路由噪声)")
    moe_layer.train()
    output_train = moe_layer(input_tensor)
    print(f"• 训练模式输出形状: {output_train.shape}")
    train_output_mean = output_train.mean().item()
    train_output_std = output_train.std().item()
    print(f"• 训练输出统计: 均值 = {train_output_mean:.4f}, 标准差 = {train_output_std:.4f}")

    # 4. 评估模式前向传播 (无噪声)
    print("\n第4步: 评估模式前向传播 (无路由噪声)")
    moe_layer.eval()
    output_eval = moe_layer(input_tensor)
    print(f"• 评估模式输出形状: {output_eval.shape}")
    eval_output_mean = output_eval.mean().item()
    eval_output_std = output_eval.std().item()
    print(f"• 评估输出统计: 均值 = {eval_output_mean:.4f}, 标准差 = {eval_output_std:.4f}")

    # 验证输出尺寸
    assert output_train.shape == input_tensor.shape
    assert output_eval.shape == input_tensor.shape
    print("• 验证: 输入输出形状一致 ✓")

    # 5. 分析路由决策
    print("\n第5步: 分析路由决策")
    with torch.no_grad():
        # 扁平化输入
        flat_input = input_tensor.reshape(-1, hidden_size)

        # 获取路由logits
        router_logits = moe_layer.router(flat_input)

        # 应用softmax获取路由概率
        router_probs = F.softmax(router_logits, dim=-1)

        # 获取top-k专家及其概率
        top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)

        # 分析路由分布
        expert_assignment = router_probs.argmax(dim=-1)
        expert_counts = torch.bincount(expert_assignment, minlength=num_experts)

        # 打印每个专家被分配的token数
        print("专家负载分布 (基于最高概率):")
        for i, count in enumerate(expert_counts):
            percent = count.item() / (batch_size * seq_len) * 100
            bar = "█" * int(percent / 5)  # 简单的条形图
            print(f"  专家 {i}: {count.item()} tokens ({percent:.1f}%) {bar}")

        # 计算负载不平衡度 (使用变异系数)
        cv = expert_counts.float().std() / expert_counts.float().mean()
        print(f"\n• 负载不平衡度 (变异系数): {cv.item():.4f}")
        print("  - 0表示完全平衡，值越大表示越不平衡")
        print("  - 在大规模模型中，理想值<0.2")

    # 6. 详细跟踪单个token的处理流程
    print("\n第6步: 跟踪单个token的处理流程")
    # 选择第一个批次的第3个token作为示例
    sample_token_idx = 2
    track_token_flow(input_tensor, output_eval, moe_layer, sample_token_idx)

    # 7. 展示MoE的效率优势
    print("\n第7步: MoE与标准FFN的计算效率对比")

    # 创建等效的标准FFN (无条件计算)
    standard_ffn = nn.Sequential(
        nn.Linear(hidden_size, 4 * hidden_size),
        nn.GELU(),
        nn.Linear(4 * hidden_size, hidden_size),
    )

    # 计算FLOPs (浮点运算数)
    # 简化计算: 线性层FLOPs ≈ 2 * in_features * out_features

    # 标准FFN的FLOPs
    ffn_flops = 2 * hidden_size * (4 * hidden_size) + 2 * (4 * hidden_size) * hidden_size
    ffn_total_flops = ffn_flops * batch_size * seq_len

    # MoE的FLOPs (简化估计)
    # 路由器FLOPs
    router_flops = 2 * hidden_size * num_experts
    # 每个token只使用top_k个专家
    expert_flops_per_token = top_k * (2 * hidden_size * (4 * hidden_size) + 2 * (4 * hidden_size) * hidden_size)
    moe_total_flops = (router_flops + expert_flops_per_token) * batch_size * seq_len

    # 参数量比较
    ffn_params = sum(p.numel() for p in standard_ffn.parameters())

    print("计算效率对比:")
    print("• 标准FFN:")
    print(f"  - 参数量: {ffn_params:,}")
    print(f"  - 计算量: {ffn_total_flops:,} FLOPs")

    print(f"\n• MoE (使用 {num_experts} 专家, top-k={top_k}):")
    print(f"  - 参数量: {total_params:,} ({total_params / ffn_params:.1f}倍)")
    print(f"  - 计算量: {moe_total_flops:,} FLOPs ({moe_total_flops / ffn_total_flops:.2f}倍)")
    print(f"  - 理论参数效率提升: {(total_params / ffn_params) / (moe_total_flops / ffn_total_flops):.2f}倍")

    # 结论
    print()
    print("=" * 60)
    print("MoE 模型的关键优势:")
    print("1. 大幅增加参数量，而只略微增加计算量")
    print("2. 专家可以专门化，处理不同类型的输入")
    print("3. 可以通过路由器动态选择合适的计算路径")
    print("4. 在大规模模型中尤为有效，是扩展模型容量的关键技术")
    print("=" * 60)

    return output_train, output_eval


if __name__ == "__main__":
    moe_example()
