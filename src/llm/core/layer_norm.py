import numpy as np
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    自定义 Layer Normalization 实现 (更接近 PyTorch 内置版本)

    Layer Normalization 通过对单个样本内的特征维度进行归一化来稳定训练过程.
    与 Batch Normalization 不同, LayerNorm 对每个样本独立操作, 不依赖于 Batch Size,
    因此特别适用于序列模型 (RNN, Transformer).

    数学公式 (对于一个样本 x 中的一个元素 x_i):
        μ = (1/H) * Σ(x_i)  (在归一化维度 H 上求均值)
        σ² = (1/H) * Σ((x_i - μ)²) (在归一化维度 H 上求方差)
        x_normalized = (x - μ) / sqrt(σ² + ε)
        output = γ * x_normalized + β

    参数:
        normalized_shape (int 或 list/tuple of ints):
            需要进行归一化的输入张量的结尾维度形状.
            例如, 如果输入形状是 (N, C, H, W) 且希望对最后两个维度 (H, W) 进行归一化,
            则 normalized_shape 应为 (H, W) 或 `[H, W]`.
            如果只对最后一个维度进行归一化, 可以传入一个整数, 如 `W`.
        eps (float):
            加在分母中的小常数, 防止除零错误并提高数值稳定性. 默认为 1e-5.
        elementwise_affine (bool):
            如果为 True, 则此模块包含可学习的仿射参数 γ (gamma/weight) 和 β (beta/bias),
            形状与 `normalized_shape` 相同. γ 初始化为 1, β 初始化为 0.
            默认为 True.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        # 将 normalized_shape 统一处理为 tuple
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # 初始化可学习的缩放参数 γ (gamma) 和偏移参数 β (beta)
            # 形状与需要归一化的维度一致
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))  # gamma
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))  # beta
        else:
            # 如果不使用仿射变换, 则注册为 None, 这是标准做法
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        参数:
            hidden_states: 输入张量, 其尾部维度应与 `normalized_shape` 匹配.
               例如, 形状可以是 [batch_size, ..., *normalized_shape]

        返回:
            归一化后的张量, 形状与输入 hidden_states 相同.
        """
        # 1. 确定需要计算均值和方差的维度
        # normalized_shape 定义了最后几个维度, 我们需要在这些维度上计算统计量
        # 例如, 如果 hidden_states.shape = (N, C, H, W) 且 normalized_shape = (H, W)
        # 则 dims_to_normalize = (-2, -1)
        num_normalized_dims = len(self.normalized_shape)
        dims_to_normalize = tuple(range(hidden_states.ndim - num_normalized_dims, hidden_states.ndim))

        # 2. 计算均值 (μ) 和方差 (σ²)
        # 在指定的维度上计算, 并保持维度以便广播
        # 注意: 计算方差时使用 unbiased=False, 与 PyTorch 官方实现一致
        mean = torch.mean(hidden_states, dim=dims_to_normalize, keepdim=True)
        # var = torch.var(hidden_states, dim=dims_to_normalize, unbiased=False, keepdim=True) # 简洁写法
        # 或者, 使用定义式计算方差(对初学者更清晰):
        var = ((hidden_states - mean) ** 2).mean(dim=dims_to_normalize, keepdim=True)

        # 3. 归一化 (x_normalized)
        # (x - μ) / sqrt(σ² + ε)
        x_normalized = (hidden_states - mean) / torch.sqrt(var + self.eps)

        # 4. 应用仿射变换 (γ * x_normalized + β)
        if self.elementwise_affine:
            # self.weight (gamma) 和 self.bias (beta) 的形状是 normalized_shape
            # PyTorch 的广播机制会自动将它们应用到 x_normalized 的对应维度上
            return self.weight * x_normalized + self.bias
        else:
            return x_normalized

    def extra_repr(self) -> str:
        # 自定义打印模块信息时的显示内容, 使其更像官方版本
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# NumPy 实现 (简化版, 仅用于概念理解, 固定在最后一个轴归一化)
def layer_norm_numpy(
    x: np.ndarray, gamma: np.ndarray | None = None, beta: np.ndarray | None = None, eps: float = 1e-5
) -> np.ndarray:
    """
    Layer Normalization 的 NumPy 实现 (简化版)

    注意: 此版本为了简洁, *固定*在最后一个轴 (axis=-1) 上进行归一化.
    主要用于帮助理解 LayerNorm 的核心计算步骤.

    参数:
        x: 输入 NumPy 数组, 形状例如 [batch_size, ..., feature_dim]
        gamma: 缩放参数 γ (如果提供), 形状应为 [feature_dim]
        beta: 偏移参数 β (如果提供), 形状应为 [feature_dim]
        eps: 防止除零错误的小常数

    返回:
        归一化后的 NumPy 数组, 形状与输入 x 相同.
    """
    # 1. 在最后一个轴上计算均值 μ
    mean = np.mean(x, axis=-1, keepdims=True)

    # 2. 在最后一个轴上计算方差 σ²
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    # 或者 var = np.var(x, axis=-1, keepdims=True)

    # 3. 归一化 x_normalized
    x_normalized = (x - mean) / np.sqrt(var + eps)

    # 4. 应用仿射变换
    if gamma is not None and beta is not None:
        # gamma 和 beta 的形状需要能广播到 x_normalized 的最后一个维度
        return gamma * x_normalized + beta
    else:
        return x_normalized


# --- 演示部分 ---
def layer_norm_demo():
    """Layer Normalization 使用示例与对比"""
    print("-" * 40)
    print("Layer Normalization Demo")
    print("-" * 40)

    # --- 参数设置 ---
    batch_size = 2
    seq_length = 3
    hidden_size = 4  # 为了方便查看, 使用较小的维度
    # 定义需要归一化的形状
    normalized_shape = (hidden_size,)  # 对最后一个维度归一化

    # --- 创建输入数据 ---
    # 使用确定性的随机种子, 方便复现
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(batch_size, seq_length, hidden_size)
    x_numpy = x.detach().numpy().copy()  # 转换为NumPy数组

    print(f"输入张量 x 形状: {x.shape}")
    print(f"归一化形状 (normalized_shape): {normalized_shape}")
    print(f"输入张量 x (第一个样本):\n{x[0]}")

    # --- 初始化 LayerNorm 模块 ---
    # 1. PyTorch 内置 LayerNorm
    torch_layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=True)

    # 2. 自定义 LayerNorm
    custom_layer_norm = LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=True)

    # 3. 自定义 LayerNorm (无仿射变换)
    custom_layer_norm_no_affine = LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=False)

    # 注意: 不再需要手动创建 gamma_numpy 和 beta_numpy, 因为我们将从 torch_layer_norm 获取

    # --- 前向传播 ---
    # 1. PyTorch 内置
    torch_output = torch_layer_norm(x)

    # 2. 自定义 (带仿射)
    # 注意: 为了公平比较, 将自定义 LN 的权重/偏置设置为与 PyTorch 内置 LN 相同
    custom_layer_norm.weight.data = torch_layer_norm.weight.data.clone()
    custom_layer_norm.bias.data = torch_layer_norm.bias.data.clone()
    custom_output = custom_layer_norm(x)

    # 3. 自定义 (无仿射)
    custom_output_no_affine = custom_layer_norm_no_affine(x)

    # 4. NumPy 实现 (带仿射)
    # 使用从 PyTorch LN 提取的 gamma 和 beta
    gamma_numpy_torch = torch_layer_norm.weight.data.detach().numpy()
    beta_numpy_torch = torch_layer_norm.bias.data.detach().numpy()
    numpy_output = layer_norm_numpy(x_numpy, gamma=gamma_numpy_torch, beta=beta_numpy_torch, eps=1e-5)

    # 5. NumPy 实现 (无仿射)
    numpy_output_no_affine = layer_norm_numpy(x_numpy, gamma=None, beta=None, eps=1e-5)

    print("\n--- 输出形状比较 ---")
    print(f"PyTorch 内置 LN 输出形状: {torch_output.shape}")
    print(f"自定义 LN 输出形状:      {custom_output.shape}")
    print(f"NumPy LN 输出形状:         {numpy_output.shape}")

    # --- 比较实现 ---
    print("\n--- 实现一致性比较 ---")

    # 比较 自定义 PyTorch 实现 vs PyTorch 内置实现
    # 使用 allclose 检查数值是否在容忍范围内一致
    is_close_torch_custom = torch.allclose(torch_output, custom_output, atol=1e-5)  # 提高容忍度以应对浮点误差
    print(f"自定义 PyTorch 实现与 PyTorch 内置实现输出是否接近? {is_close_torch_custom}")
    if not is_close_torch_custom:
        print(f"  - 最大差异: {torch.max(torch.abs(torch_output - custom_output)).item():.6f}")

    # 比较 自定义 PyTorch 实现 vs NumPy 实现
    custom_output_np = custom_output.detach().numpy()
    is_close_custom_numpy = np.allclose(custom_output_np, numpy_output, atol=1e-5)
    print(f"自定义 PyTorch 实现与 NumPy 实现输出是否接近?       {is_close_custom_numpy}")
    if not is_close_custom_numpy:
        print(f"  - 最大差异: {np.max(np.abs(custom_output_np - numpy_output)):.6f}")

    # 比较 自定义 PyTorch (无仿射) vs NumPy (无仿射)
    custom_output_no_affine_np = custom_output_no_affine.detach().numpy()
    is_close_no_affine = np.allclose(custom_output_no_affine_np, numpy_output_no_affine, atol=1e-5)
    print(f"无仿射变换的自定义 PyTorch 与 NumPy 实现输出是否接近? {is_close_no_affine}")
    if not is_close_no_affine:
        print(f"  - 最大差异: {np.max(np.abs(custom_output_no_affine_np - numpy_output_no_affine)):.6f}")

    # --- 观察输出统计特性 ---
    print("\n--- 输出统计特性 (沿归一化维度计算) ---")
    # 理想情况下:
    # - 对于无仿射变换 (或 gamma=1, beta=0) 的输出, 均值应接近 0, 标准差应接近 1.
    # - 对于有仿射变换的输出, 均值可能不再是 0, 标准差可能不再是 1, 取决于学习到的 gamma 和 beta.
    # 注意: 这里 gamma=1, beta=0, 所以理论上带仿射和不带仿射的统计特性应该相似.

    def print_stats(name: str, output_tensor: torch.Tensor):
        # 计算每个样本在归一化维度上的均值和标准差
        num_normalized_dims = len(normalized_shape)
        dims_to_normalize = tuple(range(output_tensor.ndim - num_normalized_dims, output_tensor.ndim))

        means = output_tensor.mean(dim=dims_to_normalize)
        stds = output_tensor.std(dim=dims_to_normalize, unbiased=False)  # 使用 unbiased=False 与计算过程一致

        print(f"{name}:")
        # 打印所有样本统计量的均值和标准差, 以了解整体分布
        print(f"  均值 (所有样本的均值): {means.mean().item():.4f} +/- {means.std().item():.4f}")
        print(f"  标准差 (所有样本的标准差): {stds.mean().item():.4f} +/- {stds.std().item():.4f}")

    print_stats("PyTorch 内置 LN 输出", torch_output)
    print_stats("自定义 LN 输出     ", custom_output)
    print_stats("自定义 LN (无仿射) ", custom_output_no_affine)
    # NumPy 结果的统计 (需要转换为 torch tensor 或用 numpy 计算)
    numpy_means = np.mean(numpy_output, axis=tuple(range(x_numpy.ndim - len(normalized_shape), x_numpy.ndim)))
    numpy_stds = np.std(numpy_output, axis=tuple(range(x_numpy.ndim - len(normalized_shape), x_numpy.ndim)))
    print("NumPy LN 输出:")
    print(f"  均值 (所有样本的均值): {numpy_means.mean():.4f} +/- {numpy_means.std():.4f}")
    print(f"  标准差 (所有样本的标准差): {numpy_stds.mean():.4f} +/- {numpy_stds.std():.4f}")

    # --- 可视化一个样本的输出 ---
    sample_idx = 0
    seq_idx = 0
    print(f"\n--- 比较样本 [{sample_idx}, {seq_idx}] 的输出特征值 ---")
    print(f"原始输入:        {x[sample_idx, seq_idx].detach().numpy()}")
    print(f"PyTorch 内置 LN: {torch_output[sample_idx, seq_idx].detach().numpy()}")
    print(f"自定义 LN:       {custom_output[sample_idx, seq_idx].detach().numpy()}")
    print(f"NumPy LN:        {numpy_output[sample_idx, seq_idx]}")
    print(f"自定义 (无仿射): {custom_output_no_affine[sample_idx, seq_idx].detach().numpy()}")
    print(f"NumPy (无仿射):  {numpy_output_no_affine[sample_idx, seq_idx]}")


if __name__ == "__main__":
    layer_norm_demo()
