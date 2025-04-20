import numpy as np
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) 实现。

    RMSNorm 是 LayerNorm 的一种简化形式。它仅使用激活值的均方根统计量进行归一化，
    而不进行中心化（减去均值）。通常只包含一个可学习的缩放参数（gamma/weight），
    不包含可学习的偏置参数（beta）。

    参考文献: Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." Advances in Neural Information Processing Systems 32 (2019).
    论文链接: https://arxiv.org/abs/1910.07467

    数学公式 (对于特征向量 x):
        RMS(x) = sqrt( (1/H) * Σ(x_i²) + ε )  (H = 归一化维度的大小)
        x_normalized = x / RMS(x)
        output = γ * x_normalized

    参数:
        normalized_shape (int 或 list/tuple of ints):
            需要进行归一化的输入张量的结尾维度形状。
            与 LayerNorm 中的定义相同。
        eps (float):
            加在均方根计算中的小常数，防止除零错误并提高数值稳定性。
            默认为 1e-6 (常见于 RMSNorm 实现)。
        elementwise_affine (bool):
            如果为 True，则此模块包含可学习的缩放参数 γ (gamma/weight)，
            形状与 `normalized_shape` 相同。γ 初始化为 1。
            注意：RMSNorm 通常不使用偏置项 (beta)。
            默认为 True。
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-6,  # 注意: RMSNorm 文献中 eps 常为 1e-6
        elementwise_affine: bool = True,
    ):
        super().__init__()

        # 存储配置
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # 初始化可学习的缩放参数 γ (gamma)
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))  # gamma
        else:
            # 如果无仿射变换，将 weight 注册为 None
            self.register_parameter("weight", None)

        # RMSNorm 通常没有偏置项 (beta)
        # self.register_parameter("bias", None) # 显式表明无偏置

    def _compute_rms(self, x: torch.Tensor) -> torch.Tensor:
        """计算 RMS 的辅助函数"""
        # 确定需要归一化的维度
        num_normalized_dims = len(self.normalized_shape)
        dims_to_normalize = tuple(range(x.ndim - num_normalized_dims, x.ndim))

        # 计算均方值 (不减去均值的方差)
        # mean(x^2)
        mean_square = torch.mean(x.pow(2), dim=dims_to_normalize, keepdim=True)

        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(mean_square + self.eps)
        return rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        参数:
            x: 输入张量，其尾部维度应与 `normalized_shape` 匹配。
               形状例如: [batch_size, ..., *normalized_shape]

        返回:
            归一化后的张量，形状与输入 x 相同。
        """
        # 1. 计算 RMS
        rms = self._compute_rms(x)

        # 2. 归一化: x / RMS(x)
        x_normalized = x / rms

        # 3. 应用缩放因子 γ (gamma/weight)（如果启用）
        if self.elementwise_affine:
            return self.weight * x_normalized
        else:
            return x_normalized

    def extra_repr(self) -> str:
        # 自定义打印模块信息时的显示内容
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# NumPy 实现的 RMSNorm (简化版，用于理解)
def rms_norm_numpy(x: np.ndarray, gamma: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """
    RMS Normalization 的 NumPy 实现 (简化版)

    注意：此版本为了简洁，*固定*在最后一个轴 (axis=-1) 上进行归一化。
    主要用于帮助理解 RMSNorm 的核心计算步骤。不包含偏置项。

    参数:
        x: 输入 NumPy 数组，形状例如 [batch_size, ..., feature_dim]
        gamma: 缩放参数 γ (如果提供)，形状应为 [feature_dim]
        eps: 防止除零错误的小常数

    返回:
        归一化后的 NumPy 数组，形状与输入 x 相同。
    """
    # 1. 沿最后一个轴计算均方值
    mean_square = np.mean(np.square(x), axis=-1, keepdims=True)

    # 2. 计算 RMS
    rms = np.sqrt(mean_square + eps)

    # 3. 归一化
    x_normalized = x / rms

    # 4. 应用缩放因子 gamma
    if gamma is not None:
        return gamma * x_normalized
    else:
        return x_normalized


# --- 演示部分 ---
def rms_norm_demo():
    """RMS Normalization 使用示例与对比"""
    print("-" * 40)
    print("RMS Normalization (均方根归一化) 演示")
    print("-" * 40)

    # --- 参数设置 ---
    batch_size = 2
    seq_length = 3
    hidden_size = 4  # 使用较小维度方便查看
    normalized_shape = (hidden_size,)
    eps = 1e-6

    # --- 创建输入数据 ---
    torch.manual_seed(43)  # 使用与 LayerNorm 演示不同的种子
    np.random.seed(43)
    x = torch.randn(batch_size, seq_length, hidden_size) * 2  # 稍微缩放输入
    x_numpy = x.detach().numpy().copy()

    print(f"输入张量 x 形状: {x.shape}")
    print(f"归一化形状 (normalized_shape): {normalized_shape}")
    print(f"输入张量 x (第一个样本):\n{x[0]}")

    # --- 初始化 RMSNorm 模块 ---
    # 1. 自定义 RMSNorm (带仿射变换 - gamma)
    custom_rms_norm = RMSNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=True)

    # 2. 自定义 RMSNorm (无仿射变换)
    custom_rms_norm_no_affine = RMSNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=False)

    # 3. NumPy 版本参数 (gamma)
    # 初始化 gamma 以匹配 PyTorch 模块的默认值 (全 1)
    gamma_numpy = custom_rms_norm.weight.data.detach().numpy().copy()

    # --- 前向传播 ---
    # 1. 自定义 PyTorch RMSNorm (带仿射)
    custom_output = custom_rms_norm(x)

    # 2. 自定义 PyTorch RMSNorm (无仿射)
    custom_output_no_affine = custom_rms_norm_no_affine(x)

    # 3. NumPy RMSNorm (带仿射)
    numpy_output = rms_norm_numpy(x_numpy, gamma=gamma_numpy, eps=eps)

    # 4. NumPy RMSNorm (无仿射)
    numpy_output_no_affine = rms_norm_numpy(x_numpy, gamma=None, eps=eps)

    print("\n--- 输出形状比较 ---")
    print(f"自定义 RMSNorm (带仿射) 输出形状:    {custom_output.shape}")
    print(f"自定义 RMSNorm (无仿射) 输出形状: {custom_output_no_affine.shape}")
    print(f"NumPy RMSNorm (带仿射) 输出形状:     {numpy_output.shape}")
    print(f"NumPy RMSNorm (无仿射) 输出形状:  {numpy_output_no_affine.shape}")

    # --- 实现一致性比较 ---
    print("\n--- 实现一致性比较 ---")

    # 比较 自定义 PyTorch (带仿射) vs NumPy (带仿射)
    custom_output_np = custom_output.detach().numpy()
    is_close_affine = np.allclose(custom_output_np, numpy_output, atol=1e-5)
    print(f"自定义 PyTorch (带仿射) 与 NumPy (带仿射) 输出是否接近? {is_close_affine}")
    if not is_close_affine:
        print(f"  - 最大差异: {np.max(np.abs(custom_output_np - numpy_output)):.6f}")

    # 比较 自定义 PyTorch (无仿射) vs NumPy (无仿射)
    custom_output_no_affine_np = custom_output_no_affine.detach().numpy()
    is_close_no_affine = np.allclose(custom_output_no_affine_np, numpy_output_no_affine, atol=1e-5)
    print(f"自定义 PyTorch (无仿射) 与 NumPy (无仿射) 输出是否接近? {is_close_no_affine}")
    if not is_close_no_affine:
        print(f"  - 最大差异: {np.max(np.abs(custom_output_no_affine_np - numpy_output_no_affine)):.6f}")

    # --- 观察输出统计特性 ---
    print("\n--- 输出统计特性 (沿归一化维度计算 RMS) ---")
    # 对于 RMSNorm (特别是无仿射或 gamma=1 时)，沿归一化维度计算的输出特征的 RMS 值应接近 1.0。
    # 输出的均值不一定为 0。

    def print_rms_stats(name: str, output_tensor: torch.Tensor):
        num_normalized_dims = len(normalized_shape)
        dims_to_normalize = tuple(range(output_tensor.ndim - num_normalized_dims, output_tensor.ndim))

        # 计算输出张量本身的 RMS
        output_mean_square = torch.mean(
            output_tensor.pow(2), dim=dims_to_normalize, keepdim=False
        )  # keepdim=False 便于统计
        output_rms = torch.sqrt(output_mean_square)  # 此处无需 eps，仅计算结果的 RMS

        # 计算均值和标准差作为一般信息
        means = output_tensor.mean(dim=dims_to_normalize)
        stds = output_tensor.std(dim=dims_to_normalize, unbiased=False)

        print(f"{name}:")
        # 打印所有样本 RMS 统计量的均值和标准差
        print(f"  输出 RMS (样本均值 +/- 标准差): {output_rms.mean().item():.4f} +/- {output_rms.std().item():.4f}")
        # 打印所有样本均值和标准差的统计量
        print(f"  输出均值 (样本均值 +/- 标准差): {means.mean().item():.4f} +/- {means.std().item():.4f}")
        print(f"  输出标准差 (样本均值 +/- 标准差): {stds.mean().item():.4f} +/- {stds.std().item():.4f}")

    print_rms_stats("自定义 RMSNorm (带仿射, gamma=1)", custom_output)
    print_rms_stats("自定义 RMSNorm (无仿射)", custom_output_no_affine)

    # NumPy 统计
    numpy_rms = np.sqrt(np.mean(np.square(numpy_output), axis=-1))
    numpy_means = np.mean(numpy_output, axis=-1)
    numpy_stds = np.std(numpy_output, axis=-1)
    print("NumPy RMSNorm (带仿射, gamma=1):")
    print(f"  输出 RMS (样本均值 +/- 标准差): {numpy_rms.mean():.4f} +/- {numpy_rms.std():.4f}")
    print(f"  输出均值 (样本均值 +/- 标准差): {numpy_means.mean():.4f} +/- {numpy_means.std():.4f}")
    print(f"  输出标准差 (样本均值 +/- 标准差): {numpy_stds.mean():.4f} +/- {numpy_stds.std():.4f}")

    numpy_no_affine_rms = np.sqrt(np.mean(np.square(numpy_output_no_affine), axis=-1))
    numpy_no_affine_means = np.mean(numpy_output_no_affine, axis=-1)
    numpy_no_affine_stds = np.std(numpy_output_no_affine, axis=-1)
    print("NumPy RMSNorm (无仿射):")
    print(f"  输出 RMS (样本均值 +/- 标准差): {numpy_no_affine_rms.mean():.4f} +/- {numpy_no_affine_rms.std():.4f}")
    print(f"  输出均值 (样本均值 +/- 标准差): {numpy_no_affine_means.mean():.4f} +/- {numpy_no_affine_means.std():.4f}")
    print(f"  输出标准差 (样本均值 +/- 标准差): {numpy_no_affine_stds.mean():.4f} +/- {numpy_no_affine_stds.std():.4f}")

    # --- 可视化一个样本切片的输出 ---
    sample_idx = 0
    seq_idx = 0
    print(f"\n--- 比较样本 [{sample_idx}, {seq_idx}] 的输出特征值 ---")
    print(f"原始输入:        {x[sample_idx, seq_idx].detach().numpy()}")
    print(f"自定义 (带仿射): {custom_output[sample_idx, seq_idx].detach().numpy()}")
    print(f"NumPy (带仿射):  {numpy_output[sample_idx, seq_idx]}")
    print(f"自定义 (无仿射): {custom_output_no_affine[sample_idx, seq_idx].detach().numpy()}")
    print(f"NumPy (无仿射):  {numpy_output_no_affine[sample_idx, seq_idx]}")


if __name__ == "__main__":
    rms_norm_demo()
