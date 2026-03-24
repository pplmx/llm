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
            self.normalized_shape: tuple[int, ...] = (normalized_shape,)
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

    # 3. 归一化 x_normalized
    x_normalized = (x - mean) / np.sqrt(var + eps)

    # 4. 应用仿射变换
    if gamma is not None and beta is not None:
        return gamma * x_normalized + beta
    else:
        return x_normalized
