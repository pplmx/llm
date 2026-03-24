import numpy as np
import torch
import torch.nn as nn

from llm.utils.common import make_factory_kwargs


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) 实现.

    RMSNorm 是 LayerNorm 的一种简化形式. 它仅使用激活值的均方根统计量进行归一化,
    而不进行中心化 (减去均值). 通常只包含一个可学习的缩放参数 (gamma/weight),
    不包含可学习的偏置参数 (beta).

    参考文献: Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." Advances in Neural Information Processing Systems 32 (2019).
    论文链接: https://arxiv.org/abs/1910.07467

    数学公式 (对于特征向量 x):
        RMS(x) = sqrt( (1/H) * Σ(x_i²) + ε )  (H = 归一化维度的大小)
        x_normalized = x / RMS(x)
        output = gamma * x_normalized

    参数:
        normalized_shape (int 或 list/tuple of ints):
            需要进行归一化的输入张量的结尾维度形状.
            与 LayerNorm 中的定义相同.
        eps (float):
            加在均方根计算中的小常数, 防止除零错误并提高数值稳定性.
            默认为 1e-6 (常见于 RMSNorm 实现).
        elementwise_affine (bool):
            如果为 True, 则此模块包含可学习的缩放参数 gamma (gamma/weight),
            形状与 `normalized_shape` 相同. gamma 初始化为 1.
            注意: RMSNorm 通常不使用偏置项 (beta).
            默认为 True.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-6,  # 注意: RMSNorm 文献中 eps 常为 1e-6
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # 存储配置
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        factory_kwargs = make_factory_kwargs(device, dtype)
        if self.elementwise_affine:
            # 初始化可学习的缩放参数 gamma
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))  # gamma
        else:
            # 如果无仿射变换, 将 weight 注册为 None
            self.register_parameter("weight", None)

        # RMSNorm 通常没有偏置项 (beta)
        # self.register_parameter("bias", None) # 显式表明无偏置

    def _compute_rms(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算 RMS 的辅助函数"""
        # 确定需要归一化的维度
        num_normalized_dims = len(self.normalized_shape)
        dims_to_normalize = tuple(range(hidden_states.ndim - num_normalized_dims, hidden_states.ndim))

        # 计算均方值 (不减去均值的方差)
        # mean(x^2)
        mean_square = torch.mean(hidden_states.pow(2), dim=dims_to_normalize, keepdim=True)

        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(mean_square + self.eps)
        return rms

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        参数:
            hidden_states: 输入张量, 其尾部维度应与 `normalized_shape` 匹配.
               形状例如: [batch_size, ..., *normalized_shape]

        返回:
            归一化后的张量, 形状与输入 hidden_states 相同.
        """
        # 1. 计算 RMS
        rms = self._compute_rms(hidden_states)

        # 2. 归一化: x / RMS(x)
        x_normalized = hidden_states / rms

        # 3. 应用缩放因子 (gamma/weight) (如果启用)
        if self.elementwise_affine:
            return self.weight * x_normalized
        else:
            return x_normalized

    def extra_repr(self) -> str:
        # 自定义打印模块信息时的显示内容
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# NumPy 实现的 RMSNorm (简化版, 用于理解)
def rms_norm_numpy(x: np.ndarray, gamma: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """
    RMS Normalization 的 NumPy 实现 (简化版)

    注意: 此版本为了简洁, *固定*在最后一个轴 (axis=-1) 上进行归一化.
    主要用于帮助理解 RMSNorm 的核心计算步骤. 不包含偏置项.

    参数:
        x: 输入 NumPy 数组, 形状例如 [batch_size, ..., feature_dim]
        gamma: 缩放参数 (如果提供), 形状应为 [feature_dim]
        eps: 防止除零错误的小常数

    返回:
        归一化后的 NumPy 数组, 形状与输入 x 相同.
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
