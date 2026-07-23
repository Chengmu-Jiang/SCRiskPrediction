from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.scnews_sentiment.structured.common import *

def make_projection_block(
    input_dim: int,
    output_dim: int,
    dropout: float,
    *,
    use_layer_norm: bool = True,
) -> Any:
    """构造单层投影网络块。

    Parameters
    ----------
    input_dim : int
        输入特征维度。
    output_dim : int
        输出投影维度。
    dropout : float
        Dropout 概率。
    use_layer_norm : bool, optional
        是否在线性层后加入 LayerNorm，默认值为 ``True``。

    Returns
    -------
    torch.nn.Sequential
        由 Linear、可选 LayerNorm、GELU 和 Dropout 组成的模块。

    Raises
    ------
    ImportError
        当 PyTorch 未安装时抛出。

    Notes
    -----
    该模块用于实现公式中的 ``phi_g`` 和 ``phi_z``，将结构化特征和文本
    表示投影到同一隐空间。
    """
    # 延迟导入 torch.nn，避免未使用阶段强制加载依赖。
    import torch.nn as nn

    # 初始化带类型标注的 layers，明确后续存储内容。
    layers: List[Any] = [nn.Linear(input_dim, output_dim)]
    # 根据条件 use_layer_norm 选择对应处理分支。
    if use_layer_norm:
        # 执行 append，完成当前步骤的副作用。
        layers.append(nn.LayerNorm(output_dim))
    # 调用 layers.extend，执行当前步骤的副作用。
    layers.extend([nn.GELU(), nn.Dropout(dropout)])
    # 返回当前函数的处理结果。
    return nn.Sequential(*layers)


def make_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    dropout: float,
    *,
    final_layer_norm: bool,
) -> Any:
    """构造通用 MLP。

    Parameters
    ----------
    input_dim : int
        输入维度。
    hidden_sizes : sequence of int
        隐藏层维度序列；为空时直接使用线性输出层。
    output_dim : int
        输出维度。
    dropout : float
        每个隐藏层后的 Dropout 概率。
    final_layer_norm : bool
        是否在最终输出后添加 LayerNorm。

    Returns
    -------
    torch.nn.Sequential
        可直接参与反向传播的 MLP 模块。

    Raises
    ------
    ImportError
        当 PyTorch 未安装时抛出。
    """
    # 延迟导入 torch.nn，避免未使用阶段强制加载依赖。
    import torch.nn as nn

    # 初始化带类型标注的 layers，明确后续存储内容。
    layers: List[Any] = []
    # 计算并保存 previous_dim，供后续逻辑复用。
    previous_dim = input_dim
    # 遍历 hidden_sizes，逐项执行处理逻辑。
    for hidden_dim in hidden_sizes:
        # 执行 extend，完成当前步骤的副作用。
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        # 计算并保存 previous_dim，供当前分支后续使用。
        previous_dim = hidden_dim
    # 调用 layers.append，执行当前步骤的副作用。
    layers.append(nn.Linear(previous_dim, output_dim))
    # 根据条件 final_layer_norm 选择对应处理分支。
    if final_layer_norm:
        # 执行 append，完成当前步骤的副作用。
        layers.append(nn.LayerNorm(output_dim))
    # 返回当前函数的处理结果。
    return nn.Sequential(*layers)


class NewsInitializationEncoder:
    """新闻级初始化编码器。

    Parameters
    ----------
    gdelt_dim : int
        输入 GDELT 结构化特征维度。
    text_dim : int
        输入文本嵌入维度。
    projection_dim : int
        ``phi_g`` 和 ``phi_z`` 的共同投影维度。
    sent_hidden_sizes : sequence of int
        情绪感知编码器 ``f_sent`` 的隐藏层维度。
    representation_dim : int
        输出新闻表示 ``r_{t,u}`` 的维度。
    dropout : float, optional
        MLP 中使用的 Dropout 概率，默认值为 0.1。
    final_layer_norm : bool, optional
        是否对 ``f_sent`` 输出执行 LayerNorm，默认值为 ``True``。

    Attributes
    ----------
    module : torch.nn.Module
        实际可训练的 PyTorch 模块。

    Notes
    -----
    核心逻辑是分别投影结构化事件特征和文本语义特征，然后拼接并通过
    ``f_sent`` 得到新闻风险表示。监督训练阶段会直接调用 ``module``，因此
    梯度可以回传到 ``phi_g``、``phi_z`` 和 ``f_sent``。
    """

    def __init__(
        self,
        gdelt_dim: int,
        text_dim: int,
        projection_dim: int,
        sent_hidden_sizes: Sequence[int],
        representation_dim: int,
        dropout: float = 0.1,
        final_layer_norm: bool = True,
    ) -> None:
        # 延迟导入 torch.nn，避免未使用阶段强制加载依赖。
        import torch.nn as nn

        # 定义内部类 _Module，封装局部模块逻辑。
        class _Module(nn.Module):
            def __init__(self) -> None:
                # 调用 super().__init__，执行当前步骤的副作用。
                super().__init__()
                # 计算并保存 gdelt_projector，供后续逻辑复用。
                self.gdelt_projector = make_projection_block(
                    gdelt_dim, projection_dim, dropout
                )
                # 计算并保存 text_projector，供后续逻辑复用。
                self.text_projector = make_projection_block(
                    text_dim, projection_dim, dropout
                )
                # 计算并保存 sent_encoder，供后续逻辑复用。
                self.sent_encoder = make_mlp(
                    input_dim=projection_dim * 2,
                    hidden_sizes=sent_hidden_sizes,
                    output_dim=representation_dim,
                    dropout=dropout,
                    final_layer_norm=final_layer_norm,
                )

            def forward(self, gdelt_features: Any, text_embeddings: Any) -> Any:
                # 延迟导入 torch，避免未使用阶段强制加载依赖。
                import torch

                # 计算并保存 gdelt_view，供后续逻辑复用。
                gdelt_view = self.gdelt_projector(gdelt_features)
                # 计算并保存 text_view，供后续逻辑复用。
                text_view = self.text_projector(text_embeddings)
                # 计算并保存 fused，供后续逻辑复用。
                fused = torch.cat([gdelt_view, text_view], dim=1)
                # 返回当前函数的处理结果。
                return self.sent_encoder(fused)

        # 计算并保存 module，供后续逻辑复用。
        self.module = _Module()

    def to(self, device: str) -> "NewsInitializationEncoder":
        """将内部 PyTorch 模块移动到指定设备。

        Parameters
        ----------
        device : str
            PyTorch 设备字符串。

        Returns
        -------
        NewsInitializationEncoder
            当前对象，便于链式调用。

        Raises
        ------
        RuntimeError
            当设备不可用或张量移动失败时由 PyTorch 抛出。
        """
        # 调用 self.module.to，执行当前步骤的副作用。
        self.module.to(device)
        # 返回当前函数的处理结果。
        return self

    def eval(self) -> "NewsInitializationEncoder":
        """将内部模块切换为评估模式。

        Returns
        -------
        NewsInitializationEncoder
            当前对象。

        Raises
        ------
        None
            本函数不抛出自定义异常。
        """
        # 调用 self.module.eval，执行当前步骤的副作用。
        self.module.eval()
        # 返回当前函数的处理结果。
        return self

    def load_checkpoint(self, checkpoint_path: Optional[Path], device: str) -> None:
        """从 checkpoint 加载新闻初始化编码器权重。

        Parameters
        ----------
        checkpoint_path : pathlib.Path or None
            checkpoint 路径；为 ``None`` 时不执行加载。
        device : str
            加载权重时使用的映射设备。

        Returns
        -------
        None
            加载完成后不返回值。

        Raises
        ------
        FileNotFoundError
            当 checkpoint 路径不存在时由 ``torch.load`` 抛出。
        RuntimeError
            当权重键或张量形状与当前模型不匹配时抛出。
        """
        # 根据条件 checkpoint_path is None 选择对应处理分支。
        if checkpoint_path is None:
            # 返回当前分支或函数的处理结果。
            return
        # 延迟导入 torch，避免未使用阶段强制加载依赖。
        import torch

        # 计算并保存 checkpoint，供后续逻辑复用。
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 根据条件 isinstance(checkpoint, dict) and "model_st 选择对应处理分支。
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 计算并保存 state_dict，供当前分支后续使用。
            state_dict = checkpoint["model_state_dict"]
        # 判断 f isinstance(checkpoint, dict) and "state_dict" in che...，处理对应情况。
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # 计算并保存 state_dict，供当前分支后续使用。
            state_dict = checkpoint["state_dict"]
        else:
            # 计算并保存 state_dict，供当前分支后续使用。
            state_dict = checkpoint
        # 调用 self.module.load_state_dict，执行当前步骤的副作用。
        self.module.load_state_dict(state_dict)

    def encode(
        self,
        gdelt_features: np.ndarray,
        text_embeddings: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        """在无梯度模式下批量生成新闻表示。

        Parameters
        ----------
        gdelt_features : numpy.ndarray
            标准化后的结构化特征矩阵，形状为 ``(n_news, gdelt_dim)``。
        text_embeddings : numpy.ndarray
            文本嵌入矩阵，形状为 ``(n_news, text_dim)``。
        batch_size : int
            前向推理批大小。
        device : str
            PyTorch 设备字符串。

        Returns
        -------
        numpy.ndarray
            新闻级表示矩阵，形状为 ``(n_news, representation_dim)``。

        Raises
        ------
        RuntimeError
            当输入维度与模型结构不匹配或设备不可用时由 PyTorch 抛出。

        Notes
        -----
        该方法用于非训练阶段的表示生成。监督训练阶段不会使用此无梯度路径，
        以保证损失可以回传到初始化编码器参数。
        """
        # 延迟导入 torch，避免未使用阶段强制加载依赖。
        import torch

        # 初始化带类型标注的 outputs，明确后续存储内容。
        outputs: List[np.ndarray] = []
        # 调用 self.module.eval，执行当前步骤的副作用。
        self.module.eval()
        # 使用上下文管理器处理资源，确保结束时自动释放。
        with torch.no_grad():
            # 遍历 start in range(0, len(gdelt_features), batch_size)，逐项处理数据。
            for start in range(0, len(gdelt_features), batch_size):
                # 计算并保存 gdelt_batch，供当前分支后续使用。
                gdelt_batch = torch.tensor(
                    gdelt_features[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                # 计算并保存 text_batch，供当前分支后续使用。
                text_batch = torch.tensor(
                    text_embeddings[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                # 计算并保存 encoded，供当前分支后续使用。
                encoded = self.module(gdelt_batch, text_batch)
                # 执行 append，完成当前步骤的副作用。
                outputs.append(encoded.cpu().numpy())
        # 返回当前函数的处理结果。
        return np.vstack(outputs).astype(np.float32)

def softplus_inverse(value: float) -> float:
    """计算 softplus 的数值稳定近似反函数。

    Parameters
    ----------
    value : float
        需要反变换的正数。

    Returns
    -------
    float
        使 ``softplus(x)`` 近似等于 ``value`` 的实数。

    Raises
    ------
    None
        本函数不抛出自定义异常。

    Notes
    -----
    图传播中的 ``alpha``、``beta``、``gamma`` 和 ``eta`` 使用 softplus 约束
    为非负参数，因此初始化时需要该反函数。
    """
    # 计算并保存 value，供后续逻辑复用。
    value = max(float(value), 1e-8)
    # 根据条件 value > 20.0 选择对应处理分支。
    if value > 20.0:
        # 返回当前分支或函数的处理结果。
        return value
    # 返回当前函数的处理结果。
    return float(np.log(np.expm1(value)))


class IndustryGraphPropagationModule:
    """物理启发的产业链图风险传播模块。

    Parameters
    ----------
    node_igpc : numpy.ndarray
        每个图节点的 IGPC 值，形状为 ``(N,)`` 或 ``(N, 1)``。
    normalized_adjacency : numpy.ndarray
        加自环并归一化后的邻接矩阵，形状为 ``(N, N)``。
    risk_dim : int
        新闻风险表示维度。
    state_dim : int
        图节点状态维度。
    risk_hidden_sizes : sequence of int
        新闻风险注入投影网络的隐藏层维度。
    recovery_hidden_sizes : sequence of int
        恢复速度网络的隐藏层维度。
    dropout : float
        MLP Dropout 概率。
    dt : float
        离散传播步长。
    alpha_init : float
        恢复项系数的正值初始化。
    beta_init : float
        扩散项系数的正值初始化。
    gamma_init : float
        二跳扩散系数的正值初始化。
    eta_init : float
        新闻注入系数的正值初始化。
    eps : float
        恢复速度下界，避免除零。
    diffusion_mode : str
        ``"deviation"`` 表示扩散 ``h-h0``，``"direct"`` 表示直接扩散 ``h``。

    Attributes
    ----------
    module : torch.nn.Module
        实际执行传播的 PyTorch 模块。

    Raises
    ------
    ImportError
        当 PyTorch 未安装时抛出。

    Notes
    -----
    核心逻辑包括 IGPC 初始化、新闻注入、恢复项、一跳扩散和二跳扩散。
    ``alpha``、``beta``、``gamma`` 和 ``eta`` 都通过 softplus 参数化为非负
    可训练参数，保证后续监督训练可以稳定反向传播。
    """

    def __init__(
        self,
        *,
        node_igpc: np.ndarray,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        risk_hidden_sizes: Sequence[int],
        recovery_hidden_sizes: Sequence[int],
        dropout: float,
        dt: float,
        alpha_init: float,
        beta_init: float,
        gamma_init: float,
        eta_init: float,
        eps: float,
        diffusion_mode: str,
        initial_state_mode: str = "igpc",
        enable_graph_diffusion: bool = True,
        enable_two_hop_diffusion: bool = True,
        enable_recovery: bool = True,
    ) -> None:
        # 延迟导入 torch，避免未使用阶段强制加载依赖。
        import torch
        # 延迟导入 torch.nn，避免未使用阶段强制加载依赖。
        import torch.nn as nn

        valid_initial_state_modes = {"igpc", "learnable", "fixed_zero"}
        if initial_state_mode not in valid_initial_state_modes:
            raise ValueError(
                "initial_state_mode must be one of "
                f"{sorted(valid_initial_state_modes)}, got {initial_state_mode!r}."
            )

        # 定义内部类 _Module，封装局部模块逻辑。
        class _Module(nn.Module):
            def __init__(self) -> None:
                # 调用 super().__init__，执行当前步骤的副作用。
                super().__init__()
                # 调用 self.register_buffer，执行当前步骤的副作用。
                self.register_buffer(
                    "node_igpc",
                    torch.tensor(node_igpc.reshape(-1, 1), dtype=torch.float32),
                )
                # 调用 self.register_buffer，执行当前步骤的副作用。
                self.register_buffer(
                    "normalized_adjacency",
                    torch.tensor(normalized_adjacency, dtype=torch.float32),
                )
                self.initial_state_mode = initial_state_mode
                self.state_dim = int(state_dim)
                if self.initial_state_mode == "igpc":
                    # 计算并保存 init_projector，供后续逻辑复用。
                    self.init_projector = nn.Linear(1, state_dim)
                elif self.initial_state_mode == "learnable":
                    self.raw_h0 = nn.Parameter(
                        torch.zeros(
                            self.node_igpc.shape[0],
                            state_dim,
                            dtype=torch.float32,
                        )
                    )
                else:
                    self.register_buffer(
                        "fixed_h0",
                        torch.zeros(
                            self.node_igpc.shape[0],
                            state_dim,
                            dtype=torch.float32,
                        ),
                    )
                # 计算并保存 risk_projector，供后续逻辑复用。
                self.risk_projector = make_mlp(
                    input_dim=risk_dim,
                    hidden_sizes=risk_hidden_sizes,
                    output_dim=state_dim,
                    dropout=dropout,
                    final_layer_norm=False,
                )
                # 计算并保存 recovery_net，供后续逻辑复用。
                self.recovery_net = make_mlp(
                    input_dim=state_dim * 2,
                    hidden_sizes=recovery_hidden_sizes,
                    output_dim=state_dim,
                    dropout=dropout,
                    final_layer_norm=False,
                )
                # 计算并保存 raw_alpha，供后续逻辑复用。
                self.raw_alpha = nn.Parameter(
                    torch.tensor(softplus_inverse(alpha_init), dtype=torch.float32)
                )
                # 计算并保存 raw_beta，供后续逻辑复用。
                self.raw_beta = nn.Parameter(
                    torch.tensor(softplus_inverse(beta_init), dtype=torch.float32)
                )
                # 计算并保存 raw_gamma，供后续逻辑复用。
                self.raw_gamma = nn.Parameter(
                    torch.tensor(softplus_inverse(gamma_init), dtype=torch.float32)
                )
                # 计算并保存 raw_eta，供后续逻辑复用。
                self.raw_eta = nn.Parameter(
                    torch.tensor(softplus_inverse(eta_init), dtype=torch.float32)
                )
                # 计算并保存 dt，供后续逻辑复用。
                self.dt = float(dt)
                # 计算并保存 eps，供后续逻辑复用。
                self.eps = float(eps)
                # 计算并保存 diffusion_mode，供后续逻辑复用。
                self.diffusion_mode = diffusion_mode
                self.enable_graph_diffusion = bool(enable_graph_diffusion)
                self.enable_two_hop_diffusion = bool(enable_two_hop_diffusion)
                self.enable_recovery = bool(enable_recovery)

            def positive_scalars(self) -> Tuple[Any, Any, Any, Any]:
                # 延迟导入 torch.nn.functional，避免未使用阶段强制加载依赖。
                import torch.nn.functional as functional

                # 返回当前函数的处理结果。
                return (
                    functional.softplus(self.raw_alpha),
                    functional.softplus(self.raw_beta),
                    functional.softplus(self.raw_gamma),
                    functional.softplus(self.raw_eta),
                )

            def initial_state(self) -> Any:
                # 延迟导入 torch，避免未使用阶段强制加载依赖。
                import torch

                if self.initial_state_mode == "igpc":
                    return torch.sigmoid(self.init_projector(torch.log1p(self.node_igpc)))
                if self.initial_state_mode == "learnable":
                    return torch.sigmoid(self.raw_h0)
                return self.fixed_h0

            def diffusion_operator(self, values: Any) -> Any:
                # 计算并保存 neighbor_values，供后续逻辑复用。
                neighbor_values = self.normalized_adjacency @ values
                # 计算并保存 row_weight，供后续逻辑复用。
                row_weight = self.normalized_adjacency.sum(dim=1, keepdim=True)
                # 返回当前函数的处理结果。
                return neighbor_values - row_weight * values

            def forward(
                self,
                node_risk_sequence: Any,
                node_risk_mask: Optional[Any] = None,
                initial_state: Optional[Any] = None,
                return_sequence: bool = True,
            ) -> Any:
                # 延迟导入 torch，避免未使用阶段强制加载依赖。
                import torch
                # 延迟导入 torch.nn.functional，避免未使用阶段强制加载依赖。
                import torch.nn.functional as functional

                # 根据条件 node_risk_sequence.dim() == 2 选择对应处理分支。
                if node_risk_sequence.dim() == 2:
                    # 计算并保存 node_risk_sequence，供当前分支后续使用。
                    node_risk_sequence = node_risk_sequence.unsqueeze(0)
                # 根据条件 node_risk_mask is not None and node_risk_m 选择对应处理分支。
                if node_risk_mask is not None and node_risk_mask.dim() == 2:
                    # 计算并保存 node_risk_mask，供当前分支后续使用。
                    node_risk_mask = node_risk_mask.unsqueeze(0)

                # 计算并保存 h0，供后续逻辑复用。
                h0 = self.initial_state() if initial_state is None else initial_state
                # 计算并保存 h，供后续逻辑复用。
                h = h0
                # 初始化带类型标注的 states，明确后续存储内容。
                states: List[Any] = []
                # 计算并保存 alpha, beta, gamma, eta，供后续逻辑复用。
                alpha, beta, gamma, eta = self.positive_scalars()

                # 遍历 range(node_risk_sequence.shape[0])，逐项执行处理逻辑。
                for time_index in range(node_risk_sequence.shape[0]):
                    # 计算并保存 risk，供当前分支后续使用。
                    risk = node_risk_sequence[time_index]
                    # 计算并保存 z，供当前分支后续使用。
                    z = self.risk_projector(risk)
                    # 判断 node_risk_mask is not None，处理对应情况。
                    if node_risk_mask is not None:
                        # 计算并保存 z，供当前分支后续使用。
                        z = z * node_risk_mask[time_index]

                    # 计算并保存 diffusion_source，供当前分支后续使用。
                    if self.enable_graph_diffusion:
                        diffusion_source = h - h0 if self.diffusion_mode == "deviation" else h
                    # 计算并保存 one_hop，供当前分支后续使用。
                        one_hop = self.diffusion_operator(diffusion_source)
                    # 计算并保存 two_hop，供当前分支后续使用。
                        if self.enable_two_hop_diffusion:
                            two_hop = self.normalized_adjacency @ one_hop
                    # 计算并保存 h_diff，供当前分支后续使用。
                            h_diff = self.dt * beta * (one_hop + gamma * two_hop)
                        else:
                            h_diff = self.dt * beta * one_hop
                    else:
                        h_diff = h.new_zeros(h.shape)

                    # 计算并保存 recovery_input，供当前分支后续使用。
                    if self.enable_recovery:
                        recovery_input = torch.cat([h, z], dim=1)
                    # 计算并保存 v_rec，供当前分支后续使用。
                        v_rec = functional.softplus(self.recovery_net(recovery_input)) + self.eps
                    # 计算并保存 h_react，供当前分支后续使用。
                        h_react = self.dt * v_rec * torch.tanh(alpha * (h0 - h) / v_rec)
                    else:
                        h_react = h.new_zeros(h.shape)
                    # 计算并保存 h_inject，供当前分支后续使用。
                    h_inject = self.dt * eta * z

                    # 计算并保存 h，供当前分支后续使用。
                    h = h + h_inject + h_react + h_diff
                    # 执行 append，完成当前步骤的副作用。
                    states.append(h)

                # 根据条件 return_sequence 选择对应处理分支。
                if return_sequence:
                    # 返回当前分支或函数的处理结果。
                    return torch.stack(states, dim=0)
                # 返回当前函数的处理结果。
                return h

        # 计算并保存 module，供后续逻辑复用。
        self.module = _Module()

    def to(self, device: str) -> "IndustryGraphPropagationModule":
        """将传播模块移动到指定设备。

        Parameters
        ----------
        device : str
            PyTorch 设备字符串。

        Returns
        -------
        IndustryGraphPropagationModule
            当前对象。

        Raises
        ------
        RuntimeError
            当设备不可用或张量移动失败时由 PyTorch 抛出。
        """
        # 调用 self.module.to，执行当前步骤的副作用。
        self.module.to(device)
        # 返回当前函数的处理结果。
        return self

    def eval(self) -> "IndustryGraphPropagationModule":
        """将传播模块切换到评估模式。

        Returns
        -------
        IndustryGraphPropagationModule
            当前对象。

        Raises
        ------
        None
            本函数不抛出自定义异常。
        """
        # 调用 self.module.eval，执行当前步骤的副作用。
        self.module.eval()
        # 返回当前函数的处理结果。
        return self

    def load_checkpoint(self, checkpoint_path: Optional[Path], device: str) -> None:
        """加载图传播模块 checkpoint。

        Parameters
        ----------
        checkpoint_path : pathlib.Path or None
            权重文件路径；为 ``None`` 时不加载。
        device : str
            权重加载设备。

        Returns
        -------
        None
            加载完成后不返回值。

        Raises
        ------
        FileNotFoundError
            当 checkpoint 文件不存在时由 ``torch.load`` 抛出。
        RuntimeError
            当权重结构与当前模块不匹配时抛出。
        """
        # 根据条件 checkpoint_path is None 选择对应处理分支。
        if checkpoint_path is None:
            # 返回当前分支或函数的处理结果。
            return
        # 延迟导入 torch，避免未使用阶段强制加载依赖。
        import torch

        # 计算并保存 checkpoint，供后续逻辑复用。
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 根据条件 isinstance(checkpoint, dict) and "model_st 选择对应处理分支。
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 计算并保存 state_dict，供当前分支后续使用。
            state_dict = checkpoint["model_state_dict"]
        # 判断 f isinstance(checkpoint, dict) and "state_dict" in che...，处理对应情况。
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # 计算并保存 state_dict，供当前分支后续使用。
            state_dict = checkpoint["state_dict"]
        else:
            # 计算并保存 state_dict，供当前分支后续使用。
            state_dict = checkpoint
        # 调用 self.module.load_state_dict，执行当前步骤的副作用。
        self.module.load_state_dict(state_dict)

    def run_numpy(
        self,
        node_risk_sequence: np.ndarray,
        node_risk_mask: np.ndarray,
        device: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在无梯度模式下运行图传播并返回 NumPy 结果。

        Parameters
        ----------
        node_risk_sequence : numpy.ndarray
            节点风险序列，形状为 ``(T, N, risk_dim)``。
        node_risk_mask : numpy.ndarray
            节点风险掩码，形状为 ``(T, N, 1)``。
        device : str
            PyTorch 设备字符串。

        Returns
        -------
        initial_state : numpy.ndarray
            初始节点状态 ``h0``，形状为 ``(N, state_dim)``。
        states : numpy.ndarray
            传播后的状态序列，形状为 ``(T, N, state_dim)``。

        Raises
        ------
        RuntimeError
            当输入维度不匹配或设备不可用时由 PyTorch 抛出。

        Notes
        -----
        该方法用于推理和中间结果导出；监督训练阶段直接调用内部 ``module``，
        保持梯度链路。
        """
        # 延迟导入 torch，避免未使用阶段强制加载依赖。
        import torch

        # 调用 self.module.eval，执行当前步骤的副作用。
        self.module.eval()
        # 计算并保存 risk_tensor，供后续逻辑复用。
        risk_tensor = torch.tensor(node_risk_sequence, dtype=torch.float32, device=device)
        # 计算并保存 mask_tensor，供后续逻辑复用。
        mask_tensor = torch.tensor(node_risk_mask, dtype=torch.float32, device=device)
        # 使用上下文管理器处理资源，确保结束时自动释放。
        with torch.no_grad():
            # 计算并保存 initial_state，供当前分支后续使用。
            initial_state = self.module.initial_state()
            # 计算并保存 states，供当前分支后续使用。
            states = self.module(risk_tensor, mask_tensor, initial_state=initial_state)
        # 返回当前函数的处理结果。
        return (
            initial_state.cpu().numpy().astype(np.float32),
            states.cpu().numpy().astype(np.float32),
        )


class RiskEvaluationScorer:
    """将传播后的节点状态映射为标量风险分数的 MLP。

    Parameters
    ----------
    state_dim : int
        输入节点状态维度。
    hidden_sizes : sequence of int
        评估 MLP 的隐藏层维度。
    dropout : float
        Dropout 概率。

    Attributes
    ----------
    module : torch.nn.Module
        实际的可训练 MLP。
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        # 计算并保存 module，供后续逻辑复用。
        self.module = make_mlp(
            input_dim=state_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            dropout=dropout,
            final_layer_norm=False,
        )

    def to(self, device: str) -> "RiskEvaluationScorer":
        """将打分器移动到指定设备。

        Parameters
        ----------
        device : str
            PyTorch 设备字符串。

        Returns
        -------
        RiskEvaluationScorer
            当前对象。

        Raises
        ------
        RuntimeError
            当设备不可用或张量移动失败时由 PyTorch 抛出。
        """
        # 调用 self.module.to，执行当前步骤的副作用。
        self.module.to(device)
        # 返回当前函数的处理结果。
        return self
