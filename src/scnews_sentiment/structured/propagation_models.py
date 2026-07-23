from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scnews_sentiment.structured.common import parse_hidden_sizes
from src.scnews_sentiment.structured.models import (
    IndustryGraphPropagationModule,
    make_mlp,
    softplus_inverse,
)


LABEL_HISTORY_PROPAGATION_MODELS = {"itransformer", "timesnet"}


def is_label_history_propagation_model(model_name: Any) -> bool:
    return str(model_name).lower() in LABEL_HISTORY_PROPAGATION_MODELS


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    row_sum = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        np.maximum(row_sum, 1e-12),
        out=np.zeros_like(values, dtype=np.float32),
        where=row_sum > 1e-12,
    ).astype(np.float32)


def _load_module_checkpoint(module: nn.Module, checkpoint_path: Optional[Path], device: str) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "graph_module_state_dict" in checkpoint:
        state_dict = checkpoint["graph_module_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict)


class _PropagationWrapper:
    def __init__(self, module: nn.Module) -> None:
        self.module = module

    def to(self, device: str) -> "_PropagationWrapper":
        self.module.to(device)
        return self

    def eval(self) -> "_PropagationWrapper":
        self.module.eval()
        return self

    def load_checkpoint(self, checkpoint_path: Optional[Path], device: str) -> None:
        _load_module_checkpoint(self.module, checkpoint_path, device)


class NodeInitialState(nn.Module):
    def __init__(
        self,
        node_igpc: np.ndarray,
        state_dim: int,
        *,
        mode: str,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        valid_modes = {"igpc", "learnable", "fixed_zero"}
        if mode not in valid_modes:
            raise ValueError(f"initial_state_mode must be one of {sorted(valid_modes)}.")
        valid_activations = {"sigmoid", "tanh", "none"}
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {sorted(valid_activations)}.")

        self.mode = mode
        self.activation = activation
        self.register_buffer(
            "node_igpc",
            torch.tensor(node_igpc.reshape(-1, 1), dtype=torch.float32),
        )
        if mode == "igpc":
            self.projector = nn.Linear(1, state_dim)
        elif mode == "learnable":
            self.raw_state = nn.Parameter(
                torch.zeros(self.node_igpc.shape[0], state_dim, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "fixed_state",
                torch.zeros(self.node_igpc.shape[0], state_dim, dtype=torch.float32),
            )

    def forward(self) -> Any:
        if self.mode == "igpc":
            state = self.projector(torch.log1p(self.node_igpc))
        elif self.mode == "learnable":
            state = self.raw_state
        else:
            state = self.fixed_state

        if self.activation == "sigmoid":
            return torch.sigmoid(state)
        if self.activation == "tanh":
            return torch.tanh(state)
        return state


def _parse_kernel_sizes(text: Any) -> Tuple[int, ...]:
    if text is None:
        return (1, 3, 5)
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("timesnet_kernel_sizes must contain positive integers.")
        values.append(value)
    return tuple(values) if values else (1, 3, 5)


def _causal_conv_time(x: Any, conv: nn.Conv1d) -> Any:
    """Apply a causal Conv1d along time to a ``(T, N, D)`` tensor."""
    steps = x.shape[0]
    temporal = x.permute(1, 2, 0)
    pad = int(conv.kernel_size[0]) - 1
    if pad > 0:
        temporal = F.pad(temporal, (pad, 0))
    out = conv(temporal)
    return out[..., :steps].permute(2, 0, 1)


class STGCNPropagationNetwork(nn.Module):
    """Causal STGCN-style baseline over news representations and the supply-chain graph."""

    def __init__(
        self,
        *,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        dropout: float,
        kernel_size: int,
        **_: Any,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("stgcn_kernel_size must be positive.")
        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        self.input_projection = nn.Linear(risk_dim, state_dim)
        self.temporal_in = nn.Conv1d(state_dim, state_dim, kernel_size=kernel_size)
        self.temporal_out = nn.Conv1d(state_dim, state_dim, kernel_size=kernel_size)
        self.graph_linear = nn.Linear(state_dim, state_dim)
        self.layer_norm = nn.LayerNorm(state_dim)
        self.output_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)

        x = self.input_projection(node_risk_sequence)
        if node_risk_mask is not None:
            x = x * node_risk_mask
        residual = x
        x = self.dropout(F.gelu(_causal_conv_time(x, self.temporal_in)))
        x = torch.einsum("ij,tjd->tid", self.normalized_adjacency, x)
        x = self.dropout(F.gelu(self.layer_norm(self.graph_linear(x))))
        x = self.dropout(F.gelu(_causal_conv_time(x, self.temporal_out)))
        states = self.output_norm(x + residual)
        if return_sequence:
            return states
        return states[-1]


class ITransformerPropagationNetwork(nn.Module):
    """Causal iTransformer-style baseline over ArticleScore history."""

    def __init__(
        self,
        *,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        dropout: float,
        lookback: int,
        heads: int,
        **_: Any,
    ) -> None:
        super().__init__()
        if lookback <= 0:
            raise ValueError("ts_baseline_lookback must be positive.")
        if heads <= 0:
            raise ValueError("itransformer_heads must be positive.")
        if state_dim % heads != 0:
            raise ValueError("graph_state_dim must be divisible by itransformer_heads.")
        self.lookback = int(lookback)
        self.risk_dim = int(risk_dim)
        self.register_buffer(
            "normalized_adjacency",
            torch.zeros_like(torch.tensor(normalized_adjacency, dtype=torch.float32)),
        )
        self.token_embedding = nn.Linear(self.lookback * self.risk_dim, state_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = make_mlp(
            input_dim=state_dim,
            hidden_sizes=(state_dim * 2,),
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.norm_attn = nn.LayerNorm(state_dim)
        self.norm_ffn = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def _window_at(self, sequence: Any, time_index: int) -> Any:
        start = max(0, time_index - self.lookback + 1)
        window = sequence[start : time_index + 1]
        if window.shape[0] < self.lookback:
            pad = window.new_zeros(
                self.lookback - window.shape[0],
                window.shape[1],
                window.shape[2],
            )
            window = torch.cat([pad, window], dim=0)
        return window.permute(1, 0, 2).reshape(window.shape[1], -1)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)

        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            tokens = self.token_embedding(self._window_at(node_risk_sequence, time_index))
            attn_out, _ = self.attention(
                tokens.unsqueeze(0),
                tokens.unsqueeze(0),
                tokens.unsqueeze(0),
                need_weights=False,
            )
            h = self.norm_attn(tokens + self.dropout(attn_out.squeeze(0)))
            h = self.norm_ffn(h + self.dropout(self.feed_forward(h)))
            states.append(h)

        stacked = torch.stack(states, dim=0)
        if return_sequence:
            return stacked
        return stacked[-1]


class TimesNetPropagationNetwork(nn.Module):
    """Causal multi-scale temporal CNN baseline inspired by TimesNet."""

    def __init__(
        self,
        *,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        dropout: float,
        lookback: int,
        kernel_sizes: Sequence[int],
        **_: Any,
    ) -> None:
        super().__init__()
        if lookback <= 0:
            raise ValueError("ts_baseline_lookback must be positive.")
        kernels = tuple(int(k) for k in kernel_sizes)
        if not kernels or any(k <= 0 for k in kernels):
            raise ValueError("timesnet_kernel_sizes must contain positive integers.")
        self.lookback = int(lookback)
        self.register_buffer(
            "normalized_adjacency",
            torch.zeros_like(torch.tensor(normalized_adjacency, dtype=torch.float32)),
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    risk_dim,
                    state_dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
                for kernel in kernels
            ]
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(kernels), dtype=torch.float32))
        self.residual_projection = nn.Linear(risk_dim, state_dim)
        self.layer_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def _window_at(self, sequence: Any, time_index: int) -> Any:
        start = max(0, time_index - self.lookback + 1)
        window = sequence[start : time_index + 1]
        if window.shape[0] < self.lookback:
            pad = window.new_zeros(
                self.lookback - window.shape[0],
                window.shape[1],
                window.shape[2],
            )
            window = torch.cat([pad, window], dim=0)
        return window.permute(1, 2, 0)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)

        states: List[Any] = []
        weights = torch.softmax(self.branch_logits, dim=0)
        for time_index in range(node_risk_sequence.shape[0]):
            window = self._window_at(node_risk_sequence, time_index)
            branch_values = []
            for branch in self.branches:
                branch_values.append(F.gelu(branch(window)).mean(dim=-1))
            stacked = torch.stack(branch_values, dim=0)
            h = torch.einsum("k,knd->nd", weights, stacked)
            residual = self.residual_projection(node_risk_sequence[time_index])
            states.append(self.layer_norm(self.dropout(h) + residual))

        output = torch.stack(states, dim=0)
        if return_sequence:
            return output
        return output[-1]


class GCNPropagationNetwork(nn.Module):
    def __init__(
        self,
        *,
        node_igpc: np.ndarray,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        risk_hidden_sizes: Sequence[int],
        dropout: float,
        initial_state_mode: str,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        self.initial_state = NodeInitialState(
            node_igpc,
            state_dim,
            mode=initial_state_mode,
            activation="sigmoid",
        )
        self.risk_projector = make_mlp(
            input_dim=risk_dim,
            hidden_sizes=risk_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.input_linear = nn.Linear(state_dim * 2, state_dim)
        self.layer_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)

        h = self.initial_state() if initial_state is None else initial_state
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            z = self.risk_projector(node_risk_sequence[time_index])
            if node_risk_mask is not None:
                z = z * node_risk_mask[time_index]
            fused = self.input_linear(torch.cat([h, z], dim=1))
            h = self.normalized_adjacency @ fused
            h = self.dropout(F.gelu(self.layer_norm(h)))
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class GATPropagationNetwork(nn.Module):
    def __init__(
        self,
        *,
        node_igpc: np.ndarray,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        risk_hidden_sizes: Sequence[int],
        dropout: float,
        initial_state_mode: str,
        heads: int,
    ) -> None:
        super().__init__()
        if heads <= 0:
            raise ValueError("gat_heads must be positive.")
        if state_dim % heads != 0:
            raise ValueError("graph_state_dim must be divisible by gat_heads.")

        self.heads = int(heads)
        self.head_dim = state_dim // self.heads
        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        self.register_buffer(
            "attention_mask",
            torch.tensor(normalized_adjacency > 0, dtype=torch.bool),
        )
        self.initial_state = NodeInitialState(
            node_igpc,
            state_dim,
            mode=initial_state_mode,
            activation="sigmoid",
        )
        self.risk_projector = make_mlp(
            input_dim=risk_dim,
            hidden_sizes=risk_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.input_linear = nn.Linear(state_dim * 2, state_dim)
        self.head_projection = nn.Linear(state_dim, state_dim, bias=False)
        self.attn_target = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_source = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.output_linear = nn.Linear(state_dim, state_dim)
        self.layer_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn_target)
        nn.init.xavier_uniform_(self.attn_source)

    def _attention_layer(self, node_features: Any) -> Any:
        node_count = node_features.shape[0]
        projected = self.head_projection(node_features).view(
            node_count,
            self.heads,
            self.head_dim,
        )
        target_scores = (projected * self.attn_target).sum(dim=-1)
        source_scores = (projected * self.attn_source).sum(dim=-1)
        scores = self.leaky_relu(target_scores[:, None, :] + source_scores[None, :, :])
        scores = scores.masked_fill(~self.attention_mask[:, :, None], -1e9)
        attention = torch.softmax(scores, dim=1)
        attention = self.dropout(attention)
        head_output = torch.einsum("ijh,jhd->ihd", attention, projected)
        return head_output.reshape(node_count, self.heads * self.head_dim)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)

        h = self.initial_state() if initial_state is None else initial_state
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            z = self.risk_projector(node_risk_sequence[time_index])
            if node_risk_mask is not None:
                z = z * node_risk_mask[time_index]
            fused = F.gelu(self.input_linear(torch.cat([h, z], dim=1)))
            attended = self._attention_layer(fused)
            h = self.output_linear(attended)
            h = self.dropout(F.gelu(self.layer_norm(h)))
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class DiffusionGraphConv(nn.Module):
    def __init__(
        self,
        *,
        supports: Sequence[np.ndarray],
        input_dim: int,
        output_dim: int,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        if diffusion_steps <= 0:
            raise ValueError("dcrnn_diffusion_steps must be positive.")
        self.diffusion_steps = int(diffusion_steps)
        self.support_names: List[str] = []
        for index, support in enumerate(supports):
            name = f"support_{index}"
            self.register_buffer(name, torch.tensor(support, dtype=torch.float32))
            self.support_names.append(name)
        feature_multiplier = 1 + len(self.support_names) * self.diffusion_steps
        self.linear = nn.Linear(input_dim * feature_multiplier, output_dim)

    def forward(self, inputs: Any) -> Any:
        pieces = [inputs]
        for name in self.support_names:
            support = getattr(self, name)
            propagated = inputs
            for _ in range(self.diffusion_steps):
                propagated = support @ propagated
                pieces.append(propagated)
        return self.linear(torch.cat(pieces, dim=1))


class AdaptiveGraphConv(nn.Module):
    def __init__(
        self,
        *,
        node_embeddings: nn.Parameter,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.node_embeddings = node_embeddings
        self.weight_pool = nn.Parameter(
            torch.empty(embed_dim, input_dim, output_dim, dtype=torch.float32)
        )
        self.bias_pool = nn.Parameter(
            torch.empty(embed_dim, output_dim, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def adaptive_adjacency(self) -> Any:
        logits = F.relu(self.node_embeddings @ self.node_embeddings.T)
        return torch.softmax(logits, dim=1)

    def forward(self, inputs: Any) -> Any:
        support = self.adaptive_adjacency()
        graph_inputs = inputs + support @ inputs
        weights = torch.einsum(
            "ne,eio->nio",
            self.node_embeddings,
            self.weight_pool,
        )
        bias = self.node_embeddings @ self.bias_pool
        return torch.einsum("ni,nio->no", graph_inputs, weights) + bias


class AGCRNPropagationNetwork(nn.Module):
    """AGCRN baseline with adaptive graph generation and node-adaptive GRU gates."""

    def __init__(
        self,
        *,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        risk_hidden_sizes: Sequence[int],
        dropout: float,
        embed_dim: int,
        **_: Any,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("agcrn_embed_dim must be positive.")
        node_count = int(normalized_adjacency.shape[0])
        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        self.node_embeddings = nn.Parameter(
            torch.empty(node_count, embed_dim, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        self.risk_projector = make_mlp(
            input_dim=risk_dim,
            hidden_sizes=risk_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.gate_conv = AdaptiveGraphConv(
            node_embeddings=self.node_embeddings,
            input_dim=state_dim * 2,
            output_dim=state_dim * 2,
            embed_dim=embed_dim,
        )
        self.candidate_conv = AdaptiveGraphConv(
            node_embeddings=self.node_embeddings,
            input_dim=state_dim * 2,
            output_dim=state_dim,
            embed_dim=embed_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def adaptive_adjacency(self) -> Any:
        return self.gate_conv.adaptive_adjacency()

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)

        h = (
            node_risk_sequence.new_zeros(
                node_risk_sequence.shape[1],
                self.candidate_conv.bias_pool.shape[-1],
            )
            if initial_state is None
            else initial_state
        )
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            z = self.risk_projector(node_risk_sequence[time_index])
            if node_risk_mask is not None:
                z = z * node_risk_mask[time_index]
            reset_gate, update_gate = torch.chunk(
                torch.sigmoid(self.gate_conv(torch.cat([z, h], dim=1))),
                2,
                dim=1,
            )
            candidate = torch.tanh(
                self.candidate_conv(torch.cat([z, reset_gate * h], dim=1))
            )
            h = update_gate * h + (1.0 - update_gate) * candidate
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class DCRNNPropagationNetwork(nn.Module):
    def __init__(
        self,
        *,
        node_igpc: np.ndarray,
        normalized_adjacency: np.ndarray,
        adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        risk_hidden_sizes: Sequence[int],
        dropout: float,
        initial_state_mode: str,
        diffusion_steps: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        supports = [_row_normalize(adjacency)]
        if bidirectional:
            supports.append(_row_normalize(adjacency.T))
        self.initial_state = NodeInitialState(
            node_igpc,
            state_dim,
            mode=initial_state_mode,
            activation="sigmoid",
        )
        self.risk_projector = make_mlp(
            input_dim=risk_dim,
            hidden_sizes=risk_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.gate_conv = DiffusionGraphConv(
            supports=supports,
            input_dim=state_dim * 2,
            output_dim=state_dim * 2,
            diffusion_steps=diffusion_steps,
        )
        self.candidate_conv = DiffusionGraphConv(
            supports=supports,
            input_dim=state_dim * 2,
            output_dim=state_dim,
            diffusion_steps=diffusion_steps,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)

        h = self.initial_state() if initial_state is None else initial_state
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            z = self.risk_projector(node_risk_sequence[time_index])
            if node_risk_mask is not None:
                z = z * node_risk_mask[time_index]
            reset_gate, update_gate = torch.chunk(
                torch.sigmoid(self.gate_conv(torch.cat([z, h], dim=1))),
                2,
                dim=1,
            )
            candidate = torch.tanh(
                self.candidate_conv(torch.cat([z, reset_gate * h], dim=1))
            )
            h = update_gate * h + (1.0 - update_gate) * candidate
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class LSTMPropagationNetwork(nn.Module):
    def __init__(
        self,
        *,
        node_igpc: np.ndarray,
        normalized_adjacency: np.ndarray,
        risk_dim: int,
        state_dim: int,
        risk_hidden_sizes: Sequence[int],
        dropout: float,
        initial_state_mode: str,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        self.initial_hidden = NodeInitialState(
            node_igpc,
            state_dim,
            mode=initial_state_mode,
            activation="sigmoid",
        )
        self.initial_cell = NodeInitialState(
            node_igpc,
            state_dim,
            mode=initial_state_mode,
            activation="tanh",
        )
        self.risk_projector = make_mlp(
            input_dim=risk_dim,
            hidden_sizes=risk_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.cell = nn.LSTMCell(state_dim, state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)

        if initial_state is None:
            h = self.initial_hidden()
            c = self.initial_cell()
        else:
            h = initial_state
            c = torch.zeros_like(h)

        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            z = self.risk_projector(node_risk_sequence[time_index])
            if node_risk_mask is not None:
                z = z * node_risk_mask[time_index]
            h, c = self.cell(z, (h, c))
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class _GatedReactionDiffusionBase(nn.Module):
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
        initial_state_mode: str,
        enable_graph_diffusion: bool,
        enable_two_hop_diffusion: bool,
        enable_recovery: bool,
    ) -> None:
        super().__init__()
        valid_diffusion_modes = {"deviation", "direct"}
        if diffusion_mode not in valid_diffusion_modes:
            raise ValueError(
                "diffusion_mode must be one of "
                f"{sorted(valid_diffusion_modes)}, got {diffusion_mode!r}."
            )

        self.register_buffer(
            "normalized_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
        )
        self.initial_state = NodeInitialState(
            node_igpc,
            state_dim,
            mode=initial_state_mode,
            activation="sigmoid",
        )
        self.risk_projector = make_mlp(
            input_dim=risk_dim,
            hidden_sizes=risk_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.recovery_net = make_mlp(
            input_dim=state_dim * 2,
            hidden_sizes=recovery_hidden_sizes,
            output_dim=state_dim,
            dropout=dropout,
            final_layer_norm=False,
        )
        self.raw_alpha = nn.Parameter(
            torch.tensor(softplus_inverse(alpha_init), dtype=torch.float32)
        )
        self.raw_beta = nn.Parameter(
            torch.tensor(softplus_inverse(beta_init), dtype=torch.float32)
        )
        self.raw_gamma = nn.Parameter(
            torch.tensor(softplus_inverse(gamma_init), dtype=torch.float32)
        )
        self.raw_eta = nn.Parameter(
            torch.tensor(softplus_inverse(eta_init), dtype=torch.float32)
        )
        self.dt = float(dt)
        self.eps = float(eps)
        self.diffusion_mode = diffusion_mode
        self.enable_graph_diffusion = bool(enable_graph_diffusion)
        self.enable_two_hop_diffusion = bool(enable_two_hop_diffusion)
        self.enable_recovery = bool(enable_recovery)
        self.dropout = nn.Dropout(dropout)

    def positive_scalars(self) -> Tuple[Any, Any, Any, Any]:
        return (
            F.softplus(self.raw_alpha),
            F.softplus(self.raw_beta),
            F.softplus(self.raw_gamma),
            F.softplus(self.raw_eta),
        )

    def diffusion_operator(self, values: Any) -> Any:
        neighbor_values = self.normalized_adjacency @ values
        row_weight = self.normalized_adjacency.sum(dim=1, keepdim=True)
        return neighbor_values - row_weight * values

    def _project_risk(self, risk: Any, mask: Optional[Any]) -> Any:
        z = self.risk_projector(risk)
        if mask is not None:
            z = z * mask
        return z

    def _reaction_diffusion_components(self, h: Any, h0: Any, z: Any) -> Tuple[Any, Any, Any]:
        alpha, beta, gamma, eta = self.positive_scalars()

        if self.enable_graph_diffusion:
            if self.diffusion_mode == "deviation":
                diffusion_source = h - h0
            else:
                diffusion_source = h
            one_hop = self.diffusion_operator(diffusion_source)
            if self.enable_two_hop_diffusion:
                two_hop = self.normalized_adjacency @ one_hop
                h_diff = self.dt * beta * (one_hop + gamma * two_hop)
            else:
                h_diff = self.dt * beta * one_hop
        else:
            h_diff = h.new_zeros(h.shape)

        if self.enable_recovery:
            recovery_input = torch.cat([h, z], dim=1)
            v_rec = F.softplus(self.recovery_net(recovery_input)) + self.eps
            h_react = self.dt * v_rec * torch.tanh(alpha * (h0 - h) / v_rec)
        else:
            h_react = h.new_zeros(h.shape)

        h_inject = self.dt * eta * z
        return h_inject, h_react, h_diff

    def _reaction_diffusion_residual(self, h: Any, h0: Any, z: Any) -> Any:
        h_inject, h_react, h_diff = self._reaction_diffusion_components(h, h0, z)
        return h_inject + h_react + h_diff

    @staticmethod
    def _prepare_sequence(
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any],
    ) -> Tuple[Any, Optional[Any]]:
        if node_risk_sequence.dim() == 2:
            node_risk_sequence = node_risk_sequence.unsqueeze(0)
        if node_risk_mask is not None and node_risk_mask.dim() == 2:
            node_risk_mask = node_risk_mask.unsqueeze(0)
        return node_risk_sequence, node_risk_mask


class GRUGatedReactionDiffusionPropagationNetwork(_GatedReactionDiffusionBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        state_dim = int(kwargs["state_dim"])
        self.update_gate = nn.Linear(state_dim * 2, state_dim)
        self.reset_gate = nn.Linear(state_dim * 2, state_dim)
        self.candidate_linear = nn.Linear(state_dim * 2, state_dim)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        node_risk_sequence, node_risk_mask = self._prepare_sequence(
            node_risk_sequence,
            node_risk_mask,
        )

        h0 = self.initial_state() if initial_state is None else initial_state
        h = h0
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            mask = None if node_risk_mask is None else node_risk_mask[time_index]
            z = self._project_risk(node_risk_sequence[time_index], mask)
            residual = self._reaction_diffusion_residual(h, h0, z)
            gate_input = torch.cat([h, residual], dim=1)
            update_gate = torch.sigmoid(self.update_gate(gate_input))
            reset_gate = torch.sigmoid(self.reset_gate(gate_input))
            candidate_input = torch.cat([reset_gate * h, residual], dim=1)
            candidate = torch.tanh(self.candidate_linear(candidate_input))
            h = update_gate * h + (1.0 - update_gate) * candidate
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class LSTMGatedReactionDiffusionPropagationNetwork(_GatedReactionDiffusionBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        state_dim = int(kwargs["state_dim"])
        self.gate_linear = nn.Linear(state_dim * 2, state_dim * 4)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        node_risk_sequence, node_risk_mask = self._prepare_sequence(
            node_risk_sequence,
            node_risk_mask,
        )

        h0 = self.initial_state() if initial_state is None else initial_state
        h = h0
        c = h0
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            mask = None if node_risk_mask is None else node_risk_mask[time_index]
            z = self._project_risk(node_risk_sequence[time_index], mask)
            residual = self._reaction_diffusion_residual(h, h0, z)
            gate_input = torch.cat([h, residual], dim=1)
            forget_raw, input_raw, output_raw, candidate_raw = torch.chunk(
                self.gate_linear(gate_input),
                4,
                dim=1,
            )
            forget_gate = torch.sigmoid(forget_raw)
            input_gate = torch.sigmoid(input_raw)
            output_gate = torch.sigmoid(output_raw)
            candidate = torch.tanh(candidate_raw)
            c = forget_gate * c + input_gate * candidate
            h = output_gate * torch.tanh(c)
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class GRUZGatedReactionDiffusionPropagationNetwork(_GatedReactionDiffusionBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        state_dim = int(kwargs["state_dim"])
        self.update_gate = nn.Linear(state_dim * 2, state_dim)
        self.reset_gate = nn.Linear(state_dim * 2, state_dim)
        self.candidate_linear = nn.Linear(state_dim * 2, state_dim)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        node_risk_sequence, node_risk_mask = self._prepare_sequence(
            node_risk_sequence,
            node_risk_mask,
        )

        h0 = self.initial_state() if initial_state is None else initial_state
        h = h0
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            mask = None if node_risk_mask is None else node_risk_mask[time_index]
            z = self._project_risk(node_risk_sequence[time_index], mask)
            residual = self._reaction_diffusion_residual(h, h0, z)
            gate_input = torch.cat([h, z], dim=1)
            update_gate = torch.sigmoid(self.update_gate(gate_input))
            reset_gate = torch.sigmoid(self.reset_gate(gate_input))
            candidate_input = torch.cat([reset_gate * h, residual], dim=1)
            candidate = torch.tanh(self.candidate_linear(candidate_input))
            h = update_gate * h + (1.0 - update_gate) * candidate
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class LSTMZGatedReactionDiffusionPropagationNetwork(_GatedReactionDiffusionBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        state_dim = int(kwargs["state_dim"])
        self.forget_gate = nn.Linear(state_dim * 2, state_dim)
        self.input_gate = nn.Linear(state_dim * 2, state_dim)
        self.output_gate = nn.Linear(state_dim * 2, state_dim)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        node_risk_sequence, node_risk_mask = self._prepare_sequence(
            node_risk_sequence,
            node_risk_mask,
        )

        h0 = self.initial_state() if initial_state is None else initial_state
        h = h0
        c = h0
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            mask = None if node_risk_mask is None else node_risk_mask[time_index]
            z = self._project_risk(node_risk_sequence[time_index], mask)
            residual = self._reaction_diffusion_residual(h, h0, z)
            gate_input = torch.cat([h, z], dim=1)
            forget_gate = torch.sigmoid(self.forget_gate(gate_input))
            input_gate = torch.sigmoid(self.input_gate(gate_input))
            output_gate = torch.sigmoid(self.output_gate(gate_input))
            c = forget_gate * c + input_gate * residual
            h = output_gate * torch.tanh(c)
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class LSTMComponentGatedReactionDiffusionPropagationNetwork(_GatedReactionDiffusionBase):
    def __init__(
        self,
        *,
        component_scale_mode: str,
        component_scale_init: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        valid_scale_modes = {"learnable", "none"}
        if component_scale_mode not in valid_scale_modes:
            raise ValueError(
                "component_scale_mode must be one of "
                f"{sorted(valid_scale_modes)}, got {component_scale_mode!r}."
            )
        if component_scale_init <= 0:
            raise ValueError("component_scale_init must be positive.")

        state_dim = int(kwargs["state_dim"])
        self.component_scale_mode = component_scale_mode
        self.forget_gate = nn.Linear(state_dim * 2, state_dim)
        self.output_gate = nn.Linear(state_dim * 2, state_dim)
        self.lstm_input_gate = nn.Linear(state_dim * 2, state_dim)
        self.lstm_candidate = nn.Linear(state_dim * 2, state_dim)
        self.inject_gate = nn.Linear(state_dim * 2, state_dim)
        self.react_gate = nn.Linear(state_dim * 2, state_dim)
        self.diff_gate = nn.Linear(state_dim * 2, state_dim)
        if component_scale_mode == "learnable":
            self.raw_component_scales = nn.Parameter(
                torch.full(
                    (3,),
                    softplus_inverse(component_scale_init),
                    dtype=torch.float32,
                )
            )

    def _component_scales(self, h: Any) -> Tuple[Any, Any, Any]:
        if self.component_scale_mode == "learnable":
            inject_scale, react_scale, diff_scale = F.softplus(
                self.raw_component_scales
            )
            return inject_scale, react_scale, diff_scale
        scale = h.new_tensor(1.0)
        return scale, scale, scale

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        node_risk_sequence, node_risk_mask = self._prepare_sequence(
            node_risk_sequence,
            node_risk_mask,
        )

        h0 = self.initial_state() if initial_state is None else initial_state
        h = h0
        c = h0
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            mask = None if node_risk_mask is None else node_risk_mask[time_index]
            z = self._project_risk(node_risk_sequence[time_index], mask)
            h_inject, h_react, h_diff = self._reaction_diffusion_components(h, h0, z)
            gate_input = torch.cat([h, z], dim=1)

            forget_gate = torch.sigmoid(self.forget_gate(gate_input))
            output_gate = torch.sigmoid(self.output_gate(gate_input))
            lstm_input_gate = torch.sigmoid(self.lstm_input_gate(gate_input))
            lstm_candidate = torch.tanh(self.lstm_candidate(gate_input))
            inject_gate = torch.sigmoid(self.inject_gate(gate_input))
            react_gate = torch.sigmoid(self.react_gate(gate_input))
            diff_gate = torch.sigmoid(self.diff_gate(gate_input))
            inject_scale, react_scale, diff_scale = self._component_scales(h)

            c = (
                forget_gate * c
                + lstm_input_gate * lstm_candidate
                + inject_scale * inject_gate * h_inject
                + react_scale * react_gate * h_react
                + diff_scale * diff_gate * h_diff
            )
            h = output_gate * torch.tanh(c)
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class LSTMComponentFusionGatedReactionDiffusionPropagationNetwork(
    LSTMComponentGatedReactionDiffusionPropagationNetwork
):
    def __init__(
        self,
        *,
        rd_fusion_gate_init: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not (0.0 < rd_fusion_gate_init < 1.0):
            raise ValueError("rd_fusion_gate_init must be in (0, 1).")

        state_dim = int(kwargs["state_dim"])
        self.rd_fusion_gate = nn.Linear(state_dim * 2, state_dim)
        bias_value = float(
            np.log(rd_fusion_gate_init / (1.0 - rd_fusion_gate_init))
        )
        # Make the configured initial probability independent of h/z at epoch 0.
        nn.init.zeros_(self.rd_fusion_gate.weight)
        nn.init.constant_(self.rd_fusion_gate.bias, bias_value)

    def forward(
        self,
        node_risk_sequence: Any,
        node_risk_mask: Optional[Any] = None,
        initial_state: Optional[Any] = None,
        return_sequence: bool = True,
    ) -> Any:
        node_risk_sequence, node_risk_mask = self._prepare_sequence(
            node_risk_sequence,
            node_risk_mask,
        )

        h0 = self.initial_state() if initial_state is None else initial_state
        h = h0
        c = h0
        states: List[Any] = []
        for time_index in range(node_risk_sequence.shape[0]):
            mask = None if node_risk_mask is None else node_risk_mask[time_index]
            z = self._project_risk(node_risk_sequence[time_index], mask)
            h_inject, h_react, h_diff = self._reaction_diffusion_components(h, h0, z)
            gate_input = torch.cat([h, z], dim=1)

            forget_gate = torch.sigmoid(self.forget_gate(gate_input))
            output_gate = torch.sigmoid(self.output_gate(gate_input))
            lstm_input_gate = torch.sigmoid(self.lstm_input_gate(gate_input))
            lstm_candidate = torch.tanh(self.lstm_candidate(gate_input))
            inject_gate = torch.sigmoid(self.inject_gate(gate_input))
            react_gate = torch.sigmoid(self.react_gate(gate_input))
            diff_gate = torch.sigmoid(self.diff_gate(gate_input))
            rd_fusion_gate = torch.sigmoid(self.rd_fusion_gate(gate_input))
            inject_scale, react_scale, diff_scale = self._component_scales(h)
            rd_component = (
                inject_scale * inject_gate * h_inject
                + react_scale * react_gate * h_react
                + diff_scale * diff_gate * h_diff
            )

            c = (
                forget_gate * c
                + lstm_input_gate * lstm_candidate
                + rd_fusion_gate * rd_component
            )
            h = output_gate * torch.tanh(c)
            h = self.dropout(h)
            states.append(h)

        if return_sequence:
            return torch.stack(states, dim=0)
        return h


class GCNPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(GCNPropagationNetwork(**kwargs))


class GATPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(GATPropagationNetwork(**kwargs))


class DCRNNPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(DCRNNPropagationNetwork(**kwargs))


class LSTMPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(LSTMPropagationNetwork(**kwargs))


class GRUGatedReactionDiffusionPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(GRUGatedReactionDiffusionPropagationNetwork(**kwargs))


class LSTMGatedReactionDiffusionPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(LSTMGatedReactionDiffusionPropagationNetwork(**kwargs))


class GRUZGatedReactionDiffusionPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(GRUZGatedReactionDiffusionPropagationNetwork(**kwargs))


class LSTMZGatedReactionDiffusionPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(LSTMZGatedReactionDiffusionPropagationNetwork(**kwargs))


class LSTMComponentGatedReactionDiffusionPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            LSTMComponentGatedReactionDiffusionPropagationNetwork(**kwargs)
        )


class LSTMComponentFusionGatedReactionDiffusionPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            LSTMComponentFusionGatedReactionDiffusionPropagationNetwork(**kwargs)
        )


class STGCNPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(STGCNPropagationNetwork(**kwargs))


class AGCRNPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(AGCRNPropagationNetwork(**kwargs))


class ITransformerPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ITransformerPropagationNetwork(**kwargs))


class TimesNetPropagationModule(_PropagationWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(TimesNetPropagationNetwork(**kwargs))


def build_propagation_module(args: Any, data: Any) -> Any:
    model_name = str(getattr(args, "propagation_model", "reaction_diffusion")).lower()
    node_igpc = data.graph.node_metadata["igpc_value"].to_numpy(dtype=np.float32)
    risk_hidden_sizes = parse_hidden_sizes(args.graph_risk_hidden_sizes)
    risk_dim = 2 if is_label_history_propagation_model(model_name) else args.representation_dim

    if model_name == "reaction_diffusion":
        return IndustryGraphPropagationModule(
            node_igpc=node_igpc,
            normalized_adjacency=data.graph.normalized_adjacency,
            risk_dim=args.representation_dim,
            state_dim=args.graph_state_dim,
            risk_hidden_sizes=risk_hidden_sizes,
            recovery_hidden_sizes=parse_hidden_sizes(args.graph_recovery_hidden_sizes),
            dropout=args.graph_dropout,
            dt=args.propagation_dt,
            alpha_init=args.alpha_init,
            beta_init=args.beta_init,
            gamma_init=args.gamma_init,
            eta_init=args.eta_init,
            eps=args.recovery_eps,
            diffusion_mode=args.diffusion_mode,
            initial_state_mode=args.initial_state_mode,
            enable_graph_diffusion=args.enable_graph_diffusion,
            enable_two_hop_diffusion=args.enable_two_hop_diffusion,
            enable_recovery=args.enable_recovery,
        )

    common_kwargs = {
        "node_igpc": node_igpc,
        "normalized_adjacency": data.graph.normalized_adjacency,
        "risk_dim": risk_dim,
        "state_dim": args.graph_state_dim,
        "risk_hidden_sizes": risk_hidden_sizes,
        "dropout": args.graph_dropout,
        "initial_state_mode": args.initial_state_mode,
    }
    if model_name == "stgcn":
        return STGCNPropagationModule(
            **common_kwargs,
            kernel_size=args.stgcn_kernel_size,
        )
    if model_name == "agcrn":
        return AGCRNPropagationModule(
            **common_kwargs,
            embed_dim=args.agcrn_embed_dim,
        )
    if model_name == "itransformer":
        return ITransformerPropagationModule(
            **common_kwargs,
            lookback=args.ts_baseline_lookback,
            heads=args.itransformer_heads,
        )
    if model_name == "timesnet":
        return TimesNetPropagationModule(
            **common_kwargs,
            lookback=args.ts_baseline_lookback,
            kernel_sizes=_parse_kernel_sizes(args.timesnet_kernel_sizes),
        )
    if model_name == "gcn":
        return GCNPropagationModule(**common_kwargs)
    if model_name == "gat":
        return GATPropagationModule(
            **common_kwargs,
            heads=args.gat_heads,
        )
    if model_name == "dcrnn":
        return DCRNNPropagationModule(
            **common_kwargs,
            adjacency=data.graph.adjacency,
            diffusion_steps=args.dcrnn_diffusion_steps,
            bidirectional=args.dcrnn_bidirectional,
        )
    if model_name == "lstm":
        return LSTMPropagationModule(**common_kwargs)
    if model_name in {
        "gru_gated_reaction_diffusion",
        "lstm_gated_reaction_diffusion",
        "gru_z_gated_reaction_diffusion",
        "lstm_z_gated_reaction_diffusion",
        "lstm_component_gated_reaction_diffusion",
        "lstm_component_fusion_gated_reaction_diffusion",
    }:
        gated_kwargs = {
            **common_kwargs,
            "recovery_hidden_sizes": parse_hidden_sizes(
                args.graph_recovery_hidden_sizes
            ),
            "dt": args.propagation_dt,
            "alpha_init": args.alpha_init,
            "beta_init": args.beta_init,
            "gamma_init": args.gamma_init,
            "eta_init": args.eta_init,
            "eps": args.recovery_eps,
            "diffusion_mode": args.diffusion_mode,
            "enable_graph_diffusion": args.enable_graph_diffusion,
            "enable_two_hop_diffusion": args.enable_two_hop_diffusion,
            "enable_recovery": args.enable_recovery,
        }
        if model_name == "gru_gated_reaction_diffusion":
            return GRUGatedReactionDiffusionPropagationModule(**gated_kwargs)
        if model_name == "lstm_gated_reaction_diffusion":
            return LSTMGatedReactionDiffusionPropagationModule(**gated_kwargs)
        if model_name == "gru_z_gated_reaction_diffusion":
            return GRUZGatedReactionDiffusionPropagationModule(**gated_kwargs)
        if model_name == "lstm_z_gated_reaction_diffusion":
            return LSTMZGatedReactionDiffusionPropagationModule(**gated_kwargs)
        if model_name == "lstm_component_fusion_gated_reaction_diffusion":
            return LSTMComponentFusionGatedReactionDiffusionPropagationModule(
                **gated_kwargs,
                component_scale_mode=args.component_scale_mode,
                component_scale_init=args.component_scale_init,
                rd_fusion_gate_init=args.rd_fusion_gate_init,
            )
        return LSTMComponentGatedReactionDiffusionPropagationModule(
            **gated_kwargs,
            component_scale_mode=args.component_scale_mode,
            component_scale_init=args.component_scale_init,
        )

    raise ValueError(
        "propagation_model must be one of "
        "'reaction_diffusion', 'gcn', 'gat', 'dcrnn', 'agcrn', 'lstm', "
        "'stgcn', 'itransformer', 'timesnet', "
        "'gru_gated_reaction_diffusion', 'lstm_gated_reaction_diffusion', "
        "'gru_z_gated_reaction_diffusion', "
        "'lstm_z_gated_reaction_diffusion', or "
        "'lstm_component_gated_reaction_diffusion', or "
        "'lstm_component_fusion_gated_reaction_diffusion'."
    )
