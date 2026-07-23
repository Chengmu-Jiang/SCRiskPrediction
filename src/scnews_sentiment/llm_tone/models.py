from __future__ import annotations

import argparse
from typing import Any, List, Sequence

import numpy as np

from src.scnews_sentiment.llm_tone.utils import parse_hidden_sizes, resolve_device


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.encoder_device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    model = AutoModel.from_pretrained(args.encoder_model)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), args.embedding_batch_size):
            batch = list(texts[start : start + args.embedding_batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.encoder_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            if args.pooling == "mean":
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            elif args.pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                raise ValueError(f"Unsupported --pooling: {args.pooling}")
            pooled = functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def build_tone_regressor(input_dim: int, hidden_sizes: str, dropout: float) -> Any:
    import torch.nn as nn

    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in parse_hidden_sizes(hidden_sizes):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*layers)
