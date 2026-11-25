import math
from typing import Literal
import torch
import torch.nn as nn


class NumericalEmbedding(nn.Module):
    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.02)
        self.biases = nn.Parameter(torch.zeros(num_features, embedding_dim))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class PeriodicNumericalEmbedding(nn.Module):
    """Periodic-Linear-ReLU (PLR) numerical tokenizer (Gorishniy et al., NeurIPS 2022)."""

    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        n_frequencies: int = 16,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_features, n_frequencies) * sigma)
        self.weight = nn.Parameter(
            torch.randn(num_features, 2 * n_frequencies, embedding_dim) * (1.0 / math.sqrt(2 * n_frequencies))
        )
        self.bias = nn.Parameter(torch.zeros(num_features, embedding_dim))
        self.act = nn.ReLU()

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * x_num.unsqueeze(-1) * self.freqs.unsqueeze(0)
        periodic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.act(torch.einsum("bfk,fkd->bfd", periodic, self.weight) + self.bias.unsqueeze(0))


class CategoricalEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list[int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=v, embedding_dim=embedding_dim)
            for v in vocab_sizes
        ])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)


class FeatureIdentityEmbedding(nn.Module):
    def __init__(self, feature_identities: int, embedding_dim: int):
        super().__init__()
        self.identity = nn.Parameter(torch.randn(1, feature_identities, embedding_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.identity


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        vocab_sizes: list[int],
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        pooling: Literal["cls", "mean"] = "cls",
        num_tokenizer: Literal["linear", "plr"] = "plr",
        plr_frequencies: int = 16,
        plr_sigma: float = 1.0,
    ):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unsupported pooling mode: {pooling!r}")
        if num_tokenizer not in ("linear", "plr"):
            raise ValueError(f"Unsupported num_tokenizer: {num_tokenizer!r}")

        total_tokens = num_features + len(vocab_sizes)
        self.pooling = pooling
        self.num_features = num_features

        if num_features == 0:
            self.num_embed = None
        elif num_tokenizer == "linear":
            self.num_embed = NumericalEmbedding(num_features=num_features, embedding_dim=d_model)
        else:
            self.num_embed = PeriodicNumericalEmbedding(
                num_features=num_features,
                embedding_dim=d_model,
                n_frequencies=plr_frequencies,
                sigma=plr_sigma,
            )

        self.cat_embed = CategoricalEmbedding(vocab_sizes=vocab_sizes, embedding_dim=d_model) if vocab_sizes else None
        self.feat_id_embed = FeatureIdentityEmbedding(feature_identities=total_tokens, embedding_dim=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02) if pooling == "cls" else None

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.size(0) if self.num_features > 0 else x_cat.size(0)
        token_parts = []

        if self.num_embed is not None:
            token_parts.append(self.num_embed(x_num))
        if self.cat_embed is not None:
            token_parts.append(self.cat_embed(x_cat))

        tokens = self.feat_id_embed(torch.cat(token_parts, dim=1))

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        encoded = self.encoder(tokens)
        pooled = encoded[:, 0, :] if self.pooling == "cls" else encoded.mean(dim=1)
        return self.head(pooled)