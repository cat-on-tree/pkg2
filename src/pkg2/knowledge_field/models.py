"""Models for Temporal Knowledge Field experiments.

This module contains baseline and physics-inspired graph neural models for
Knowledge Unit (KU) temporal forecasting.

Model families
--------------

1. MLPRegressor

    No-graph baseline.

        score_i(t) = MLP(x_i(t))

2. GraphSAGERegressor

    PyG GraphSAGE baseline using sampled neighborhoods.

        h_i(t) = SAGE(x_i(t), A_KU)
        score_i(t) = Head(h_i(t))

3. WeightedDiffusionRegressor

    A lightweight weighted diffusion model using edge weights.

        m_i(t) = sum_j normalized(w_ij) W_neigh x_j(t)
        h_i(t) = MLP([W_self x_i(t), m_i(t)])
        score_i(t) = Head(h_i(t))

    This is closer to a physical diffusion operator than vanilla GraphSAGE.

4. ReactionDiffusionSourceRegressor

    Scaffold for a future Knowledge Field physical model:

        score_i(t) =
            reaction_i(local_i(t))
          + source_i(source_i(t))
          + diffusion_i(neighbor states)

    The v0 implementation uses feature-group slicing if provided, otherwise it
    falls back to treating all features as local/source jointly.

Notes
-----

- GraphSAGERegressor uses torch_geometric.nn.SAGEConv.
- WeightedDiffusionRegressor uses torch_geometric.nn.MessagePassing.
- Edge weights are optional. If unavailable, weighted layers default to weight 1.
- The first training script can use GraphSAGE as the stable neural graph
  baseline, then later switch to weighted_diffusion or reaction_diffusion_source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing, SAGEConv
    from torch_geometric.utils import add_self_loops, degree
except Exception:  # pragma: no cover
    MessagePassing = None
    SAGEConv = None
    add_self_loops = None
    degree = None


SUPPORTED_MODELS = (
    "mlp",
    "graphsage",
    "weighted_diffusion",
    "reaction_diffusion_source",
)


@dataclass(frozen=True)
class FeatureGroups:
    """Column index groups for physics-inspired model branches.

    Indices refer to the dense feature matrix after numeric normalization and
    categorical one-hot concatenation.

    local_indices:
        Focal KU state / reaction features.
    source_indices:
        Carrier graph source/exposure features.
    proxy_indices:
        Handcrafted KU diffusion proxy features, if present.
    categorical_indices:
        Optional categorical one-hot columns.
    """

    local_indices: list[int]
    source_indices: list[int]
    proxy_indices: list[int]
    categorical_indices: list[int]

    @property
    def has_source(self) -> bool:
        return len(self.source_indices) > 0

    @property
    def has_proxy(self) -> bool:
        return len(self.proxy_indices) > 0

    @property
    def has_categorical(self) -> bool:
        return len(self.categorical_indices) > 0


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int = 1,
    *,
    num_layers: int = 2,
    dropout: float = 0.1,
    activation: type[nn.Module] = nn.ReLU,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Create a simple feed-forward MLP."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    if num_layers <= 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = []

    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class MLPRegressor(nn.Module):
    """No-graph neural baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GraphSAGERegressor(nn.Module):
    """GraphSAGE regression model using PyG SAGEConv."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        *,
        project_input: bool = False,
    ) -> None:
        super().__init__()

        if SAGEConv is None:
            raise ImportError(
                "torch_geometric.nn.SAGEConv is unavailable. "
                "Please install PyTorch Geometric."
            )

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.project_input = bool(project_input)

        if project_input:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            conv_input_dim = hidden_dim
        else:
            self.input_proj = None
            conv_input_dim = input_dim

        self.convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = conv_input_dim if layer_idx == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = x

        if self.input_proj is not None:
            h = self.input_proj(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # edge_weight intentionally unused by vanilla GraphSAGE.
        h = self.encode(x, edge_index)
        return self.head(h).squeeze(-1)


class WeightedDiffusionConv(MessagePassing if MessagePassing is not None else nn.Module):
    """Weighted diffusion message passing layer.

    Aggregates weighted neighbor messages:

        m_i = sum_{j -> i} norm(w_ji) * W_neigh x_j

    and combines with transformed self state:

        h_i = W_self x_i + m_i

    Normalization is by target-node weighted in-degree.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        add_self_loop: bool = True,
        eps: float = 1e-12,
    ) -> None:
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric.nn.MessagePassing is unavailable. "
                "Please install PyTorch Geometric."
            )

        super().__init__(aggr="add", node_dim=0)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.add_self_loop = bool(add_self_loop)
        self.eps = float(eps)

        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_neigh.reset_parameters()
        self.lin_self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                device=edge_index.device,
                dtype=x.dtype,
            )
        else:
            edge_weight = edge_weight.to(device=edge_index.device, dtype=x.dtype)

        if self.add_self_loop:
            if add_self_loops is None:
                raise ImportError("torch_geometric.utils.add_self_loops unavailable.")

            edge_index, edge_weight = add_self_loops(
                edge_index,
                edge_weight,
                fill_value=1.0,
                num_nodes=x.size(0),
            )

        row, col = edge_index[0], edge_index[1]

        # PyG MessagePassing with source_to_target uses edge_index[0] as source
        # and edge_index[1] as target. Normalize by target weighted degree.
        deg = torch.zeros(
            x.size(0),
            device=x.device,
            dtype=x.dtype,
        )
        deg.scatter_add_(0, col, edge_weight)
        norm = edge_weight / deg[col].clamp_min(self.eps)

        neigh = self.propagate(
            edge_index=edge_index,
            x=self.lin_neigh(x),
            norm=norm,
        )

        return self.lin_self(x) + neigh

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        return x_j * norm.view(-1, 1)


class WeightedDiffusionRegressor(nn.Module):
    """Weighted diffusion GNN regression model.

    This model explicitly uses edge_weight and therefore is a stronger starting
    point for a physical KU diffusion operator than vanilla GraphSAGE.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        *,
        add_self_loop: bool = True,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.convs.append(
                WeightedDiffusionConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    add_self_loop=add_self_loop,
                )
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x

        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_weight=edge_weight)
        return self.head(h).squeeze(-1)


class ReactionDiffusionSourceRegressor(nn.Module):
    """Scaffold for a Reaction-Diffusion-Source Knowledge Field model.

    Conceptual decomposition:

        score_i(t) =
            Head(
                R(local_i(t))
              + S(source_i(t))
              + D(neighbor states)
              + optional P(proxy_i(t))
            )

    v0 behavior
    -----------

    - If feature_groups is provided, local/source/proxy branches use the
      specified indices.
    - If feature_groups is None, all input features are treated as local.
    - Diffusion is computed from the full input feature vector using weighted
      diffusion convolution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        *,
        feature_groups: FeatureGroups | None = None,
        use_source_branch: bool = True,
        use_proxy_branch: bool = True,
        add_self_loop: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.feature_groups = feature_groups
        self.use_source_branch = bool(use_source_branch)
        self.use_proxy_branch = bool(use_proxy_branch)

        if feature_groups is None:
            local_dim = input_dim
            source_dim = 0
            proxy_dim = 0
        else:
            local_dim = max(len(feature_groups.local_indices), 1)
            source_dim = len(feature_groups.source_indices)
            proxy_dim = len(feature_groups.proxy_indices)

        self.local_dim = int(local_dim)
        self.source_dim = int(source_dim)
        self.proxy_dim = int(proxy_dim)

        self.reaction = make_mlp(
            input_dim=self.local_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        if self.use_source_branch and source_dim > 0:
            self.source = make_mlp(
                input_dim=source_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
        else:
            self.source = None

        if self.use_proxy_branch and proxy_dim > 0:
            self.proxy = make_mlp(
                input_dim=proxy_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
        else:
            self.proxy = None

        self.diffusion = WeightedDiffusionRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            add_self_loop=add_self_loop,
        )

        self.combine_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _select(
        self,
        x: torch.Tensor,
        indices: list[int],
        *,
        fallback_all: bool = False,
    ) -> torch.Tensor:
        if indices:
            idx = torch.as_tensor(indices, device=x.device, dtype=torch.long)
            return x.index_select(dim=1, index=idx)

        if fallback_all:
            return x

        # Return an empty feature matrix with zero columns.
        return x.new_zeros((x.size(0), 0))

    def split_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.feature_groups is None:
            return x, x.new_zeros((x.size(0), 0)), x.new_zeros((x.size(0), 0))

        local_x = self._select(
            x,
            self.feature_groups.local_indices,
            fallback_all=True,
        )
        source_x = self._select(x, self.feature_groups.source_indices)
        proxy_x = self._select(x, self.feature_groups.proxy_indices)

        return local_x, source_x, proxy_x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_x, source_x, proxy_x = self.split_features(x)

        h = self.reaction(local_x)

        if self.source is not None and source_x.size(1) > 0:
            h = h + self.source(source_x)

        if self.proxy is not None and proxy_x.size(1) > 0:
            h = h + self.proxy(proxy_x)

        # Use diffusion encoder's representation, not its scalar head.
        h_diff = self.diffusion.encode(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        h = h + h_diff
        h = self.combine_norm(h)

        return self.head(h).squeeze(-1)


def infer_feature_groups_from_names(
    numeric_features: list[str],
    categorical_one_hot_dim: int = 0,
) -> FeatureGroups:
    """Infer approximate feature groups from feature names.

    This is intentionally heuristic. It is useful for initial RDS experiments,
    but formal experiments should write explicit feature group metadata.

    Rules
    -----
    proxy:
        names starting with kud_, diffusion_, neighbor_, nbr_
    source:
        names containing carrier/source/structural/exposure or prefixes cgs_/src_
    local:
        remaining numeric features
    categorical:
        one-hot columns appended after numeric features
    """

    local_indices: list[int] = []
    source_indices: list[int] = []
    proxy_indices: list[int] = []

    for idx, name in enumerate(numeric_features):
        lname = name.lower()

        is_proxy = (
            lname.startswith("kud_")
            or lname.startswith("diffusion_")
            or lname.startswith("neighbor_")
            or lname.startswith("nbr_")
            or "diffusion_proxy" in lname
        )

        is_source = (
            "carrier" in lname
            or "source" in lname
            or "structural" in lname
            or "exposure" in lname
            or lname.startswith("cgs_")
            or lname.startswith("src_")
        )

        if is_proxy:
            proxy_indices.append(idx)
        elif is_source:
            source_indices.append(idx)
        else:
            local_indices.append(idx)

    start_cat = len(numeric_features)
    categorical_indices = list(range(start_cat, start_cat + categorical_one_hot_dim))

    # Categorical one-hot features are safest to include in local branch too.
    local_indices = local_indices + categorical_indices

    return FeatureGroups(
        local_indices=local_indices,
        source_indices=source_indices,
        proxy_indices=proxy_indices,
        categorical_indices=categorical_indices,
    )


def build_model(
    model_name: str,
    *,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    feature_groups: FeatureGroups | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Factory for Knowledge Field models."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name={model_name}. "
            f"Expected one of {SUPPORTED_MODELS}."
        )

    if model_name == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_name == "graphsage":
        return GraphSAGERegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            project_input=bool(kwargs.get("project_input", False)),
        )

    if model_name == "weighted_diffusion":
        return WeightedDiffusionRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            add_self_loop=bool(kwargs.get("add_self_loop", True)),
        )

    if model_name == "reaction_diffusion_source":
        return ReactionDiffusionSourceRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            feature_groups=feature_groups,
            use_source_branch=bool(kwargs.get("use_source_branch", True)),
            use_proxy_branch=bool(kwargs.get("use_proxy_branch", True)),
            add_self_loop=bool(kwargs.get("add_self_loop", True)),
        )

    raise AssertionError("Unreachable model factory branch.")


def count_parameters(
    model: nn.Module,
    *,
    trainable_only: bool = True,
) -> int:
    if trainable_only:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in model.parameters()))


def model_summary_dict(model: nn.Module) -> dict[str, Any]:
    """Return compact model summary metadata."""

    return {
        "class_name": model.__class__.__name__,
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
        "modules": {
            name: module.__class__.__name__
            for name, module in model.named_modules()
            if name
        },
    }