"""Type-token curves averaged over random text orders, with a Heaps' law fit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .structure import geometric_points, heaps_fit, type_token_curve
from .text import Doc


@dataclass
class Curve:
    name: str
    tokens: np.ndarray
    types: np.ndarray
    k: float
    beta: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tokens": self.tokens.tolist(),
            "types": self.types.tolist(),
            "heaps_k": self.k,
            "heaps_beta": self.beta,
        }


def average_curve(
    docs: Sequence[Doc],
    max_tokens: int,
    name: str = "",
    orders: int = 5,
    seed: int = 0,
    start: int = 10,
    points: int = 25,
) -> Curve:
    """Mean number of distinct types after N tokens, over ``orders`` shuffles of the texts."""
    pool = [d for d in docs if len(d)]
    total = sum(len(d) for d in pool)
    if total < max_tokens:
        raise ValueError(f"{name or 'dataset'} holds {total} tokens, fewer than {max_tokens}")
    grid = geometric_points(start, max_tokens, points)
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(orders):
        order = rng.permutation(len(pool))
        ns, vs = type_token_curve([pool[int(i)].tokens for i in order], grid)
        runs.append(vs)
    types = np.mean(runs, axis=0)
    ns = np.asarray(grid, dtype=float)
    k, beta = heaps_fit(ns, types)
    return Curve(name, ns, types, k, beta)


def plot_curves(curves: Sequence[Curve], out: Path, title: str = "") -> None:
    """Log-log plot of the curves and their Heaps fits (needs matplotlib)."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("plotting needs matplotlib: pip install 'divscore[plot]'") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=120)
    for c in curves:
        (line,) = ax.plot(
            c.tokens,
            c.types,
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=f"{c.name} (beta {c.beta:.3f})",
        )
        ax.plot(
            c.tokens, c.k * c.tokens**c.beta, linestyle="--", linewidth=0.8, color=line.get_color()
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens read")
    ax.set_ylabel("distinct word types")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
