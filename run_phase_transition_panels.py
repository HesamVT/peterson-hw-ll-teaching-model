"""
Author: Hesam Mahmoudi, PhD
Affiliation: MGB Center for Health Technology Assessment, Harvard Medical School

Project:
Teaching Demonstration for UiB
"Systems Thinking and Modelling as a Framework for Sustainability Analysis and Action"

Presentation:
Feedback, Resilience, and Thresholds in Sustainability Systems
A system dynamics reading of Peterson's longleaf pine-hardwood model

Date: 2026-07-09

Purpose:
Generate three side-by-side state transition panels plotting hardwood density at time t + dt
against hardwood density at time t for selected parameter combinations, in order to identify
equilibrium points and their stability.

Notes:
- This script supports the analytical figures used in the teaching presentation.
- The analysis is based on the simplified longleaf pine-hardwood model discussed in
  Peterson (2002), Appendix 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdg-cache"))
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = ANALYSIS_DIR / "phase_transition_panels.png"
PROPORTION_KILLED_PER_FIRE = 0.5
DT = 10.0


@dataclass(frozen=True)
class Case:
    """Parameter bundle for one state-transition scenario."""

    label: str
    growth_rate: float
    avg_time_between_fires: float

    @property
    def fire_frequency(self) -> float:
        return 1.0 / self.avg_time_between_fires

    @property
    def destruction_rate(self) -> float:
        return PROPORTION_KILLED_PER_FIRE * self.fire_frequency


CASES = [
    Case("Case 1", 0.075, 3.0),
    Case("Case 2", 0.075, 10.0),
    Case("Case 3", 0.12, 3.0),
]


def growth(x: np.ndarray, growth_rate: float) -> np.ndarray:
    return growth_rate * x * (1.0 - x)


def destruction(x: np.ndarray, destruction_rate: float) -> np.ndarray:
    return destruction_rate * x * (1.0 - x) ** 3


def netflow(x: np.ndarray, growth_rate: float, destruction_rate: float) -> np.ndarray:
    return growth(x, growth_rate) - destruction(x, destruction_rate)


def next_state(x: np.ndarray, growth_rate: float, destruction_rate: float, dt: float) -> np.ndarray:
    """Explicit Euler step used to visualize state-to-state movement."""
    return x + dt * netflow(x, growth_rate, destruction_rate)


def equilibria(growth_rate: float, destruction_rate: float) -> list[float]:
    roots = [0.0, 1.0]
    if growth_rate <= destruction_rate:
        interior_root = 1.0 - np.sqrt(growth_rate / destruction_rate)
        if 0.0 <= interior_root <= 1.0:
            roots.append(float(interior_root))
    return sorted(roots)


def interior_equilibrium(growth_rate: float, destruction_rate: float) -> float | None:
    roots = equilibria(growth_rate, destruction_rate)
    if len(roots) == 3:
        return roots[1]
    return None


def stable_equilibria(growth_rate: float, destruction_rate: float) -> tuple[list[float], list[float]]:
    """Classify equilibrium roots into locally stable and unstable sets."""
    roots = equilibria(growth_rate, destruction_rate)
    stable = [1.0]
    unstable: list[float] = []

    for root in roots:
        if np.isclose(root, 1.0):
            continue
        if np.isclose(root, 0.0):
            if growth_rate < destruction_rate:
                stable.append(0.0)
            else:
                unstable.append(0.0)
        else:
            unstable.append(root)

    return sorted(stable), sorted(unstable)


def add_tangent_arrows(
    axis: plt.Axes,
    x: np.ndarray,
    x_next: np.ndarray,
    growth_rate: float,
    destruction_rate: float,
) -> None:
    arrow_positions = np.arange(0.1, 1.0, 0.1)

    for position in arrow_positions:
        idx = int(np.argmin(np.abs(x - position)))
        delta = float(netflow(np.array([x[idx]]), growth_rate, destruction_rate)[0])
        if abs(delta) < 1e-12:
            continue

        step = 12
        # Darker arrows indicate rising trajectories; lighter arrows indicate falling ones.
        if delta > 0:
            start_idx = max(0, idx - step)
            end_idx = min(len(x) - 1, idx + step)
            color = "#0b3c8c"
        else:
            start_idx = min(len(x) - 1, idx + step)
            end_idx = max(0, idx - step)
            color = "#5b84c4"

        axis.annotate(
            "",
            xy=(x[end_idx], x_next[end_idx]),
            xytext=(x[start_idx], x_next[start_idx]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.5,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=6,
        )


def plot_cases(output_file: Path) -> None:
    x = np.linspace(0.0, 1.0, 1001)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for axis, case in zip(axes, CASES):
        # Compare each state to its one-step-ahead value to reveal attraction and repulsion.
        x_next = next_state(x, case.growth_rate, case.destruction_rate, DT)
        stable_points, unstable_points = stable_equilibria(case.growth_rate, case.destruction_rate)
        middle_equilibrium = interior_equilibrium(case.growth_rate, case.destruction_rate)
        delta = x_next - x

        axis.plot(
            x,
            np.ma.masked_where(delta <= 0.0, x_next),
            color="#0b3c8c",
            linewidth=2.2,
            label="Increasing" if case.label == "Case 1" else None,
        )
        axis.plot(
            x,
            np.ma.masked_where(delta >= 0.0, x_next),
            color="#5b84c4",
            linewidth=2.2,
            label="Decreasing" if case.label == "Case 1" else None,
        )
        axis.plot([0.0, 1.0], [0.0, 1.0], color="0.45", linewidth=1.2, linestyle="--", label="Steady line")
        if middle_equilibrium is not None:
            axis.plot(
                [middle_equilibrium, middle_equilibrium],
                [0.0, 1.0],
                linewidth=1.0,
                color="red",
                linestyle="--",
                zorder=1,
            )
        axis.scatter(
            stable_points,
            stable_points,
            color="green",
            edgecolors="black",
            linewidths=0.7,
            s=36,
            zorder=5,
            label="Stable equilibrium" if case.label == "Case 1" else None,
        )
        axis.scatter(
            unstable_points,
            unstable_points,
            color="red",
            edgecolors="black",
            linewidths=0.7,
            s=36,
            zorder=5,
            label="Unstable equilibrium" if case.label == "Case 1" else None,
        )
        add_tangent_arrows(axis, x, x_next, case.growth_rate, case.destruction_rate)
        axis.set_title(
            f"{case.label}\n"
            f"r={case.growth_rate:g}, avg fire interval={case.avg_time_between_fires:g}"
        )
        axis.set_xlabel("Hardwood Density at time t")
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Hardwood Density at time t + dt")
    axes[0].legend(loc="best")
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def main() -> None:
    plot_cases(OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
