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
Generate bifurcation and resilience panels showing how equilibrium structure and the resilience
of longleaf- and hardwood-dominated states change as average fire interval or hardwood growth
rate is varied.

Notes:
- This script supports the analytical figures used in the teaching presentation.
- The analysis is based on the simplified longleaf pine-hardwood model discussed in
  Peterson (2002), Appendix 1.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdg-cache"))
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = ANALYSIS_DIR / "bifurcation_panels.png"
PROPORTION_KILLED_PER_FIRE = 0.5
BASE_GROWTH_RATE = 0.075
BASE_AVG_FIRE_INTERVAL = 3.0


def interior_equilibrium(growth_rate: np.ndarray, destruction_rate: np.ndarray) -> np.ndarray:
    """Interior unstable branch that separates the two stable basins when it exists."""
    values = np.full_like(growth_rate, np.nan, dtype=float)
    mask = growth_rate <= destruction_rate
    values[mask] = 1.0 - np.sqrt(growth_rate[mask] / destruction_rate[mask])
    return values


def stable_zero(growth_rate: np.ndarray, destruction_rate: np.ndarray) -> np.ndarray:
    values = np.full_like(growth_rate, np.nan, dtype=float)
    mask = growth_rate < destruction_rate
    values[mask] = 0.0
    return values


def unstable_zero(growth_rate: np.ndarray, destruction_rate: np.ndarray) -> np.ndarray:
    values = np.full_like(growth_rate, np.nan, dtype=float)
    mask = growth_rate >= destruction_rate
    values[mask] = 0.0
    return values


def stable_one(reference: np.ndarray) -> np.ndarray:
    return np.ones_like(reference, dtype=float)


def ll_resilience(interior_branch: np.ndarray, growth_rate: np.ndarray, destruction_rate: np.ndarray) -> np.ndarray:
    values = np.zeros_like(growth_rate, dtype=float)
    stable_mask = growth_rate < destruction_rate
    values[stable_mask] = interior_branch[stable_mask]
    return values


def hw_resilience(interior_branch: np.ndarray) -> np.ndarray:
    values = np.zeros_like(interior_branch, dtype=float)
    valid = ~np.isnan(interior_branch)
    values[valid] = 1.0 - interior_branch[valid]
    return values


def hw_resilience_with_zero_unstable(
    interior_branch: np.ndarray,
    zero_unstable_branch: np.ndarray,
) -> np.ndarray:
    """Measure distance from the hardwood attractor to the relevant unstable boundary."""
    values = np.zeros_like(interior_branch, dtype=float)
    interior_valid = ~np.isnan(interior_branch)
    values[interior_valid] = 1.0 - interior_branch[interior_valid]

    zero_unstable_valid = ~np.isnan(zero_unstable_branch)
    zero_only = zero_unstable_valid & ~interior_valid
    values[zero_only] = 1.0 - zero_unstable_branch[zero_only]
    return values


def plot_panel(
    axis: plt.Axes,
    x: np.ndarray,
    growth_rate: np.ndarray,
    destruction_rate: np.ndarray,
    xlabel: str,
    title: str,
    show_ylabel: bool,
) -> None:
    hw_stable_color = "blue"
    ll_stable_color = "green"
    unstable_color = "red"

    axis.plot(x, stable_one(x), color=hw_stable_color, linewidth=2.2, label="HW Stable Equilibrium")
    axis.plot(
        x,
        stable_zero(growth_rate, destruction_rate),
        color=ll_stable_color,
        linewidth=2.2,
        label="LL Stable Equilibrium",
    )
    axis.plot(
        x,
        unstable_zero(growth_rate, destruction_rate),
        color=unstable_color,
        linewidth=2.2,
        linestyle="--",
        label="Unstable equilibrium",
    )
    axis.plot(
        x,
        interior_equilibrium(growth_rate, destruction_rate),
        color=unstable_color,
        linewidth=2.2,
        linestyle="--",
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    if show_ylabel:
        axis.set_ylabel("Equilibrium Hardwood Density")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.3)


def add_resilience_annotation(
    axis: plt.Axes,
    x: np.ndarray,
    interior_branch: np.ndarray,
    x_pos: float,
    label: str,
) -> None:
    """Draw a double-headed arrow from a stable state to the nearby unstable separator."""
    valid = ~np.isnan(interior_branch)
    if not np.any(valid):
        return
    x_valid = x[valid]
    if x_pos < x_valid[0] or x_pos > x_valid[-1]:
        return

    y_pos = float(np.interp(x_pos, x_valid, interior_branch[valid]))
    axis.annotate(
        "",
        xy=(x_pos, y_pos),
        xytext=(x_pos, 0.0),
        arrowprops={
            "arrowstyle": "<|-|>",
            "color": "black",
            "lw": 1.0,
            "mutation_scale": 10,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )
    axis.text(x_pos + (x[-1] - x[0]) * 0.03, y_pos * 0.45, label, fontsize=9, va="center")


def add_instability_arrows(
    axis: plt.Axes,
    x: np.ndarray,
    unstable_branch: np.ndarray,
    zero_unstable_branch: np.ndarray,
    lower_stable_branch: np.ndarray,
    x_positions: np.ndarray,
) -> None:
    """Add directional arrows to show which equilibrium branch attracts nearby states."""
    interior_valid = ~np.isnan(unstable_branch)
    zero_unstable_valid = ~np.isnan(zero_unstable_branch)
    lower_valid = ~np.isnan(lower_stable_branch)

    lower_x = x[lower_valid] if np.any(lower_valid) else np.array([])
    lower_y = lower_stable_branch[lower_valid] if np.any(lower_valid) else np.array([])

    for x_pos in x_positions:
        if np.any(interior_valid) and x_pos >= x[interior_valid][0] and x_pos <= x[interior_valid][-1]:
            separating_y = float(np.interp(x_pos, x[interior_valid], unstable_branch[interior_valid]))
        elif np.any(zero_unstable_valid):
            separating_y = float(np.interp(x_pos, x[zero_unstable_valid], zero_unstable_branch[zero_unstable_valid]))
        else:
            continue

        upper_start = min(1.0, separating_y + 0.16)
        upper_end = 0.92
        if upper_start < upper_end:
            axis.annotate(
                "",
                xy=(x_pos, upper_end),
                xytext=(x_pos, upper_start),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#8fbbe8",
                    "lw": 1.2,
                    "mutation_scale": 10,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=5,
            )

        if np.any(lower_valid) and x_pos >= lower_x[0] and x_pos <= lower_x[-1]:
            lower_target = float(np.interp(x_pos, lower_x, lower_y))
            lower_start = max(0.0, separating_y - 0.16)
            lower_end = max(lower_target + 0.08, 0.0)
            if lower_end < lower_start:
                axis.annotate(
                    "",
                    xy=(x_pos, lower_end),
                    xytext=(x_pos, lower_start),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#8fcf9b",
                        "lw": 1.2,
                        "mutation_scale": 10,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    zorder=5,
                )


def plot_resilience_panel(
    axis: plt.Axes,
    x: np.ndarray,
    ll_values: np.ndarray,
    hw_values: np.ndarray,
    xlabel: str,
    show_ylabel: bool,
) -> None:
    axis.plot(x, ll_values, color="green", linewidth=2.2, label="LL Resilience")
    axis.plot(x, hw_values, color="blue", linewidth=2.2, label="HW Resilience")
    axis.set_xlabel(xlabel)
    if show_ylabel:
        axis.set_ylabel("Resilience")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.3)


def main() -> None:
    # Left column: vary fire interval while holding hardwood growth fixed.
    avg_fire_interval = np.linspace(0.1, 10.0, 1000)
    destruction_rate_fire = PROPORTION_KILLED_PER_FIRE / avg_fire_interval
    growth_rate_fire = np.full_like(avg_fire_interval, BASE_GROWTH_RATE)
    interior_fire = interior_equilibrium(growth_rate_fire, destruction_rate_fire)
    zero_unstable_fire = unstable_zero(growth_rate_fire, destruction_rate_fire)

    # Right column: vary hardwood growth while holding fire interval fixed.
    growth_rate_r = np.linspace(0.001, 0.25, 1000)
    destruction_rate_r = np.full_like(
        growth_rate_r,
        PROPORTION_KILLED_PER_FIRE / BASE_AVG_FIRE_INTERVAL,
    )
    interior_r = interior_equilibrium(growth_rate_r, destruction_rate_r)
    zero_unstable_r = unstable_zero(growth_rate_r, destruction_rate_r)
    ll_resilience_fire = ll_resilience(interior_fire, growth_rate_fire, destruction_rate_fire)
    hw_resilience_fire = hw_resilience_with_zero_unstable(interior_fire, zero_unstable_fire)
    ll_resilience_r = ll_resilience(interior_r, growth_rate_r, destruction_rate_r)
    hw_resilience_r = hw_resilience_with_zero_unstable(interior_r, zero_unstable_r)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey="row", constrained_layout=True)

    # Top row shows equilibrium branches; bottom row converts the same structure into resilience.
    lower_stable_fire = stable_zero(growth_rate_fire, destruction_rate_fire)
    zero_unstable_fire = unstable_zero(growth_rate_fire, destruction_rate_fire)
    plot_panel(
        axes[0, 0],
        avg_fire_interval,
        growth_rate_fire,
        destruction_rate_fire,
        xlabel="Average Fire Interval (Years)",
        title="r = 0.075",
        show_ylabel=True,
    )
    add_resilience_annotation(axes[0, 0], avg_fire_interval, interior_fire, 3.0, "LL\nResilience")
    add_instability_arrows(
        axes[0, 0],
        avg_fire_interval,
        interior_fire,
        zero_unstable_fire,
        lower_stable_fire,
        np.linspace(0.5, 9.5, 10),
    )
    lower_stable_r = stable_zero(growth_rate_r, destruction_rate_r)
    zero_unstable_r = unstable_zero(growth_rate_r, destruction_rate_r)
    plot_panel(
        axes[0, 1],
        growth_rate_r,
        growth_rate_r,
        destruction_rate_r,
        xlabel="Hardwood Growth Parameter",
        title="Average Fire Interval = 3",
        show_ylabel=False,
    )
    add_resilience_annotation(axes[0, 1], growth_rate_r, interior_r, 0.075, "LL\nResilience")
    add_instability_arrows(
        axes[0, 1],
        growth_rate_r,
        interior_r,
        zero_unstable_r,
        lower_stable_r,
        np.linspace(0.02, 0.245, 10),
    )
    plot_resilience_panel(
        axes[1, 0],
        avg_fire_interval,
        ll_resilience_fire,
        hw_resilience_fire,
        xlabel="Average Fire Interval (Years)",
        show_ylabel=True,
    )
    plot_resilience_panel(
        axes[1, 1],
        growth_rate_r,
        ll_resilience_r,
        hw_resilience_r,
        xlabel="Hardwood Growth Parameter",
        show_ylabel=False,
    )

    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
