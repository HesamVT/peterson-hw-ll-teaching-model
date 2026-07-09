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
Generate a clean version of the bifurcation and resilience panels without resilience annotations
or directional arrows, for use in incremental slide builds and presentation overlays.

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
OUTPUT_FILE = ANALYSIS_DIR / "bifurcation_panels_clean.png"
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

    # This variant keeps the equilibrium and resilience curves only, without overlays.
    plot_panel(
        axes[0, 0],
        avg_fire_interval,
        growth_rate_fire,
        destruction_rate_fire,
        xlabel="Average Fire Interval (Years)",
        title="r = 0.075",
        show_ylabel=True,
    )
    plot_panel(
        axes[0, 1],
        growth_rate_r,
        growth_rate_r,
        destruction_rate_r,
        xlabel="Hardwood Growth Parameter",
        title="Average Fire Interval = 3",
        show_ylabel=False,
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
