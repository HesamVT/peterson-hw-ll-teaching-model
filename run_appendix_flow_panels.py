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
Generate three side-by-side flow panels showing hardwood growth, fire-driven destruction,
and net flow as functions of hardwood density for selected parameter combinations.

Notes:
- This script supports the analytical figures used in the teaching presentation.
- The analysis is based on the simplified longleaf pine-hardwood model discussed in
  Peterson (2002), Appendix 1.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdg-cache"))
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = ANALYSIS_DIR / "appendix_flow_panels.png"
PROPORTION_KILLED_PER_FIRE = 0.5
Y_LIMITS = (-0.010, 0.035)


@dataclass(frozen=True)
class Case:
    """Parameter bundle for one flow-panel scenario."""

    label: str
    growth_rate: float
    avg_time_between_fires: float

    @property
    def fire_frequency(self) -> float:
        return 1.0 / self.avg_time_between_fires

    @property
    def destruction_rate(self) -> float:
        return PROPORTION_KILLED_PER_FIRE * self.fire_frequency


DEFAULT_CASES = [
    Case("Case 1", 0.075, 3.0),
    Case("Case 2", 0.075, 10.0),
    Case("Case 3", 0.12, 3.0),
]


def parse_case(raw_value: str) -> Case:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Each --case must be 'label,growth_rate,avg_time_between_fires'."
        )

    label, growth_rate_text, avg_time_text = parts
    try:
        growth_rate = float(growth_rate_text)
        avg_time_between_fires = float(avg_time_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Growth rate and average time between fires must be numeric."
        ) from exc

    if avg_time_between_fires <= 0:
        raise argparse.ArgumentTypeError("Average time between fires must be positive.")

    return Case(label, growth_rate, avg_time_between_fires)


def build_cases(parsed_cases: list[Case] | None) -> list[Case]:
    if not parsed_cases:
        return DEFAULT_CASES
    if len(parsed_cases) != 3:
        raise SystemExit("Provide exactly three --case values.")
    return parsed_cases


def growth(x: np.ndarray, growth_rate: float) -> np.ndarray:
    return growth_rate * x * (1.0 - x)


def destruction(x: np.ndarray, destruction_rate: float) -> np.ndarray:
    return destruction_rate * x * (1.0 - x) ** 3


def netflow(x: np.ndarray, growth_rate: float, destruction_rate: float) -> np.ndarray:
    return growth(x, growth_rate) - destruction(x, destruction_rate)


def equilibria(growth_rate: float, destruction_rate: float) -> list[float]:
    """Return equilibrium hardwood shares implied by the analytical net-flow equation."""
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


def plot_cases(cases: list[Case], output_file: Path) -> None:
    x = np.linspace(0.0, 1.0, 1001)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for axis, case in zip(axes, cases):
        # Compute the three flow curves directly from the analytical model equations.
        growth_values = growth(x, case.growth_rate)
        destruction_values = destruction(x, case.destruction_rate)
        netflow_values = netflow(x, case.growth_rate, case.destruction_rate)
        middle_equilibrium = interior_equilibrium(case.growth_rate, case.destruction_rate)

        axis.plot(x, growth_values, label="HW Growth", linewidth=2.0, color="blue")
        axis.plot(x, destruction_values, label="Destruction", linewidth=2.0, color="red")
        axis.plot(x, netflow_values, label="Netflow", linewidth=2.2, color="green")
        if middle_equilibrium is not None:
            # The interior equilibrium is shown separately because it acts as the threshold root.
            axis.axvline(middle_equilibrium, linewidth=1.0, color="0.45", linestyle="--")
        eq_points = equilibria(case.growth_rate, case.destruction_rate)
        axis.scatter(
            eq_points,
            [0.0] * len(eq_points),
            color="green",
            edgecolors="black",
            linewidths=0.7,
            s=36,
            zorder=5,
        )
        axis.axhline(0.0, linewidth=1.0, color="0.65")
        axis.set_title(
            f"{case.label}\n"
            f"r={case.growth_rate:g}, avg fire interval={case.avg_time_between_fires:g}"
        )
        axis.set_xlabel("Hardwood Density")
        axis.set_ylim(*Y_LIMITS)
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Flows")
    axes[0].legend(loc="best")
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot growth, destruction, and netflow versus state X for three parameter sets "
            "from the Peterson appendix equations."
        )
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=parse_case,
        help=(
            "Parameter set in the form 'label,growth_rate,avg_time_between_fires'. "
            "Provide exactly three to override the defaults."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output image path. Defaults to {OUTPUT_FILE}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(args.cases)
    plot_cases(cases, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
