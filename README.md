# Peterson HW-LL Teaching Model

This repository contains a simplified Vensim and Python-based teaching package built around Peterson's longleaf pine-hardwood model from:

- Peterson, G. D. 2002. *Estimating resilience across landscapes*. *Conservation Ecology* 6(1): 17.

The materials were prepared for a teaching demonstration for the University of Bergen (UiB) on:

**Systems Thinking and Modelling as a Framework for Sustainability Analysis and Action**

## Repository Contents

### Vensim model

- `Peterson HW LL.mdl`

This is the Vensim implementation of the simplified hardwood-longleaf model used in the teaching presentation.

### Python scripts

- `run_appendix_flow_panels.py`
  Generates flow panels showing hardwood growth, fire-driven destruction, and net flow as functions of hardwood density.
- `run_phase_transition_panels.py`
  Generates state transition panels plotting hardwood density at time `t + dt` against hardwood density at time `t`.
- `run_bifurcation_panels.py`
  Generates annotated bifurcation and resilience panels for presentation use.
- `run_bifurcation_panels_clean.py`
  Generates a clean version of the bifurcation and resilience panels without overlays.

### Generated figures

- `appendix_flow_panels.png`
- `phase_transition_panels.png`
- `bifurcation_panels.png`
- `bifurcation_panels_clean.png`

## Model Summary

The simplified model tracks the proportion of the landscape dominated by hardwood (`HW`). Longleaf (`LL`) is represented as the remaining share of the landscape:

- `LL = 1 - HW`

The two main processes are:

- hardwood growth
- hardwood destruction by fire

In the simplified analytical form:

- `HW growth = r × HW × LL`
- `HW destruction = f × HW × LL^3`

where:

- `r` is the hardwood growth parameter,
- `f` represents fire-driven hardwood loss,
- fire frequency is defined as the inverse of average time between fires.

## Clarification relative to Peterson's original wording

In Peterson's appendix, the term **fire frequency** is used in a way that is potentially confusing: it refers to the **average time between fires**, not to the number of fires per unit time.

In this teaching version of the model, that quantity is made explicit by separating:

- `avg time between fires` in units of `Year`, and
- `fire frequency = 1 / avg time between fires` in units of `1/Year`.

This change does not alter the intended model behavior. It is a clarification of terminology and dimensional meaning so that the model reads more clearly in system dynamics form.

## What the Python analyses do

The Python scripts reproduce the main analytical figures used in the teaching module:

1. **Flow plots**
   Show how growth, destruction, and net flow vary with hardwood density.

2. **State transition plots**
   Show how the system moves from one state to the next and identify stable and unstable equilibria.

3. **Bifurcation and resilience plots**
   Show how equilibrium structure and resilience change as:
   - average fire interval varies, or
   - hardwood growth rate varies.

## Requirements

The Python scripts require:

- Python 3
- `numpy`
- `matplotlib`

## Teaching Use

This repository is intended for teaching and analytical illustration. Its main purpose is to show how a non-system-dynamics formulation can be:

- translated into stock-flow language,
- simulated in Vensim,
- analyzed through equilibrium, transition, and resilience plots,
- interpreted in terms of thresholds, alternative stable states, and resilience.
