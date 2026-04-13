# FPGA Monte Carlo Workload Accelerator Simulator

A discrete-event simulation of an FPGA-style parallel pipeline for Monte Carlo workloads, built in Python. The FPGA model is the system under evaluation: we measure how throughput scales with lane count under realistic hardware contention, apply simple timing and resource-feasibility models, and compare against CPU baselines.

Developed for **EECE5643 Simulation and Performance Evaluation** at Northeastern University (Spring 2026).

---

## Motivation

Monte Carlo simulations are embarrassingly parallel in theory, but real hardware introduces contention: shared memory buses, reduction tree bottlenecks, output buffer backpressure, and RNG reseeding overhead. A simple formula like `throughput = N_lanes * clock_freq` ignores all of this. We use discrete-event simulation to model these constraints and discover empirically where throughput saturates, which resources become bottlenecks, and how different workload profiles interact with the hardware.

---

## System Under Evaluation

### FPGA Pipeline Model

Each of **L** parallel lanes executes a 4-stage pipeline per trial:

```
Stage 1        Stage 2             Stage 3            Stage 4
LFSR RNG  -->  Workload Mapping --> Evaluation -----> State Update
(1 cycle)      (workload-specific)  (workload-specific) (workload-specific)
```

After all four stages, the lane deposits its result into a shared reduction resource. All lanes share a memory bus with limited ports, a reducer with limited throughput, and a finite output buffer. These shared resources create queueing and contention that the simulator captures.

**Contention sources modeled:**

| Resource | Model | Effect |
|----------|-------|--------|
| Memory bus | Shared resource with limited ports | Lanes stall waiting for read/write access |
| Reduction tree | SimPy Resource, limited throughput | Lanes stall when reducer cannot absorb results fast enough |
| LFSR reseeding | Periodic pipeline bubble | Dead cycles every N steps when LFSR state is refreshed |
| Output buffer | Finite-depth buffer | Backpressure propagates when downstream consumer is slow |

Stage costs are workload-specific. Roulette uses a light 4 x 1-cycle pipeline, sine integration adds a heavier evaluation stage, and option pricing adds heavier mapping and evaluation stages through the timing model.

The simulator also includes:

- A workload abstraction layer for roulette, sine integration, and European option pricing
- A simple timing model that caps effective clock frequency by workload and structural complexity
- A simple resource model that marks oversized FPGA configurations infeasible

**Key parameters:** lane count, requested clock frequency, memory bus ports, reducer throughput, buffer depth, LFSR reseed interval and latency. All are configurable for parameter sweeps.

### CPU Baselines

Two CPU implementations serve as comparison points:

**Serial Python** runs one trial per iteration in a standard loop. This represents the naive, unoptimized baseline.

**NumPy-vectorized** batches the data path where possible and falls back to sequential logic when the workload is stateful. This represents a realistic optimized CPU implementation.

### LFSR Random Number Generator

Each lane has its own 32-bit Galois LFSR with a maximal-length feedback polynomial (period 2^32 - 1). The LFSR produces one pseudorandom value per clock cycle, mapped to a roulette outcome via `state % 38`. Each lane's LFSR is seeded deterministically from a base seed plus the lane index, ensuring reproducibility. Periodic reseeding (XOR with a secondary value) introduces pipeline bubbles, modeling realistic hardware RNG behavior.

---

## Workloads

The project currently supports three workloads through a shared FPGA execution model:

**Roulette** maps each random draw to one of 38 outcomes and evaluates either flat betting or martingale. This is the most contention-sensitive workload because state updates can require extra memory-bus traffic.

**Sine integration** estimates the integral of `sin(x)` over a configurable interval. This workload uses the same lane/reduction structure but shifts more cost into the arithmetic stages.

**European option pricing** estimates a Black-Scholes-style call option value using Monte Carlo sampling. This is the heaviest workload in the repository and is the main driver for timing and resource-feasibility limits.

Within roulette, we evaluate two betting strategies as workload variants:

**Flat betting** uses a constant bet size every round. The strategy update stage is essentially a no-op, so memory bus contention comes only from bet configuration reads. This produces uniform, predictable pipeline behavior.

**Martingale** doubles the bet after each loss and resets after a win. This requires a read-modify-write to strategy state via the memory bus on every round. Losing streaks create bursts of heavy bus traffic. Pipeline throughput becomes outcome-dependent, creating variable-intensity workload behavior that interacts differently with hardware constraints.

Comparing these workloads and roulette strategies reveals how arithmetic intensity, statefulness, and shared-resource demand affect pipeline utilization, bottleneck locations, and scaling behavior.

---

## Comparison Methodology

The FPGA model and the CPU baselines measure performance in fundamentally different ways. The CPU baselines measure actual wall-clock execution time on real hardware. The FPGA model produces a predicted throughput: the simulator determines how many clock cycles the pipeline would require, applies the effective clock from the timing model, and converts that to modeled time.

This means the "speedup" metric compares a model prediction against an empirical measurement. This approach is standard in FPGA acceleration research (see Related Work), since building and benchmarking real FPGA hardware is outside the scope of this project. The credibility of the prediction depends on how realistically the simulator models hardware constraints. A naive formula like `throughput = clock_freq * n_lanes` would trivially predict linear scaling. Our model captures shared-resource contention, workload-dependent stage costs, timing limits, and infeasible resource footprints, producing non-trivial scaling curves that depend on both hardware configuration and workload.

The primary findings of this project are about relative scaling behavior: how throughput changes with lane count, where it saturates, and how workload characteristics (flat vs. martingale) interact with hardware bottlenecks. The absolute speedup numbers depend on the assumed clock frequency and should be interpreted as order-of-magnitude estimates rather than precise predictions.

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Throughput | Modeled trials per second (trials / simulated wall-clock time) |
| Speedup | FPGA throughput relative to serial and NumPy CPU baselines |
| Per-stage utilization | Fraction of time each pipeline stage is active vs. stalled |
| Memory bus utilization | Fraction of time the bus is fully occupied |
| Reducer utilization | Fraction of reducer capacity consumed per cycle |
| Trial latency distribution | Histogram of per-trial cycle counts (reveals contention effects) |
| Saturation point | Lane count at which throughput stops scaling linearly |

### Statistical Analysis

**Convergence analysis** tracks the running win-rate estimate as a function of N, plotted with 95% confidence intervals. This validates that the simulation produces stable, converging results.

**LFSR quality validation** applies a chi-squared goodness-of-fit test to the outcome histogram across all 38 roulette slots. A p-value above 0.05 confirms the LFSR output is sufficiently uniform.

**Throughput confidence intervals** are computed across multiple runs with different random seeds to quantify measurement variability.

### Parameter Sweeps

| Parameter | Values |
|-----------|--------|
| Lane count | 1, 2, 4, 8, 16, 32, 64 |
| Memory bus ports | 1, 2, 4 |
| Reducer throughput | 1, 2, 4, 8 |
| Requested clock frequency | 100, 200, 250 MHz |
| Trial count | 10K, 100K, 1M |
| Workload | Roulette, sine, option pricing |
| Roulette strategy | Flat, Martingale |

## Quick start

```bash
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pytest -q

# Quick benchmark sweep with CSV + plots
# Note: for FPGA benchmark mode, main.py currently runs all workloads
# and writes a combined benchmark_results.csv to the chosen output dir.
python main.py --quick --output-dir results/quick

# CPU-only baselines
python main.py --cpu-only --workload roulette --mode both --strategy flat
python main.py --cpu-only --workload sine --mode both
python main.py --cpu-only --workload option --mode both

# Full FPGA sweep across all workloads
python main.py --output-dir results/full

# Re-generate plots from an existing benchmark CSV
python main.py --plot-only --output-dir results/quick
```

### Output Artifacts

1. Throughput vs. lane count (with ideal linear scaling reference)
2. Speedup vs. lane count (relative to both CPU baselines)
3. Pipeline utilization heatmap (stages x lane counts)
4. Memory bus utilization vs. lane count (for different bus port configurations)
5. Reducer saturation curve (throughput vs. lane count for different reducer capacities)
6. Trial latency distribution (at selected lane counts)
7. Convergence plot (running win-rate with confidence interval bands)
8. Outcome histogram for roulette (38 bins, annotated with chi-squared p-value)

---

## Project Structure

```
fpga-monte-carlo-roulette/
├── config.py                        # Central simulation/workload configuration
├── main.py                          # CLI entry point for benchmarks, plots, CPU-only runs
├── cpu_baseline/
│   ├── runner.py                    # Dispatches CPU baseline by workload
│   ├── roulette.py                  # Roulette serial + NumPy baselines
│   ├── sine.py                      # Sine serial + NumPy baselines
│   └── option_pricing.py            # Option pricing serial + NumPy baselines
├── fpga_model/
│   ├── __init__.py
│   ├── fpga_sim.py                  # Top-level FPGA simulation orchestrator
│   ├── lane.py                      # Per-lane state and scheduling
│   ├── lfsr.py                      # 32-bit Galois LFSR RNG
│   ├── metrics.py                   # Throughput, latency, and utilization accounting
│   ├── resource_model.py            # Board-capacity feasibility model
│   ├── shared_resources.py          # Bus/reducer/output-buffer models
│   ├── timing_model.py              # Effective clock and per-stage cycle model
│   └── workloads/                   # Workload-specific stage behavior and reductions
│       ├── __init__.py
│       ├── base.py                  # Shared workload interface
│       ├── roulette.py              # Roulette workload stages and reductions
│       ├── sine.py                  # Sine integration workload
│       └── option_pricing.py        # Option pricing workload
├── evaluation/
│   ├── __init__.py
│   ├── analysis.py                  # Convergence and statistical validation
│   ├── benchmark.py                 # Parameter sweep runner and CSV generation
│   └── plots.py                     # Result visualization
├── tests/                           # Regression tests for workloads and models
│   ├── conftest.py
│   ├── test_benchmark_outputs.py
│   ├── test_fpga_option.py
│   ├── test_fpga_roulette.py
│   ├── test_fpga_sine.py
│   ├── test_resource_model.py
│   ├── test_shared_resources.py
│   └── test_timing_model.py
├── results/                         # Output plots, CSVs, and analysis notes
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Implementation language |
| NumPy | Vectorized CPU baseline, numerical operations |
| Matplotlib | Result visualization |
| SciPy | Statistical tests (chi-squared, confidence intervals) |

---

## Related Work

- Ortega-Zamorano et al. (2016) — [FPGA Monte Carlo Ising Model](https://arxiv.org/abs/1602.03016)
- Barbone et al. (2021) — [FPGA Acceleration of Monte Carlo Simulation](https://indico.cern.ch/event/1096432/contributions/4612623/attachments/2363830/4035767/13-12-2021%20HSF%20meeting.pdf)
- Tian & Benkrid (2010) — [High-Performance Quasi-Monte Carlo Financial Simulation: FPGA vs. GPP vs. GPU](https://dl.acm.org/doi/10.1145/1862648.1862656)
- Gothandaraman et al. (2008) — [FPGA Acceleration of a Quantum Monte Carlo Application](https://www.sciencedirect.com/science/article/abs/pii/S0167819108000227)

---

## Team

- Govind Luck
- John Bergin
- Wonhee Lee

## Course

EECE5643 Simulation and Performance Evaluation
Department of Electrical and Computer Engineering
Northeastern University, Spring 2026
Instructor: Prof. Ningfang Mi
