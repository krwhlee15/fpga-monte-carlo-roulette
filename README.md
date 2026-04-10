# FPGA Monte Carlo Roulette Accelerator

A discrete-event simulation of an FPGA-style parallel pipeline for Monte Carlo roulette, built in Python with SimPy. The FPGA pipeline model is the system under evaluation: we measure how throughput scales with lane count under realistic hardware contention, and compare against CPU baselines.

Developed for **EECE5643 Simulation and Performance Evaluation** at Northeastern University (Spring 2026).

---

## Motivation

Monte Carlo simulations are embarrassingly parallel in theory, but real hardware introduces contention: shared memory buses, reduction tree bottlenecks, output buffer backpressure, and RNG reseeding overhead. A simple formula like `throughput = N_lanes * clock_freq` ignores all of this. We use discrete-event simulation to model these constraints and discover empirically where throughput saturates, which resources become bottlenecks, and how different workload profiles interact with the hardware.

---

## System Under Evaluation

### FPGA Pipeline Model

Each of **L** parallel lanes executes a 4-stage pipeline per roulette trial:

```
Stage 1        Stage 2             Stage 3            Stage 4
LFSR RNG  -->  Outcome Mapping -->  Bet Evaluation -->  Strategy Update
(1 cycle)      (1 cycle)           (1 cycle + bus)     (1 cycle + bus)
```

After all four stages, the lane deposits its result into a shared **reduction tree**. All lanes share a **memory bus** with limited ports and a **reducer** with limited throughput. These shared resources create queueing and contention that the DES captures.

**Contention sources modeled:**

| Resource | Model | Effect |
|----------|-------|--------|
| Memory bus | SimPy Resource, limited ports | Lanes stall waiting for bus access during bet read / strategy write |
| Reduction tree | SimPy Resource, limited throughput | Lanes stall when reducer cannot absorb results fast enough |
| LFSR reseeding | Periodic pipeline bubble | Dead cycles every N steps when LFSR state is refreshed |
| Output buffer | Finite-depth buffer | Backpressure propagates when downstream consumer is slow |

**Key parameters:** lane count, clock frequency, memory bus ports, reducer throughput, buffer depth, LFSR reseed interval and latency. All are configurable for parameter sweeps.

### CPU Baselines

Two CPU implementations serve as comparison points:

**Serial Python** runs one trial per iteration in a standard loop using Python's `random` module. This represents the naive, unoptimized baseline.

**NumPy-vectorized** generates all random outcomes in a single batch and evaluates bets with vectorized operations. For state-dependent strategies like martingale, a sequential loop handles the round-by-round updates. This represents a realistic optimized CPU implementation.

### LFSR Random Number Generator

Each lane has its own 32-bit Galois LFSR with a maximal-length feedback polynomial (period 2^32 - 1). The LFSR produces one pseudorandom value per clock cycle, mapped to a roulette outcome via `state % 38`. Each lane's LFSR is seeded deterministically from a base seed plus the lane index, ensuring reproducibility. Periodic reseeding (XOR with a secondary value) introduces pipeline bubbles, modeling realistic hardware RNG behavior.

---

## Workload Profiles

We evaluate two betting strategies as distinct workload profiles, enabling workload characterization analysis:

**Flat betting** uses a constant bet size every round. The strategy update stage is essentially a no-op, so memory bus contention comes only from bet configuration reads. This produces uniform, predictable pipeline behavior.

**Martingale** doubles the bet after each loss and resets after a win. This requires a read-modify-write to strategy state via the memory bus on every round. Losing streaks create bursts of heavy bus traffic. Pipeline throughput becomes outcome-dependent, creating variable-intensity workload behavior that interacts differently with hardware constraints.

Comparing these two profiles reveals how workload characteristics affect pipeline utilization, bottleneck locations, and scaling behavior.

---

## Comparison Methodology

The FPGA pipeline model and the CPU baselines measure performance in fundamentally different ways. The CPU baselines (serial Python and NumPy) measure actual wall-clock execution time on real hardware. The FPGA model, on the other hand, produces a predicted throughput: the discrete-event simulation determines how many clock cycles the pipeline would require, and we convert that to time using an assumed clock frequency (e.g., 200 MHz for a mid-range FPGA).

This means the "speedup" metric compares a model prediction against an empirical measurement. This approach is standard in FPGA acceleration research (see Related Work), since building and benchmarking real FPGA hardware is outside the scope of this project. The credibility of the prediction depends on how realistically the DES models hardware constraints. A naive formula like `throughput = clock_freq * n_lanes` would trivially predict linear scaling. Our DES captures shared resource contention (memory bus queueing, reducer saturation, LFSR reseeding stalls) that reduce throughput below the theoretical maximum, producing non-trivial scaling curves that depend on both hardware configuration and workload profile.

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
| Trial count | 10K, 100K, 1M |
| Betting strategy | Flat, Martingale |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python main.py --quick
```

### Planned Figures

1. Throughput vs. lane count (with ideal linear scaling reference)
2. Speedup vs. lane count (relative to both CPU baselines)
3. Pipeline utilization heatmap (stages x lane counts)
4. Memory bus utilization vs. lane count (for different bus port configurations)
5. Reducer saturation curve (throughput vs. lane count for different reducer capacities)
6. Trial latency distribution (at selected lane counts)
7. Convergence plot (running win-rate with confidence interval bands)
8. Outcome histogram (38 bins, annotated with chi-squared p-value)

---

## Project Structure

```
fpga-monte-carlo-roulette/
├── config.py                  # Central parameter configuration
├── cpu_baseline/
│   ├── serial_sim.py          # Serial Python roulette simulation
│   └── numpy_sim.py           # NumPy-vectorized simulation
├── fpga_model/
│   ├── lfsr.py                # 32-bit Galois LFSR RNG
│   ├── pipeline.py            # Single lane: 4-stage pipeline (SimPy process)
│   ├── memory_bus.py          # Shared memory bus (SimPy Resource)
│   ├── reducer.py             # Reduction tree aggregator (SimPy Resource)
│   └── fpga_sim.py            # Top-level DES orchestrator
├── evaluation/
│   ├── benchmark.py           # Parameter sweep runner, metric collection
│   ├── analysis.py            # Statistical tests (chi-squared, CI, convergence)
│   └── plots.py               # Result visualization
├── results/                   # Output plots and CSV data
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Implementation language |
| SimPy | Discrete-event simulation engine |
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
