# fpga-monte-carlo-roulette

A cycle-accurate Python model of an FPGA-style parallel Monte Carlo pipeline for roulette simulation, benchmarked against a CPU baseline.

Developed for **EECE5643 Simulation and Performance Evaluation** at Northeastern University (Spring 2026).

---

## Overview

This project models and evaluates an FPGA-style hardware accelerator for Monte Carlo roulette simulation. Rather than implementing on physical FPGA hardware, we use a cycle-accurate Python pipeline model to simulate the behavior of L parallel execution lanes.

The core question we answer is: **how does simulation throughput scale with the number of parallel hardware lanes?**

---

## How It Works

### CPU Baseline
- Serial Python simulation — one trial per iteration
- NumPy-vectorized simulation — batched trial execution

### FPGA Pipeline Model
Each lane in the FPGA model consists of three stages:

```
RNG (LFSR) → Roulette Outcome Mapping → Bet Evaluation
```

- **RNG:** Linear Feedback Shift Register (LFSR)-based pseudo-random number generator, producing one random value per clock cycle
- **Roulette Mapping:** Maps the random value to one of 38 outcomes (0, 00, 1–36)
- **Bet Evaluation:** Evaluates the outcome against a fixed bet type (e.g., red/black)
- **Streaming Reducer:** Aggregates wins, losses, and payout across all lanes each clock cycle

Modeled execution time is computed as:

```
time = (cycles / clock_frequency)
cycles = (N_trials / N_lanes) + pipeline_depth
```

---

## Project Structure

```
fpga-monte-carlo-roulette/
├── cpu_baseline/
│   ├── serial_sim.py         # Serial Python CPU simulation
│   └── numpy_sim.py          # NumPy-vectorized simulation
├── fpga_model/
│   ├── pipeline.py           # Cycle-accurate FPGA pipeline model
│   ├── lfsr.py               # LFSR-based RNG implementation
│   └── reducer.py            # Streaming result accumulator
├── evaluation/
│   ├── benchmark.py          # Throughput and speedup benchmarks
│   └── plots.py              # Result visualization
├── results/                  # Output plots and data
└── README.md
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| Throughput | Trials per second |
| Speedup | Relative to single-threaded CPU baseline |
| Lane scaling | Throughput vs. number of parallel lanes (1, 4, 8, 16, 32) |

---

## Expected Results

- Near-linear speedup as lane count increases
- Target: 10x–100x speedup over single-threaded CPU baseline
- Based on related work in FPGA Monte Carlo acceleration

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

---

## Course

EECE5643 Simulation and Performance Evaluation
Department of Electrical and Computer Engineering
Northeastern University, Spring 2026
Instructor: Prof. Ningfang Mi
