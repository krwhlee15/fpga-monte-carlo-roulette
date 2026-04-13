# Results Analysis: FPGA Monte Carlo Accelerator Simulation

## Experimental Setup

All experiments were run using the cycle-accurate FPGA simulator with N = 100,000 trials per configuration. The simulator models a Basys3 board with configurable clock frequency, a four-stage pipeline per lane, shared memory bus and reducer contention, output buffer backpressure, and LFSR reseeding overhead. A resource feasibility model checks whether a given configuration exceeds the Basys3's LUT, DSP, BRAM, and FF limits, and a timing model estimates the maximum achievable clock frequency based on workload complexity and structural parameters.

The parameter sweep covered 720 configurations across the following variables: three workloads (roulette, sine integral, European option pricing), lane counts of 1, 4, 8, 16, and 32, memory bus ports of 1, 2, and 4, reducer throughputs of 1, 2, 4, and 8, clock frequencies of 100, 200, and 250 MHz, and two betting strategies (flat and Martingale) for roulette. CPU baselines were measured on real hardware using both a serial Python loop and a NumPy-vectorized implementation.

## 1. CPU Baseline Performance

| Workload | Serial (trials/s) | NumPy (trials/s) | NumPy Speedup |
|---|---|---|---|
| Roulette | 4,955,622 | 76,318,401 | 15.4x |
| Sine integral | 11,509,069 | 87,773,194 | 7.6x |
| Option pricing | 3,197,166 | 41,225,213 | 12.9x |

The serial roulette and option pricing baselines are comparably slow because both involve per-trial Python-loop overhead with conditional logic and function calls. The sine integral serial baseline runs faster because each trial is a single `math.sin()` call with minimal branching. NumPy achieves the largest absolute throughput on sine (87.8M/s) because `np.sin()` on a contiguous array maps directly to vectorized SIMD instructions, while roulette requires discrete lookups and conditional masking that are harder to vectorize efficiently.

## 2. Pipeline Depth and Single-Lane Throughput

The timing model assigns different per-stage cycle costs to each workload based on its arithmetic complexity.

| Workload | Stage cycles (RNG, Map, Eval, Update) | Total per trial | Single-lane throughput (100 MHz) |
|---|---|---|---|
| Roulette | (1, 1, 1, 1) | 6 cycles | 16.7M trials/s |
| Sine integral | (1, 1, 2, 1) | 7 cycles | 14.3M trials/s |
| Option pricing | (1, 2, 4, 1) | 10 cycles | 10.0M trials/s |

The total cycles per trial include the pipeline stages plus overhead from the reducer deposit and output buffer. Roulette uses simple comparison logic (1 cycle per stage), sine requires a 2-cycle CORDIC-style evaluation for sin(x), and option pricing requires 2 cycles for the mapping stage (Box-Muller transform) and 4 cycles for the evaluation stage (exponentiation, square root, and multiply in the GBM formula). The stage utilization data confirms this: at L=8, roulette shows uniform 16.7% utilization across all stages, sine shows 28.6% on the eval stage (2x the others), and option shows 40.0% on eval and 20.0% on map, reflecting the heavier arithmetic in those stages.

## 3. Throughput Scaling with Lane Count

The following results use the default resource configuration (bus_ports=2, reducer_throughput=4, flat strategy, 100 MHz).

### Roulette (flat)

| Lanes | Throughput (trials/s) | vs Serial | vs NumPy | Contention Rate |
|---|---|---|---|---|
| 1 | 16.7M | 3.4x | 0.2x | 0.00 |
| 4 | 66.7M | 13.5x | 0.9x | 0.00 |
| 8 | 133.0M | 26.9x | 1.7x | 0.00 |
| 16 | 200.0M | 40.4x | 2.6x | 1.00 |
| 32 | 200.0M | 40.4x | 2.6x | 1.00 |

Scaling is perfectly linear through 8 lanes with zero contention. At 16 lanes, throughput saturates at 200M trials/s and the contention rate jumps to 1.0, meaning every cycle has at least one lane stalled. The bus utilization climbs from 8.3% at L=1 to 99.9% at L=16, confirming that the memory bus is the bottleneck. Adding more lanes beyond 16 provides no additional throughput under this bus configuration.

### Sine Integral

| Lanes | Throughput (trials/s) | vs Serial | vs NumPy | Contention Rate |
|---|---|---|---|---|
| 1 | 14.3M | 1.2x | 0.2x | 0.00 |
| 4 | 57.1M | 5.0x | 0.7x | 0.00 |
| 8 | 114.0M | 9.9x | 1.3x | 0.00 |
| 16 | 200.0M | 17.4x | 2.3x | 1.00 |
| 32 | 200.0M | 17.4x | 2.3x | 1.00 |

Sine integral follows a similar scaling pattern. It runs contention-free through 8 lanes, then saturates at 16. The speedup over serial is smaller than roulette's because the serial Python baseline is faster for sine (11.5M/s vs 5.0M/s), leaving less room for the FPGA to differentiate.

### Option Pricing

| Lanes | Throughput (trials/s) | vs Serial | vs NumPy | Contention Rate |
|---|---|---|---|---|
| 1 | 10.0M | 3.1x | 0.2x | 0.00 |
| 4 | 40.0M | 12.5x | 1.0x | 0.00 |
| 8 | 80.0M | 25.0x | 1.9x | 0.00 |
| 16 | infeasible | - | - | - |
| 32 | infeasible | - | - | - |

Option pricing has the slowest single-lane throughput due to its 10-cycle pipeline, but scales linearly through the feasible range (1-8 lanes). At 16 and 32 lanes, the resource model reports the configuration as infeasible, meaning the combined LUT, DSP, and BRAM requirements of 16 option-pricing lanes exceed the Basys3 board's capacity. This is a realistic constraint: each option pricing lane requires DSP slices for the exp/sqrt/multiply operations, and the Basys3 has only 90 DSP slices.

## 4. Shared Resource Contention

### 4.1 Memory Bus Contention (Martingale vs. Flat)

The Martingale strategy requires an additional bus access per trial for updating the bet amount, doubling the bus demand compared to flat betting.

| Strategy | L=1 (trials/s) | L=4 (trials/s) | L=8 (trials/s) | L=16 (trials/s) | L=32 (trials/s) |
|---|---|---|---|---|---|
| Flat | 16.7M | 66.7M | 133.0M | 200.0M | 200.0M |
| Martingale | 14.3M | 57.1M | 66.7M | 66.7M | 66.7M |

Martingale throughput saturates at L=8 (66.7M) and stays pinned there, while flat continues scaling to L=16 (200M). This is a 3x difference at L=16. The contention rate for Martingale reaches 1.0 at L=8 (vs L=16 for flat), meaning the extra bus transaction per trial shifts the saturation point down by a factor of 2. The bus utilization at L=8 is 66.7% for both strategies, but Martingale's higher per-trial bus demand means the 2-port bus is fully contested at just 8 lanes.

### 4.2 Bus Port Scaling

Increasing bus ports directly alleviates contention. For roulette Martingale at L=8:

| Bus Ports | Throughput (trials/s) | Bus Utilization |
|---|---|---|
| 1 | 33.3M | 66.7% |
| 2 | 66.7M | 66.7% |
| 4 | 114.0M | 57.1% |

Each doubling of bus ports roughly doubles throughput until bus utilization drops below the saturation point. With 4 ports, the bus is no longer the bottleneck (utilization drops to 57.1%), and throughput approaches the contention-free rate.

### 4.3 Reducer Saturation

For workloads without bus access (sine, option), the reducer is the only shared resource. The sine integral at 100 MHz with bus_ports=2 demonstrates this:

| Reducer Throughput | L=4 (trials/s) | L=8 (trials/s) | L=16 (trials/s) |
|---|---|---|---|
| 1 | 57.1M | 100.0M | 100.0M |
| 2 | 57.1M | 114.0M | 200.0M |
| 4 | 57.1M | 114.0M | 200.0M |
| 8 | 57.1M | 114.0M | 200.0M |

With reducer_throughput=1, throughput caps at 100M at L=8 (reducer utilization = 99.9%). Increasing to throughput=2 pushes the saturation point to L=16. At throughput=4 and above, the reducer utilization at L=16 is only 50%, indicating the reducer is no longer the binding constraint and saturation is caused by a different resource (output buffer or structural limits).

## 5. Clock Frequency Sensitivity

The timing model caps the effective clock at the estimated fmax, which varies by workload complexity and structural parameters.

| Workload | 100 MHz | 200 MHz (eff) | 250 MHz (eff) | Speedup (max/100) |
|---|---|---|---|---|
| Roulette L=8 | 133M | 267M (200 MHz) | 312M (234 MHz) | 2.3x |
| Sine L=8 | 114M | 229M (200 MHz) | 244M (214 MHz) | 2.1x |
| Option L=8 | 80M | 147M (184 MHz) | 147M (184 MHz) | 1.8x |

Roulette achieves an effective clock of 234 MHz at the 250 MHz setting because its simple comparison logic has the highest fmax. Sine reaches 214 MHz, and option pricing caps at 184 MHz for both the 200 and 250 MHz settings due to its complex arithmetic path. This means requesting 250 MHz for option pricing provides no benefit over 200 MHz, which matches the expected behavior of an FPGA where the critical path through exp/sqrt operations limits the achievable clock.

The throughput scales proportionally with effective clock frequency, confirming that contention behavior operates in the cycle domain rather than the time domain.

## 6. Resource Feasibility

72 out of 720 configurations (10%) were marked infeasible by the resource model. All infeasible configs are option pricing at L=16 and L=32 (36 configs each), regardless of bus/reducer/clock settings. The per-lane resource cost for option pricing (DSP slices for exp, sqrt, multiply) is high enough that 16 lanes exceed the Basys3's 90 DSP slices and/or 33,280 logic cells. Roulette and sine remain feasible at all lane counts up to 32.

This is a meaningful result: it demonstrates that the FPGA's finite resources impose a hard upper bound on parallelism for arithmetic-heavy workloads, and that the simulator correctly captures this constraint before synthesis.

## 7. Statistical Accuracy

| Workload | FPGA Estimate (L=8) | Ground Truth | Absolute Error | Relative Error |
|---|---|---|---|---|
| Roulette (win rate) | 0.47173 | 0.47368 | 0.00195 | 0.41% |
| Sine integral | 2.00399 | 2.00000 | 0.00399 | 0.20% |
| Option pricing | 10.41581 | 10.45058 | 0.03477 | 0.33% |

All three workloads converge within 0.5% of their analytical ground truth at 100K trials. The LFSR chi-squared test for roulette passed (chi2=33.43, p=0.637), confirming that the LFSR produces a sufficiently uniform distribution across the 38 outcomes. The convergence analysis with 95% confidence intervals shows all three estimates tracking toward their ground truth values, with the CI bands narrowing as expected with increasing trial count.

## 8. Maximum Achievable Speedups

Under the best feasible configuration for each workload:

| Workload | Config | Throughput | vs Serial | vs NumPy |
|---|---|---|---|---|
| Roulette | L=32, 250 MHz, bus=4, red=4 | 835M | 168.4x | 10.9x |
| Sine | L=32, 200 MHz, bus=4, red=4 | 755M | 65.6x | 8.6x |
| Option | L=8, 200 MHz, bus=1, red=1 | 156M | 48.6x | 3.8x |

Roulette achieves the highest speedup (168x over serial) because its simple pipeline allows 32 lanes at high clock speeds. Option pricing is limited to 8 lanes by resource constraints, capping its speedup at 48.6x. The speedups over NumPy are more modest (3.8-10.9x) because NumPy's vectorized operations already eliminate most Python overhead.

## 9. Summary

The simulator captures the three key performance-limiting factors in FPGA Monte Carlo accelerators: pipeline depth (which determines single-lane throughput), shared resource contention (which caps multi-lane scaling), and resource feasibility (which limits maximum parallelism for arithmetic-heavy workloads). Roulette scales to 32 lanes with sub-microsecond trial latency, sine integral follows similar patterns but with reduced speedups due to its already-fast CPU baseline, and option pricing hits a hard resource ceiling at 8 lanes on the Basys3 board. The Martingale strategy halves roulette throughput at high lane counts by doubling bus demand, demonstrating how stateful trial-to-trial dependencies create contention that cannot be parallelized away. Clock frequency sensitivity shows diminishing returns for complex workloads due to timing constraints on the critical arithmetic path.
