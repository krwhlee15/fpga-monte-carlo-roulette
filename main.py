import argparse
import os
import numpy as np

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim
from evaluation.benchmark import run_benchmark
from evaluation.analysis import (
    convergence_analysis, convergence_analysis_sine, convergence_analysis_option,
    lfsr_chi_squared,
)
from evaluation.plots import generate_all_plots, plot_latency_histogram, plot_outcome_histogram, plot_convergence
from fpga_model.lfsr import LFSR

from cpu_baseline.runner import run_cpu_serial, run_cpu_numpy

def print_result(result):
    # Keep CLI output compact but consistent across CPU modes and workloads.
    print("=" * 60)
    print(f"Workload   : {result.workload}")
    print(f"Mode       : {result.mode}")
    print(f"Trials     : {result.trials}")
    print(f"Elapsed    : {result.elapsed_sec:.6f} s")
    print(f"Throughput : {result.throughput_trials_per_sec:,.2f} trials/s")
    print(f"Estimate   : {result.estimate:.8f}")
    if result.extra:
        print("Extra:")
        for k, v in result.extra.items():
            print(f"  {k}: {v}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="FPGA Monte Carlo Simulator")
    # Benchmark/plot orchestration flags.
    parser.add_argument("--quick", action="store_true", help="Run a small quick benchmark (fewer configs)")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation, only generate plots from existing CSV")
    parser.add_argument("--output-dir", default="results", help="Output directory for results and plots")

    # CPU-only path for comparing the software baselines without the FPGA model.
    parser.add_argument("--cpu-only", action="store_true", help="Run CPU baseline only and skip FPGA flow")
    parser.add_argument("--workload", choices=["roulette", "sine", "option"], default="roulette")
    parser.add_argument("--mode", choices=["serial", "numpy", "both"], default="both")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    
    # Workload-specific parameters exposed on the CLI.
    # Roulette args
    parser.add_argument("--strategy", choices=["flat", "martingale"], default="flat")
    parser.add_argument("--bet-type", choices=["red_black", "single_number"], default="red_black")
    parser.add_argument("--single-number-choice", type=int, default=17)
    parser.add_argument("--base-bet", type=float, default=10.0)

    # Sine args
    parser.add_argument("--sine-a", type=float, default=0.0)
    parser.add_argument("--sine-b", type=float, default=3.141592653589793)

    # Option args
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--T", type=float, default=1.0)

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")

    # Short-circuit into the baseline implementations when the user only wants CPU results.
    if args.cpu_only:
        config = SimConfig(
            n_trials=args.trials if args.trials is not None else 100_000,
            seed=args.seed,
            workload=args.workload,
            strategy=args.strategy,
            bet_type=args.bet_type,
            single_number_choice=args.single_number_choice,
            base_bet=args.base_bet,
            sine_a=args.sine_a,
            sine_b=args.sine_b,
            S0=args.S0,
            K=args.K,
            r=args.r,
            sigma=args.sigma,
            T=args.T,
        )

        if args.mode in ("serial", "both"):
            result = run_cpu_serial(config)
            print_result(result)

        if args.mode in ("numpy", "both"):
            result = run_cpu_numpy(config)
            print_result(result)

        return

    all_workloads = ["roulette", "sine", "option"]

    if not args.plot_only:
        # Sweep a smaller parameter grid for quick checks, otherwise use the fuller study.
        if args.quick:
            lane_counts = [1, 4, 16]
            bus_ports_list = [2]
            reducer_throughputs = [2, 4]
            clock_freqs = [100.0]
            n_trials = 10_000
        else:
            lane_counts = [1, 4, 8, 16, 32]
            bus_ports_list = [1, 2, 4]
            reducer_throughputs = [1, 2, 4, 8]
            clock_freqs = [100.0, 200.0, 250.0]
            n_trials = 100_000

        # Run each workload independently, then combine the rows into one report CSV.
        all_results = []
        all_cpu = {}
        for wl in all_workloads:
            strategies = ["flat", "martingale"] if wl == "roulette" else ["flat"]
            print("=" * 60)
            print(f"FPGA Monte Carlo Benchmark Suite ({wl})")
            print("=" * 60)

            results, cpu_results = run_benchmark(
                lane_counts=lane_counts,
                bus_ports_list=bus_ports_list,
                reducer_throughputs=reducer_throughputs,
                strategies=strategies,
                clock_freqs=clock_freqs,
                n_trials=n_trials,
                workload=wl,
                output_dir=output_dir,
            )
            all_results.extend(results)
            all_cpu[wl] = cpu_results

        # Write one normalized CSV so the plotting helpers only need a single input file.
        import csv as csv_mod
        assert len(all_results) > 0
        fieldnames = list(all_results[0].keys())
        # Ensure all rows have the same keys (roulette has fpga_win_rate)
        for r in all_results:
            for k in fieldnames:
                r.setdefault(k, "")
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCombined results saved to {csv_path}")

        # Run one representative configuration per workload to generate deeper analysis plots.
        print("\n" + "=" * 60)
        print("Running detailed analysis per workload...")
        print("=" * 60)

        fpga_result = None
        conv_data = {}
        conv_n = min(n_trials, 50_000)

        # loop through workload tests
        for wl in all_workloads:
            print(f"\n--- {wl} ---")
            detail_config = SimConfig(
                n_trials=n_trials,
                n_lanes=8,
                memory_bus_ports=2,
                reducer_throughput=4,
                strategy="flat",
                workload=wl,
            )
            result = run_fpga_sim(detail_config)

            # Use last workload's result for latency histogram (or roulette if available)
            if wl == "roulette":
                fpga_result = result

            truth = result.get("ground_truth")
            err = result.get("absolute_error")
            print(f"  Estimate={result['estimate']:.6f}", end="")
            if truth is not None:
                print(f", truth={truth:.6f}", end="")
            if err is not None:
                print(f", error={err:.6f}", end="")
            print()

            if wl == "roulette":
                chi2, p_val = lfsr_chi_squared(np.array(result["outcome_histogram"]))
                print(f"  LFSR chi-squared: chi2={chi2:.2f}, p={p_val:.4f}")
                if p_val > 0.05:
                    print("    -> PASS: uniform distribution (p > 0.05)")
                else:
                    print("    -> FAIL: non-uniform distribution (p <= 0.05)")

                conv_config = SimConfig(
                    n_trials=conv_n, n_lanes=1, memory_bus_ports=2,
                    reducer_throughput=4, strategy="flat", workload="roulette",
                )
                conv_result = run_fpga_sim(conv_config)
                outcomes_seq = np.array(conv_result["ordered_outcomes"])
                conv_data["roulette"] = convergence_analysis(outcomes_seq, bet_type=conv_config.bet_type)

            elif wl == "sine":
                import math
                lfsr = LFSR(seed=detail_config.seed + 1)
                interval = detail_config.sine_b - detail_config.sine_a
                values = np.array([
                    math.sin(detail_config.sine_a + (lfsr.step() / 4294967296.0) * interval)
                    for _ in range(conv_n)
                ])
                conv_data["sine"] = convergence_analysis_sine(values, detail_config)

            elif wl == "option":
                import math
                lfsr = LFSR(seed=detail_config.seed + 1)
                payoffs = []
                for _ in range(conv_n):
                    u1 = lfsr.step() / 4294967296.0
                    u2 = lfsr.step() / 4294967296.0
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    ST = detail_config.S0 * math.exp(
                        (detail_config.r - 0.5 * detail_config.sigma ** 2) * detail_config.T
                        + detail_config.sigma * math.sqrt(detail_config.T) * z
                    )
                    payoffs.append(max(ST - detail_config.K, 0.0))
                conv_data["option"] = convergence_analysis_option(np.array(payoffs), detail_config)

    else:
        fpga_result = None
        conv_data = None

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    generate_all_plots(
        csv_path=csv_path,
        output_dir=output_dir,
        fpga_result=fpga_result if not args.plot_only else None,
        convergence_data=conv_data if not args.plot_only else None,
    )

    print("\nDone! Results in:", output_dir)


if __name__ == "__main__":
    main()
