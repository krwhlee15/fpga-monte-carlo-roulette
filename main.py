import argparse
import os
import numpy as np

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim
from evaluation.benchmark import run_benchmark
from evaluation.analysis import convergence_analysis, lfsr_chi_squared
from evaluation.plots import generate_all_plots, plot_latency_histogram, plot_outcome_histogram, plot_convergence

from cpu_baseline.runner import run_cpu_serial, run_cpu_numpy

def print_result(result):
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
    # EXISTING ARGS
    parser.add_argument("--quick", action="store_true", help="Run a small quick benchmark (fewer configs)")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation, only generate plots from existing CSV")
    parser.add_argument("--output-dir", default="results", help="Output directory for results and plots")

    # NEW CPU-ONLY ARGS
    parser.add_argument("--cpu-only", action="store_true", help="Run CPU baseline only and skip FPGA flow")
    parser.add_argument("--workload", choices=["roulette", "sine", "option"], default="roulette")
    parser.add_argument("--mode", choices=["serial", "numpy", "both"], default="both")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    
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

    # NEW: CPU-only branch
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

    if not args.plot_only:
        # Run benchmark sweep
        if args.quick:
            lane_counts = [1, 4, 16]
            bus_ports_list = [2]
            reducer_throughputs = [2, 4]
            strategies = ["flat", "martingale"] if args.workload == "roulette" else ["flat"]
            n_trials = 10_000
        else:
            lane_counts = [1, 4, 8, 16, 32]
            bus_ports_list = [1, 2, 4]
            reducer_throughputs = [1, 2, 4, 8]
            strategies = ["flat", "martingale"] if args.workload == "roulette" else ["flat"]
            n_trials = 1_000_000

        print("=" * 60)
        print(f"FPGA Monte Carlo Benchmark Suite ({args.workload})")
        print("=" * 60)

        results, cpu_results = run_benchmark(
            lane_counts=lane_counts,
            bus_ports_list=bus_ports_list,
            reducer_throughputs=reducer_throughputs,
            strategies=strategies,
            n_trials=n_trials,
            workload=args.workload,
            output_dir=output_dir,
        )

        # Run a detailed single config for latency + convergence data
        print("\n" + "=" * 60)
        print(f"Running detailed FPGA analysis for workload={args.workload}...")
        print("=" * 60)

        detail_config = SimConfig(
            n_trials=n_trials,
            n_lanes=8,
            memory_bus_ports=2,
            reducer_throughput=4,
            strategy="flat",
            workload=args.workload,
        )
        fpga_result = run_fpga_sim(detail_config)

        conv_data = None
        if args.workload == "roulette":
            convergence_config = SimConfig(
                n_trials=min(n_trials, 50_000),
                n_lanes=1,
                memory_bus_ports=2,
                reducer_throughput=4,
                strategy="flat",
                workload="roulette",
            )
            conv_result = run_fpga_sim(convergence_config)
            outcomes_seq = np.array(conv_result["ordered_outcomes"])
            conv_data = convergence_analysis(outcomes_seq, bet_type=convergence_config.bet_type)

            chi2, p_val = lfsr_chi_squared(np.array(fpga_result["outcome_histogram"]))
            print(f"LFSR chi-squared test: chi2={chi2:.2f}, p-value={p_val:.4f}")
            if p_val > 0.05:
                print("  -> PASS: outcome distribution is consistent with uniform (p > 0.05)")
            else:
                print("  -> FAIL: outcome distribution deviates from uniform (p <= 0.05)")

            print(f"Win rate (8 lanes, flat): {fpga_result['win_rate']:.4f} "
                  f"(theoretical: {18/38:.4f})")
        else:
            truth = fpga_result.get("ground_truth")
            err = fpga_result.get("absolute_error")
            if truth is not None and err is not None:
                print(f"Estimate={fpga_result['estimate']:.6f}, truth={truth:.6f}, abs_error={err:.6f}")

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
