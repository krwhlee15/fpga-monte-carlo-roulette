import argparse
import os
import numpy as np

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim
from fpga_model.lfsr import LFSR
from evaluation.benchmark import run_benchmark
from evaluation.analysis import convergence_analysis, lfsr_chi_squared, run_convergence_study
from evaluation.plots import generate_all_plots, plot_latency_histogram, plot_outcome_histogram, plot_convergence


def main():
    parser = argparse.ArgumentParser(description="FPGA Monte Carlo Roulette Simulator")
    parser.add_argument("--quick", action="store_true",
                        help="Run a small quick benchmark (fewer configs)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation, only generate plots from existing CSV")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results and plots")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")

    if not args.plot_only:
        # Run benchmark sweep
        if args.quick:
            lane_counts = [1, 4, 16]
            bus_ports_list = [2]
            reducer_throughputs = [2, 4]
            strategies = ["flat", "martingale"]
            n_trials = 10_000
        else:
            lane_counts = [1, 2, 4, 8, 16, 32, 64]
            bus_ports_list = [1, 2, 4]
            reducer_throughputs = [1, 2, 4, 8]
            strategies = ["flat", "martingale"]
            n_trials = 100_000

        print("=" * 60)
        print("FPGA Monte Carlo Roulette - Benchmark Suite")
        print("=" * 60)

        results, cpu_results = run_benchmark(
            lane_counts=lane_counts,
            bus_ports_list=bus_ports_list,
            reducer_throughputs=reducer_throughputs,
            strategies=strategies,
            n_trials=n_trials,
            output_dir=output_dir,
        )

        # Run a detailed single config for latency + convergence data
        print("\n" + "=" * 60)
        print("Running detailed analysis (latency, convergence, chi-squared)...")
        print("=" * 60)

        detail_config = SimConfig(
            n_trials=n_trials,
            n_lanes=8,
            memory_bus_ports=2,
            reducer_throughput=4,
            strategy="flat",
        )
        fpga_result = run_fpga_sim(detail_config)

        # Convergence: need per-trial outcomes. Run a single-lane sim to get ordered outcomes.
        convergence_config = SimConfig(
            n_trials=min(n_trials, 50_000),
            n_lanes=1,
            memory_bus_ports=2,
            reducer_throughput=4,
            strategy="flat",
        )
        conv_result = run_fpga_sim(convergence_config)
        # Generate outcome sequence from a fresh LFSR for convergence analysis
        lfsr = LFSR(seed=convergence_config.seed + 1)
        outcomes_seq = np.array([lfsr.step() % 38 for _ in range(convergence_config.n_trials)])
        conv_data = convergence_analysis(outcomes_seq, bet_type=convergence_config.bet_type)

        # Chi-squared test
        chi2, p_val = lfsr_chi_squared(fpga_result["outcome_histogram"])
        print(f"LFSR chi-squared test: chi2={chi2:.2f}, p-value={p_val:.4f}")
        if p_val > 0.05:
            print("  -> PASS: outcome distribution is consistent with uniform (p > 0.05)")
        else:
            print("  -> FAIL: outcome distribution deviates from uniform (p <= 0.05)")

        print(f"Win rate (8 lanes, flat): {fpga_result['win_rate']:.4f} "
              f"(theoretical: {18/38:.4f})")

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
