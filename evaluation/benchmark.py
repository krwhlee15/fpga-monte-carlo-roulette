import csv
import os
import itertools
from dataclasses import replace

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim
from cpu_baseline.runner import run_cpu_serial, run_cpu_numpy



def run_benchmark(
    lane_counts=None,
    bus_ports_list=None,
    reducer_throughputs=None,
    strategies=None,
    clock_freqs=None,
    n_trials=100_000,
    workload="roulette",
    output_dir="results",
):
    """Run full parameter sweep and save results to CSV."""
    if lane_counts is None:
        lane_counts = [1, 2, 4, 8, 16, 32, 64]
    if bus_ports_list is None:
        bus_ports_list = [1, 2, 4]
    if reducer_throughputs is None:
        reducer_throughputs = [1, 2, 4, 8]
    if strategies is None:
        strategies = ["flat", "martingale"]
    if clock_freqs is None:
        clock_freqs = [100.0]
    if workload != "roulette":
        strategies = ["flat"]

    os.makedirs(output_dir, exist_ok=True)

    # CPU baselines provide the software reference point for every FPGA configuration.
    base_config = SimConfig(
        n_trials=n_trials,
        workload=workload,
    )

    # CPU work does not depend on lane/bus/reducer counts, so compute it once per strategy.
    cpu_results = {}
    for strategy in strategies:
        cfg = replace(base_config, strategy=strategy)
        serial = run_cpu_serial(cfg)
        numpy_res = run_cpu_numpy(cfg)
        cpu_results[strategy] = {
            "serial_throughput": serial.throughput_trials_per_sec,
            "serial_elapsed": serial.elapsed_sec,
            "serial_estimate": serial.estimate,
            "serial_extra": serial.extra,
            "numpy_throughput": numpy_res.throughput_trials_per_sec,
            "numpy_elapsed": numpy_res.elapsed_sec,
            "numpy_estimate": numpy_res.estimate,
            "numpy_extra": numpy_res.extra,
        }

        print(
            f"CPU baselines ({workload}, {strategy}): "
            f"serial={serial.throughput_trials_per_sec:.0f} trials/s, "
            f"numpy={numpy_res.throughput_trials_per_sec:.0f} trials/s"
        )

    # Sweep the FPGA design space and join each modeled run with the baseline data.
    fpga_base_config = SimConfig(
        n_trials=n_trials,
        workload=workload,
    )
    results = []
    combos = list(itertools.product(lane_counts, bus_ports_list, reducer_throughputs, strategies, clock_freqs))
    total = len(combos)

    for idx, (n_lanes, bus_ports, red_tput, strategy, clock_freq) in enumerate(combos):
        cfg = replace(
            fpga_base_config,
            n_lanes=n_lanes,
            memory_bus_ports=bus_ports,
            reducer_throughput=red_tput,
            strategy=strategy,
            clock_freq_mhz=clock_freq,
        )

        print(f"[{idx + 1}/{total}] lanes={n_lanes}, bus={bus_ports}, "
              f"red={red_tput}, strat={strategy}, freq={clock_freq}...", end=" ", flush=True)

        fpga = run_fpga_sim(cfg)
        cpu = cpu_results[strategy]

        # Store one flat row per configuration so downstream plotting stays simple.
        row = {
            "workload": workload,
            "n_lanes": n_lanes,
            "bus_ports": bus_ports,
            "reducer_throughput": red_tput,
            "strategy": strategy,
            "clock_freq_mhz": clock_freq,
            "n_trials": cfg.n_trials,
            "feasible": float(fpga["feasible"]),
            "contention_rate": fpga["contention_rate"],

            "fpga_throughput": fpga["throughput"],
            "fpga_total_cycles": fpga["total_cycles"],
            "fpga_modeled_time_sec": fpga["modeled_time_sec"],
            "fpga_effective_clock_mhz": fpga["clock_mhz"],
            "fpga_fmax_mhz": fpga["fmax_mhz"],
            "fpga_estimate": fpga["estimate"],
            "fpga_ground_truth": fpga["ground_truth"] if fpga["ground_truth"] is not None else "",
            "fpga_absolute_error": fpga.get("absolute_error", ""),
            "fpga_mean_latency_cycles": fpga["mean_latency_cycles"],
            "fpga_max_latency_cycles": fpga["max_latency_cycles"],
            "bus_utilization": fpga["bus_utilization"],
            "bus_total_wait": fpga["bus_total_wait"],
            "buffer_total_wait": fpga["buffer_wait_cycles"],
            "reducer_utilization": fpga["reducer_utilization"],
            "reducer_total_wait": fpga["reducer_total_wait"],
            "rng_utilization": fpga["stage_utilization"]["rng"],
            "map_utilization": fpga["stage_utilization"]["map"],
            "eval_utilization": fpga["stage_utilization"]["eval"],
            "update_utilization": fpga["stage_utilization"]["update"],

            "serial_throughput": cpu["serial_throughput"],
            "numpy_throughput": cpu["numpy_throughput"],
            "speedup_vs_serial": fpga["throughput"] / cpu["serial_throughput"] if cpu["serial_throughput"] > 0 else 0,
            "speedup_vs_numpy": fpga["throughput"] / cpu["numpy_throughput"] if cpu["numpy_throughput"] > 0 else 0,
        }
        if workload == "roulette":
            row["fpga_win_rate"] = fpga["win_rate"]
        results.append(row)
        print(f"throughput={fpga['throughput']:.2e}, speedup_serial={row['speedup_vs_serial']:.1f}x")

    # Persist the sweep so plotting/analysis can be rerun without recomputing simulations.
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    assert len(results) > 0
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")
    return results, cpu_results
