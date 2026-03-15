import csv
import os
import itertools
from dataclasses import replace

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim
from cpu_baseline.serial_sim import run_serial
from cpu_baseline.numpy_sim import run_numpy


def run_benchmark(
    lane_counts=None,
    bus_ports_list=None,
    reducer_throughputs=None,
    strategies=None,
    n_trials=100_000,
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

    os.makedirs(output_dir, exist_ok=True)

    base_config = SimConfig(n_trials=n_trials)

    # Run CPU baselines once per strategy
    cpu_results = {}
    for strategy in strategies:
        cfg = replace(base_config, strategy=strategy)
        serial = run_serial(cfg)
        numpy_res = run_numpy(cfg)
        cpu_results[strategy] = {
            "serial_throughput": serial["throughput"],
            "serial_elapsed": serial["elapsed_sec"],
            "serial_win_rate": serial["win_rate"],
            "numpy_throughput": numpy_res["throughput"],
            "numpy_elapsed": numpy_res["elapsed_sec"],
            "numpy_win_rate": numpy_res["win_rate"],
        }
        print(f"CPU baselines ({strategy}): serial={serial['throughput']:.0f} trials/s, "
              f"numpy={numpy_res['throughput']:.0f} trials/s")

    # FPGA sweep
    results = []
    combos = list(itertools.product(lane_counts, bus_ports_list, reducer_throughputs, strategies))
    total = len(combos)

    for idx, (n_lanes, bus_ports, red_tput, strategy) in enumerate(combos):
        cfg = replace(
            base_config,
            n_lanes=n_lanes,
            memory_bus_ports=bus_ports,
            reducer_throughput=red_tput,
            strategy=strategy,
        )

        print(f"[{idx + 1}/{total}] lanes={n_lanes}, bus_ports={bus_ports}, "
              f"reducer_tput={red_tput}, strategy={strategy}...", end=" ", flush=True)

        fpga = run_fpga_sim(cfg)
        cpu = cpu_results[strategy]

        row = {
            "n_lanes": n_lanes,
            "bus_ports": bus_ports,
            "reducer_throughput": red_tput,
            "strategy": strategy,
            "n_trials": cfg.n_trials,
            "fpga_throughput": fpga["throughput"],
            "fpga_total_cycles": fpga["total_cycles"],
            "fpga_modeled_time_sec": fpga["modeled_time_sec"],
            "fpga_win_rate": fpga["win_rate"],
            "bus_utilization": fpga["bus_utilization"],
            "bus_total_wait": fpga["bus_total_wait"],
            "reducer_utilization": fpga["reducer_utilization"],
            "reducer_total_wait": fpga["reducer_total_wait"],
            "serial_throughput": cpu["serial_throughput"],
            "numpy_throughput": cpu["numpy_throughput"],
            "speedup_vs_serial": fpga["throughput"] / cpu["serial_throughput"] if cpu["serial_throughput"] > 0 else 0,
            "speedup_vs_numpy": fpga["throughput"] / cpu["numpy_throughput"] if cpu["numpy_throughput"] > 0 else 0,
        }
        results.append(row)
        print(f"throughput={fpga['throughput']:.2e}, speedup_serial={row['speedup_vs_serial']:.1f}x")

    # Save to CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    assert len(results) > 0
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")
    return results, cpu_results
