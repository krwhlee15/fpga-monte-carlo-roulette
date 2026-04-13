import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from evaluation.analysis import lfsr_chi_squared


def load_results(csv_path):
    """Load benchmark results from CSV into a list of dicts."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in row:
                try:
                    # CSV is string-based; convert numeric-looking fields back to floats.
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def filter_rows(rows, **kwargs):
    """Filter rows by matching key-value pairs."""
    result = rows
    for k, v in kwargs.items():
        result = [r for r in result if r[k] == v]
    return result


def plot_throughput_vs_lanes(rows, output_dir, bus_ports=2, reducer_tput=4):
    """Plot 1: Throughput vs lane count for both strategies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = sorted({row["strategy"] for row in rows if "strategy" in row})

    for strategy in strategies:
        subset = filter_rows(rows, strategy=strategy, bus_ports=bus_ports, reducer_throughput=reducer_tput)
        subset.sort(key=lambda r: r["n_lanes"])
        if not subset:
            continue
        lanes = [r["n_lanes"] for r in subset]
        tputs = [r["fpga_throughput"] for r in subset]
        ax.plot(lanes, tputs, "o-", label=f"{strategy}")

    # Overlay the ideal linear scale-up from the single-lane flat-betting point.
    if rows:
        base_subset = filter_rows(rows, strategy="flat", bus_ports=bus_ports, reducer_throughput=reducer_tput, n_lanes=1.0)
        if base_subset:
            base_tput = base_subset[0]["fpga_throughput"]
            lanes_ref = [r["n_lanes"] for r in filter_rows(rows, strategy="flat", bus_ports=bus_ports, reducer_throughput=reducer_tput)]
            lanes_ref.sort()
            ax.plot(lanes_ref, [base_tput * l for l in lanes_ref], "--", color="gray",
                    alpha=0.5, label="Ideal linear")

    ax.set_xlabel("Number of Lanes")
    ax.set_ylabel("Throughput (trials/sec)")
    ax.set_title("FPGA Throughput vs Lane Count")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "throughput_vs_lanes.png"), dpi=150)
    plt.close(fig)


def plot_speedup_vs_lanes(rows, output_dir, bus_ports=2, reducer_tput=4):
    """Plot 2: Speedup vs lane count relative to CPU baselines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = sorted({row["strategy"] for row in rows if "strategy" in row})

    for strategy in strategies:
        subset = filter_rows(rows, strategy=strategy, bus_ports=bus_ports, reducer_throughput=reducer_tput)
        subset.sort(key=lambda r: r["n_lanes"])
        if not subset:
            continue
        lanes = [r["n_lanes"] for r in subset]
        speedup_serial = [r["speedup_vs_serial"] for r in subset]
        speedup_numpy = [r["speedup_vs_numpy"] for r in subset]
        ax.plot(lanes, speedup_serial, "o-", label=f"{strategy} vs serial")
        ax.plot(lanes, speedup_numpy, "s--", label=f"{strategy} vs numpy")

    ax.set_xlabel("Number of Lanes")
    ax.set_ylabel("Speedup")
    ax.set_title("FPGA Speedup vs CPU Baselines")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "speedup_vs_lanes.png"), dpi=150)
    plt.close(fig)


def plot_bus_utilization(rows, output_dir, reducer_tput=4, strategy="flat"):
    """Plot 3: Memory bus utilization vs lane count for different bus port configs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for bp in [1, 2, 4]:
        subset = filter_rows(rows, strategy=strategy, bus_ports=bp, reducer_throughput=reducer_tput)
        subset.sort(key=lambda r: r["n_lanes"])
        if not subset:
            continue
        lanes = [r["n_lanes"] for r in subset]
        util = [r["bus_utilization"] for r in subset]
        ax.plot(lanes, util, "o-", label=f"bus_ports={int(bp)}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Number of Lanes")
    ax.set_ylabel("Bus Utilization")
    ax.set_title(f"Memory Bus Utilization ({strategy} betting)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"bus_utilization_{strategy}.png"), dpi=150)
    plt.close(fig)


def plot_reducer_saturation(rows, output_dir, bus_ports=2, strategy="flat"):
    """Plot 4: Throughput vs lane count for different reducer capacities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for rt in [1, 2, 4, 8]:
        subset = filter_rows(rows, strategy=strategy, bus_ports=bus_ports, reducer_throughput=rt)
        subset.sort(key=lambda r: r["n_lanes"])
        if not subset:
            continue
        lanes = [r["n_lanes"] for r in subset]
        tputs = [r["fpga_throughput"] for r in subset]
        ax.plot(lanes, tputs, "o-", label=f"reducer_tput={int(rt)}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Number of Lanes")
    ax.set_ylabel("Throughput (trials/sec)")
    ax.set_title(f"Reducer Saturation ({strategy} betting)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reducer_saturation_{strategy}.png"), dpi=150)
    plt.close(fig)


def plot_utilization_heatmap(rows, output_dir, bus_ports=2, reducer_tput=4):
    """Plot 5: Heatmap of bus and reducer utilization across lanes and strategies."""
    strategies = sorted({row["strategy"] for row in rows if "strategy" in row})
    if not strategies:
        return

    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]

    im = None
    for ax, strategy in zip(axes, strategies):
        subset = filter_rows(rows, strategy=strategy, bus_ports=bus_ports, reducer_throughput=reducer_tput)
        subset.sort(key=lambda r: r["n_lanes"])
        if not subset:
            continue
        lanes = [int(r["n_lanes"]) for r in subset]
        bus_util = [r["bus_utilization"] for r in subset]
        red_util = [r["reducer_utilization"] for r in subset]

        data = np.array([bus_util, red_util])
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(lanes)))
        ax.set_xticklabels([str(l) for l in lanes])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Memory Bus", "Reducer"])
        ax.set_xlabel("Number of Lanes")
        ax.set_title(f"Resource Utilization ({strategy})")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Utilization")
    fig.savefig(os.path.join(output_dir, "utilization_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_latency_histogram(lane_latencies, output_dir, title_suffix=""):
    """Plot 6: Trial latency distribution from FPGA sim results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_latencies = []
    for lane_id, lats in lane_latencies.items():
        # Collapse per-lane latency lists into one distribution for the histogram.
        all_latencies.extend(lats)

    if not all_latencies:
        plt.close(fig)
        return

    ax.hist(all_latencies, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Trial Latency (cycles)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Trial Latency Distribution {title_suffix}")
    ax.axvline(np.mean(all_latencies), color="red", linestyle="--",
               label=f"Mean={np.mean(all_latencies):.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"latency_histogram{title_suffix.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)


def plot_convergence(n_values, win_rates, ci_lower, ci_upper, output_dir, bet_type="red_black"):
    """Plot 7: Running win rate with confidence interval bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Downsample dense series so the plot remains legible and fast to render.
    step = max(1, len(n_values) // 1000)
    idx = slice(None, None, step)

    ax.plot(n_values[idx], win_rates[idx], linewidth=0.8, label="Win rate")
    ax.fill_between(n_values[idx], ci_lower[idx], ci_upper[idx], alpha=0.2, label="95% CI")

    # Draw the analytical target so the running estimate has a fixed reference line.
    if bet_type == "red_black":
        theoretical = 18.0 / 38.0
    elif bet_type == "single_number":
        theoretical = 1.0 / 38.0
    else:
        theoretical = None

    if theoretical is not None:
        ax.axhline(theoretical, color="red", linestyle="--", alpha=0.7,
                    label=f"Theoretical={theoretical:.4f}")

    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Win Rate")
    ax.set_title("Convergence Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence.png"), dpi=150)
    plt.close(fig)


def plot_outcome_histogram(histogram, output_dir):
    """Plot 8: Outcome histogram with chi-squared annotation."""
    histogram = np.asarray(histogram)
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [str(i) for i in range(37)] + ["00"]
    colors = []
    from fpga_model.lfsr import RED_NUMBERS, BLACK_NUMBERS
    for i in range(38):
        if i == 0 or i == 37:
            colors.append("green")
        elif i in RED_NUMBERS:
            colors.append("red")
        else:
            colors.append("black")

    ax.bar(range(38), histogram, color=colors, edgecolor="gray", alpha=0.8)
    ax.set_xticks(range(38))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Frequency")

    chi2, p_val = lfsr_chi_squared(histogram)
    ax.set_title(f"Outcome Distribution (chi-squared={chi2:.2f}, p={p_val:.4f})")

    # A uniform RNG would make every pocket appear equally often in expectation.
    expected = histogram.sum() / 38.0
    ax.axhline(expected, color="blue", linestyle="--", alpha=0.5, label=f"Expected={expected:.0f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "outcome_histogram.png"), dpi=150)
    plt.close(fig)


def plot_convergence_generic(conv_data, output_dir, workload):
    """Convergence plot for any workload with CI bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_values = conv_data["n_values"]
    estimates = conv_data["estimates"]
    ci_lower = conv_data["ci_lower"]
    ci_upper = conv_data["ci_upper"]
    truth = conv_data["ground_truth"]
    label = conv_data["label"]

    step = max(1, len(n_values) // 1000)
    idx = slice(None, None, step)

    ax.plot(n_values[idx], estimates[idx], linewidth=0.8, label=label)
    ax.fill_between(n_values[idx], ci_lower[idx], ci_upper[idx], alpha=0.2, label="95% CI")
    ax.axhline(truth, color="red", linestyle="--", alpha=0.7,
               label=f"Truth={truth:.4f}")

    ax.set_xlabel("Number of Trials")
    ax.set_ylabel(label)
    ax.set_title(f"Convergence Analysis ({workload})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"convergence_{workload}.png"), dpi=150)
    plt.close(fig)


def plot_clock_freq_sensitivity(rows, output_dir, bus_ports=2, reducer_tput=4, strategy="flat"):
    """Throughput vs lane count at different clock frequencies."""
    freqs = sorted(set(r.get("clock_freq_mhz", 100.0) for r in rows))
    if len(freqs) <= 1:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for freq in freqs:
        subset = filter_rows(rows, bus_ports=bus_ports, reducer_throughput=reducer_tput,
                            strategy=strategy, clock_freq_mhz=freq)
        subset.sort(key=lambda r: r["n_lanes"])
        if not subset:
            continue
        lanes = [r["n_lanes"] for r in subset]
        tputs = [r["fpga_throughput"] for r in subset]
        ax.plot(lanes, tputs, "o-", label=f"{int(freq)} MHz")

    ax.set_xlabel("Number of Lanes")
    ax.set_ylabel("Throughput (trials/sec)")
    ax.set_title("Clock Frequency Sensitivity")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "clock_freq_sensitivity.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(csv_path, output_dir, fpga_result=None, convergence_data=None):
    """Generate all plots from benchmark CSV and optional detailed run data."""
    os.makedirs(output_dir, exist_ok=True)
    rows = load_results(csv_path)
    if not rows:
        return

    # These plots cover scaling, bottlenecks, and one representative detailed run.
    print("Generating plots...")
    plot_throughput_vs_lanes(rows, output_dir)
    plot_speedup_vs_lanes(rows, output_dir)
    plot_bus_utilization(rows, output_dir, strategy="flat")
    plot_bus_utilization(rows, output_dir, strategy="martingale")
    plot_reducer_saturation(rows, output_dir)
    plot_utilization_heatmap(rows, output_dir)
    plot_clock_freq_sensitivity(rows, output_dir)

    if fpga_result is not None:
        plot_latency_histogram(fpga_result["lane_latencies"], output_dir,
                               title_suffix=f" (L={fpga_result['n_lanes']}, {fpga_result['strategy']})")
        if "outcome_histogram" in fpga_result:
            plot_outcome_histogram(fpga_result["outcome_histogram"], output_dir)

    if convergence_data is not None:
        for wl, conv in convergence_data.items():
            if wl == "roulette":
                plot_convergence(
                    conv["n_values"], conv["win_rates"],
                    conv["ci_lower"], conv["ci_upper"], output_dir,
                )
            else:
                plot_convergence_generic(conv, output_dir, wl)

    print(f"Plots saved to {output_dir}/")
