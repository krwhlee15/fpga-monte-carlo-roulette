import numpy as np
from scipy import stats

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim
from fpga_model.lfsr import RED_NUMBERS


def convergence_analysis(outcome_histogram_running, bet_type="red_black"):
    """Compute running win rate and 95% confidence intervals.

    Args:
        outcome_histogram_running: array of outcomes (one per trial, in order)
        bet_type: "red_black" or "single_number"

    Returns dict with arrays: n_values, win_rates, ci_lower, ci_upper
    """
    n = len(outcome_histogram_running)
    assert n > 0

    if bet_type == "red_black":
        wins_cumulative = np.cumsum(np.isin(outcome_histogram_running, list(RED_NUMBERS)))
    elif bet_type == "single_number":
        wins_cumulative = np.cumsum(outcome_histogram_running == 17)
    else:
        assert False, f"Unknown bet type: {bet_type}"

    n_values = np.arange(1, n + 1)
    win_rates = wins_cumulative / n_values

    # 95% CI using normal approximation for proportion
    z = 1.96
    se = np.sqrt(win_rates * (1 - win_rates) / n_values)
    # Avoid division by zero at n=0 (already asserted n>0, but first element could be edge case)
    ci_lower = win_rates - z * se
    ci_upper = win_rates + z * se

    return {
        "n_values": n_values,
        "win_rates": win_rates,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def convergence_analysis_sine(values, config):
    """Running integral estimate convergence for sine workload."""
    n = len(values)
    assert n > 0
    interval = config.sine_b - config.sine_a
    cumsum = np.cumsum(values)
    n_values = np.arange(1, n + 1)
    estimates = interval * cumsum / n_values

    z = 1.96
    cumsum_sq = np.cumsum(values ** 2)
    running_var = cumsum_sq / n_values - (cumsum / n_values) ** 2
    running_var = np.maximum(running_var, 0)
    se = interval * np.sqrt(running_var / n_values)

    import math
    ground_truth = -math.cos(config.sine_b) + math.cos(config.sine_a)

    return {
        "n_values": n_values,
        "estimates": estimates,
        "ci_lower": estimates - z * se,
        "ci_upper": estimates + z * se,
        "ground_truth": ground_truth,
        "label": "Integral Estimate",
    }


def convergence_analysis_option(payoffs, config):
    """Running option price convergence for stock pricing workload."""
    import math
    n = len(payoffs)
    assert n > 0
    discount = math.exp(-config.r * config.T)
    cumsum = np.cumsum(payoffs)
    n_values = np.arange(1, n + 1)
    estimates = discount * cumsum / n_values

    z = 1.96
    cumsum_sq = np.cumsum(payoffs ** 2)
    running_var = cumsum_sq / n_values - (cumsum / n_values) ** 2
    running_var = np.maximum(running_var, 0)
    se = discount * np.sqrt(running_var / n_values)

    from fpga_model.workloads import get_workload_model
    wl = get_workload_model(config)
    ground_truth = wl.ground_truth(config)

    return {
        "n_values": n_values,
        "estimates": estimates,
        "ci_lower": estimates - z * se,
        "ci_upper": estimates + z * se,
        "ground_truth": ground_truth,
        "label": "Option Price",
    }


def lfsr_chi_squared(histogram):
    """Chi-squared goodness-of-fit test for uniform distribution over 38 outcomes.

    Args:
        histogram: array of length 38 with counts per outcome

    Returns (chi2_stat, p_value)
    """
    histogram = np.asarray(histogram)
    assert len(histogram) == 38
    total = histogram.sum()
    assert total > 0
    expected = np.full(38, total / 38.0)
    chi2_stat, p_value = stats.chisquare(histogram, f_exp=expected)
    return float(chi2_stat), float(p_value)


def throughput_ci(throughputs):
    """Compute mean throughput and 95% confidence interval.

    Args:
        throughputs: list of throughput measurements from multiple runs

    Returns (mean, ci_lower, ci_upper)
    """
    arr = np.array(throughputs)
    assert len(arr) >= 2, "Need at least 2 measurements for CI"
    mean = arr.mean()
    se = stats.sem(arr)
    ci = stats.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=se)
    return float(mean), float(ci[0]), float(ci[1])


def run_convergence_study(config, n_seeds=5):
    """Run FPGA sim with multiple seeds and collect convergence + throughput CI data.

    Returns dict with convergence data from first seed and throughput CI across seeds.
    """
    throughputs = []
    first_histogram = None

    for seed_offset in range(n_seeds):
        cfg = SimConfig(
            n_trials=config.n_trials,
            n_lanes=config.n_lanes,
            clock_freq_mhz=config.clock_freq_mhz,
            workload=config.workload,
            memory_bus_ports=config.memory_bus_ports,
            reducer_throughput=config.reducer_throughput,
            strategy=config.strategy,
            bet_type=config.bet_type,
            seed=config.seed + seed_offset * 1000,
            pipeline_depth=config.pipeline_depth,
            output_buffer_size=config.output_buffer_size,
            lfsr_reseed_interval=config.lfsr_reseed_interval,
            lfsr_reseed_latency=config.lfsr_reseed_latency,
            initial_bankroll=config.initial_bankroll,
            base_bet=config.base_bet,
            single_number_choice=config.single_number_choice,
            sine_a=config.sine_a,
            sine_b=config.sine_b,
            S0=config.S0,
            K=config.K,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
        )
        result = run_fpga_sim(cfg)
        throughputs.append(result["throughput"])
        if first_histogram is None:
            first_histogram = result["outcome_histogram"]

    mean_tput, ci_lo, ci_hi = throughput_ci(throughputs)

    return {
        "throughputs": throughputs,
        "mean_throughput": mean_tput,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "first_histogram": first_histogram,
    }
