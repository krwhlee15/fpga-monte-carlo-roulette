import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# ============================================================
# Utility / common helpers
# ============================================================

@dataclass
class BenchmarkResult:
    workload: str
    mode: str
    trials: int
    elapsed_sec: float
    throughput_trials_per_sec: float
    estimate: float
    extra: Optional[Dict] = None


def timed_run(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed


# ============================================================
# Workload 1: Roulette
# From report:
# - American roulette => 38 outcomes
# - Win rate target for even-money bet = 18/38
# - Two sub-workloads: flat betting and Martingale
# ============================================================

def roulette_serial_flat(
    n_trials: int,
    seed: int = 12345,
    bet_amount: float = 1.0,
    win_pockets: int = 18
) -> BenchmarkResult:
    """
    Serial Python baseline for roulette with flat betting.

    Assumes an even-money bet:
    - win if outcome in [0, win_pockets-1]
    - payout = +bet_amount on win, -bet_amount on loss

    Returns:
        estimate = empirical win rate
        extra = total payout, total wins
    """
    rng = random.Random(seed)

    wins = 0
    total_payout = 0.0

    for _ in range(n_trials):
        outcome = rng.randrange(38)   # maps RNG to 0..37
        win = outcome < win_pockets

        if win:
            wins += 1
            total_payout += bet_amount
        else:
            total_payout -= bet_amount

    estimate = wins / n_trials
    elapsed_dummy = 1.0  # replaced by timed_run wrapper

    return BenchmarkResult(
        workload="roulette_flat",
        mode="serial",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "wins": wins,
            "total_payout": total_payout,
            "expected_win_rate": 18 / 38
        }
    )


def roulette_numpy_flat(
    n_trials: int,
    seed: int = 12345,
    bet_amount: float = 1.0,
    win_pockets: int = 18,
    batch_size: int = 1_000_000
) -> BenchmarkResult:
    """
    NumPy-vectorized baseline for roulette flat betting.

    Generates trials in batches to avoid huge memory spikes for large N.
    """
    rng = np.random.default_rng(seed)

    wins = 0
    total_payout = 0.0
    done = 0

    while done < n_trials:
        m = min(batch_size, n_trials - done)
        outcomes = rng.integers(0, 38, size=m, dtype=np.int32)
        win_mask = outcomes < win_pockets

        batch_wins = int(np.sum(win_mask))
        wins += batch_wins

        # +bet on win, -bet on loss
        total_payout += bet_amount * (2 * batch_wins - m)
        done += m

    estimate = wins / n_trials
    elapsed_dummy = 1.0

    return BenchmarkResult(
        workload="roulette_flat",
        mode="numpy",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "wins": wins,
            "total_payout": total_payout,
            "expected_win_rate": 18 / 38
        }
    )


def roulette_serial_martingale(
    n_trials: int,
    seed: int = 12345,
    base_bet: float = 1.0,
    max_bet: Optional[float] = None,
    win_pockets: int = 18
) -> BenchmarkResult:
    """
    Serial Python baseline for Martingale roulette.

    Martingale:
    - start with base_bet
    - after loss: double bet
    - after win: reset to base_bet

    max_bet can cap the doubling to avoid unrealistic blow-up.
    """
    rng = random.Random(seed)

    wins = 0
    total_payout = 0.0
    current_bet = base_bet
    max_bet_seen = current_bet

    for _ in range(n_trials):
        outcome = rng.randrange(38)
        win = outcome < win_pockets

        if win:
            wins += 1
            total_payout += current_bet
            current_bet = base_bet
        else:
            total_payout -= current_bet
            current_bet *= 2.0
            if max_bet is not None:
                current_bet = min(current_bet, max_bet)

        if current_bet > max_bet_seen:
            max_bet_seen = current_bet

    estimate = wins / n_trials
    elapsed_dummy = 1.0

    return BenchmarkResult(
        workload="roulette_martingale",
        mode="serial",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "wins": wins,
            "total_payout": total_payout,
            "expected_win_rate": 18 / 38,
            "base_bet": base_bet,
            "max_bet_cap": max_bet,
            "max_bet_seen": max_bet_seen
        }
    )


def roulette_numpy_martingale(
    n_trials: int,
    seed: int = 12345,
    base_bet: float = 1.0,
    max_bet: Optional[float] = None,
    win_pockets: int = 18
) -> BenchmarkResult:
    """
    Martingale is stateful, so full vectorization is not valid.
    This function intentionally falls back to sequential logic,
    matching the limitation described in the report.
    """
    return roulette_serial_martingale(
        n_trials=n_trials,
        seed=seed,
        base_bet=base_bet,
        max_bet=max_bet,
        win_pockets=win_pockets
    )


# ============================================================
# Workload 2: Sine Integral Approximation
# Estimate integral of sin(x) over [a, b]
# report suggests [0, pi], exact value = 2
# ============================================================

def sine_integral_serial(
    n_trials: int,
    a: float = 0.0,
    b: float = math.pi,
    seed: int = 12345
) -> BenchmarkResult:
    rng = random.Random(seed)

    s = 0.0
    width = b - a

    for _ in range(n_trials):
        u = rng.random()          # uniform in [0,1)
        x = a + width * u
        s += math.sin(x)

    estimate = width * (s / n_trials)
    exact = math.cos(a) - math.cos(b)  # integral of sin(x) from a to b
    elapsed_dummy = 1.0

    return BenchmarkResult(
        workload="sine_integral",
        mode="serial",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "a": a,
            "b": b,
            "exact_value": exact,
            "absolute_error": abs(estimate - exact)
        }
    )


def sine_integral_numpy(
    n_trials: int,
    a: float = 0.0,
    b: float = math.pi,
    seed: int = 12345,
    batch_size: int = 1_000_000
) -> BenchmarkResult:
    rng = np.random.default_rng(seed)

    width = b - a
    s = 0.0
    done = 0

    while done < n_trials:
        m = min(batch_size, n_trials - done)
        u = rng.random(m)
        x = a + width * u
        s += np.sin(x).sum()
        done += m

    estimate = width * (s / n_trials)
    exact = math.cos(a) - math.cos(b)
    elapsed_dummy = 1.0

    return BenchmarkResult(
        workload="sine_integral",
        mode="numpy",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "a": a,
            "b": b,
            "exact_value": exact,
            "absolute_error": abs(estimate - exact)
        }
    )


# ============================================================
# Workload 3: European Call Option Pricing
# Report uses Black-Scholes Monte Carlo terminal price:
# ST = S0 * exp((r - 0.5*sigma^2)T + sigma*sqrt(T)*Z)
# payoff = max(ST - K, 0)
# price = exp(-rT) * mean(payoff)
# ============================================================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return math.exp(-r * T) * max(S0 * math.exp(r * T) - K, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)


def european_call_serial(
    n_trials: int,
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    seed: int = 12345
) -> BenchmarkResult:
    rng = random.Random(seed)

    payoff_sum = 0.0
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion_scale = sigma * math.sqrt(T)

    for _ in range(n_trials):
        z = rng.gauss(0.0, 1.0)
        ST = S0 * math.exp(drift + diffusion_scale * z)
        payoff = max(ST - K, 0.0)
        payoff_sum += payoff

    estimate = math.exp(-r * T) * (payoff_sum / n_trials)
    exact = black_scholes_call_price(S0, K, r, sigma, T)
    elapsed_dummy = 1.0

    return BenchmarkResult(
        workload="european_call",
        mode="serial",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "S0": S0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "black_scholes_exact": exact,
            "absolute_error": abs(estimate - exact)
        }
    )


def european_call_numpy(
    n_trials: int,
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    seed: int = 12345,
    batch_size: int = 1_000_000
) -> BenchmarkResult:
    rng = np.random.default_rng(seed)

    payoff_sum = 0.0
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion_scale = sigma * math.sqrt(T)
    done = 0

    while done < n_trials:
        m = min(batch_size, n_trials - done)
        z = rng.standard_normal(m)
        ST = S0 * np.exp(drift + diffusion_scale * z)
        payoff_sum += np.maximum(ST - K, 0.0).sum()
        done += m

    estimate = math.exp(-r * T) * (payoff_sum / n_trials)
    exact = black_scholes_call_price(S0, K, r, sigma, T)
    elapsed_dummy = 1.0

    return BenchmarkResult(
        workload="european_call",
        mode="numpy",
        trials=n_trials,
        elapsed_sec=elapsed_dummy,
        throughput_trials_per_sec=0.0,
        estimate=estimate,
        extra={
            "S0": S0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "black_scholes_exact": exact,
            "absolute_error": abs(estimate - exact)
        }
    )


# ============================================================
# Benchmark wrappers
# ============================================================

def finalize_timing(result: BenchmarkResult, elapsed: float) -> BenchmarkResult:
    result.elapsed_sec = elapsed
    result.throughput_trials_per_sec = result.trials / elapsed if elapsed > 0 else float("inf")
    return result


def run_and_print(name: str, fn, *args, **kwargs):
    result, elapsed = timed_run(fn, *args, **kwargs)
    result = finalize_timing(result, elapsed)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Workload    : {result.workload}")
    print(f"Mode        : {result.mode}")
    print(f"Trials      : {result.trials}")
    print(f"Elapsed (s) : {result.elapsed_sec:.6f}")
    print(f"Throughput  : {result.throughput_trials_per_sec:,.2f} trials/s")
    print(f"Estimate    : {result.estimate:.8f}")
    if result.extra:
        for k, v in result.extra.items():
            print(f"{k:16}: {v}")

    return result


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    N = 1_000_000
    SEED = 12345

    # Workload 1: roulette
    run_and_print(
        "Roulette - Serial Flat",
        roulette_serial_flat,
        N, SEED, 1.0, 18
    )
    run_and_print(
        "Roulette - NumPy Flat",
        roulette_numpy_flat,
        N, SEED, 1.0, 18
    )
    run_and_print(
        "Roulette - Serial Martingale",
        roulette_serial_martingale,
        N, SEED, 1.0, 1024.0, 18
    )
    run_and_print(
        "Roulette - NumPy Martingale (falls back to sequential)",
        roulette_numpy_martingale,
        N, SEED, 1.0, 1024.0, 18
    )

    # Workload 2: sine integral
    run_and_print(
        "Sine Integral - Serial",
        sine_integral_serial,
        N, 0.0, math.pi, SEED
    )
    run_and_print(
        "Sine Integral - NumPy",
        sine_integral_numpy,
        N, 0.0, math.pi, SEED
    )

    # Workload 3: European call option
    run_and_print(
        "European Call - Serial",
        european_call_serial,
        N, 100.0, 100.0, 0.05, 0.2, 1.0, SEED
    )
    run_and_print(
        "European Call - NumPy",
        european_call_numpy,
        N, 100.0, 100.0, 0.05, 0.2, 1.0, SEED
    )