import math
import random
import time
import numpy as np
from .common import BenchmarkResult

def normal_cdf(x):
    # Standard normal CDF used by the analytical Black-Scholes formula.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call_price(S0, K, r, sigma, T):
    # Closed-form price for a European call under geometric Brownian motion.
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)

def run_option_serial(config):
    rng = random.Random(config.seed)
    # Precompute the deterministic part of the log-price evolution.
    drift = (config.r - 0.5 * config.sigma * config.sigma) * config.T
    diffusion = config.sigma * math.sqrt(config.T)
    payoff_sum = 0.0

    start = time.perf_counter()
    for _ in range(config.n_trials):
        # z is the standard normal shock for one simulated terminal price.
        z = rng.gauss(0.0, 1.0)
        # Simulated stock price at maturity using constants and z
        ST = config.S0 * math.exp(drift + diffusion * z)
        # European call payoff: exercise only when the option finishes in the money.
        payoff_sum += max(ST - config.K, 0.0)
    elapsed = time.perf_counter() - start

    # Discount the average payoff back to present value.
    estimate = math.exp(-config.r * config.T) * (payoff_sum / config.n_trials)
    # Compare the Monte Carlo estimate against the analytical benchmark, aka ground truth
    exact = black_scholes_call_price(config.S0, config.K, config.r, config.sigma, config.T)

    return BenchmarkResult(
        workload="european_call",
        mode="serial",
        trials=config.n_trials,
        elapsed_sec=elapsed,
        throughput_trials_per_sec=config.n_trials / elapsed,
        estimate=estimate,
        extra={"black_scholes_exact": exact, "absolute_error": abs(estimate - exact)}
    )

# same as above but vectorized for n_trials
def run_option_numpy(config, batch_size=1_000_000):
    rng = np.random.default_rng(config.seed)
    # Match the serial formulation, but evaluate many paths per NumPy batch.
    drift = (config.r - 0.5 * config.sigma * config.sigma) * config.T
    diffusion = config.sigma * math.sqrt(config.T)

    payoff_sum = 0.0
    done = 0

    start = time.perf_counter()
    while done < config.n_trials:
        # Chunk the workload so very large trial counts do not exhaust memory.
        m = min(batch_size, config.n_trials - done)
        # array of m random variables
        z = rng.standard_normal(m)
        # array calc
        ST = config.S0 * np.exp(drift + diffusion * z)
        payoff_sum += np.maximum(ST - config.K, 0.0).sum() # sum max each element
        done += m
    elapsed = time.perf_counter() - start

    estimate = math.exp(-config.r * config.T) * (payoff_sum / config.n_trials)
    exact = black_scholes_call_price(config.S0, config.K, config.r, config.sigma, config.T)

    return BenchmarkResult(
        workload="european_call",
        mode="numpy",
        trials=config.n_trials,
        elapsed_sec=elapsed,
        throughput_trials_per_sec=config.n_trials / elapsed,
        estimate=estimate,
        extra={"black_scholes_exact": exact, "absolute_error": abs(estimate - exact)}
    )
