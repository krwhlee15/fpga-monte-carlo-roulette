import math
import random
import time
import numpy as np
from .common import BenchmarkResult

def run_sine_serial(config):
    rng = random.Random(config.seed)
    # Monte Carlo integration samples x uniformly over [a, b].
    width = config.sine_b - config.sine_a
    s = 0.0

    start = time.perf_counter()
    for _ in range(config.n_trials):
        u = rng.random()
        x = config.sine_a + width * u
        s += math.sin(x)
    elapsed = time.perf_counter() - start

    # Multiply the sample mean by the interval width to estimate the integral.
    estimate = width * (s / config.n_trials)
    exact = math.cos(config.sine_a) - math.cos(config.sine_b)

    return BenchmarkResult(
        workload="sine_integral",
        mode="serial",
        trials=config.n_trials,
        elapsed_sec=elapsed,
        throughput_trials_per_sec=config.n_trials / elapsed,
        estimate=estimate,
        extra={"exact_value": exact, "absolute_error": abs(estimate - exact)}
    )

# vectorzied version
def run_sine_numpy(config, batch_size=1_000_000):
    rng = np.random.default_rng(config.seed)
    width = config.sine_b - config.sine_a
    s = 0.0
    done = 0

    start = time.perf_counter()
    while done < config.n_trials:
        # Process the integral estimate in chunks to bound memory use.
        m = min(batch_size, config.n_trials - done)
        # array of size m
        u = rng.random(m)
        x = config.sine_a + width * u
        s += np.sin(x).sum()
        done += m
    elapsed = time.perf_counter() - start

    estimate = width * (s / config.n_trials)
    exact = math.cos(config.sine_a) - math.cos(config.sine_b)

    return BenchmarkResult(
        workload="sine_integral",
        mode="numpy",
        trials=config.n_trials,
        elapsed_sec=elapsed,
        throughput_trials_per_sec=config.n_trials / elapsed,
        estimate=estimate,
        extra={"exact_value": exact, "absolute_error": abs(estimate - exact)}
    )
