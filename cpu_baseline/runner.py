from .roulette import run_roulette_serial, run_roulette_numpy
from .sine import run_sine_serial, run_sine_numpy
from .option_pricing import run_option_serial, run_option_numpy

# used to run all serial and numpy functions
# acts as a wrapper
def run_cpu_serial(config):
    # Dispatch to the matching scalar Python baseline.
    if config.workload == "roulette":
        return run_roulette_serial(config)
    if config.workload == "sine":
        return run_sine_serial(config)
    if config.workload == "option":
        return run_option_serial(config)
    raise ValueError(f"Unknown workload: {config.workload}")

def run_cpu_numpy(config):
    # Dispatch to the vectorized baseline when a workload supports it.
    if config.workload == "roulette":
        return run_roulette_numpy(config)
    if config.workload == "sine":
        return run_sine_numpy(config)
    if config.workload == "option":
        return run_option_numpy(config)
    raise ValueError(f"Unknown workload: {config.workload}")
