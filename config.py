from dataclasses import dataclass
import math

@dataclass
class SimConfig:
    # Shared knobs used by both the CPU baselines and FPGA timing/resource models.
    # Common
    n_trials: int = 100_000
    seed: int = 42
    workload: str = "roulette"   # "roulette", "sine", "option"

    # FPGA hardware parameters
    n_lanes: int = 8
    clock_freq_mhz: float = 100.0
    pipeline_depth: int = 4
    memory_bus_ports: int = 2
    reducer_throughput: int = 4
    output_buffer_size: int = 16

    # LFSR parameters
    lfsr_reseed_interval: int = 10_000
    lfsr_reseed_latency: int = 3

    # Roulette parameters
    bet_type: str = "red_black"          # "red_black" or "single_number"
    single_number_choice: int = 17
    strategy: str = "flat"               # "flat" or "martingale"
    initial_bankroll: int = 1000
    base_bet: float = 10.0

    # Sine workload parameters
    sine_a: float = 0.0
    sine_b: float = math.pi

    # Option pricing parameters
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
