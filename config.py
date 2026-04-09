from dataclasses import dataclass


@dataclass
class SimConfig:
    # Trial count
    n_trials: int = 100_000

    # FPGA hardware parameters
    n_lanes: int = 8
    clock_freq_mhz: float = 200.0
    pipeline_depth: int = 4
    memory_bus_ports: int = 2
    reducer_throughput: int = 4
    output_buffer_size: int = 16

    # LFSR parameters
    lfsr_reseed_interval: int = 10_000
    lfsr_reseed_latency: int = 3

    # Roulette parameters
    bet_type: str = "red_black"  # "red_black" or "single_number"
    single_number_choice: int = 17  # used when bet_type == "single_number"

    # Betting strategy
    strategy: str = "flat"  # "flat" or "martingale"
    initial_bankroll: int = 1000
    base_bet: int = 10

    # Reproducibility
    seed: int = 42
