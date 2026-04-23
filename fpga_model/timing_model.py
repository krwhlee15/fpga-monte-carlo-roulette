def stage_cycle_cost(config):
    """
    Cycle cost model per stage.
    Keeps 4 logical stages fixed, but lets heavier arithmetic occupy more cycles.
    """
    # estimated clock cycles used for each pipeline stage per workload
    # more cycles for more complex computation
    # more cycles may seem worse but it also allows for higher clock speeds as less work is done per cycle
    if config.workload == "roulette":
        return {"rng": 1, "map": 1, "eval": 1, "update": 1}
    if config.workload == "sine":
        return {"rng": 1, "map": 1, "eval": 2, "update": 1}
    if config.workload == "option":
        return {"rng": 1, "map": 2, "eval": 4, "update": 1}
    raise ValueError(f"Unknown workload: {config.workload}")


def estimate_max_clock_mhz(config):
    """
    Conservative timing model aligned with report sensitivity study.
    Baseline experiments are at 100 MHz; higher clocks become harder as
    arithmetic intensity and structural complexity increase.
    """
    base = 250.0

    if config.workload == "roulette":
        workload_penalty = 0.0
    elif config.workload == "sine":
        workload_penalty = 20.0
    elif config.workload == "option":
        workload_penalty = 50.0
    else:
        raise ValueError(f"Unknown workload: {config.workload}")

    structural_penalty = 0.8 * max(config.n_lanes - 1, 0)
    structural_penalty += 3.0 * max(config.memory_bus_ports - 1, 0)
    structural_penalty += 2.5 * max(config.reducer_throughput - 1, 0)

    fmax = max(50.0, base - workload_penalty - structural_penalty)
    return fmax


def effective_clock_mhz(config):
    # The simulator cannot exceed the modeled timing limit for a design point.
    return min(config.clock_freq_mhz, estimate_max_clock_mhz(config))
