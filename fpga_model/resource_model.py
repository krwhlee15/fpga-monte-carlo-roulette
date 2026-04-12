BOARD_LIMITS = {
    "lut": 33280,
    "dsp": 90,
    "bram_kbits": 1800,
    "ff": 66560,  # rough modeled budget
}


def estimate_resources(config):
    if config.workload == "roulette":
        lut_per_lane = 180
        dsp_per_lane = 0
        bram_per_lane = 4
        ff_per_lane = 120
    elif config.workload == "sine":
        lut_per_lane = 320
        dsp_per_lane = 2
        bram_per_lane = 6
        ff_per_lane = 180
    elif config.workload == "option":
        lut_per_lane = 650
        dsp_per_lane = 6
        bram_per_lane = 10
        ff_per_lane = 320
    else:
        raise ValueError(f"Unknown workload: {config.workload}")

    shared_lut = 300 + 80 * config.memory_bus_ports + 100 * config.reducer_throughput
    shared_dsp = 0
    shared_bram = 20 + max(config.output_buffer_size, 1)
    shared_ff = 200 + 40 * config.memory_bus_ports + 40 * config.reducer_throughput

    return {
        "lut": config.n_lanes * lut_per_lane + shared_lut,
        "dsp": config.n_lanes * dsp_per_lane + shared_dsp,
        "bram_kbits": config.n_lanes * bram_per_lane + shared_bram,
        "ff": config.n_lanes * ff_per_lane + shared_ff,
    }


def config_is_feasible(config):
    usage = estimate_resources(config)
    return (
        usage["lut"] <= BOARD_LIMITS["lut"]
        and usage["dsp"] <= BOARD_LIMITS["dsp"]
        and usage["bram_kbits"] <= BOARD_LIMITS["bram_kbits"]
        and usage["ff"] <= BOARD_LIMITS["ff"]
    )