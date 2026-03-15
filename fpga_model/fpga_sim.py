import simpy

from fpga_model.lfsr import LFSR
from fpga_model.memory_bus import MemoryBus
from fpga_model.reducer import Reducer
from fpga_model.pipeline import lane_process


def run_fpga_sim(config):
    """Run the full FPGA pipeline DES and return metrics."""
    env = simpy.Environment()

    memory_bus = MemoryBus(env, capacity=config.memory_bus_ports)
    reducer = Reducer(env, capacity=config.reducer_throughput)

    trials_per_lane = config.n_trials // config.n_lanes
    assert trials_per_lane > 0, "Not enough trials for the number of lanes"

    # Track per-lane latency data via a mutable container
    lane_latencies = {}

    def lane_wrapper(lane_id):
        """Wrapper to capture the return value from lane_process."""
        lfsr = LFSR(seed=config.seed + lane_id + 1)  # +1 to avoid seed=0
        latencies = yield from lane_process(
            env, lane_id, trials_per_lane, lfsr, memory_bus, reducer, config
        )
        lane_latencies[lane_id] = latencies

    for lane_id in range(config.n_lanes):
        env.process(lane_wrapper(lane_id))

    env.run()

    total_cycles = env.now
    modeled_time_sec = total_cycles / (config.clock_freq_mhz * 1e6)
    actual_trials = trials_per_lane * config.n_lanes
    throughput = actual_trials / modeled_time_sec if modeled_time_sec > 0 else 0

    return {
        "throughput": throughput,
        "total_cycles": total_cycles,
        "modeled_time_sec": modeled_time_sec,
        "actual_trials": actual_trials,
        "wins": reducer.wins,
        "losses": reducer.losses,
        "win_rate": reducer.win_rate,
        "total_payout": reducer.total_payout,
        "outcome_histogram": reducer.outcome_histogram.copy(),
        "bus_utilization": memory_bus.utilization(total_cycles),
        "bus_total_wait": memory_bus.total_wait_time,
        "bus_total_requests": memory_bus.total_requests,
        "reducer_utilization": reducer.utilization(total_cycles),
        "reducer_total_wait": reducer.total_wait_time,
        "reducer_total_requests": reducer.total_requests,
        "lane_latencies": lane_latencies,
        "n_lanes": config.n_lanes,
        "strategy": config.strategy,
    }
