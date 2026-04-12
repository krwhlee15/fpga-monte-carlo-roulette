from fpga_model.workloads import get_workload_model
from fpga_model.shared_resources import OutputBuffer, SharedResourcePool
from fpga_model.metrics import MetricsCollector
from fpga_model.lane import Lane
from fpga_model.resource_model import config_is_feasible, estimate_resources
from fpga_model.timing_model import effective_clock_mhz, estimate_max_clock_mhz, stage_cycle_cost


def _apply_resource_accesses(resource, request_cycle, service_cycles, metrics, stall_type):
    cycle = request_cycle
    total_wait = 0
    for _ in range(service_cycles):
        start_cycle, wait_cycles = resource.acquire(cycle, service_cycles=1)
        total_wait += wait_cycles
        metrics.record_stall(stall_type, cycle, wait_cycles)
        cycle = start_cycle + 1
    return cycle, total_wait


def run_fpga_sim(config):
    workload = get_workload_model(config)
    resource_usage = estimate_resources(config)
    feasible = config_is_feasible(config)
    fmax_mhz = estimate_max_clock_mhz(config)
    clock_mhz = effective_clock_mhz(config)

    if not feasible:
        return {
            "workload": config.workload,
            "estimate": 0.0,
            "ground_truth": workload.ground_truth(config),
            "total_cycles": 0,
            "sim_seconds": 0.0,
            "modeled_time_sec": 0.0,
            "throughput": 0.0,
            "actual_trials": config.n_trials,
            "contention_rate": 0.0,
            "mean_latency_cycles": 0.0,
            "max_latency_cycles": 0.0,
            "trial_latencies": [],
            "lane_latencies": {},
            "stage_busy_cycles": {"rng": 0, "map": 0, "eval": 0, "update": 0},
            "stage_utilization": {"rng": 0.0, "map": 0.0, "eval": 0.0, "update": 0.0},
            "bus_wait_cycles": 0,
            "bus_total_wait": 0,
            "bus_total_requests": 0,
            "bus_utilization": 0.0,
            "reducer_wait_cycles": 0,
            "reducer_total_wait": 0,
            "reducer_total_requests": 0,
            "reducer_utilization": 0.0,
            "buffer_wait_cycles": 0,
            "reseed_stall_cycles": 0,
            "stall_counters": {"bus": 0, "reducer": 0, "buffer": 0, "reseed": 0},
            "feasible": False,
            "resource_usage": resource_usage,
            "fmax_mhz": fmax_mhz,
            "clock_mhz": clock_mhz,
            "requested_clock_mhz": config.clock_freq_mhz,
            "status": "infeasible_resources",
        }

    stage_cost = stage_cycle_cost(config)
    lanes = [
        Lane(
            lane_id=lane_id,
            initial_state=workload.init_lane_state(config),
            seed=config.seed + lane_id + 1,
        )
        for lane_id in range(config.n_lanes)
    ]
    bus = SharedResourcePool(config.memory_bus_ports)
    reducer = SharedResourcePool(config.reducer_throughput)
    output_buffer = OutputBuffer(config.output_buffer_size)
    metrics = MetricsCollector()
    aggregates = workload.init_aggregates(config)
    ordered_outcomes = []

    for trial_id in range(config.n_trials):
        lane = lanes[trial_id % config.n_lanes]
        cycle = lane.next_free_cycle
        start_cycle = cycle

        metrics.record_stage("rng", stage_cost["rng"])
        raw_rng = lane.lfsr.step()
        cycle += stage_cost["rng"]

        if (
            config.lfsr_reseed_interval > 0
            and lane.lfsr.steps_since_reseed >= config.lfsr_reseed_interval
        ):
            metrics.stall_counters["reseed"] += config.lfsr_reseed_latency
            metrics.total_reseed_stall_cycles += config.lfsr_reseed_latency
            cycle += config.lfsr_reseed_latency
            lane.lfsr.reseed(config.seed + lane.lane_id + trial_id + 1)

        metrics.record_stage("map", stage_cost["map"])
        mapped_value = workload.stage2_map(raw_rng, lane.state, config)
        cycle += stage_cost["map"]

        cycle, bus_wait = _apply_resource_accesses(
            bus,
            cycle,
            workload.eval_bus_accesses(config),
            metrics,
            "bus",
        )
        metrics.total_bus_wait_cycles += bus_wait
        metrics.record_stage("eval", stage_cost["eval"])
        eval_result = workload.stage3_evaluate(mapped_value, lane.state, config)
        cycle += stage_cost["eval"]

        cycle, bus_wait = _apply_resource_accesses(
            bus,
            cycle,
            workload.update_bus_accesses(config),
            metrics,
            "bus",
        )
        metrics.total_bus_wait_cycles += bus_wait
        metrics.record_stage("update", stage_cost["update"])
        final_result, lane.state = workload.stage4_update(eval_result, lane.state, config)
        cycle += stage_cost["update"]

        push_cycle, buffer_wait = output_buffer.push(
            cycle,
            drain_cycles=workload.output_drain_cycles(config),
        )
        metrics.total_buffer_wait_cycles += buffer_wait
        metrics.record_stall("buffer", cycle, buffer_wait)
        cycle = push_cycle

        payload = workload.reduce_payload(final_result, config)
        reducer_start, reducer_wait = reducer.acquire(
            cycle,
            service_cycles=workload.reducer_service_cycles(config),
        )
        metrics.total_reducer_wait_cycles += reducer_wait
        metrics.record_stall("reducer", cycle, reducer_wait)
        cycle = reducer_start + workload.reducer_service_cycles(config)
        workload.update_aggregates(aggregates, payload, config)

        if config.workload == "roulette":
            ordered_outcomes.append(final_result["outcome"])

        latency = cycle - start_cycle
        lane.trial_latencies.append(latency)
        lane.next_free_cycle = cycle
        metrics.record_latency(latency)

    total_cycles = max((lane.next_free_cycle for lane in lanes), default=0)
    sim_seconds = total_cycles / (clock_mhz * 1e6) if total_cycles > 0 else 0.0
    throughput = config.n_trials / sim_seconds if sim_seconds > 0 else 0.0
    estimate = workload.finalize_estimate(aggregates, config)
    ground_truth = workload.ground_truth(config)
    stage_utilization = metrics.stage_utilization(total_cycles, config.n_lanes)
    lane_latencies = {
        lane.lane_id: list(lane.trial_latencies)
        for lane in lanes
    }

    result = {
        "workload": config.workload,
        "estimate": estimate,
        "ground_truth": ground_truth,
        "total_cycles": total_cycles,
        "sim_seconds": sim_seconds,
        "modeled_time_sec": sim_seconds,
        "throughput": throughput,
        "actual_trials": config.n_trials,
        "contention_rate": metrics.contention_rate(total_cycles),
        "mean_latency_cycles": metrics.mean_latency(),
        "max_latency_cycles": metrics.max_latency(),
        "trial_latencies": list(metrics.trial_latencies),
        "lane_latencies": lane_latencies,
        "stage_busy_cycles": dict(metrics.stage_busy_cycles),
        "stage_utilization": stage_utilization,
        "bus_wait_cycles": metrics.total_bus_wait_cycles,
        "bus_total_wait": metrics.total_bus_wait_cycles,
        "bus_total_requests": bus.total_requests,
        "bus_utilization": bus.utilization(total_cycles),
        "reducer_wait_cycles": metrics.total_reducer_wait_cycles,
        "reducer_total_wait": metrics.total_reducer_wait_cycles,
        "reducer_total_requests": reducer.total_requests,
        "reducer_utilization": reducer.utilization(total_cycles),
        "buffer_wait_cycles": metrics.total_buffer_wait_cycles,
        "reseed_stall_cycles": metrics.total_reseed_stall_cycles,
        "stall_counters": dict(metrics.stall_counters),
        "feasible": True,
        "resource_usage": resource_usage,
        "fmax_mhz": fmax_mhz,
        "clock_mhz": clock_mhz,
        "requested_clock_mhz": config.clock_freq_mhz,
        "n_lanes": config.n_lanes,
        "strategy": config.strategy,
        "status": "ok",
    }

    if config.workload == "roulette":
        losses = aggregates["count"] - aggregates["wins"]
        result.update({
            "wins": aggregates["wins"],
            "losses": losses,
            "win_rate": estimate,
            "total_payout": aggregates["total_payout"],
            "outcome_histogram": list(aggregates["outcome_histogram"]),
            "ordered_outcomes": ordered_outcomes,
        })
    elif config.workload == "sine":
        result.update({
            "sum_fx": aggregates["sum_fx"],
            "absolute_error": abs(estimate - ground_truth) if ground_truth is not None else None,
        })
    elif config.workload == "option":
        result.update({
            "sum_discounted_payoff": aggregates["sum_discounted_payoff"],
            "absolute_error": abs(estimate - ground_truth) if ground_truth is not None else None,
        })

    return result
