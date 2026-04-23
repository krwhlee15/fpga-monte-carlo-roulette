from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    # Accumulates stage activity and stall information across the whole run.
    # Used for final evaluation metrics
    total_bus_wait_cycles: int = 0
    total_reducer_wait_cycles: int = 0
    total_buffer_wait_cycles: int = 0
    total_reseed_stall_cycles: int = 0

    stage_busy_cycles: dict = field(default_factory=lambda: {
        "rng": 0,
        "map": 0,
        "eval": 0,
        "update": 0,
    })
    stall_counters: dict = field(default_factory=lambda: {
        "bus": 0,
        "reducer": 0,
        "buffer": 0,
        "reseed": 0,
    })
    stall_intervals: list = field(default_factory=list)

    trial_latencies: list = field(default_factory=list)

    def record_stage(self, stage_name: str, cycles: int):
        self.stage_busy_cycles[stage_name] += cycles

    def record_latency(self, cycles: int):
        self.trial_latencies.append(cycles)

    def record_stall(self, stall_type: str, start_cycle: int, stall_cycles: int):
        if stall_cycles <= 0:
            return
        self.stall_counters[stall_type] += stall_cycles
        # Keep raw intervals so overlapping stalls can be merged later.
        self.stall_intervals.append((start_cycle, start_cycle + stall_cycles))

    def mean_latency(self):
        return sum(self.trial_latencies) / len(self.trial_latencies) if self.trial_latencies else 0.0

    def max_latency(self):
        return max(self.trial_latencies) if self.trial_latencies else 0

    def contention_rate(self, total_cycles: int):
        if total_cycles <= 0:
            return 0.0
        if not self.stall_intervals:
            return 0.0

        # Merge overlaps to avoid double-counting cycles where multiple resources stall at once.
        merged = []
        for start, end in sorted(self.stall_intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        stalled_cycles = sum(end - start for start, end in merged)
        return stalled_cycles / total_cycles

    def stage_utilization(self, total_cycles: int, n_lanes: int):
        if total_cycles <= 0 or n_lanes <= 0:
            return {stage: 0.0 for stage in self.stage_busy_cycles}
        denom = float(total_cycles * n_lanes)
        return {
            stage: busy_cycles / denom
            for stage, busy_cycles in self.stage_busy_cycles.items()
        }
