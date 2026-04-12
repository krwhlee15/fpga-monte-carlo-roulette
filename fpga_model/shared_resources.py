class SharedResourcePool:
    """
    Simple cycle-based shared resource with fixed capacity per cycle.
    We model each port/unit as a 'next free cycle' slot.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Resource capacity must be positive")
        self.capacity = capacity
        self.next_free = [0] * capacity
        self.total_requests = 0
        self.total_wait_cycles = 0
        self.total_busy_cycles = 0

    def acquire(self, request_cycle: int, service_cycles: int = 1):
        if service_cycles <= 0:
            raise ValueError("Service cycles must be positive")
        self.total_requests += 1
        idx = min(range(self.capacity), key=lambda i: max(self.next_free[i], request_cycle))
        start_cycle = max(self.next_free[idx], request_cycle)
        wait_cycles = start_cycle - request_cycle
        self.next_free[idx] = start_cycle + service_cycles
        self.total_wait_cycles += wait_cycles
        self.total_busy_cycles += service_cycles
        return start_cycle, wait_cycles

    def max_busy_cycle(self):
        return max(self.next_free) if self.next_free else 0

    def utilization(self, total_cycles: int):
        if total_cycles <= 0:
            return 0.0
        return self.total_busy_cycles / float(total_cycles * self.capacity)


class OutputBuffer:
    """
    Simple finite-depth output buffer model. If full, producer stalls until
    the oldest slot is effectively drained one cycle later.
    """

    def __init__(self, depth: int):
        if depth <= 0:
            raise ValueError("Output buffer depth must be positive")
        self.depth = depth
        self.release_cycles = []

    def push(self, request_cycle: int, drain_cycles: int = 1):
        if drain_cycles <= 0:
            raise ValueError("Drain cycles must be positive")
        # Drop any entries already released
        self.release_cycles = [c for c in self.release_cycles if c > request_cycle]

        if len(self.release_cycles) < self.depth:
            self.release_cycles.append(request_cycle + drain_cycles)
            return request_cycle, 0

        earliest = min(self.release_cycles)
        stall = max(0, earliest - request_cycle)
        self.release_cycles = [c for c in self.release_cycles if c > earliest]
        push_cycle = earliest
        self.release_cycles.append(push_cycle + drain_cycles)
        return push_cycle, stall
