# UNUSED
# --------------
import simpy


class MemoryBus:
    """Shared memory bus modeled as a SimPy Resource with limited ports.

    Lanes must acquire the bus to read bet config or write strategy state.
    At high lane counts, this becomes a contention bottleneck.
    """

    def __init__(self, env, capacity):
        self.env = env
        self.resource = simpy.Resource(env, capacity=capacity)
        self.total_requests = 0
        self.total_wait_time = 0.0
        self._busy_time = 0.0

    def request(self):
        self.total_requests += 1
        return self.resource.request()

    def release(self, req):
        self.resource.release(req)

    def record_wait(self, wait_cycles):
        """Record how many cycles a lane waited for bus access."""
        self.total_wait_time += wait_cycles

    def utilization(self, current_time):
        if current_time == 0:
            return 0.0
        # Utilization approximated by: how often is the bus fully occupied?
        # We track this via queue length snapshots, but for simplicity
        # use: total_wait_time / (current_time * capacity) as a proxy
        capacity = self.resource.capacity
        return self.total_wait_time / (current_time * capacity)

    @property
    def queue_length(self):
        return len(self.resource.queue)
