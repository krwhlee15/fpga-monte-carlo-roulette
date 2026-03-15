import simpy
import numpy as np


class Reducer:
    """Reduction tree aggregator modeled as a SimPy Resource.

    All lanes deposit results here after completing a trial.
    Limited throughput means lanes stall when the reducer is saturated.
    """

    def __init__(self, env, capacity):
        self.env = env
        self.resource = simpy.Resource(env, capacity=capacity)
        self.wins = 0
        self.losses = 0
        self.total_payout = 0
        self.outcome_histogram = np.zeros(38, dtype=np.int64)
        self.total_requests = 0
        self.total_wait_time = 0.0

    def request(self):
        self.total_requests += 1
        return self.resource.request()

    def release(self, req):
        self.resource.release(req)

    def record(self, win, payout, outcome):
        """Record one trial result."""
        if win:
            self.wins += 1
        else:
            self.losses += 1
        self.total_payout += payout
        self.outcome_histogram[outcome] += 1

    def record_wait(self, wait_cycles):
        self.total_wait_time += wait_cycles

    def utilization(self, current_time):
        if current_time == 0:
            return 0.0
        capacity = self.resource.capacity
        return self.total_wait_time / (current_time * capacity)

    @property
    def win_rate(self):
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def queue_length(self):
        return len(self.resource.queue)
