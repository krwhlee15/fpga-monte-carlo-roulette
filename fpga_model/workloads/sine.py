import math
from .base import BaseWorkload


class SineWorkload(BaseWorkload):
    name = "sine"

    def stage2_map(self, raw_rng: int, lane_state: dict, config):
        # Convert a 32-bit integer into a uniform sample over the integration interval.
        u = (raw_rng & 0xFFFFFFFF) / float(2**32)
        return config.sine_a + (config.sine_b - config.sine_a) * u

    def stage3_evaluate(self, x: float, lane_state: dict, config):
        return {"x": x, "fx": math.sin(x)}

    def eval_bus_accesses(self, config):
        return 1

    def reduce_payload(self, final_result, config):
        return {"sum_fx": final_result["fx"]}

    def init_aggregates(self, config):
        return {"sum_fx": 0.0, "count": 0}

    def update_aggregates(self, aggregates: dict, payload: dict, config):
        aggregates["sum_fx"] += payload["sum_fx"]
        aggregates["count"] += 1

    def finalize_estimate(self, aggregates: dict, config):
        if aggregates["count"] == 0:
            return 0.0
        # Monte Carlo integral = interval width times the average sampled function value.
        width = config.sine_b - config.sine_a
        return width * (aggregates["sum_fx"] / aggregates["count"])

    def ground_truth(self, config):
        return math.cos(config.sine_a) - math.cos(config.sine_b)
