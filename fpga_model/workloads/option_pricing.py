import math
from .base import BaseWorkload


class OptionPricingWorkload(BaseWorkload):
    name = "option"

    def init_lane_state(self, config):
        return {"box_muller_spare": None}

    @staticmethod
    def _uniform01(raw_rng: int) -> float:
        u = (raw_rng & 0xFFFFFFFF) / float(2**32)
        return min(max(u, 1e-12), 1.0 - 1e-12)

    def _next_standard_normal(self, raw_rng: int, lane_state: dict):
        if lane_state["box_muller_spare"] is not None:
            z = lane_state["box_muller_spare"]
            lane_state["box_muller_spare"] = None
            return z, lane_state

        u1 = self._uniform01(raw_rng)
        mixed = ((raw_rng * 1664525 + 1013904223) & 0xFFFFFFFF)
        u2 = self._uniform01(mixed)

        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2

        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)

        lane_state["box_muller_spare"] = z1
        return z0, lane_state

    def stage2_map(self, raw_rng: int, lane_state: dict, config):
        z, lane_state = self._next_standard_normal(raw_rng, lane_state)
        return {"z": z}

    def stage3_evaluate(self, mapped_value: dict, lane_state: dict, config):
        z = mapped_value["z"]
        drift = (config.r - 0.5 * config.sigma * config.sigma) * config.T
        diffusion = config.sigma * math.sqrt(config.T) * z
        ST = config.S0 * math.exp(drift + diffusion)
        payoff = max(ST - config.K, 0.0)
        discounted_payoff = math.exp(-config.r * config.T) * payoff

        return {
            "ST": ST,
            "payoff": payoff,
            "discounted_payoff": discounted_payoff,
        }

    def eval_bus_accesses(self, config):
        return 1

    def reduce_payload(self, final_result, config):
        return {"sum_discounted_payoff": final_result["discounted_payoff"]}

    def init_aggregates(self, config):
        return {"sum_discounted_payoff": 0.0, "count": 0}

    def update_aggregates(self, aggregates: dict, payload: dict, config):
        aggregates["sum_discounted_payoff"] += payload["sum_discounted_payoff"]
        aggregates["count"] += 1

    def finalize_estimate(self, aggregates: dict, config):
        return aggregates["sum_discounted_payoff"] / aggregates["count"] if aggregates["count"] else 0.0

    def ground_truth(self, config):
        d1 = (math.log(config.S0 / config.K) + (config.r + 0.5 * config.sigma**2) * config.T) / (
            config.sigma * math.sqrt(config.T)
        )
        d2 = d1 - config.sigma * math.sqrt(config.T)

        def normal_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        return config.S0 * normal_cdf(d1) - config.K * math.exp(-config.r * config.T) * normal_cdf(d2)
