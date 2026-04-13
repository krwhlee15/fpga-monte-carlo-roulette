from .base import BaseWorkload


class RouletteWorkload(BaseWorkload):
    name = "roulette"

    def init_lane_state(self, config):
        # Track per-lane betting state so Martingale can evolve independently by lane.
        return {"current_bet": config.base_bet}

    def stage2_map(self, raw_rng: int, lane_state: dict, config):
        # Compress the raw RNG output into one roulette pocket.
        return raw_rng % 38

    def stage3_evaluate(self, outcome: int, lane_state: dict, config):
        current_bet = lane_state["current_bet"]

        if config.bet_type == "red_black":
            win = outcome < 18
            payout = current_bet if win else -current_bet
        elif config.bet_type == "single_number":
            win = (outcome == config.single_number_choice)
            payout = current_bet * 35 if win else -current_bet
        else:
            raise ValueError(f"Unknown bet type: {config.bet_type}")

        return {
            "outcome": outcome,
            "win": win,
            "payout": payout,
        }

    def stage4_update(self, eval_result, lane_state: dict, config):
        if config.strategy == "martingale":
            # Strategy state changes are modeled after the bet outcome is known.
            if eval_result["win"]:
                lane_state["current_bet"] = config.base_bet
            else:
                lane_state["current_bet"] *= 2
        return eval_result, lane_state

    def eval_bus_accesses(self, config):
        return 1

    def update_bus_accesses(self, config):
        return 1 if config.strategy == "martingale" else 0

    def reduce_payload(self, final_result, config):
        return {
            "wins": 1 if final_result["win"] else 0,
            "total_payout": final_result["payout"],
            "outcome": final_result["outcome"],
        }

    def init_aggregates(self, config):
        return {
            "wins": 0,
            "total_payout": 0.0,
            "count": 0,
            "outcome_histogram": [0] * 38,
        }

    def update_aggregates(self, aggregates: dict, payload: dict, config):
        aggregates["wins"] += payload["wins"]
        aggregates["total_payout"] += payload["total_payout"]
        aggregates["count"] += 1
        aggregates["outcome_histogram"][payload["outcome"]] += 1

    def finalize_estimate(self, aggregates: dict, config):
        return aggregates["wins"] / aggregates["count"] if aggregates["count"] else 0.0

    def ground_truth(self, config):
        if config.bet_type == "red_black":
            return 18 / 38
        if config.bet_type == "single_number":
            return 1 / 38
        return None
