class BaseWorkload:
    name = "base"

    def init_lane_state(self, config):
        return {}

    def stage2_map(self, raw_rng: int, lane_state: dict, config):
        raise NotImplementedError

    def stage3_evaluate(self, mapped_value, lane_state: dict, config):
        raise NotImplementedError

    def stage4_update(self, eval_result, lane_state: dict, config):
        return eval_result, lane_state

    def eval_bus_accesses(self, config):
        return 1

    def update_bus_accesses(self, config):
        return 0

    def output_drain_cycles(self, config):
        return 1

    def reducer_service_cycles(self, config):
        return 1

    def reduce_payload(self, final_result, config):
        raise NotImplementedError

    def init_aggregates(self, config):
        raise NotImplementedError

    def update_aggregates(self, aggregates: dict, payload: dict, config):
        raise NotImplementedError

    def finalize_estimate(self, aggregates: dict, config):
        raise NotImplementedError

    def ground_truth(self, config):
        return None
