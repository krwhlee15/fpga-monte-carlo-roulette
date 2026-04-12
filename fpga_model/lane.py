from fpga_model.lfsr import LFSR


class Lane:
    def __init__(self, lane_id: int, initial_state: dict, seed: int):
        self.lane_id = lane_id
        self.state = initial_state
        self.next_free_cycle = 0
        self.lfsr = LFSR(seed=seed)
        self.trial_latencies = []
