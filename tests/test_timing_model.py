import unittest

from config import SimConfig
from fpga_model.timing_model import effective_clock_mhz, estimate_max_clock_mhz, stage_cycle_cost


class TestTimingModel(unittest.TestCase):
    def test_stage_costs_reflect_workload_complexity(self):
        roulette = stage_cycle_cost(SimConfig(workload="roulette"))
        option = stage_cycle_cost(SimConfig(workload="option"))

        self.assertGreater(option["map"], roulette["map"])
        self.assertGreater(option["eval"], roulette["eval"])

    def test_effective_clock_is_capped_by_fmax(self):
        config = SimConfig(workload="option", n_lanes=32, clock_freq_mhz=250.0)

        fmax = estimate_max_clock_mhz(config)
        effective = effective_clock_mhz(config)

        self.assertLessEqual(effective, config.clock_freq_mhz)
        self.assertEqual(effective, fmax)


if __name__ == "__main__":
    unittest.main()
