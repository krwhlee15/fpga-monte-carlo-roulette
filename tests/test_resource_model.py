import unittest

from config import SimConfig
from fpga_model.resource_model import config_is_feasible, estimate_resources


class TestResourceModel(unittest.TestCase):
    def test_resource_usage_grows_with_heavier_workloads(self):
        roulette = estimate_resources(SimConfig(workload="roulette", n_lanes=8))
        option = estimate_resources(SimConfig(workload="option", n_lanes=8))

        self.assertGreater(option["lut"], roulette["lut"])
        self.assertGreater(option["dsp"], roulette["dsp"])
        self.assertGreater(option["ff"], roulette["ff"])

    def test_large_option_config_can_become_infeasible(self):
        infeasible_cfg = SimConfig(
            workload="option",
            n_lanes=64,
            memory_bus_ports=4,
            reducer_throughput=8,
        )

        self.assertFalse(config_is_feasible(infeasible_cfg))


if __name__ == "__main__":
    unittest.main()
