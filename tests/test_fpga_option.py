import unittest

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim


class TestFPGAOptionPricing(unittest.TestCase):
    def test_option_estimate_is_reasonable(self):
        # This is a sanity check, not a precision benchmark, so the error tolerance is loose.
        config = SimConfig(
            workload="option",
            n_trials=30000,
            n_lanes=8,
            memory_bus_ports=2,
            reducer_throughput=2,
        )

        result = run_fpga_sim(config)

        self.assertTrue(result["feasible"])
        self.assertGreater(result["ground_truth"], 0.0)
        self.assertLess(result["absolute_error"], 2.0)
        self.assertGreater(result["throughput"], 0.0)
        self.assertGreater(result["mean_latency_cycles"], 0.0)


if __name__ == "__main__":
    unittest.main()
