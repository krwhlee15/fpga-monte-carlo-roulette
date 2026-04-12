import unittest

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim


class TestFPGASine(unittest.TestCase):
    def test_sine_estimate_tracks_ground_truth(self):
        config = SimConfig(
            workload="sine",
            n_trials=20000,
            n_lanes=8,
            memory_bus_ports=2,
            reducer_throughput=2,
        )

        result = run_fpga_sim(config)

        self.assertTrue(result["feasible"])
        self.assertEqual(result["ground_truth"], 2.0)
        self.assertLess(result["absolute_error"], 0.15)
        self.assertGreater(result["throughput"], 0.0)
        self.assertGreater(result["total_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
