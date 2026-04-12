import math
import unittest

from config import SimConfig
from fpga_model.fpga_sim import run_fpga_sim


class TestFPGARoulette(unittest.TestCase):
    def test_roulette_outputs_and_metrics(self):
        config = SimConfig(
            workload="roulette",
            n_trials=2000,
            n_lanes=4,
            memory_bus_ports=2,
            reducer_throughput=2,
            strategy="flat",
        )

        result = run_fpga_sim(config)

        self.assertTrue(result["feasible"])
        self.assertEqual(result["actual_trials"], config.n_trials)
        self.assertEqual(sum(result["outcome_histogram"]), config.n_trials)
        self.assertEqual(sum(len(v) for v in result["lane_latencies"].values()), config.n_trials)
        self.assertAlmostEqual(
            result["sim_seconds"],
            result["total_cycles"] / (result["clock_mhz"] * 1e6),
        )
        self.assertAlmostEqual(
            result["throughput"],
            config.n_trials / result["sim_seconds"],
        )
        self.assertGreaterEqual(result["win_rate"], 0.0)
        self.assertLessEqual(result["win_rate"], 1.0)

    def test_martingale_increases_bus_pressure(self):
        shared = {
            "workload": "roulette",
            "n_trials": 1500,
            "n_lanes": 8,
            "memory_bus_ports": 1,
            "reducer_throughput": 2,
        }

        flat = run_fpga_sim(SimConfig(strategy="flat", **shared))
        martingale = run_fpga_sim(SimConfig(strategy="martingale", **shared))

        self.assertGreater(martingale["bus_total_requests"], flat["bus_total_requests"])
        self.assertGreaterEqual(martingale["bus_total_wait"], flat["bus_total_wait"])


if __name__ == "__main__":
    unittest.main()
