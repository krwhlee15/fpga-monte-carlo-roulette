import csv
import os
import tempfile
import unittest

from evaluation.benchmark import run_benchmark


class TestBenchmarkOutputs(unittest.TestCase):
    def test_benchmark_writes_fpga_and_cpu_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results, cpu_results = run_benchmark(
                lane_counts=[1, 2],
                bus_ports_list=[1],
                reducer_throughputs=[1],
                strategies=["flat", "martingale"],
                n_trials=200,
                workload="roulette",
                output_dir=tmpdir,
            )

            csv_path = os.path.join(tmpdir, "benchmark_results.csv")
            self.assertEqual(len(results), 4)
            self.assertTrue(os.path.exists(csv_path))
            self.assertIn("flat", cpu_results)
            self.assertIn("martingale", cpu_results)

            with open(csv_path, "r", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []

            self.assertIn("fpga_total_cycles", fieldnames)
            self.assertIn("fpga_effective_clock_mhz", fieldnames)
            self.assertIn("speedup_vs_serial", fieldnames)
            self.assertIn("contention_rate", fieldnames)


if __name__ == "__main__":
    unittest.main()
