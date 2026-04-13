import unittest

from fpga_model.shared_resources import OutputBuffer, SharedResourcePool


class TestSharedResources(unittest.TestCase):
    def test_shared_resource_waits_when_capacity_is_busy(self):
        # A second request at the same cycle should queue behind the first one.
        pool = SharedResourcePool(capacity=1)

        start0, wait0 = pool.acquire(0, service_cycles=2)
        start1, wait1 = pool.acquire(0, service_cycles=1)

        self.assertEqual((start0, wait0), (0, 0))
        self.assertEqual((start1, wait1), (2, 2))
        self.assertEqual(pool.total_requests, 2)
        self.assertEqual(pool.total_wait_cycles, 2)

    def test_output_buffer_backpressures_when_full(self):
        # Depth-1 means the second push must wait for the first item to drain.
        buffer = OutputBuffer(depth=1)

        push0, stall0 = buffer.push(5, drain_cycles=2)
        push1, stall1 = buffer.push(5, drain_cycles=2)

        self.assertEqual((push0, stall0), (5, 0))
        self.assertEqual((push1, stall1), (7, 2))


if __name__ == "__main__":
    unittest.main()
