from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class BenchmarkResult:
    workload: str
    mode: str
    trials: int
    elapsed_sec: float
    throughput_trials_per_sec: float
    estimate: float
    extra: Optional[Dict] = None