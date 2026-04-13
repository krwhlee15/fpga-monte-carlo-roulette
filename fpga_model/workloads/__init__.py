from .roulette import RouletteWorkload
from .sine import SineWorkload
from .option_pricing import OptionPricingWorkload


def get_workload_model(config):
    # Instantiate the workload-specific stage logic requested by the config.
    if config.workload == "roulette":
        return RouletteWorkload()
    if config.workload == "sine":
        return SineWorkload()
    if config.workload == "option":
        return OptionPricingWorkload()
    raise ValueError(f"Unknown workload: {config.workload}")
