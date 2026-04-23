import random
import time
import numpy as np
from .common import BenchmarkResult
from fpga_model.lfsr import RED_NUMBERS

def run_roulette_serial(config):
    rng = random.Random(config.seed)
    wins = 0
    total_payout = 0.0
    current_bet = config.base_bet

    start = time.perf_counter()
    for _ in range(config.n_trials):
        # Model American roulette as 38 equally likely pockets: 0, 00, and 1-36.
        outcome = rng.randrange(38)

        if config.bet_type == "red_black":
            # Reuse the same red-pocket set as the FPGA model so both paths agree on winners.
            win = outcome in RED_NUMBERS
            payout_win = current_bet
        else:
            win = (outcome == config.single_number_choice)
            payout_win = current_bet * 35

        if win:
            wins += 1
            total_payout += payout_win
            if config.strategy == "martingale":
                # Martingale resets after a win.
                current_bet = config.base_bet
        else:
            total_payout -= current_bet
            if config.strategy == "martingale":
                # Martingale doubles after a loss, creating stateful dependence across trials.
                current_bet *= 2

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        workload=f"roulette_{config.strategy}",
        mode="serial",
        trials=config.n_trials,
        elapsed_sec=elapsed,
        throughput_trials_per_sec=config.n_trials / elapsed,
        estimate=wins / config.n_trials,
        extra={"wins": wins, "total_payout": total_payout}
    )


# vectorized version
def run_roulette_numpy(config, batch_size=1_000_000):
    rng = np.random.default_rng(config.seed)
    start = time.perf_counter()

    # Martingale depends on the previous outcome, so it falls back to the serial loop.
    if config.strategy == "martingale":
        elapsed = time.perf_counter() - start
        result = run_roulette_serial(config)
        result.mode = "numpy_fallback"
        return result

    done = 0
    wins = 0
    total_payout = 0.0

    while done < config.n_trials:
        # Batch the random draws so large experiments do not allocate one giant array.
        m = min(batch_size, config.n_trials - done)
        # array or results
        outcomes = rng.integers(0, 38, size=m)

        if config.bet_type == "red_black":
            # Match the serial baseline and FPGA model's definition of a winning red pocket.
            win_mask = np.isin(outcomes, list(RED_NUMBERS))
            win_payout = config.base_bet
        else:
            win_mask = (outcomes == config.single_number_choice)
            win_payout = config.base_bet * 35

        batch_wins = int(win_mask.sum()) # sum array to get total number of hits
        wins += batch_wins
        total_payout += batch_wins * win_payout - (m - batch_wins) * config.base_bet
        done += m

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        workload=f"roulette_{config.strategy}",
        mode="numpy",
        trials=config.n_trials,
        elapsed_sec=elapsed,
        throughput_trials_per_sec=config.n_trials / elapsed,
        estimate=wins / config.n_trials,
        extra={"wins": wins, "total_payout": total_payout}
    )
