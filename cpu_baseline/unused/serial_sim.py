# UNUSED
# --------------
import time
import random
import numpy as np

from fpga_model.lfsr import RED_NUMBERS, OUTCOME_MAP


def run_serial(config):
    """Serial CPU roulette simulation. One trial per iteration."""
    rng = random.Random(config.seed)

    wins = 0
    losses = 0
    total_payout = 0
    outcome_histogram = np.zeros(38, dtype=np.int64)
    bankroll = config.initial_bankroll
    current_bet = config.base_bet

    strategy = config.strategy
    bet_type = config.bet_type

    t_start = time.perf_counter()

    for _ in range(config.n_trials):
        # Sample one roulette pocket and update the running histogram.
        outcome = rng.randint(0, 37)
        outcome_histogram[outcome] += 1

        number, color = OUTCOME_MAP[outcome]

        # Resolve the configured bet against the sampled pocket.
        if bet_type == "red_black":
            win = color == "red"
            payout = current_bet if win else -current_bet
        elif bet_type == "single_number":
            win = number == config.single_number_choice
            payout = current_bet * 35 if win else -current_bet
        else:
            assert False, f"Unknown bet type: {bet_type}"

        if win:
            wins += 1
        else:
            losses += 1
        total_payout += payout
        bankroll += payout

        # Strategy update happens after the payout is known.
        if strategy == "martingale":
            current_bet = config.base_bet if win else current_bet * 2
        # flat: current_bet stays the same

    elapsed = time.perf_counter() - t_start
    throughput = config.n_trials / elapsed if elapsed > 0 else 0

    return {
        "throughput": throughput,
        "elapsed_sec": elapsed,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / config.n_trials if config.n_trials > 0 else 0,
        "total_payout": total_payout,
        "final_bankroll": bankroll,
        "outcome_histogram": outcome_histogram,
    }
