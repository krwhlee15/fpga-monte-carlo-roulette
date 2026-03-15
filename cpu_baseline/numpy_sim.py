import time
import numpy as np

from fpga_model.lfsr import RED_NUMBERS


def run_numpy(config):
    """NumPy-vectorized CPU roulette simulation."""
    rng = np.random.default_rng(config.seed)

    t_start = time.perf_counter()

    # Generate all outcomes at once
    outcomes = rng.integers(0, 38, size=config.n_trials)
    outcome_histogram = np.bincount(outcomes, minlength=38)

    # Map outcomes to colors: 0 and 37 are green, rest are red/black
    red_mask = np.isin(outcomes, list(RED_NUMBERS))
    green_mask = (outcomes == 0) | (outcomes == 37)

    bet_type = config.bet_type

    if bet_type == "red_black":
        win_mask = red_mask
    elif bet_type == "single_number":
        win_mask = outcomes == config.single_number_choice
    else:
        assert False, f"Unknown bet type: {bet_type}"

    if config.strategy == "flat":
        # Fully vectorized for flat betting
        payouts = np.where(
            win_mask,
            config.base_bet if bet_type == "red_black" else config.base_bet * 35,
            -config.base_bet,
        )
        wins = int(win_mask.sum())
        losses = config.n_trials - wins
        total_payout = int(payouts.sum())
        final_bankroll = config.initial_bankroll + total_payout

    elif config.strategy == "martingale":
        # Must be sequential due to state dependency
        wins = 0
        losses = 0
        total_payout = 0
        current_bet = config.base_bet
        bankroll = config.initial_bankroll

        for i in range(config.n_trials):
            w = bool(win_mask[i])
            if w:
                if bet_type == "red_black":
                    payout = current_bet
                else:
                    payout = current_bet * 35
                wins += 1
                current_bet = config.base_bet
            else:
                payout = -current_bet
                losses += 1
                current_bet = current_bet * 2

            total_payout += payout
            bankroll += payout

        final_bankroll = bankroll
    else:
        assert False, f"Unknown strategy: {config.strategy}"

    elapsed = time.perf_counter() - t_start
    throughput = config.n_trials / elapsed if elapsed > 0 else 0

    return {
        "throughput": throughput,
        "elapsed_sec": elapsed,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / config.n_trials if config.n_trials > 0 else 0,
        "total_payout": total_payout,
        "final_bankroll": final_bankroll,
        "outcome_histogram": outcome_histogram,
    }
