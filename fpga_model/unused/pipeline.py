# UNUSED
# --------------
from fpga_model.lfsr import map_outcome, evaluate_bet


def flat_strategy(win, current_bet, base_bet):
    """Flat betting: always bet the same amount."""
    return base_bet


def martingale_strategy(win, current_bet, base_bet):
    """Martingale: double after loss, reset to base after win."""
    if win:
        return base_bet
    return current_bet * 2


STRATEGIES = {
    "flat": flat_strategy,
    "martingale": martingale_strategy,
}


def lane_process(env, lane_id, n_trials, lfsr, memory_bus, reducer, config):
    """SimPy process for a single FPGA pipeline lane.

    Each trial goes through 4 pipeline stages, with contention on
    the shared memory bus and reducer. Returns per-trial latency data.
    """
    strategy_fn = STRATEGIES[config.strategy]
    current_bet = config.base_bet
    needs_bus_for_strategy = config.strategy != "flat"
    trial_latencies = []

    for trial_idx in range(n_trials):
        cycle_start = env.now

        # Stage 1: RNG (1 cycle)
        lfsr.step()
        raw_outcome = lfsr.get_outcome()
        yield env.timeout(1)

        # Stage 2: Outcome mapping (1 cycle)
        number, color = map_outcome(raw_outcome)
        yield env.timeout(1)

        # Stage 3: Bet evaluation — requires memory bus read
        before_wait = env.now
        req = memory_bus.request()
        yield req
        wait = env.now - before_wait
        memory_bus.record_wait(wait)
        win, payout = evaluate_bet(number, color, config.bet_type, current_bet)
        yield env.timeout(1)  # 1 cycle for evaluation
        memory_bus.release(req)

        # Stage 4: Strategy update
        if needs_bus_for_strategy:
            # Martingale needs bus access for read-modify-write
            before_wait = env.now
            req = memory_bus.request()
            yield req
            wait = env.now - before_wait
            memory_bus.record_wait(wait)
            current_bet = strategy_fn(win, current_bet, config.base_bet)
            yield env.timeout(1)
            memory_bus.release(req)
        else:
            # Flat betting: no bus needed, just 1 cycle
            current_bet = strategy_fn(win, current_bet, config.base_bet)
            yield env.timeout(1)

        # Deposit the completed trial into the shared reducer, which may become the bottleneck.
        before_wait = env.now
        req = reducer.request()
        yield req
        wait = env.now - before_wait
        reducer.record_wait(wait)
        reducer.record(win, payout, raw_outcome)
        yield env.timeout(1)  # 1 cycle for reduction
        reducer.release(req)

        # Periodic reseeding injects additional pipeline bubbles in this older SimPy model.
        if lfsr.steps_since_reseed >= config.lfsr_reseed_interval:
            lfsr.reseed(config.seed + lane_id + trial_idx)
            yield env.timeout(config.lfsr_reseed_latency)

        trial_latencies.append(env.now - cycle_start)

    return trial_latencies
