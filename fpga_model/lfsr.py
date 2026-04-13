# American roulette layout
# 38 outcomes: 0 (green), 00 (green), 1-36 (red or black)

RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

# Outcome map: index 0-37 -> (display_number, color)
# Index 0 -> "0" green, index 37 -> "00" green, indices 1-36 -> their number
OUTCOME_MAP = {}
for i in range(38):
    if i == 0:
        OUTCOME_MAP[i] = (0, "green")
    elif i == 37:
        OUTCOME_MAP[i] = (37, "green")  # represents "00"
    elif i in RED_NUMBERS:
        OUTCOME_MAP[i] = (i, "red")
    else:
        assert i in BLACK_NUMBERS, f"Number {i} not in red or black"
        OUTCOME_MAP[i] = (i, "black")


def map_outcome(raw):
    """Map a raw LFSR output (0-37) to (number, color)."""
    return OUTCOME_MAP[raw]


def evaluate_bet(number, color, bet_type, bet_amount):
    """Evaluate a roulette bet. Returns (win: bool, payout: int).

    Payout is net: positive for wins, negative (lost bet) for losses.
    Red/black pays 1:1, single number pays 35:1.
    """
    if bet_type == "red_black":
        # The simplified even-money bet wins on the red pockets only.
        win = color == "red"
        payout = bet_amount if win else -bet_amount
    elif bet_type == "single_number":
        win = number == 17  # default single number bet
        payout = bet_amount * 35 if win else -bet_amount
    else:
        assert False, f"Unknown bet type: {bet_type}"
    return win, payout


# 32-bit Galois LFSR
# Polynomial: x^32 + x^31 + x^29 + x + 1 -> feedback taps 0x80200003
LFSR_POLY = 0x80200003
LFSR_MASK = 0xFFFFFFFF


class LFSR:
    def __init__(self, seed):
        assert seed != 0, "LFSR seed must be nonzero"
        # Mask to 32 bits so the software model matches the intended hardware width.
        self.state = seed & LFSR_MASK
        self.steps_since_reseed = 0

    def step(self):
        """Advance the LFSR by one clock cycle. Returns the new state."""
        # Galois form applies feedback only when the outgoing low bit is 1.
        lsb = self.state & 1
        self.state >>= 1
        if lsb:
            self.state ^= LFSR_POLY
        self.state &= LFSR_MASK
        self.steps_since_reseed += 1
        return self.state

    def get_outcome(self):
        """Map current state to a roulette outcome in [0, 37]."""
        # Modulo reduction is a simple stand-in for mapping raw RNG bits to table entries.
        return self.state % 38

    def reseed(self, xor_val):
        """XOR the state with a secondary value to refresh randomness."""
        self.state ^= (xor_val & LFSR_MASK)
        if self.state == 0:
            self.state = 1  # LFSR must never be zero
        self.steps_since_reseed = 0
