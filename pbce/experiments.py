"""Game definitions and settings for the five benchmarks in the PBCE paper."""

GAME_MAP = {
    "kuhn_3p": "kuhn_poker(players=3)",
    "leduc": "leduc_poker",
    "random_goofspiel": "goofspiel(imp_info=True,num_cards=4,points_order=random)",
    "sheriff": "sheriff",
    "tiny_bridge": "tiny_bridge_2p",
    # A smaller diagnostic instance used by the bounded executable smoke test.
    "kuhn_2p": "kuhn_poker(players=2)",
}

DEFAULT_GAMES = ("kuhn_3p", "leduc", "random_goofspiel", "sheriff", "tiny_bridge")
NUM_ITERATIONS_MAP = dict.fromkeys(GAME_MAP, 100_000)
FIXED_EPSILONS = ("1e-1", "5e-2", "1e-2", "5e-3", "1e-3", "0")

# Historical output labels are preserved so the bundled data stay readable.
# Paper names: ACT-IN, CSPS, CFPS, CF, and CF-IN, respectively.
# Keep this set in sync with the algorithm factory in src/bin/ltbr.h.
PBCE_ALGORITHMS = frozenset({"A-EFR_IN", "CSPS-EFR", "CFPS-EFR", "CFR", "CFR_IN"})
