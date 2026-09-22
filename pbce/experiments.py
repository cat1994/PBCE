"""Game definitions and defaults used by the PBCE experiment launcher."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GAME_MAP = {
    "sheriff": "sheriff",
    "leduc": "leduc_poker",
    # This bundled Leduc version has fixed raise sizes (2, 4), not a raise_sizes option.
    "leduc_3p": "leduc_poker(players=3)",
    "goofspiel": "goofspiel(imp_info=True,num_cards=5,points_order=descending)",
    "goofspiel_ascending": "goofspiel(imp_info=True,num_cards=5,points_order=ascending)",
    "random_goofspiel": "goofspiel(imp_info=True,num_cards=4,points_order=random)",
    "goofspiel_3p": (
        "goofspiel(imp_info=True,num_cards=4,points_order=descending,players=3)"
    ),
    "goofspiel_ascending_3p": (
        "goofspiel(imp_info=True,num_cards=4,points_order=ascending,players=3)"
    ),
    "tiny_bridge": "tiny_bridge_2p",
    "kuhn_2p": "kuhn_poker(players=2)",
    "kuhn_3p": "kuhn_poker(players=3)",
    "kuhn_4p": "kuhn_poker(players=4)",
    "tiny_hanabi": "tiny_hanabi",
    "shapley": str(PROJECT_ROOT / "games/efg/extended_shapleys.efg"),
    "test_game": str(PROJECT_ROOT / "games/efg/test_game.efg"),
    "battleship": (
        "battleship(board_width=3,board_height=2,ship_sizes=[1],"
        "ship_values=[1.0],num_shots=2,allow_repeated_shots=False,"
        "loss_multiplier=2.0)"
    ),
}

NUM_ITERATIONS_MAP = dict.fromkeys(GAME_MAP, 100_000)
DEFAULT_GAMES = ("kuhn_3p",)
FIXED_EPSILONS = ("1e-1", "5e-2", "1e-2", "5e-3", "1e-3", "0")

# Keep this set in sync with the PBCE algorithm factory in bin/ltbr.h.
PBCE_ALGORITHMS = frozenset({"A-EFR_IN", "CSPS-EFR", "CFPS-EFR", "CFR", "CFR_IN"})
