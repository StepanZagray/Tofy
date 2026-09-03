"""Public ARC-AGI-3 game ids that generated games must never collide with.

Source: ``runs/p2/_pod_handoffs/6zp5oip7tvokfl-20260827-foundation-v2/
foundation-v2/arc3_live_report.json`` (``discovered_games[].game_id``).  The
list is copied here so the generator never depends on run artifacts.  The
generator only attests exclusion of ids; it does not read, adapt or imitate
the public games' code or layouts.
"""

from __future__ import annotations

PUBLIC_GAME_IDS: tuple[str, ...] = (
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
)

# Public ids use a 4-character prefix followed by "-" and 8 hex digits.  Our
# prefix is 4 characters as well ("tsyn") but is not used by any public game,
# so no generated id can collide with a public one.
GENERATED_GAME_ID_PREFIX = "tsyn"

PUBLIC_GAME_PREFIXES: frozenset[str] = frozenset(g.split("-", 1)[0] for g in PUBLIC_GAME_IDS)
assert GENERATED_GAME_ID_PREFIX not in PUBLIC_GAME_PREFIXES


def assert_not_public(game_ids: list[str] | tuple[str, ...]) -> None:
    """Fail closed if any generated id intersects the public list."""
    clash = sorted(set(game_ids) & set(PUBLIC_GAME_IDS))
    if clash:
        raise ValueError(f"generated game ids collide with public games: {clash}")
    for gid in game_ids:
        if not gid.startswith(GENERATED_GAME_ID_PREFIX + "-"):
            raise ValueError(f"generated game id {gid!r} lacks prefix {GENERATED_GAME_ID_PREFIX!r}")
