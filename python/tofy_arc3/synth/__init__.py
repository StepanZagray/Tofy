"""Synthetic ARCEngine training-game generator (ADR 0005 §2.6).

Games are authored against ``arcengine.ARCBaseGame`` from a seeded parameter
record.  Every game carries a Hidden Rule and is emitted together with a twin
that shares the byte-identical level-0 initial frame but a different rule.
"""

from .public_games import GENERATED_GAME_ID_PREFIX, PUBLIC_GAME_IDS

__all__ = ["GENERATED_GAME_ID_PREFIX", "PUBLIC_GAME_IDS"]
