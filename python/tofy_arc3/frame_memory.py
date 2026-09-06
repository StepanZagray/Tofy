# SPDX-License-Identifier: MIT-0
"""Lossless visible-frame encoding and bounded factual observation memory.

The API handles one caller-selected 64x64 animation frame at a time. Palette
values are categorical symbols; this module assigns no color or object meaning.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

FRAME_SIZE = 64
PALETTE = "0123456789ABCDEF"
_HEADER = f"frame-v1 x=0 y=0 width={FRAME_SIZE} height={FRAME_SIZE} palette={PALETTE}"
_SPAN_LINE = re.compile(r"\A([PR])([1-9][0-9]*):([0-9A-F*,]+)\Z")
_RUN = re.compile(r"\A([0-9A-F])(?:\*([1-9][0-9]*))?\Z")
_MAX_ENCODING_CHARS = 8192
_HARD_MAX_ENTRIES = 4096
_HARD_MAX_NOTE_CHARS = 1024
_HARD_MAX_COMPONENTS = FRAME_SIZE * FRAME_SIZE

Frame = tuple[tuple[int, ...], ...]


def _bounded_integer(value: object, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in 0..{maximum}")
    return value


def _freeze_frame(frame: Sequence[Sequence[int]]) -> Frame:
    if isinstance(frame, (str, bytes)) or len(frame) != FRAME_SIZE:
        raise ValueError(f"frame must contain exactly {FRAME_SIZE} rows")
    rows: list[tuple[int, ...]] = []
    for y, row in enumerate(frame):
        if isinstance(row, (str, bytes)) or len(row) != FRAME_SIZE:
            raise ValueError(f"frame row {y} must contain exactly {FRAME_SIZE} cells")
        values: list[int] = []
        for x, value in enumerate(row):
            values.append(_bounded_integer(value, f"frame[{y}][{x}]", maximum=15))
        rows.append(tuple(values))
    return tuple(rows)


def _encode_row(row: tuple[int, ...], span: int) -> str:
    raw = "".join(PALETTE[value] for value in row)
    runs: list[str] = []
    start = 0
    while start < FRAME_SIZE:
        end = start + 1
        while end < FRAME_SIZE and row[end] == row[start]:
            end += 1
        count = end - start
        symbol = PALETTE[row[start]]
        runs.append(symbol if count == 1 else f"{symbol}*{count}")
        start = end
    rle = ",".join(runs)
    raw_record = raw if span == 1 else f"P{span}:{raw}"
    rle_record = f"R{span}:{rle}"
    return rle_record if len(rle_record) < len(raw_record) else raw_record


def encode_frame(frame: Sequence[Sequence[int]]) -> str:
    """Encode every cell of one 64x64 categorical frame deterministically."""

    rows = _freeze_frame(frame)
    lines = [_HEADER]
    y = 0
    while y < FRAME_SIZE:
        end = y + 1
        while end < FRAME_SIZE and rows[end] == rows[y]:
            end += 1
        lines.append(_encode_row(rows[y], end - y))
        y = end
    return "\n".join(lines)


def decode_frame(encoded: str) -> Frame:
    """Decode and validate a frame produced by :func:`encode_frame`."""

    if not isinstance(encoded, str):
        raise TypeError("encoded frame must be a string")
    if len(encoded) > _MAX_ENCODING_CHARS:
        raise ValueError("encoded frame exceeds the bounded format size")
    lines = encoded.splitlines()
    if not lines or lines[0] != _HEADER:
        raise ValueError("encoded frame header is invalid")

    rows: list[tuple[int, ...]] = []
    expected_y = 0
    for line in lines[1:]:
        if len(line) == FRAME_SIZE and all(symbol in PALETTE for symbol in line):
            mode, span, payload = "P", 1, line
        else:
            match = _SPAN_LINE.fullmatch(line)
            if match is None:
                raise ValueError("encoded frame row is invalid")
            mode, span, payload = match.group(1), int(match.group(2)), match.group(3)
        if expected_y + span > FRAME_SIZE:
            raise ValueError("encoded frame row spans are invalid")

        if mode == "P":
            if len(payload) != FRAME_SIZE or any(
                symbol not in PALETTE for symbol in payload
            ):
                raise ValueError("raw row must contain exactly 64 palette symbols")
            row = tuple(PALETTE.index(symbol) for symbol in payload)
        else:
            values: list[int] = []
            for token in payload.split(","):
                run = _RUN.fullmatch(token)
                if run is None:
                    raise ValueError("RLE row contains an invalid run")
                count = int(run.group(2) or "1")
                if len(values) + count > FRAME_SIZE:
                    raise ValueError("RLE row exceeds 64 cells")
                values.extend([PALETTE.index(run.group(1))] * count)
            if len(values) != FRAME_SIZE:
                raise ValueError("RLE row must decode to exactly 64 cells")
            row = tuple(values)

        rows.extend([row] * span)
        expected_y += span

    if expected_y != FRAME_SIZE:
        raise ValueError("encoded frame does not contain all 64 rows")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    """Geometry of one four-connected same-palette component.

    ``bbox`` is inclusive ``(x_min, y_min, x_max, y_max)``. ``centroid`` is
    the coordinate-wise arithmetic mean rounded down to an integer.
    """

    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ComponentSummaries:
    components: tuple[ComponentSummary, ...]
    total_components: int
    truncated_count: int


def summarize_components(
    frame: Sequence[Sequence[int]], *, max_components: int = 64
) -> ComponentSummaries:
    """Summarize four-connected same-color regions in row-major order."""

    rows = _freeze_frame(frame)
    limit = _bounded_integer(
        max_components, "max_components", maximum=_HARD_MAX_COMPONENTS
    )
    visited = [[False] * FRAME_SIZE for _ in range(FRAME_SIZE)]
    summaries: list[ComponentSummary] = []
    total = 0

    for start_y in range(FRAME_SIZE):
        for start_x in range(FRAME_SIZE):
            if visited[start_y][start_x]:
                continue
            total += 1
            color = rows[start_y][start_x]
            pending = deque([(start_x, start_y)])
            visited[start_y][start_x] = True
            area = sum_x = sum_y = 0
            min_x = max_x = start_x
            min_y = max_y = start_y

            while pending:
                x, y = pending.popleft()
                area += 1
                sum_x += x
                sum_y += y
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                for neighbor_x, neighbor_y in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if not (
                        0 <= neighbor_x < FRAME_SIZE and 0 <= neighbor_y < FRAME_SIZE
                    ):
                        continue
                    if visited[neighbor_y][neighbor_x]:
                        continue
                    if rows[neighbor_y][neighbor_x] != color:
                        continue
                    visited[neighbor_y][neighbor_x] = True
                    pending.append((neighbor_x, neighbor_y))

            if len(summaries) < limit:
                summaries.append(
                    ComponentSummary(
                        color=color,
                        area=area,
                        bbox=(min_x, min_y, max_x, max_y),
                        centroid=(sum_x // area, sum_y // area),
                    )
                )

    return ComponentSummaries(
        components=tuple(summaries),
        total_components=total,
        truncated_count=total - len(summaries),
    )


@dataclass(frozen=True, slots=True)
class ActionFact:
    """An observed ARC interface action, with no inferred game semantics."""

    action_id: int
    x: int | None = None
    y: int | None = None

    def __post_init__(self) -> None:
        _bounded_integer(self.action_id, "action_id", maximum=7)
        if self.action_id == 6:
            if self.x is None or self.y is None:
                raise ValueError("action 6 requires x and y coordinates")
            _bounded_integer(self.x, "action x", maximum=FRAME_SIZE - 1)
            _bounded_integer(self.y, "action y", maximum=FRAME_SIZE - 1)
        elif self.x is not None or self.y is not None:
            raise ValueError("only action 6 accepts coordinates")


@dataclass(frozen=True, slots=True)
class FactualEntry:
    index: int
    encoded_frame: str
    preceding_action: ActionFact | None


@dataclass(frozen=True, slots=True)
class PixelChange:
    x: int
    y: int
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class FrameDiff:
    before_index: int
    after_index: int
    changes: tuple[PixelChange, ...]

    @property
    def changed_count(self) -> int:
        return len(self.changes)


@dataclass(frozen=True, slots=True)
class FrameMemoryConfig:
    max_entries: int = 128
    max_note_chars: int = 256

    def __post_init__(self) -> None:
        entries = _bounded_integer(
            self.max_entries, "max_entries", maximum=_HARD_MAX_ENTRIES
        )
        if entries == 0:
            raise ValueError("max_entries must be positive")
        _bounded_integer(
            self.max_note_chars,
            "max_note_chars",
            maximum=_HARD_MAX_NOTE_CHARS,
        )


class FrameLedger:
    """Bounded per-game memory of factual frames/actions and separate notes.

    Callers create a fresh instance for each game. An entry's action is the
    factual action observed immediately before that entry's frame. The first
    entry therefore normally has no preceding action.
    """

    def __init__(self, config: FrameMemoryConfig | None = None) -> None:
        if config is not None and not isinstance(config, FrameMemoryConfig):
            raise TypeError("config must be a FrameMemoryConfig or None")
        self.config = config or FrameMemoryConfig()
        self._entries: list[FactualEntry] = []
        self._model_notes: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def append(
        self,
        frame: Sequence[Sequence[int]],
        *,
        preceding_action: ActionFact | None = None,
    ) -> int:
        """Append a fact or raise ``OverflowError`` at the configured bound."""

        if len(self._entries) >= self.config.max_entries:
            raise OverflowError("frame ledger is full")
        if preceding_action is not None and not isinstance(
            preceding_action, ActionFact
        ):
            raise TypeError("preceding_action must be an ActionFact or None")
        index = len(self._entries)
        self._entries.append(
            FactualEntry(
                index=index,
                encoded_frame=encode_frame(frame),
                preceding_action=preceding_action,
            )
        )
        return index

    def _entry_index(self, index: object) -> int:
        value = _bounded_integer(index, "index", maximum=_HARD_MAX_ENTRIES - 1)
        if value >= len(self._entries):
            raise IndexError("frame ledger index is out of range")
        return value

    def entry(self, index: int) -> FactualEntry:
        return self._entries[self._entry_index(index)]

    def frame(self, index: int) -> Frame:
        return decode_frame(self.entry(index).encoded_frame)

    def diff(self, before_index: int, after_index: int) -> FrameDiff:
        before_slot = self._entry_index(before_index)
        after_slot = self._entry_index(after_index)
        before = decode_frame(self._entries[before_slot].encoded_frame)
        after = decode_frame(self._entries[after_slot].encoded_frame)
        changes = tuple(
            PixelChange(x=x, y=y, before=before[y][x], after=after[y][x])
            for y in range(FRAME_SIZE)
            for x in range(FRAME_SIZE)
            if before[y][x] != after[y][x]
        )
        return FrameDiff(
            before_index=before_slot,
            after_index=after_slot,
            changes=changes,
        )

    def set_model_note(self, index: int, note: str) -> None:
        slot = self._entry_index(index)
        if not isinstance(note, str):
            raise TypeError("model note must be a string")
        if len(note) > self.config.max_note_chars:
            raise ValueError("model note exceeds max_note_chars")
        self._model_notes[slot] = note

    def model_note(self, index: int) -> str | None:
        return self._model_notes.get(self._entry_index(index))
