# SPDX-License-Identifier: MIT-0
"""Non-ARC fixtures for the lossless frame-memory seam."""

from __future__ import annotations

import random
import unittest

from tofy_arc3.frame_memory import (
    ActionFact,
    FrameLedger,
    FrameMemoryConfig,
    PixelChange,
    decode_frame,
    encode_frame,
    summarize_components,
    summarize_frame_change,
)


def blank(value: int = 0) -> list[list[int]]:
    return [[value for _ in range(64)] for _ in range(64)]


class FrameEncodingTests(unittest.TestCase):
    def test_round_trips_varied_generated_frames_without_discarding_bottom_row(
        self,
    ) -> None:
        uniform = blank(3)

        all_symbols = [[(x + y) % 16 for x in range(64)] for y in range(64)]

        repeated_patterns = []
        for band in range(8):
            row = [(x + band) % 16 for x in range(64)]
            repeated_patterns.extend([row.copy() for _ in range(8)])

        generator = random.Random(78231)
        noise = [[generator.randrange(16) for _ in range(64)] for _ in range(64)]

        bottom_row = blank(1)
        bottom_row[63] = list(range(16)) * 4

        cases = [uniform, all_symbols, repeated_patterns, noise, bottom_row]
        for case, frame in enumerate(cases):
            with self.subTest(case=case):
                encoded = encode_frame(frame)
                self.assertEqual(decode_frame(encoded), tuple(map(tuple, frame)))
                self.assertEqual(encode_frame(frame), encoded)
                self.assertTrue(
                    encoded.startswith(
                        "frame-v1 x=0 y=0 width=64 height=64 palette=0123456789ABCDEF"
                    )
                )

        self.assertIn("R64:3*64", encode_frame(uniform))
        self.assertTrue(
            all(
                len(row_record) == 64
                for row_record in encode_frame(noise).splitlines()[1:]
            )
        )
        self.assertIn("P8:", encode_frame(repeated_patterns))
        self.assertLess(len(encode_frame(uniform)), 64 * 64)

    def test_rejects_malformed_shapes_values_and_encodings(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 rows"):
            encode_frame(blank()[:-1])
        invalid = blank()
        invalid[63][63] = 16
        with self.assertRaisesRegex(ValueError, r"frame\[63\]\[63\]"):
            encode_frame(invalid)
        invalid[63][63] = True
        with self.assertRaisesRegex(TypeError, r"frame\[63\]\[63\]"):
            encode_frame(invalid)

        header = "frame-v1 x=0 y=0 width=64 height=64 palette=0123456789ABCDEF"
        with self.assertRaisesRegex(ValueError, "all 64 rows"):
            decode_frame(header)
        with self.assertRaisesRegex(ValueError, "spans"):
            decode_frame(header + "\nR65:0*64")
        with self.assertRaisesRegex(ValueError, "exactly 64"):
            decode_frame(header + "\nR64:0*63")


class ComponentSummaryTests(unittest.TestCase):
    def test_reports_geometry_and_explicit_truncation(self) -> None:
        frame = blank()
        for y in (1, 2):
            for x in (1, 2):
                frame[y][x] = 1
        frame[10][10] = 2
        frame[11][10] = 2
        frame[63] = [15] * 64

        result = summarize_components(frame, max_components=10)
        by_color = {component.color: component for component in result.components}
        self.assertEqual(result.total_components, 4)
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(by_color[1].area, 4)
        self.assertEqual(by_color[1].bbox, (1, 1, 2, 2))
        self.assertEqual(by_color[1].centroid, (1, 1))
        self.assertEqual(by_color[2].centroid, (10, 10))
        self.assertEqual(by_color[15].bbox, (0, 63, 63, 63))

        checkerboard = [[(x + y) % 2 for x in range(64)] for y in range(64)]
        truncated = summarize_components(checkerboard, max_components=3)
        self.assertEqual(len(truncated.components), 3)
        self.assertEqual(truncated.total_components, 4096)
        self.assertEqual(truncated.truncated_count, 4093)
        self.assertEqual(
            [component.bbox for component in truncated.components],
            [(0, 0, 0, 0), (1, 0, 1, 0), (2, 0, 2, 0)],
        )


class FrameChangeSummaryTests(unittest.TestCase):
    def test_reports_exact_changed_count_and_inclusive_bbox(self) -> None:
        before = blank()
        after = blank()
        after[3][9] = 4
        after[60][2] = 7
        after[12][31] = 4

        summary = summarize_frame_change(before, after)

        self.assertEqual(summary.changed_pixel_count, 3)
        self.assertEqual(summary.bbox, (2, 3, 31, 60))

    def test_reports_a_nullable_bbox_for_a_no_op(self) -> None:
        summary = summarize_frame_change(blank(5), blank(5))

        self.assertEqual(summary.changed_pixel_count, 0)
        self.assertIsNone(summary.bbox)


class FrameLedgerTests(unittest.TestCase):
    def test_action_facts_enforce_the_coordinate_protocol(self) -> None:
        self.assertEqual(ActionFact(action_id=0), ActionFact(action_id=0))
        self.assertEqual(
            ActionFact(action_id=6, x=0, y=63),
            ActionFact(action_id=6, x=0, y=63),
        )
        with self.assertRaisesRegex(ValueError, "action 6 requires"):
            ActionFact(action_id=6)
        with self.assertRaisesRegex(ValueError, "action 6 requires"):
            ActionFact(action_id=6, x=1)
        with self.assertRaisesRegex(ValueError, "only action 6"):
            ActionFact(action_id=5, x=1, y=2)
        with self.assertRaisesRegex(ValueError, "only action 6"):
            ActionFact(action_id=0, x=1, y=2)

    def test_queries_factual_frames_actions_and_lossless_diffs(self) -> None:
        first = blank()
        second = blank()
        second[0][2] = 7
        second[63][63] = 15
        ledger = FrameLedger(FrameMemoryConfig(max_entries=2, max_note_chars=12))

        self.assertEqual(ledger.append(first), 0)
        action = ActionFact(action_id=6, x=3, y=63)
        self.assertEqual(ledger.append(second, preceding_action=action), 1)
        self.assertEqual(ledger.frame(0), tuple(map(tuple, first)))
        self.assertEqual(ledger.entry(1).preceding_action, action)

        difference = ledger.diff(0, 1)
        self.assertEqual(difference.changed_count, 2)
        self.assertEqual(
            difference.changes,
            (
                PixelChange(x=2, y=0, before=0, after=7),
                PixelChange(x=63, y=63, before=0, after=15),
            ),
        )

    def test_bounds_entries_and_keeps_model_notes_out_of_facts(self) -> None:
        ledger = FrameLedger(FrameMemoryConfig(max_entries=1, max_note_chars=8))
        ledger.append(blank())
        fact_before_note = ledger.entry(0)
        ledger.set_model_note(0, "try edge")

        self.assertEqual(ledger.model_note(0), "try edge")
        self.assertEqual(ledger.entry(0), fact_before_note)
        self.assertFalse(hasattr(ledger.entry(0), "note"))
        with self.assertRaisesRegex(OverflowError, "full"):
            ledger.append(blank(1))
        with self.assertRaisesRegex(ValueError, "max_note_chars"):
            ledger.set_model_note(0, "nine chars")
        with self.assertRaisesRegex(IndexError, "out of range"):
            ledger.frame(1)
        with self.assertRaises(TypeError):
            ledger.frame(True)

        with self.assertRaisesRegex(ValueError, "max_entries"):
            FrameMemoryConfig(max_entries=4097)
        with self.assertRaisesRegex(ValueError, "max_note_chars"):
            FrameMemoryConfig(max_note_chars=1025)


if __name__ == "__main__":
    unittest.main()
