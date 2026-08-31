import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terminal_tensorboard.tfevents import (
    encode_scalar_event,
    frame_record,
    parse_chunk,
)


def record(tag, step, wall, value, tensor=False):
    return frame_record(encode_scalar_event(tag, step, wall, value, tensor=tensor))


class ParseChunkTest(unittest.TestCase):
    def test_simple_value_roundtrip(self):
        data = record("loss", 10, 123.5, 0.25) + record("acc", 11, 124.0, 0.75)
        points, consumed = parse_chunk(data)
        self.assertEqual(consumed, len(data))
        self.assertEqual(len(points), 2)
        tag, step, wall, val = points[0]
        self.assertEqual((tag, step, wall), ("loss", 10, 123.5))
        self.assertAlmostEqual(val, 0.25, places=6)
        self.assertEqual(points[1][0], "acc")
        self.assertAlmostEqual(points[1][3], 0.75, places=6)

    def test_tensor_scalar_roundtrip(self):
        data = record("loss", 5, 1.0, 3.14159, tensor=True)
        points, consumed = parse_chunk(data)
        self.assertEqual(consumed, len(data))
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0][3], 3.14159, places=4)

    def test_negative_and_large_values(self):
        data = record("neg", 1, 1.0, -123.456) + record("big", 2, 2.0, 1e30)
        points, _ = parse_chunk(data)
        self.assertAlmostEqual(points[0][3], -123.456, places=2)
        self.assertAlmostEqual(points[1][3] / 1e30, 1.0, places=3)

    def test_truncated_tail_is_left_unconsumed(self):
        full = record("loss", 1, 1.0, 0.5)
        partial = record("loss", 2, 2.0, 0.6)[:-7]  # simulate an in-progress write
        points, consumed = parse_chunk(full + partial)
        self.assertEqual(len(points), 1)
        self.assertEqual(consumed, len(full))
        # once the rest arrives, parsing resumes from that offset
        rest = record("loss", 2, 2.0, 0.6)
        points2, consumed2 = parse_chunk((full + rest)[consumed:])
        self.assertEqual(len(points2), 1)
        self.assertEqual(points2[0][1], 2)
        self.assertEqual(consumed2, len(rest))

    def test_event_without_summary_is_skipped(self):
        # a file_version-style event: wall_time only
        from terminal_tensorboard.tfevents import _F64, _field

        ev = _field(1, 1) + _F64.pack(42.0)
        data = frame_record(ev) + record("loss", 1, 1.0, 0.5)
        points, consumed = parse_chunk(data)
        self.assertEqual(len(points), 1)
        self.assertEqual(consumed, len(data))

    def test_tag_interning(self):
        data = record("loss", 1, 1.0, 0.1) + record("loss", 2, 2.0, 0.2)
        points, _ = parse_chunk(data)
        self.assertIs(points[0][0], points[1][0])


class StoreTest(unittest.TestCase):
    def test_incremental_refresh(self):
        import tempfile

        from terminal_tensorboard.store import ScalarStore

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run1")
            os.makedirs(run_dir)
            path = os.path.join(run_dir, "events.out.tfevents.123.host")
            with open(path, "wb") as f:
                f.write(record("loss", 1, 1.0, 0.9))

            store = ScalarStore(tmp)
            self.assertTrue(store.refresh())
            self.assertEqual(store.run_names(), ["run1"])
            series = store.runs["run1"].series["loss"]
            self.assertEqual(list(series.steps), [1])

            with open(path, "ab") as f:
                f.write(record("loss", 2, 2.0, 0.8))
            self.assertTrue(store.refresh())
            self.assertEqual(list(series.steps), [1, 2])
            self.assertFalse(store.refresh())  # nothing new

    def test_out_of_order_steps_get_sorted(self):
        from terminal_tensorboard.store import Series

        s = Series()
        for step, val in ((5, 0.5), (1, 0.1), (3, 0.3)):
            s.append(step, float(step), val)
        s.ensure_sorted()
        self.assertEqual(list(s.steps), [1, 3, 5])
        self.assertEqual(list(s.vals), [0.1, 0.3, 0.5])


class PlotTest(unittest.TestCase):
    def test_bucketize_means(self):
        from terminal_tensorboard.plot import bucketize

        xs = list(range(100))
        ys = [float(x) for x in xs]
        pts = bucketize(xs, ys, 0, 99, 10)
        self.assertEqual(len(pts), 10)
        cols = [c for c, _ in pts]
        self.assertEqual(cols, sorted(cols))
        # means increase monotonically for a monotonic series
        vals = [v for _, v in pts]
        self.assertEqual(vals, sorted(vals))

    def test_ema_smooth_debiased_start(self):
        from terminal_tensorboard.plot import ema_smooth

        pts = [(i, 1.0) for i in range(10)]
        out = ema_smooth(pts, 0.9)
        for _, v in out:
            self.assertAlmostEqual(v, 1.0, places=9)

    def test_canvas_segments(self):
        from terminal_tensorboard.plot import BrailleCanvas

        c = BrailleCanvas(10, 2)
        c.line(0, 0, 19, 7, 3)
        segments = [list(c.row_segments(r)) for r in range(2)]
        self.assertTrue(any(segments))
        for row in segments:
            for _, text, color in row:
                self.assertEqual(color, 3)
                self.assertTrue(all(0x2800 <= ord(ch) <= 0x28FF for ch in text))


if __name__ == "__main__":
    unittest.main()
