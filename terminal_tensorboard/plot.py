"""Braille-dot canvas and series preprocessing for terminal charts.

A character cell holds a 2x4 grid of braille dots, so a WxH cell canvas has
2W x 4H addressable pixels.  Rendering millions of points stays fast because
series are first reduced to one mean value per pixel column (``bucketize``),
which is O(points) in slicing/summing done in C, and everything after that
is O(columns).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Iterator, List, Sequence, Tuple

_BRAILLE_BASE = 0x2800
# dot bit for (y in 0..3, x in 0..1) inside one cell
_DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


class BrailleCanvas:
    __slots__ = ("w", "h", "px_w", "px_h", "cells", "colors")

    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        self.px_w = w * 2
        self.px_h = h * 4
        self.cells = bytearray(w * h)
        self.colors = bytearray(w * h)

    def dot(self, x: int, y: int, color: int) -> None:
        if 0 <= x < self.px_w and 0 <= y < self.px_h:
            idx = (y >> 2) * self.w + (x >> 1)
            self.cells[idx] |= _DOT_BITS[y & 3][x & 1]
            self.colors[idx] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.dot(x0, y0, color)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def row_segments(self, row: int) -> Iterator[Tuple[int, str, int]]:
        """Yield (start_col, text, color) runs of non-empty same-color cells."""
        cells = self.cells
        colors = self.colors
        base = row * self.w
        col = 0
        w = self.w
        while col < w:
            bits = cells[base + col]
            if not bits:
                col += 1
                continue
            color = colors[base + col]
            start = col
            chars = []
            while col < w and cells[base + col] and colors[base + col] == color:
                chars.append(chr(_BRAILLE_BASE | cells[base + col]))
                col += 1
            yield start, "".join(chars), color


# --------------------------------------------------------------------------
# series preprocessing
# --------------------------------------------------------------------------

def bucketize(
    xs: Sequence[float],
    ys: Sequence[float],
    x0: float,
    x1: float,
    ncols: int,
) -> List[Tuple[int, float]]:
    """Reduce a sorted series to at most one (column, mean) point per column."""
    n = len(xs)
    if n == 0 or ncols <= 0 or x1 < x0:
        return []
    lo = bisect_left(xs, x0)
    hi = bisect_right(xs, x1)
    if hi <= lo:
        return []
    span = x1 - x0
    if span <= 0:
        seg = ys[lo:hi]
        return [(ncols // 2, sum(seg) / len(seg))]
    out: List[Tuple[int, float]] = []
    prev = lo
    for col in range(ncols):
        edge = x0 + span * (col + 1) / ncols
        nxt = bisect_right(xs, edge, prev, hi)
        if nxt > prev:
            seg = ys[prev:nxt]
            out.append((col, sum(seg) / len(seg)))
            prev = nxt
    return out


def ema_smooth(
    points: List[Tuple[int, float]], weight: float
) -> List[Tuple[int, float]]:
    """Debiased exponential moving average (TensorBoard-style smoothing)."""
    if weight <= 0 or len(points) < 2:
        return points
    last = 0.0
    out: List[Tuple[int, float]] = []
    wpow = 1.0
    for col, v in points:
        last = last * weight + (1.0 - weight) * v
        wpow *= weight
        out.append((col, last / (1.0 - wpow)))
    return out


def nice_ticks(lo: float, hi: float, count: int) -> List[float]:
    """A few round tick values covering [lo, hi]."""
    if count < 2 or not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / (count - 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mag * mult
        if step >= raw:
            break
    first = math.ceil(lo / step) * step
    ticks = []
    t = first
    while t <= hi + step * 1e-9:
        ticks.append(0.0 if abs(t) < step * 1e-9 else t)
        t += step
    return ticks


def fmt_num(v: float) -> str:
    """Compact human formatting for axis labels and readouts."""
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    a = abs(v)
    if a != 0 and (a >= 1e5 or a < 1e-3):
        return "%.3g" % v
    if a >= 100:
        return "%.1f" % v
    return "%.4g" % v


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "%.0fs" % seconds
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return "%dm%02ds" % (m, s)
    h, m = divmod(m, 60)
    if h < 48:
        return "%dh%02dm" % (h, m)
    d, h = divmod(h, 24)
    return "%dd%02dh" % (d, h)


def fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return "%.1fM" % (n / 1_000_000)
    if n >= 10_000:
        return "%.0fk" % (n / 1_000)
    return str(n)
