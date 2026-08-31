"""The curses TUI: layout, input handling and chart rendering."""

from __future__ import annotations

import curses
import threading
import time
from bisect import bisect_right
from typing import List, Optional, Tuple

from .plot import (
    BrailleCanvas,
    bucketize,
    ema_smooth,
    fmt_count,
    fmt_duration,
    fmt_num,
    nice_ticks,
)
from .store import ScalarStore, Series

XMODES = ("step", "reltime", "wall")
_ZOOM_FACTOR = 0.7
_PAN_FRACTION = 0.15
_MIN_SPAN = 1e-4

FOCUS_RUNS, FOCUS_TAGS, FOCUS_CHART = 0, 1, 2

HELP_LINES = [
    ("Tab / Shift-Tab", "cycle focus: runs -> tags -> chart"),
    ("j k / arrows", "move in lists; prev/next tag in chart"),
    ("Space", "toggle run on/off        a: toggle all runs"),
    ("Enter", "open tag in chart"),
    ("/", "filter tags (Enter apply, Esc cancel)"),
    ("h l / arrows", "move data cursor (chart focus), c/Esc clear"),
    ("+ - [ ] 0", "zoom in/out, pan left/right, reset view"),
    ("s / S", "less / more smoothing"),
    ("L", "toggle log-scale Y"),
    ("x", "cycle X axis: step -> reltime -> wall"),
    ("g", "toggle grid view (up to 4 charts)"),
    ("f", "toggle live follow    r: reload now"),
    ("b", "toggle sidebar"),
    ("q", "quit"),
]


class Loader(threading.Thread):
    """Background thread that keeps the store fresh without blocking the UI."""

    def __init__(self, store: ScalarStore, interval: float, app: "App") -> None:
        super().__init__(daemon=True, name="ttb-loader")
        self.store = store
        self.interval = interval
        self.app = app
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.busy = False

    def run(self) -> None:
        first = True
        while not self.stopping.is_set():
            manual = self.wake.is_set()
            self.wake.clear()
            if first or manual or self.app.follow:
                self.busy = True
                try:
                    if self.store.refresh():
                        self.app.data_event.set()
                finally:
                    self.busy = False
                    if first:
                        self.app.loaded = True
                        self.app.data_event.set()
                first = False
            self.wake.wait(self.interval)

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()


class App:
    def __init__(
        self,
        store: ScalarStore,
        refresh_interval: float = 2.0,
        follow: bool = True,
        smoothing: float = 0.6,
        xmode: str = "step",
    ) -> None:
        self.store = store
        self.follow = follow
        self.smoothing = max(0.0, min(0.99, smoothing))
        self.xmode = xmode if xmode in XMODES else "step"
        self.loader = Loader(store, refresh_interval, self)

        self.loaded = False
        self.data_event = threading.Event()
        self.focus = FOCUS_TAGS
        self.run_sel = 0
        self.tag_sel = 0
        self.run_scroll = 0
        self.tag_scroll = 0
        self.disabled: set = set()  # run names switched off by the user
        self.filter_text = ""
        self.filter_editing = False
        self._filter_backup = ""
        self.grid = False
        self.log_y = False
        self.sidebar = True
        self.view = (0.0, 1.0)  # fraction of full X domain
        self.cursor: Optional[float] = None  # fraction within the view
        self.help_visible = False
        self.flash_msg = ""
        self.flash_until = 0.0
        self.n_colors = 6

    # -- misc helpers ------------------------------------------------------

    def flash(self, msg: str) -> None:
        self.flash_msg = msg
        self.flash_until = time.monotonic() + 2.5

    def _color_pair(self, run_index: int) -> int:
        return 1 + (run_index % self.n_colors)

    def _snapshot(self):
        """Grab consistent lists of runs/tags under the store lock."""
        with self.store.lock:
            run_names = self.store.run_names()
            enabled = {n for n in run_names if n not in self.disabled}
            tags = self.store.tags(enabled)
            total = self.store.total_points
        if self.filter_text:
            needle = self.filter_text.lower()
            tags = [t for t in tags if needle in t.lower()]
        self.run_sel = max(0, min(self.run_sel, len(run_names) - 1))
        self.tag_sel = max(0, min(self.tag_sel, len(tags) - 1))
        return run_names, tags, total

    # ======================================================================
    # input
    # ======================================================================

    def handle_key(self, key: int, run_names: List[str], tags: List[str]) -> bool:
        """Returns False when the app should quit."""
        if self.help_visible:
            self.help_visible = False
            return True
        if self.filter_editing:
            self._handle_filter_key(key)
            return True

        if key in (ord("q"), ord("Q")):
            return False
        if key == ord("?"):
            self.help_visible = True
        elif key == ord("\t"):
            self.focus = (self.focus + 1) % 3
        elif key == curses.KEY_BTAB:
            self.focus = (self.focus - 1) % 3
        elif key == ord("b"):
            self.sidebar = not self.sidebar
        elif key == ord("g"):
            self.grid = not self.grid
        elif key == ord("f"):
            self.follow = not self.follow
            self.flash("follow %s" % ("on" if self.follow else "off"))
            if self.follow:
                self.loader.wake.set()
        elif key == ord("r"):
            self.loader.wake.set()
            self.flash("reloading…")
        elif key == ord("L"):
            self.log_y = not self.log_y
        elif key == ord("x"):
            self.xmode = XMODES[(XMODES.index(self.xmode) + 1) % len(XMODES)]
            self.view = (0.0, 1.0)
            self.cursor = None
        elif key == ord("s"):
            self.smoothing = round(max(0.0, self.smoothing - 0.05), 2)
        elif key == ord("S"):
            self.smoothing = round(min(0.99, self.smoothing + 0.05), 2)
        elif key == ord("a"):
            if self.disabled:
                self.disabled.clear()
            else:
                self.disabled = set(run_names)
        elif key in (ord("+"), ord("=")):
            self._zoom(_ZOOM_FACTOR)
        elif key in (ord("-"), ord("_")):
            self._zoom(1.0 / _ZOOM_FACTOR)
        elif key == ord("["):
            self._pan(-_PAN_FRACTION)
        elif key == ord("]"):
            self._pan(_PAN_FRACTION)
        elif key == ord("0"):
            self.view = (0.0, 1.0)
            self.cursor = None
        elif key == ord("/"):
            self.filter_editing = True
            self._filter_backup = self.filter_text
            self.focus = FOCUS_TAGS
        elif self.focus == FOCUS_RUNS:
            self._handle_list_key(key, run_names, is_runs=True)
        elif self.focus == FOCUS_TAGS:
            self._handle_list_key(key, tags, is_runs=False)
        else:
            self._handle_chart_key(key, tags)
        return True

    def _handle_filter_key(self, key: int) -> None:
        if key in (ord("\n"), curses.KEY_ENTER):
            self.filter_editing = False
            self.tag_sel = 0
        elif key == 27:  # Esc cancels
            self.filter_text = self._filter_backup
            self.filter_editing = False
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self.filter_text = self.filter_text[:-1]
        elif 32 <= key < 127:
            self.filter_text += chr(key)
            self.tag_sel = 0

    def _handle_list_key(self, key: int, items: List[str], is_runs: bool) -> None:
        if not items:
            return
        if key in (ord("j"), curses.KEY_DOWN):
            if is_runs:
                self.run_sel = min(self.run_sel + 1, len(items) - 1)
            else:
                self.tag_sel = min(self.tag_sel + 1, len(items) - 1)
        elif key in (ord("k"), curses.KEY_UP):
            if is_runs:
                self.run_sel = max(self.run_sel - 1, 0)
            else:
                self.tag_sel = max(self.tag_sel - 1, 0)
        elif key == curses.KEY_HOME:
            if is_runs:
                self.run_sel = 0
            else:
                self.tag_sel = 0
        elif key == curses.KEY_END:
            if is_runs:
                self.run_sel = len(items) - 1
            else:
                self.tag_sel = len(items) - 1
        elif is_runs and key in (ord(" "), ord("\n"), curses.KEY_ENTER):
            name = items[self.run_sel]
            if name in self.disabled:
                self.disabled.discard(name)
            else:
                self.disabled.add(name)
        elif not is_runs and key in (ord("\n"), curses.KEY_ENTER):
            self.grid = False
            self.focus = FOCUS_CHART

    def _handle_chart_key(self, key: int, tags: List[str]) -> None:
        if key in (ord("j"), curses.KEY_DOWN):
            if tags:
                self.tag_sel = min(self.tag_sel + 1, len(tags) - 1)
        elif key in (ord("k"), curses.KEY_UP):
            self.tag_sel = max(self.tag_sel - 1, 0)
        elif key in (ord("h"), curses.KEY_LEFT):
            self._move_cursor(-1)
        elif key in (ord("l"), curses.KEY_RIGHT):
            self._move_cursor(+1)
        elif key in (ord("c"), 27):
            self.cursor = None

    def _move_cursor(self, direction: int) -> None:
        step = 0.01
        if self.cursor is None:
            self.cursor = 0.5
        else:
            self.cursor = max(0.0, min(1.0, self.cursor + direction * step * 2))

    def _zoom(self, factor: float) -> None:
        lo, hi = self.view
        span = hi - lo
        center = lo + span * (self.cursor if self.cursor is not None else 0.5)
        new_span = max(_MIN_SPAN, min(1.0, span * factor))
        lo = center - new_span * ((center - lo) / span if span else 0.5)
        lo = max(0.0, min(lo, 1.0 - new_span))
        self.view = (lo, lo + new_span)

    def _pan(self, fraction: float) -> None:
        lo, hi = self.view
        span = hi - lo
        delta = span * fraction
        lo = max(0.0, min(lo + delta, 1.0 - span))
        self.view = (lo, lo + span)

    # ======================================================================
    # drawing
    # ======================================================================

    @staticmethod
    def _put(scr, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            scr.addstr(y, x, text, attr)
        except curses.error:
            pass  # writes clipped at the screen edge

    def draw(self, scr) -> None:
        scr.erase()
        rows, cols = scr.getmaxyx()
        run_names, tags, total = self._snapshot()

        if rows < 8 or cols < 40:
            self._put(scr, 0, 0, "terminal too small")
            scr.noutrefresh()
            curses.doupdate()
            return

        self._draw_header(scr, cols, run_names, tags, total)
        self._draw_footer(scr, rows, cols)

        body_y, body_h = 1, rows - 2
        chart_x = 0
        if self.sidebar:
            side_w = max(24, min(34, cols // 3))
            self._draw_sidebar(scr, body_y, 0, body_h, side_w, run_names, tags)
            chart_x = side_w
        chart_w = cols - chart_x
        self._draw_charts(scr, body_y, chart_x, body_h, chart_w, run_names, tags)

        if self.help_visible:
            self._draw_help(scr, rows, cols)
        scr.noutrefresh()
        curses.doupdate()

    def _draw_header(self, scr, cols, run_names, tags, total) -> None:
        enabled = len(run_names) - len(self.disabled & set(run_names))
        left = " ttb  %s " % self.store.logdir
        state = "live %s" % ("●" if self.loader.busy else "○") if self.follow else "paused"
        right = " runs %d/%d │ tags %d │ pts %s │ %s │ x:%s │ y:%s │ smooth %.2f " % (
            enabled,
            len(run_names),
            len(tags),
            fmt_count(total),
            state,
            self.xmode,
            "log" if self.log_y else "lin",
            self.smoothing,
        )
        bar = left + " " * max(1, cols - len(left) - len(right)) + right
        self._put(scr, 0, 0, bar[: cols - 1].ljust(cols - 1), curses.A_REVERSE)

    def _draw_footer(self, scr, rows, cols) -> None:
        if self.filter_editing:
            text = " /%s▏  (Enter apply · Esc cancel)" % self.filter_text
        elif time.monotonic() < self.flash_until:
            text = " " + self.flash_msg
        else:
            text = (
                "  Tab:focus  Space:run on/off  Enter:open  /:filter  s/S:smooth"
                "  +/-/[/]:zoom·pan  g:grid  L:log  x:axis  f:follow  ?:help  q:quit"
            )
        self._put(scr, rows - 1, 0, text[: cols - 1].ljust(cols - 1), curses.A_DIM)

    # -- sidebar -----------------------------------------------------------

    def _draw_sidebar(self, scr, y0, x0, h, w, run_names, tags) -> None:
        runs_h = max(3, min(len(run_names) + 1, h // 3))
        tags_h = h - runs_h
        self._draw_run_list(scr, y0, x0, runs_h, w, run_names)
        self._draw_tag_list(scr, y0 + runs_h, x0, tags_h, w, tags)
        for y in range(y0, y0 + h):
            self._put(scr, y, x0 + w - 1, "│", curses.A_DIM)

    def _draw_run_list(self, scr, y0, x0, h, w, run_names) -> None:
        focused = self.focus == FOCUS_RUNS
        title = " RUNS "
        self._put(scr, y0, x0, title, curses.A_BOLD | (curses.A_REVERSE if focused else 0))
        visible = h - 1
        self.run_scroll = _scroll(self.run_scroll, self.run_sel, visible)
        for i in range(visible):
            idx = self.run_scroll + i
            if idx >= len(run_names):
                break
            name = run_names[idx]
            on = name not in self.disabled
            attr = curses.A_REVERSE if (focused and idx == self.run_sel) else 0
            mark = "▣" if on else "☐"
            pair = curses.color_pair(self._color_pair(idx))
            line = " %s " % mark
            self._put(scr, y0 + 1 + i, x0, line, attr)
            self._put(scr, y0 + 1 + i, x0 + 3, "●", pair | curses.A_BOLD)
            label = " " + name
            self._put(
                scr,
                y0 + 1 + i,
                x0 + 4,
                label[: w - 6].ljust(w - 6),
                attr | (0 if on else curses.A_DIM),
            )

    def _draw_tag_list(self, scr, y0, x0, h, w, tags) -> None:
        focused = self.focus == FOCUS_TAGS
        title = " TAGS (%d)" % len(tags)
        if self.filter_text or self.filter_editing:
            title += "  /" + self.filter_text
        self._put(
            scr, y0, x0, title[: w - 2], curses.A_BOLD | (curses.A_REVERSE if focused else 0)
        )
        visible = h - 1
        self.tag_scroll = _scroll(self.tag_scroll, self.tag_sel, visible)
        for i in range(visible):
            idx = self.tag_scroll + i
            if idx >= len(tags):
                break
            attr = curses.A_REVERSE if (focused and idx == self.tag_sel) else 0
            marker = "▶" if idx == self.tag_sel else " "
            line = " %s %s" % (marker, tags[idx])
            self._put(scr, y0 + 1 + i, x0, line[: w - 2].ljust(w - 2), attr)

    # -- charts ------------------------------------------------------------

    def _draw_charts(self, scr, y0, x0, h, w, run_names, tags) -> None:
        if not self.loaded:
            self._center_msg(scr, y0, x0, h, w, "loading event files…")
            return
        if not tags:
            msg = "no scalars found" if not self.filter_text else "no tags match filter"
            self._center_msg(scr, y0, x0, h, w, msg)
            return
        if self.grid:
            shown = tags[self.tag_sel : self.tag_sel + 4]
            ncols = 2 if (w >= 100 and len(shown) > 1) else 1
            nrows = (len(shown) + ncols - 1) // ncols
            cell_h = h // nrows
            cell_w = w // ncols
            for i, tag in enumerate(shown):
                r, c = divmod(i, ncols)
                self._draw_chart(
                    scr,
                    y0 + r * cell_h,
                    x0 + c * cell_w,
                    cell_h,
                    cell_w,
                    tag,
                    run_names,
                    detailed=False,
                    highlight=(i == 0),
                )
        else:
            self._draw_chart(
                scr, y0, x0, h, w, tags[self.tag_sel], run_names, detailed=True, highlight=True
            )

    def _center_msg(self, scr, y0, x0, h, w, msg) -> None:
        self._put(scr, y0 + h // 2, x0 + max(0, (w - len(msg)) // 2), msg, curses.A_DIM)

    def _series_for_tag(self, tag: str, run_names: List[str]):
        out = []
        with self.store.lock:
            for idx, name in enumerate(run_names):
                if name in self.disabled:
                    continue
                run = self.store.runs.get(name)
                if run is None:
                    continue
                s = run.series.get(tag)
                if s is not None and len(s) > 0:
                    out.append((name, self._color_pair(idx), s, run.first_wall))
        return out

    def _xs_offset(self, s: Series, first_wall) -> Tuple:
        if self.xmode == "step":
            return s.steps, 0.0
        if self.xmode == "reltime":
            base = first_wall if first_wall is not None else (s.walls[0] if s.walls else 0.0)
            return s.walls, base
        return s.walls, 0.0

    def _draw_chart(
        self, scr, y0, x0, h, w, tag, run_names, detailed: bool, highlight: bool
    ) -> None:
        gutter = 9
        plot_h = h - 2  # title row + x-label row
        plot_w = w - gutter - 2
        if plot_h < 3 or plot_w < 10:
            return

        title_attr = curses.A_BOLD if highlight else curses.A_BOLD | curses.A_DIM
        marker = "▶ " if (highlight and self.grid) else ""
        self._put(scr, y0, x0 + gutter + 1, (marker + tag)[: plot_w], title_attr)

        series = self._series_for_tag(tag, run_names)
        if not series:
            self._center_msg(scr, y0 + 1, x0, plot_h, w, "no data in enabled runs")
            return

        # full X domain across the drawn series
        xmin = float("inf")
        xmax = float("-inf")
        for _, _, s, fw in series:
            xs, off = self._xs_offset(s, fw)
            if len(xs):
                xmin = min(xmin, xs[0] - off)
                xmax = max(xmax, xs[-1] - off)
        if xmin > xmax:
            return
        span = max(xmax - xmin, 1e-12)
        lo = xmin + span * self.view[0]
        hi = xmin + span * self.view[1]
        if hi <= lo:
            hi = lo + 1e-12

        canvas = BrailleCanvas(plot_w, plot_h)
        drawn = []  # (name, color, points_display, s, off)
        vmin = float("inf")
        vmax = float("-inf")
        for name, color, s, fw in series:
            xs, off = self._xs_offset(s, fw)
            pts = bucketize(xs, s.vals, lo + off, hi + off, canvas.px_w)
            pts = ema_smooth(pts, self.smoothing)
            if self.log_y:
                import math

                pts = [(c, math.log10(v)) for c, v in pts if v > 0]
            if not pts:
                continue
            for _, v in pts:
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
            drawn.append((name, color, pts, s, off))
        if not drawn or vmin > vmax:
            self._center_msg(scr, y0 + 1, x0, plot_h, w, "no drawable points")
            return
        if vmax - vmin < 1e-12:
            pad = abs(vmax) * 0.1 or 1.0
            vmin -= pad
            vmax += pad
        else:
            pad = (vmax - vmin) * 0.05
            vmin -= pad
            vmax += pad

        vspan = vmax - vmin
        py_max = canvas.px_h - 1

        def to_py(v: float) -> int:
            return py_max - int(round((v - vmin) / vspan * py_max))

        for name, color, pts, _, _ in drawn:
            prev = None
            for col, v in pts:
                py = to_py(v)
                if prev is not None:
                    canvas.line(prev[0], prev[1], col, py, color)
                else:
                    canvas.dot(col, py, color)
                prev = (col, py)

        # Y axis: labels + vertical rule
        axis_x = x0 + gutter
        for row in range(plot_h):
            self._put(scr, y0 + 1 + row, axis_x, "│", curses.A_DIM)
        for tick in nice_ticks(vmin, vmax, max(3, plot_h // 4)):
            row = to_py(tick) // 4
            if 0 <= row < plot_h:
                label = fmt_num(10 ** tick if self.log_y else tick)
                self._put(scr, y0 + 1 + row, x0, label[:gutter].rjust(gutter), curses.A_DIM)
                self._put(scr, y0 + 1 + row, axis_x, "┼", curses.A_DIM)

        # canvas cells
        for row in range(plot_h):
            for start, text, color in canvas.row_segments(row):
                self._put(
                    scr,
                    y0 + 1 + row,
                    axis_x + 1 + start,
                    text,
                    curses.color_pair(color) | curses.A_BOLD,
                )

        # X labels
        xl_y = y0 + 1 + plot_h
        self._put(scr, xl_y, axis_x, "└", curses.A_DIM)
        nticks = max(2, plot_w // 22)
        for i in range(nticks + 1):
            frac = i / nticks
            xv = lo + (hi - lo) * frac
            label = self._fmt_x(xv)
            px = axis_x + 1 + int(frac * (plot_w - 1)) - (len(label) if i == nticks else 0)
            px = max(axis_x + 1, min(px, x0 + w - len(label) - 1))
            self._put(scr, xl_y, px, label, curses.A_DIM)

        # cursor + legend
        cursor_x = None
        if detailed and self.cursor is not None:
            col = int(self.cursor * (canvas.px_w - 1))
            cell = col // 2
            cursor_x = lo + (hi - lo) * (col / max(1, canvas.px_w - 1))
            for row in range(plot_h):
                if canvas.cells[row * canvas.w + cell] == 0:
                    self._put(scr, y0 + 1 + row, axis_x + 1 + cell, "┊", curses.A_DIM)
        if detailed:
            self._draw_legend(scr, y0 + 1, x0 + gutter + 1, plot_w, drawn, cursor_x)

    def _fmt_x(self, xv: float) -> str:
        if self.xmode == "step":
            return fmt_count(int(round(xv)))
        if self.xmode == "reltime":
            return fmt_duration(xv)
        return time.strftime("%H:%M:%S", time.localtime(xv))

    def _draw_legend(self, scr, y0, x0, w, drawn, cursor_x: Optional[float]) -> None:
        for i, (name, color, _, s, off) in enumerate(drawn[:8]):
            text = name
            if cursor_x is not None:
                # nearest raw point at or before the cursor
                xs = s.steps if self.xmode == "step" else s.walls
                idx = bisect_right(xs, cursor_x + off) - 1
                if 0 <= idx < len(s.vals):
                    text = "%s  %s @ %s" % (
                        name,
                        fmt_num(s.vals[idx]),
                        fmt_count(s.steps[idx]),
                    )
            line = "● " + text
            self._put(scr, y0 + i, x0 + max(0, w - len(line) - 1), "●", curses.color_pair(color) | curses.A_BOLD)
            self._put(scr, y0 + i, x0 + max(0, w - len(line) - 1) + 2, text[: w - 3])

    def _draw_help(self, scr, rows, cols) -> None:
        box_w = min(64, cols - 4)
        box_h = min(len(HELP_LINES) + 4, rows - 2)
        top = (rows - box_h) // 2
        left = (cols - box_w) // 2
        for y in range(box_h):
            self._put(scr, top + y, left, " " * box_w, curses.A_REVERSE)
        self._put(scr, top + 1, left + 2, "terminal-tensorboard — keys", curses.A_REVERSE | curses.A_BOLD)
        for i, (keys, desc) in enumerate(HELP_LINES[: box_h - 4]):
            self._put(scr, top + 3 + i, left + 2, keys.ljust(18), curses.A_REVERSE | curses.A_BOLD)
            self._put(scr, top + 3 + i, left + 20, desc[: box_w - 22], curses.A_REVERSE)

    # ======================================================================
    # main loop
    # ======================================================================

    def run(self, scr) -> None:
        curses.curs_set(0)
        scr.timeout(150)
        scr.keypad(True)
        self.n_colors = _init_colors()
        self.loader.start()

        dirty = True
        last_version = -1
        try:
            while True:
                if self.data_event.is_set():
                    self.data_event.clear()
                    dirty = True
                with self.store.lock:
                    v = self.store.version
                if v != last_version:
                    last_version = v
                    dirty = True
                if dirty:
                    self.draw(scr)
                    dirty = False
                key = scr.getch()
                if key == -1:
                    continue
                if key == curses.KEY_RESIZE:
                    dirty = True
                    continue
                run_names, tags, _ = self._snapshot()
                if not self.handle_key(key, run_names, tags):
                    break
                dirty = True
        finally:
            self.loader.stop()


def _scroll(scroll: int, sel: int, visible: int) -> int:
    if visible <= 0:
        return 0
    if sel < scroll:
        return sel
    if sel >= scroll + visible:
        return sel - visible + 1
    return scroll


def _init_colors() -> int:
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    if curses.COLORS >= 256:
        palette = [39, 214, 204, 113, 177, 44, 209, 105, 190, 81, 171, 222]
    else:
        palette = [
            curses.COLOR_CYAN,
            curses.COLOR_YELLOW,
            curses.COLOR_MAGENTA,
            curses.COLOR_GREEN,
            curses.COLOR_RED,
            curses.COLOR_BLUE,
        ]
    n = min(len(palette), curses.COLOR_PAIRS - 1)
    for i in range(n):
        curses.init_pair(i + 1, palette[i], bg)
    return max(1, n)


def run_app(
    logdir: str,
    refresh_interval: float = 2.0,
    follow: bool = True,
    smoothing: float = 0.6,
    xmode: str = "step",
) -> None:
    store = ScalarStore(logdir)
    app = App(
        store,
        refresh_interval=refresh_interval,
        follow=follow,
        smoothing=smoothing,
        xmode=xmode,
    )
    curses.wrapper(app.run)
