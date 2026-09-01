//! Rendering: header/footer, sidebar, braille charts, help overlay.
//!
//! Draws straight into the ratatui buffer; ratatui diffs frames so only
//! changed cells hit the terminal.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::Frame;

use crate::app::{App, Focus, XMode};
use crate::colors::{Dash, Palette};
use crate::plot::{
    bucketize, ema_smooth, fmt_count, fmt_duration, fmt_num, nice_ticks, BrailleCanvas,
};
use crate::store::{Series, Store};

const HELP: &[(&str, &str)] = &[
    ("Tab / Shift-Tab", "cycle focus: runs -> tags -> chart"),
    ("j k / arrows", "move in lists; prev/next tag in chart"),
    ("Space", "show/hide the selected run"),
    ("Enter", "run: show only it · group: open/close · tag: chart"),
    ("→ ←  or  l h", "in the tag tree: open / close a group"),
    ("a", "show/hide every run the filter lists"),
    ("/", "filter the focused list — runs or tags"),
    ("h l / arrows", "move data cursor (chart focus), c/Esc clear"),
    ("+ - [ ] 0", "zoom in/out, pan left/right, reset view"),
    ("s / S", "less / more smoothing"),
    ("L", "toggle log-scale Y"),
    ("x", "cycle X axis: step -> reltime -> wall"),
    ("g", "grid view: as many charts as fit, up to 3x3"),
    ("f", "toggle live follow    r: reload now"),
    ("b", "toggle sidebar"),
    ("q", "quit"),
];

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn rev() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}

/// Style for a panel drawn on top of the chart.
///
/// `REVERSED` on its own inverts whatever the cell already holds, and any DIM
/// or BOLD left there by the chart survives underneath, so a panel painted
/// with it shows the curves through it as coloured and faded bands. Resetting
/// both colours and clearing those leftovers makes it opaque. (Cells apply
/// `add` before `sub`, so bold needs its own style rather than being added on
/// top of one that clears it.)
fn overlay() -> Style {
    Style::default()
        .fg(Color::Reset)
        .bg(Color::Reset)
        .add_modifier(Modifier::REVERSED)
        .remove_modifier(Modifier::DIM | Modifier::BOLD)
}

fn overlay_bold() -> Style {
    Style::default()
        .fg(Color::Reset)
        .bg(Color::Reset)
        .add_modifier(Modifier::REVERSED | Modifier::BOLD)
        .remove_modifier(Modifier::DIM)
}

fn put(buf: &mut Buffer, x: u16, y: u16, text: &str, style: Style) {
    let area = buf.area;
    if y >= area.y + area.height || x >= area.x + area.width {
        return;
    }
    let max = (area.x + area.width - x) as usize;
    buf.set_stringn(x, y, text, max, style);
}

pub struct UiState {
    pub palette: Palette,
    pub busy: bool,
}

/// Colour and stroke a run is drawn with. Derived from the run's own colour
/// slot, so it does not change when other runs appear or are toggled off.
fn run_style(store: &Store, ui: &UiState, name: &str) -> (Color, Dash) {
    let slot = store.runs.get(name).map_or(0, |r| r.color_slot);
    (ui.palette.color(slot), ui.palette.dash(slot))
}

/// Filtered tag list for the current filter text.
pub fn visible_tags(store: &Store, app: &App) -> Vec<String> {
    let enabled: std::collections::HashSet<String> =
        store.run_names().into_iter().filter(|n| !app.disabled.contains(n)).collect();
    let mut tags = store.tags(&enabled);
    if !app.filter_text.is_empty() {
        let needle = app.filter_text.to_lowercase();
        tags.retain(|t| t.to_lowercase().contains(&needle));
    }
    tags
}

pub fn draw(f: &mut Frame, app: &mut App, store: &Store, ui: &UiState) {
    let area = f.area();
    let buf = f.buffer_mut();
    if area.height < 8 || area.width < 40 {
        put(buf, area.x, area.y, "terminal too small", Style::default());
        return;
    }

    let all_runs = store.run_names();
    let run_names = app.visible_runs(&all_runs);
    let tags = visible_tags(store, app);
    app.run_sel = app.run_sel.min(run_names.len().saturating_sub(1));

    draw_header(buf, area, app, store, ui, &all_runs, &tags);
    draw_footer(buf, area, app);

    let body = Rect { x: area.x, y: area.y + 1, width: area.width, height: area.height - 2 };
    let chart_area = if app.sidebar {
        // built once and passed down: on a directory with thousands of tags
        // this walk is milliseconds, and doing it twice a frame would show
        let tag_rows = app.tag_rows(&tags);
        let side_w = sidebar_width(area.width, &run_names, &tag_rows);
        let side = Rect { width: side_w, ..body };
        draw_sidebar(buf, side, app, store, ui, &run_names, &tags, &tag_rows);
        Rect { x: body.x + side_w, width: body.width - side_w, ..body }
    } else {
        body
    };
    draw_charts(buf, chart_area, app, store, ui, &all_runs, &tags);

    if app.help_visible {
        draw_help(buf, area);
    }
}

fn draw_header(
    buf: &mut Buffer,
    area: Rect,
    app: &App,
    store: &Store,
    ui: &UiState,
    run_names: &[String],
    tags: &[String],
) {
    let enabled = run_names.iter().filter(|n| !app.disabled.contains(*n)).count();
    let state = if app.follow {
        if ui.busy {
            "live ●"
        } else {
            "live ○"
        }
    } else {
        "paused"
    };
    let left = format!(" ttb  {} ", store.logdir.display());
    let thinned =
        if store.max_stride > 1 { format!(" ÷{}", store.max_stride) } else { String::new() };
    let right = format!(
        " runs {}/{} │ tags {} │ pts {}{} │ {} │ x:{} │ y:{} │ smooth {:.2} ",
        enabled,
        run_names.len(),
        tags.len(),
        fmt_count(store.total_points),
        thinned,
        state,
        app.xmode.label(),
        if app.log_y { "log" } else { "lin" },
        app.smoothing,
    );
    let pad = (area.width as usize).saturating_sub(left.chars().count() + right.chars().count());
    let bar = format!("{}{}{}", left, " ".repeat(pad.max(1)), right);
    put(buf, area.x, area.y, &format!("{:w$}", bar, w = area.width as usize), rev());
}

fn draw_footer(buf: &mut Buffer, area: Rect, app: &App) {
    let text = if app.filter_editing {
        let (what, buf) = if app.filter_target == Focus::Runs {
            ("runs", &app.run_filter_text)
        } else {
            ("tags", &app.filter_text)
        };
        format!(" filter {}: {}▏  (Enter apply · Esc cancel)", what, buf)
    } else if std::time::Instant::now() < app.flash_until {
        format!(" {}", app.flash_msg)
    } else {
        "  Tab:focus  Space:run on/off  Enter:solo/open  a:all  /:filter  s/S:smooth  \
         +/-/[/]:zoom·pan  g:grid  L:log  x:axis  f:follow  ?:help  q:quit"
            .to_string()
    };
    put(
        buf,
        area.x,
        area.y + area.height - 1,
        &format!("{:w$}", text, w = area.width as usize),
        dim(),
    );
}

// -- sidebar ---------------------------------------------------------------

/// Width the sidebar takes: enough for the longest name it has to show,
/// within bounds. Short tag sets keep it narrow; the deep hierarchies that
/// motivated the tree get room for their names instead of eliding them, and
/// the chart never gives up more than two fifths of the pane.
fn sidebar_width(total: u16, runs: &[String], rows: &[crate::tags::Row]) -> u16 {
    const MARK: usize = 6; // cursor, checkbox, colour dot, padding
    let runs_w = runs.iter().map(|n| n.chars().count()).max().unwrap_or(0) + MARK;
    let tags_w = rows
        .iter()
        .map(|r| {
            // cursor + indent + twisty + label + " (n)"
            4 + 2 * r.depth + r.label.chars().count() + if r.leaf { 0 } else { 5 }
        })
        .max()
        .unwrap_or(0);
    let want = runs_w.max(tags_w).min(u16::MAX as usize) as u16;
    let ceiling = (total * 2 / 5).clamp(24, 46);
    want.clamp(24, ceiling)
}

fn scroll_for(scroll: usize, sel: usize, visible: usize) -> usize {
    if visible == 0 {
        0
    } else if sel < scroll {
        sel
    } else if sel >= scroll + visible {
        sel - visible + 1
    } else {
        scroll
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_sidebar(
    buf: &mut Buffer,
    area: Rect,
    app: &mut App,
    store: &Store,
    ui: &UiState,
    run_names: &[String],
    tags: &[String],
    rows: &[crate::tags::Row],
) {
    // `clamp` panics when its max is below its min, which happened on short
    // terminals (body height 6-8 gave max 2); keep the bound above the min.
    let runs_h = ((run_names.len() + 1) as u16).clamp(2, (area.height / 3).max(2));
    let tags_h = area.height - runs_h;

    // runs
    let focused = app.focus == Focus::Runs;
    let title_style = if focused { bold().add_modifier(Modifier::REVERSED) } else { bold() };
    let mut runs_title = if app.run_filter_text.is_empty() {
        format!(" RUNS ({})", run_names.len())
    } else {
        format!(" RUNS ({}/{})", run_names.len(), store.runs.len())
    };
    if !app.run_filter_text.is_empty() || (app.filter_editing && app.filter_target == Focus::Runs) {
        runs_title.push_str("  /");
        runs_title.push_str(&app.run_filter_text);
    }
    put(buf, area.x, area.y, &runs_title, title_style);
    let visible = runs_h.saturating_sub(1) as usize;
    app.run_scroll = scroll_for(app.run_scroll, app.run_sel, visible);
    for i in 0..visible {
        let idx = app.run_scroll + i;
        if idx >= run_names.len() {
            break;
        }
        let name = &run_names[idx];
        let on = !app.disabled.contains(name);
        let sel = focused && idx == app.run_sel;
        let row_style = if sel { rev() } else { Style::default() };
        let y = area.y + 1 + i as u16;
        let mark = if on { "▣" } else { "☐" };
        put(buf, area.x, y, &format!(" {} ", mark), row_style);
        let (color, dash) = run_style(store, ui, name);
        put(
            buf,
            area.x + 3,
            y,
            dash.marker(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        );
        let label_style = if on { row_style } else { row_style.add_modifier(Modifier::DIM) };
        let label = format!(" {}", name);
        let w = area.width.saturating_sub(6) as usize;
        put(buf, area.x + 4, y, &format!("{:w$}", label, w = w), label_style);
    }

    // tags
    let ty = area.y + runs_h;
    let focused = app.focus == Focus::Tags;
    let title_style = if focused { bold().add_modifier(Modifier::REVERSED) } else { bold() };
    let mut title = format!(" TAGS ({})", tags.len());
    if !app.filter_text.is_empty() || app.filter_editing {
        title.push_str("  /");
        title.push_str(&app.filter_text);
    }
    put(buf, area.x, ty, &title, title_style);
    let visible = tags_h.saturating_sub(1) as usize;
    app.tag_sel = app.tag_sel.min(rows.len().saturating_sub(1));
    app.tag_scroll = scroll_for(app.tag_scroll, app.tag_sel, visible);
    let w = area.width.saturating_sub(2) as usize;
    for i in 0..visible {
        let idx = app.tag_scroll + i;
        let Some(row) = rows.get(idx) else { break };
        let sel = focused && idx == app.tag_sel;
        let style = if sel { rev() } else { Style::default() };
        let cursor = if idx == app.tag_sel { "▶" } else { " " };
        // groups carry a twisty and their tag count; leaves are just the name
        let twisty = if row.leaf {
            "  "
        } else if row.expanded {
            "▾ "
        } else {
            "▸ "
        };
        let indent = "  ".repeat(row.depth);
        let count = if row.leaf { String::new() } else { format!(" ({})", row.leaves) };
        let fixed = 2 + indent.chars().count() + twisty.chars().count() + count.chars().count();
        let label = crate::tags::elide(&row.label, w.saturating_sub(fixed));
        let line = format!(" {}{}{}{}{}", cursor, indent, twisty, label, count);
        put(buf, area.x, ty + 1 + i as u16, &format!("{:w$}", line, w = w), style);
    }

    // separator
    for y in area.y..area.y + area.height {
        put(buf, area.x + area.width - 1, y, "│", dim());
    }
}

// -- charts ----------------------------------------------------------------

fn center_msg(buf: &mut Buffer, area: Rect, msg: &str) {
    let x = area.x + (area.width.saturating_sub(msg.chars().count() as u16)) / 2;
    put(buf, x, area.y + area.height / 2, msg, dim());
}

fn draw_charts(
    buf: &mut Buffer,
    area: Rect,
    app: &App,
    store: &Store,
    ui: &UiState,
    run_names: &[String],
    tags: &[String],
) {
    if !app.loaded {
        center_msg(buf, area, "loading event files…");
        return;
    }
    if tags.is_empty() {
        let msg =
            if app.filter_text.is_empty() { "no scalars found" } else { "no tags match filter" };
        center_msg(buf, area, msg);
        return;
    }
    // the tag list is a tree, so the selection is a row; the chart follows the
    // tag under it (or the first tag inside a group)
    let first = app.selected_tag(tags).unwrap_or(0).min(tags.len() - 1);
    if app.grid {
        let (cols, rows, n) = grid_shape(area, tags.len() - first);
        let shown = &tags[first..first + n];
        // Spread the remainder over the leading cells rather than leaving a
        // ragged strip of unused columns on the right.
        let (cell_w, extra_w) = (area.width / cols, area.width % cols);
        let (cell_h, extra_h) = (area.height / rows, area.height % rows);
        let span = |i: u16, size: u16, extra: u16| {
            let start = i * size + i.min(extra);
            (start, size + u16::from(i < extra))
        };
        for (i, tag) in shown.iter().enumerate() {
            let (r, c) = ((i as u16) / cols, (i as u16) % cols);
            let (x, width) = span(c, cell_w, extra_w);
            let (y, height) = span(r, cell_h, extra_h);
            let cell = Rect { x: area.x + x, y: area.y + y, width, height };
            draw_chart(buf, cell, app, store, ui, run_names, tag, false, i == 0);
        }
    } else {
        draw_chart(buf, area, app, store, ui, run_names, &tags[first], true, true);
    }
}

/// Chart count and arrangement for the grid view.
///
/// Columns and rows are however many fit at a size still worth reading, up to
/// three each, so a wide terminal gets 2 or 3 across instead of always one.
/// The shape is then trimmed to the tags actually available: among the
/// layouts that hold them, the one leaving fewest empty cells wins, and among
/// equals the widest — three tags side by side beat three stacked.
fn grid_shape(area: Rect, available: usize) -> (u16, u16, usize) {
    /// Below this a cell is mostly axis gutter: 9 columns of labels and the
    /// axis rule leave ~30 for the plot, which is 60 braille pixels across.
    const MIN_W: u16 = 40;
    /// Title, a few rows of plot, and the x labels.
    const MIN_H: u16 = 9;
    const MAX_SIDE: u16 = 3;

    let max_cols = (area.width / MIN_W).clamp(1, MAX_SIDE);
    let max_rows = (area.height / MIN_H).clamp(1, MAX_SIDE);
    let n = available.clamp(1, (max_cols * max_rows) as usize);

    let mut best: Option<(u16, u16, u16)> = None; // cols, rows, empty cells
    for cols in 1..=max_cols {
        let rows = (n as u16).div_ceil(cols);
        if rows > max_rows {
            continue;
        }
        let empty = cols * rows - n as u16;
        // cols ascends, so `<=` keeps the widest of the equally tight layouts
        if best.is_none_or(|(_, _, best_empty)| empty <= best_empty) {
            best = Some((cols, rows, empty));
        }
    }
    let (cols, rows, _) = best.unwrap_or((1, 1, 0));
    (cols, rows, n)
}

struct DrawnSeries<'a> {
    color: Color,
    dash: Dash,
    name: &'a str,
    pts: Vec<(usize, f64)>,
    series: &'a Series,
    off: f64,
}

fn xs_offset(app: &App, s: &Series, first_wall: Option<f64>) -> f64 {
    match app.xmode {
        XMode::Step | XMode::Wall => 0.0,
        XMode::RelTime => first_wall.unwrap_or_else(|| s.walls.first().copied().unwrap_or(0.0)),
    }
}

fn x_at<'a>(app: &App, s: &'a Series) -> Box<dyn Fn(usize) -> f64 + 'a> {
    match app.xmode {
        XMode::Step => Box::new(move |i| s.steps[i] as f64),
        _ => Box::new(move |i| s.walls[i]),
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_chart(
    buf: &mut Buffer,
    area: Rect,
    app: &App,
    store: &Store,
    ui: &UiState,
    run_names: &[String],
    tag: &str,
    detailed: bool,
    highlight: bool,
) {
    let gutter: u16 = 9;
    if area.height < 5 || area.width < gutter + 12 {
        return;
    }
    let plot_h = (area.height - 2) as usize;
    let plot_w = (area.width - gutter - 2) as usize;

    let title_style = if highlight { bold() } else { bold().add_modifier(Modifier::DIM) };
    let marker = if highlight && app.grid { "▶ " } else { "" };
    put(buf, area.x + gutter + 1, area.y, &format!("{}{}", marker, tag), title_style);

    // gather enabled series holding this tag
    let mut series: Vec<(&str, &Series, Option<f64>)> = Vec::new();
    for name in run_names.iter() {
        if app.disabled.contains(name) {
            continue;
        }
        if let Some(run) = store.runs.get(name) {
            if let Some(s) = run.series.get(tag) {
                if !s.is_empty() {
                    series.push((name, s, run.first_wall));
                }
            }
        }
    }
    if series.is_empty() {
        center_msg(
            buf,
            Rect { y: area.y + 1, height: plot_h as u16, ..area },
            "no data in enabled runs",
        );
        return;
    }

    // full X domain
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    for (_, s, fw) in &series {
        let off = xs_offset(app, s, *fw);
        let xat = x_at(app, s);
        xmin = xmin.min(xat(0) - off);
        xmax = xmax.max(xat(s.len() - 1) - off);
    }
    let span = (xmax - xmin).max(1e-12);
    let lo = xmin + span * app.view.0;
    let mut hi = xmin + span * app.view.1;
    if hi <= lo {
        hi = lo + 1e-12;
    }

    let mut canvas = BrailleCanvas::new(plot_w, plot_h);
    let mut drawn: Vec<DrawnSeries> = Vec::new();
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for (name, s, fw) in &series {
        let off = xs_offset(app, s, *fw);
        let xat = x_at(app, s);
        let mut pts = bucketize(s.len(), &xat, &s.vals, lo + off, hi + off, canvas.px_w);
        ema_smooth(&mut pts, app.smoothing);
        if app.log_y {
            pts.retain(|p| p.1 > 0.0);
            for p in pts.iter_mut() {
                p.1 = p.1.log10();
            }
        }
        if pts.is_empty() {
            continue;
        }
        for &(_, v) in &pts {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
        let (color, dash) = run_style(store, ui, name);
        drawn.push(DrawnSeries { color, dash, name, pts, series: s, off });
    }
    if drawn.is_empty() || vmin > vmax {
        center_msg(
            buf,
            Rect { y: area.y + 1, height: plot_h as u16, ..area },
            "no drawable points",
        );
        return;
    }
    if vmax - vmin < 1e-12 {
        let pad = if vmax.abs() > 0.0 { vmax.abs() * 0.1 } else { 1.0 };
        vmin -= pad;
        vmax += pad;
    } else {
        let pad = (vmax - vmin) * 0.05;
        vmin -= pad;
        vmax += pad;
    }
    let vspan = vmax - vmin;
    let py_max = (canvas.px_h - 1) as i64;
    let to_py = |v: f64| -> i64 { py_max - ((v - vmin) / vspan * py_max as f64).round() as i64 };

    // color index -> ratatui color, so the canvas stores one byte per cell
    let mut color_of: Vec<Color> = Vec::new();
    for d in &drawn {
        if !color_of.contains(&d.color) {
            color_of.push(d.color);
        }
    }
    for d in &drawn {
        let ci = color_of.iter().position(|c| *c == d.color).unwrap() as u8 + 1;
        let mut prev: Option<(i64, i64)> = None;
        let mut phase = 0u32;
        for &(col, v) in &d.pts {
            let px = col as i64;
            let py = to_py(v);
            match prev {
                Some((ax, ay)) => canvas.line_styled((ax, ay), (px, py), ci, d.dash, &mut phase),
                None => {
                    canvas.dot(px, py, ci);
                    phase = phase.wrapping_add(1);
                }
            }
            prev = Some((px, py));
        }
    }

    // Y axis rule + ticks
    let axis_x = area.x + gutter;
    for row in 0..plot_h {
        put(buf, axis_x, area.y + 1 + row as u16, "│", dim());
    }
    for tick in nice_ticks(vmin, vmax, (plot_h / 4).max(3)) {
        let row = (to_py(tick) / 4).clamp(0, plot_h as i64 - 1) as u16;
        let shown = if app.log_y { 10f64.powf(tick) } else { tick };
        let label = fmt_num(shown);
        let label = if label.len() > gutter as usize {
            label[..gutter as usize].to_string()
        } else {
            label
        };
        put(
            buf,
            area.x + gutter - label.chars().count().min(gutter as usize) as u16,
            area.y + 1 + row,
            &label,
            dim(),
        );
        put(buf, axis_x, area.y + 1 + row, "┼", dim());
    }

    // canvas cells
    for row in 0..plot_h {
        for col in 0..plot_w {
            if let Some((ch, ci)) = canvas.cell_char(row, col) {
                let color = color_of[(ci - 1) as usize];
                let style = Style::default().fg(color).add_modifier(Modifier::BOLD);
                let mut tmp = [0u8; 4];
                put(
                    buf,
                    axis_x + 1 + col as u16,
                    area.y + 1 + row as u16,
                    ch.encode_utf8(&mut tmp),
                    style,
                );
            }
        }
    }

    // X labels
    let xl_y = area.y + 1 + plot_h as u16;
    put(buf, axis_x, xl_y, "└", dim());
    let nticks = (plot_w / 22).max(2);
    for i in 0..=nticks {
        let frac = i as f64 / nticks as f64;
        let xv = lo + (hi - lo) * frac;
        let label = fmt_x(app, xv);
        let mut px = axis_x as i64 + 1 + (frac * (plot_w - 1) as f64) as i64;
        if i == nticks {
            px -= label.chars().count() as i64;
        }
        let max_x = (area.x + area.width) as i64 - label.chars().count() as i64 - 1;
        let px = px.clamp(axis_x as i64 + 1, max_x.max(axis_x as i64 + 1)) as u16;
        put(buf, px, xl_y, &label, dim());
    }

    // cursor + legend
    let mut cursor_x: Option<f64> = None;
    if detailed {
        if let Some(cur) = app.cursor {
            let col = (cur * (canvas.px_w - 1) as f64) as usize;
            let cell = col / 2;
            cursor_x = Some(lo + (hi - lo) * col as f64 / (canvas.px_w - 1).max(1) as f64);
            for row in 0..plot_h {
                if canvas.cells[row * canvas.w + cell] == 0 {
                    put(buf, axis_x + 1 + cell as u16, area.y + 1 + row as u16, "┊", dim());
                }
            }
        }
        draw_legend(buf, area, gutter, app, &drawn, cursor_x);
    }
}

fn fmt_x(app: &App, xv: f64) -> String {
    match app.xmode {
        XMode::Step => fmt_count(xv.round().max(0.0) as u64),
        XMode::RelTime => fmt_duration(xv),
        XMode::Wall => {
            let t = xv as i64;
            let secs = t.rem_euclid(86_400);
            format!("{:02}:{:02}:{:02}", secs / 3600, (secs / 60) % 60, secs % 60)
        }
    }
}

fn draw_legend(
    buf: &mut Buffer,
    area: Rect,
    gutter: u16,
    app: &App,
    drawn: &[DrawnSeries],
    cursor_x: Option<f64>,
) {
    for (i, d) in drawn.iter().take(8).enumerate() {
        let mut text = d.name.to_string();
        if let Some(cx) = cursor_x {
            // nearest raw point at or before the cursor
            let s = d.series;
            let target = cx + d.off;
            let idx = match app.xmode {
                XMode::Step => s.steps.partition_point(|&v| (v as f64) <= target),
                _ => s.walls.partition_point(|&v| v <= target),
            };
            if idx > 0 {
                let idx = idx - 1;
                text = format!(
                    "{}  {} @ {}",
                    d.name,
                    fmt_num(s.vals[idx]),
                    fmt_count(s.steps[idx].max(0) as u64)
                );
            }
        }
        let line_len = text.chars().count() as u16 + 2;
        let x = (area.x + area.width).saturating_sub(line_len + 1).max(area.x + gutter + 1);
        let y = area.y + 1 + i as u16;
        put(buf, x, y, d.dash.marker(), Style::default().fg(d.color).add_modifier(Modifier::BOLD));
        put(buf, x + 2, y, &text, Style::default());
    }
}

fn draw_help(buf: &mut Buffer, area: Rect) {
    let box_w = 64.min(area.width.saturating_sub(4));
    let box_h = (HELP.len() as u16 + 4).min(area.height.saturating_sub(2));
    let top = area.y + (area.height - box_h) / 2;
    let left = area.x + (area.width - box_w) / 2;
    for y in 0..box_h {
        put(buf, left, top + y, &" ".repeat(box_w as usize), overlay());
    }
    put(
        buf,
        left + 2,
        top + 1,
        "terminal-tensorboard — keys",
        overlay().add_modifier(Modifier::BOLD),
    );
    for (i, (keys, desc)) in HELP.iter().take(box_h.saturating_sub(4) as usize).enumerate() {
        put(buf, left + 2, top + 3 + i as u16, &format!("{:18}", keys), overlay_bold());
        put(buf, left + 20, top + 3 + i as u16, desc, overlay());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(w: u16, h: u16, tags: usize) -> (u16, u16, usize) {
        grid_shape(Rect { x: 0, y: 0, width: w, height: h }, tags)
    }

    fn rows_of(labels: &[(usize, &str, bool)]) -> Vec<crate::tags::Row> {
        labels
            .iter()
            .map(|(depth, label, leaf)| crate::tags::Row {
                depth: *depth,
                label: label.to_string(),
                path: label.to_string(),
                leaf: *leaf,
                leaves: 1,
                expanded: true,
            })
            .collect()
    }

    #[test]
    fn the_sidebar_fits_its_content_within_bounds() {
        let short = rows_of(&[(0, "train", false), (1, "loss", true)]);
        let runs = vec!["baseline".to_string()];
        assert_eq!(sidebar_width(200, &runs, &short), 24, "short names keep it narrow");

        let long = rows_of(&[(1, "swh_content-dedup-opc-filtered-code_bd912224", false)]);
        assert_eq!(sidebar_width(200, &runs, &long), 46, "long names take the ceiling");
        assert_eq!(sidebar_width(80, &runs, &long), 32, "but never more than two fifths");
        assert_eq!(sidebar_width(40, &runs, &long), 24, "and never below the floor");
    }

    #[test]
    fn columns_follow_the_width() {
        assert_eq!(shape(60, 30, 9).0, 1, "a narrow pane stays single column");
        assert_eq!(shape(100, 30, 9).0, 2);
        assert_eq!(shape(160, 30, 9).0, 3);
        assert_eq!(shape(400, 30, 9).0, 3, "never more than three across");
    }

    #[test]
    fn the_column_thresholds_are_where_they_claim_to_be() {
        // Pinned literally rather than recomputed: these are widths of the
        // chart pane, not of the terminal — the sidebar takes 24-34 columns
        // before this sees anything.
        for (width, columns) in [(39, 1), (40, 1), (79, 1), (80, 2), (119, 2), (120, 3), (200, 3)] {
            assert_eq!(shape(width, 30, 9).0, columns, "at pane width {}", width);
        }
    }

    #[test]
    fn rows_follow_the_height() {
        assert_eq!(shape(160, 10, 9).1, 1, "a short pane fits one row");
        assert_eq!(shape(160, 20, 9).1, 2);
        assert_eq!(shape(160, 30, 9).1, 3);
        assert_eq!(shape(160, 200, 9).1, 3);
    }

    #[test]
    fn the_grid_never_exceeds_its_capacity() {
        for (w, h) in [(60, 30), (100, 30), (160, 30), (160, 10), (40, 6)] {
            let (cols, rows, n) = shape(w, h, 100);
            assert_eq!(n, (cols * rows) as usize, "{}x{} left cells unused", w, h);
            assert!(cols <= 3 && rows <= 3);
        }
    }

    #[test]
    fn the_shape_shrinks_to_the_tags_available() {
        // one tag never becomes a 3x3 of blanks
        assert_eq!(shape(160, 30, 1), (1, 1, 1));
        // two go side by side, using the full height, not stacked
        assert_eq!(shape(160, 30, 2), (2, 1, 2));
        assert_eq!(shape(160, 30, 3), (3, 1, 3));
        // four square up rather than leaving a ragged row of three plus one
        assert_eq!(shape(160, 30, 4), (2, 2, 4));
        assert_eq!(shape(160, 30, 6), (3, 2, 6));
        assert_eq!(shape(160, 30, 9), (3, 3, 9));
    }

    #[test]
    fn a_narrow_pane_stacks_within_the_rows_it_has() {
        assert_eq!(shape(60, 30, 5), (1, 3, 3), "one column, three rows, three tags");
        assert_eq!(shape(60, 12, 5), (1, 1, 1));
    }

    #[test]
    fn even_a_tiny_pane_asks_for_one_chart() {
        let (cols, rows, n) = shape(10, 3, 5);
        assert_eq!((cols, rows, n), (1, 1, 1));
        // and with no tags at all the count never underflows
        assert_eq!(shape(160, 30, 0), (1, 1, 1));
    }
}
