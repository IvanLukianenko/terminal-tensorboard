//! Braille-dot canvas and series preprocessing for terminal charts.
//!
//! A character cell holds a 2x4 grid of braille dots, so a WxH cell canvas
//! has 2W x 4H addressable pixels.  Series are first reduced to one mean
//! value per pixel column (`bucketize`), so everything after that is
//! O(columns) regardless of how many points a run holds.

const BRAILLE_BASE: u32 = 0x2800;
// dot bit for (y in 0..4, x in 0..2) inside one cell
const DOT_BITS: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

pub struct BrailleCanvas {
    pub w: usize,
    pub px_w: usize,
    pub px_h: usize,
    pub cells: Vec<u8>,
    pub colors: Vec<u8>,
}

impl BrailleCanvas {
    pub fn new(w: usize, h: usize) -> Self {
        BrailleCanvas { w, px_w: w * 2, px_h: h * 4, cells: vec![0; w * h], colors: vec![0; w * h] }
    }

    #[inline]
    pub fn dot(&mut self, x: i64, y: i64, color: u8) {
        if x >= 0 && (x as usize) < self.px_w && y >= 0 && (y as usize) < self.px_h {
            let (x, y) = (x as usize, y as usize);
            let idx = (y >> 2) * self.w + (x >> 1);
            self.cells[idx] |= DOT_BITS[y & 3][x & 1];
            self.colors[idx] = color;
        }
    }

    pub fn line(&mut self, mut x0: i64, mut y0: i64, x1: i64, y1: i64, color: u8) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.dot(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                return;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    #[inline]
    pub fn cell_char(&self, row: usize, col: usize) -> Option<(char, u8)> {
        let bits = self.cells[row * self.w + col];
        if bits == 0 {
            return None;
        }
        let ch = char::from_u32(BRAILLE_BASE | u32::from(bits)).unwrap();
        Some((ch, self.colors[row * self.w + col]))
    }
}

// --------------------------------------------------------------------------
// series preprocessing
// --------------------------------------------------------------------------

/// Reduce a sorted series to at most one (column, mean) point per column.
///
/// `xat(i)` maps index -> x coordinate (steps as f64, or wall time), which
/// must be non-decreasing.  O(log n) per column boundary + O(points in view)
/// for the summing.
pub fn bucketize(
    len: usize,
    xat: impl Fn(usize) -> f64,
    ys: &[f64],
    x0: f64,
    x1: f64,
    ncols: usize,
) -> Vec<(usize, f64)> {
    if len == 0 || ncols == 0 || x1 < x0 {
        return Vec::new();
    }
    let lo = partition_point(len, |i| xat(i) < x0);
    let hi = partition_point(len, |i| xat(i) <= x1);
    if hi <= lo {
        return Vec::new();
    }
    let span = x1 - x0;
    if span <= 0.0 {
        let sum: f64 = ys[lo..hi].iter().sum();
        return vec![(ncols / 2, sum / (hi - lo) as f64)];
    }
    let mut out = Vec::with_capacity(ncols.min(hi - lo));
    let mut prev = lo;
    for col in 0..ncols {
        let edge = x0 + span * (col + 1) as f64 / ncols as f64;
        // linear scan is fine: total work over all columns is O(hi - lo)
        let mut nxt = prev;
        while nxt < hi && xat(nxt) <= edge {
            nxt += 1;
        }
        if nxt > prev {
            let sum: f64 = ys[prev..nxt].iter().sum();
            out.push((col, sum / (nxt - prev) as f64));
            prev = nxt;
        }
    }
    out
}

fn partition_point(len: usize, pred: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, len);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Debiased exponential moving average (TensorBoard-style smoothing).
pub fn ema_smooth(points: &mut [(usize, f64)], weight: f64) {
    if weight <= 0.0 || points.len() < 2 {
        return;
    }
    let mut last = 0.0f64;
    let mut wpow = 1.0f64;
    for p in points.iter_mut() {
        last = last * weight + (1.0 - weight) * p.1;
        wpow *= weight;
        p.1 = last / (1.0 - wpow);
    }
}

/// A few round tick values covering [lo, hi].
pub fn nice_ticks(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    if count < 2 || !lo.is_finite() || !hi.is_finite() {
        return Vec::new();
    }
    if hi <= lo {
        return vec![lo];
    }
    let raw = (hi - lo) / (count - 1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let mut step = mag;
    for mult in [1.0, 2.0, 2.5, 5.0, 10.0] {
        step = mag * mult;
        if step >= raw {
            break;
        }
    }
    let first = (lo / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut t = first;
    while t <= hi + step * 1e-9 {
        ticks.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        t += step;
    }
    ticks
}

/// Compact human formatting for axis labels and readouts.
pub fn fmt_num(v: f64) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "nan".into() } else if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = v.abs();
    if a != 0.0 && !(1e-3..1e5).contains(&a) {
        return format!("{:.3e}", v);
    }
    if a >= 100.0 {
        return format!("{:.1}", v);
    }
    let s = format!("{:.4}", v);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" { "0".into() } else { trimmed.to_string() }
}

pub fn fmt_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    if seconds < 60.0 {
        return format!("{:.0}s", seconds);
    }
    let total = seconds as u64;
    let (m, s) = (total / 60, total % 60);
    if m < 60 {
        return format!("{}m{:02}s", m, s);
    }
    let (h, m) = (m / 60, m % 60);
    if h < 48 {
        return format!("{}h{:02}m", h, m);
    }
    let (d, h) = (h / 24, h % 24);
    format!("{}d{:02}h", d, h)
}

pub fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketize_means_monotonic() {
        let ys: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let pts = bucketize(100, |i| i as f64, &ys, 0.0, 99.0, 10);
        assert_eq!(pts.len(), 10);
        let vals: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let mut sorted = vals.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(vals, sorted);
    }

    #[test]
    fn ema_debiased_start() {
        let mut pts: Vec<(usize, f64)> = (0..10).map(|i| (i, 1.0)).collect();
        ema_smooth(&mut pts, 0.9);
        for (_, v) in pts {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn canvas_line_sets_braille() {
        let mut c = BrailleCanvas::new(10, 2);
        c.line(0, 0, 19, 7, 3);
        let mut any = false;
        for row in 0..2 {
            for col in 0..10 {
                if let Some((ch, color)) = c.cell_char(row, col) {
                    any = true;
                    assert_eq!(color, 3);
                    assert!((0x2800..=0x28FF).contains(&(ch as u32)));
                }
            }
        }
        assert!(any);
    }

    #[test]
    fn nice_ticks_are_round() {
        let ticks = nice_ticks(0.0, 1.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks.iter().all(|t| (0.0..=1.0 + 1e-9).contains(t)));
    }
}
