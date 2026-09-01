//! Categorical colors for run series.
//!
//! Eight hues in a fixed, validated order — blue, orange, aqua, yellow,
//! magenta, green, violet, red.  The truecolor values are the reference
//! data-viz categorical palette (dark-surface steps, chosen because a
//! terminal's background is usually dark and these also stay legible on a
//! light one).  The 256-color values are those hues snapped to the nearest
//! xterm cube entry that still sits inside the lightness band and over the
//! chroma floor, so the approximation does not drift into "reads gray".
//!
//! Two rules drive the assignment:
//!
//! * **A run keeps its color.** The slot is handed out once, when the run is
//!   first discovered, and never recomputed — so toggling a run off, or a new
//!   run appearing mid-training, never repaints the others.
//! * **Hues are used in fixed order, never cycled into ambiguity.** Past the
//!   eighth run the hues repeat, but each repeat comes with a different line
//!   style (solid → dashed → dotted), so two runs sharing a hue are still
//!   told apart on the chart itself, not just by the legend.

use ratatui::style::Color;

/// Reference categorical palette, dark-surface steps.
const TRUECOLOR: [(u8, u8, u8); 8] = [
    (0x39, 0x87, 0xe5), // blue
    (0xd9, 0x59, 0x26), // orange
    (0x19, 0x9e, 0x70), // aqua
    (0xc9, 0x85, 0x00), // yellow
    (0xd5, 0x51, 0x81), // magenta
    (0x00, 0x83, 0x00), // green
    (0x90, 0x85, 0xe9), // violet
    (0xe6, 0x67, 0x67), // red
];

/// The same hues as xterm-256 cube indices, kept inside the lightness band.
const INDEXED: [u8; 8] = [32, 166, 29, 136, 168, 28, 68, 167];

/// Last-resort 8-color approximation of the same hue order.
const BASIC: [Color; 8] = [
    Color::Blue,
    Color::LightRed,
    Color::Cyan,
    Color::Yellow,
    Color::LightMagenta,
    Color::Green,
    Color::LightBlue,
    Color::Red,
];

pub const SLOTS: usize = 8;

/// How a series' line is stroked. Distinguishes runs that share a hue once
/// there are more runs than hues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dash {
    /// Pixels drawn out of every `period`.
    pub on: u32,
    pub period: u32,
}

impl Dash {
    pub const SOLID: Dash = Dash { on: 1, period: 1 };

    #[inline]
    pub fn covers(&self, phase: u32) -> bool {
        self.period <= 1 || phase % self.period < self.on
    }

    /// Legend marker: a glyph that differs per style, so identity on the
    /// chart is never carried by hue alone.
    pub fn marker(&self) -> &'static str {
        match self.period {
            0..=1 => "●",
            9 => "◆",
            _ => "▪",
        }
    }
}

const DASHES: [Dash; 3] = [
    Dash::SOLID,
    Dash { on: 6, period: 9 },
    Dash { on: 2, period: 6 },
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Depth {
    Truecolor,
    Indexed256,
    Basic,
}

/// What the terminal can render, from the usual environment hints.
pub fn detect_depth() -> Depth {
    let colorterm = std::env::var("COLORTERM").unwrap_or_default();
    if colorterm.eq_ignore_ascii_case("truecolor") || colorterm.eq_ignore_ascii_case("24bit") {
        return Depth::Truecolor;
    }
    let term = std::env::var("TERM").unwrap_or_default();
    let modern = ["kitty", "alacritty", "wezterm", "foot", "contour", "ghostty"];
    if term.contains("256color")
        || term.contains("direct")
        || modern.iter().any(|t| term.contains(t))
        || !colorterm.is_empty()
    {
        return Depth::Indexed256;
    }
    Depth::Basic
}

#[derive(Clone)]
pub struct Palette {
    depth: Depth,
}

impl Palette {
    pub fn new(depth: Depth) -> Self {
        Palette { depth }
    }

    pub fn detect() -> Self {
        Palette::new(detect_depth())
    }

    /// Hue for a run's colour slot.
    pub fn color(&self, slot: usize) -> Color {
        let i = slot % SLOTS;
        match self.depth {
            Depth::Truecolor => {
                let (r, g, b) = TRUECOLOR[i];
                Color::Rgb(r, g, b)
            }
            Depth::Indexed256 => Color::Indexed(INDEXED[i]),
            Depth::Basic => BASIC[i],
        }
    }

    /// Line style for a run's colour slot: solid for the first eight runs,
    /// then dashed, then dotted, so repeated hues stay distinguishable.
    pub fn dash(&self, slot: usize) -> Dash {
        DASHES[(slot / SLOTS).min(DASHES.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_eight_runs_get_distinct_solid_colors() {
        let p = Palette::new(Depth::Truecolor);
        let mut seen = Vec::new();
        for slot in 0..SLOTS {
            let c = p.color(slot);
            assert!(!seen.contains(&c), "slot {} repeats a hue", slot);
            assert_eq!(p.dash(slot), Dash::SOLID);
            seen.push(c);
        }
    }

    #[test]
    fn repeated_hues_get_a_different_dash() {
        let p = Palette::new(Depth::Truecolor);
        // slot 8 wraps onto slot 0's hue, so its stroke must differ
        assert_eq!(p.color(8), p.color(0));
        assert_ne!(p.dash(8), p.dash(0));
        assert_ne!(p.dash(8).marker(), p.dash(0).marker());
        assert_eq!(p.color(16), p.color(0));
        assert_ne!(p.dash(16), p.dash(8));
    }

    #[test]
    fn every_depth_offers_eight_distinct_colors() {
        for depth in [Depth::Truecolor, Depth::Indexed256, Depth::Basic] {
            let p = Palette::new(depth);
            let mut seen = Vec::new();
            for slot in 0..SLOTS {
                let c = p.color(slot);
                assert!(!seen.contains(&c), "{:?} repeats at slot {}", depth, slot);
                seen.push(c);
            }
        }
    }

    #[test]
    fn dash_pattern_draws_and_skips() {
        let solid = Dash::SOLID;
        assert!((0..20).all(|p| solid.covers(p)));
        let dashed = DASHES[1];
        assert!(dashed.covers(0));
        assert!(!dashed.covers(6));
        assert!(dashed.covers(9)); // pattern repeats
        let dotted = DASHES[2];
        assert!(dotted.covers(1));
        assert!(!dotted.covers(2));
    }
}
