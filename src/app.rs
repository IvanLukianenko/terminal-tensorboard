//! Application state and key handling.

use std::collections::HashSet;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub const ZOOM_FACTOR: f64 = 0.7;
pub const PAN_FRACTION: f64 = 0.15;
pub const MIN_SPAN: f64 = 1e-4;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Runs,
    Tags,
    Chart,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XMode {
    Step,
    RelTime,
    Wall,
}

impl XMode {
    pub fn next(self) -> Self {
        match self {
            XMode::Step => XMode::RelTime,
            XMode::RelTime => XMode::Wall,
            XMode::Wall => XMode::Step,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            XMode::Step => "step",
            XMode::RelTime => "reltime",
            XMode::Wall => "wall",
        }
    }
}

pub struct App {
    pub focus: Focus,
    pub run_sel: usize,
    pub tag_sel: usize,
    pub run_scroll: usize,
    pub tag_scroll: usize,
    pub disabled: HashSet<String>,
    pub filter_text: String,
    pub filter_editing: bool,
    filter_backup: String,
    pub grid: bool,
    pub log_y: bool,
    pub sidebar: bool,
    pub follow: bool,
    pub smoothing: f64,
    pub xmode: XMode,
    /// Visible fraction of the full X domain.
    pub view: (f64, f64),
    /// Data cursor as a fraction of the visible range.
    pub cursor: Option<f64>,
    pub help_visible: bool,
    pub loaded: bool,
    pub flash_msg: String,
    pub flash_until: Instant,
    /// Set when the user asks for an immediate reload ('r').
    pub reload_requested: bool,
    pub quit: bool,
}

impl App {
    pub fn new(follow: bool, smoothing: f64, xmode: XMode) -> Self {
        App {
            focus: Focus::Tags,
            run_sel: 0,
            tag_sel: 0,
            run_scroll: 0,
            tag_scroll: 0,
            disabled: HashSet::new(),
            filter_text: String::new(),
            filter_editing: false,
            filter_backup: String::new(),
            grid: false,
            log_y: false,
            sidebar: true,
            follow,
            smoothing: smoothing.clamp(0.0, 0.99),
            xmode,
            view: (0.0, 1.0),
            cursor: None,
            help_visible: false,
            loaded: false,
            flash_msg: String::new(),
            flash_until: Instant::now(),
            reload_requested: false,
            quit: false,
        }
    }

    pub fn flash(&mut self, msg: &str) {
        self.flash_msg = msg.to_string();
        self.flash_until = Instant::now() + std::time::Duration::from_millis(2500);
    }

    pub fn handle_key(&mut self, mut key: KeyEvent, run_names: &[String], tags: &[String]) {
        // some terminals deliver Enter as a raw LF/CR character or as ^J/^M
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if matches!(key.code, KeyCode::Char('\n') | KeyCode::Char('\r'))
            || (ctrl && matches!(key.code, KeyCode::Char('j') | KeyCode::Char('m')))
        {
            key.code = KeyCode::Enter;
            key.modifiers.remove(KeyModifiers::CONTROL);
        }
        if self.help_visible {
            self.help_visible = false;
            return;
        }
        if self.filter_editing {
            self.handle_filter_key(key);
            return;
        }
        let ctrl_c = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl_c {
            self.quit = true;
            return;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => self.quit = true,
            KeyCode::Char('?') => self.help_visible = true,
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Runs => Focus::Tags,
                    Focus::Tags => Focus::Chart,
                    Focus::Chart => Focus::Runs,
                }
            }
            KeyCode::BackTab => {
                self.focus = match self.focus {
                    Focus::Runs => Focus::Chart,
                    Focus::Tags => Focus::Runs,
                    Focus::Chart => Focus::Tags,
                }
            }
            KeyCode::Char('b') => self.sidebar = !self.sidebar,
            KeyCode::Char('g') => self.grid = !self.grid,
            KeyCode::Char('f') => {
                self.follow = !self.follow;
                self.flash(if self.follow { "follow on" } else { "follow off" });
                if self.follow {
                    self.reload_requested = true;
                }
            }
            KeyCode::Char('r') => {
                self.reload_requested = true;
                self.flash("reloading…");
            }
            KeyCode::Char('L') => self.log_y = !self.log_y,
            KeyCode::Char('x') => {
                self.xmode = self.xmode.next();
                self.view = (0.0, 1.0);
                self.cursor = None;
            }
            KeyCode::Char('s') => self.smoothing = ((self.smoothing - 0.05) * 100.0).round() / 100.0,
            KeyCode::Char('S') => self.smoothing = ((self.smoothing + 0.05) * 100.0).round() / 100.0,
            KeyCode::Char('a') => {
                if self.disabled.is_empty() {
                    self.disabled = run_names.iter().cloned().collect();
                } else {
                    self.disabled.clear();
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => self.zoom(ZOOM_FACTOR),
            KeyCode::Char('-') | KeyCode::Char('_') => self.zoom(1.0 / ZOOM_FACTOR),
            KeyCode::Char('[') => self.pan(-PAN_FRACTION),
            KeyCode::Char(']') => self.pan(PAN_FRACTION),
            KeyCode::Char('0') => {
                self.view = (0.0, 1.0);
                self.cursor = None;
            }
            KeyCode::Char('/') => {
                self.filter_editing = true;
                self.filter_backup = self.filter_text.clone();
                self.focus = Focus::Tags;
            }
            _ => match self.focus {
                Focus::Runs => self.handle_runs_key(key, run_names),
                Focus::Tags => self.handle_tags_key(key, tags),
                Focus::Chart => self.handle_chart_key(key, tags),
            },
        }
        self.smoothing = self.smoothing.clamp(0.0, 0.99);
    }

    fn handle_filter_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                self.filter_editing = false;
                self.tag_sel = 0;
            }
            KeyCode::Esc => {
                self.filter_text = self.filter_backup.clone();
                self.filter_editing = false;
            }
            KeyCode::Backspace => {
                self.filter_text.pop();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.filter_text.push(c);
                self.tag_sel = 0;
            }
            _ => {}
        }
    }

    fn handle_runs_key(&mut self, key: KeyEvent, run_names: &[String]) {
        if run_names.is_empty() {
            return;
        }
        match key.code {
            KeyCode::Char('j') | KeyCode::Down => {
                self.run_sel = (self.run_sel + 1).min(run_names.len() - 1)
            }
            KeyCode::Char('k') | KeyCode::Up => self.run_sel = self.run_sel.saturating_sub(1),
            KeyCode::Home => self.run_sel = 0,
            KeyCode::End => self.run_sel = run_names.len() - 1,
            KeyCode::Char(' ') | KeyCode::Enter => {
                let name = &run_names[self.run_sel];
                if !self.disabled.remove(name) {
                    self.disabled.insert(name.clone());
                }
            }
            _ => {}
        }
    }

    fn handle_tags_key(&mut self, key: KeyEvent, tags: &[String]) {
        match key.code {
            KeyCode::Char('j') | KeyCode::Down if !tags.is_empty() => {
                self.tag_sel = (self.tag_sel + 1).min(tags.len() - 1)
            }
            KeyCode::Char('k') | KeyCode::Up => self.tag_sel = self.tag_sel.saturating_sub(1),
            KeyCode::Home => self.tag_sel = 0,
            KeyCode::End if !tags.is_empty() => self.tag_sel = tags.len() - 1,
            KeyCode::Enter => {
                self.grid = false;
                self.focus = Focus::Chart;
            }
            _ => {}
        }
    }

    fn handle_chart_key(&mut self, key: KeyEvent, tags: &[String]) {
        match key.code {
            KeyCode::Char('j') | KeyCode::Down if !tags.is_empty() => {
                self.tag_sel = (self.tag_sel + 1).min(tags.len() - 1)
            }
            KeyCode::Char('k') | KeyCode::Up => self.tag_sel = self.tag_sel.saturating_sub(1),
            KeyCode::Char('h') | KeyCode::Left => self.move_cursor(-1.0),
            KeyCode::Char('l') | KeyCode::Right => self.move_cursor(1.0),
            KeyCode::Char('c') | KeyCode::Esc => self.cursor = None,
            _ => {}
        }
    }

    fn move_cursor(&mut self, direction: f64) {
        self.cursor = Some(match self.cursor {
            None => 0.5,
            Some(c) => (c + direction * 0.02).clamp(0.0, 1.0),
        });
    }

    fn zoom(&mut self, factor: f64) {
        let (lo, hi) = self.view;
        let span = hi - lo;
        let center = lo + span * self.cursor.unwrap_or(0.5);
        let new_span = (span * factor).clamp(MIN_SPAN, 1.0);
        let frac = if span > 0.0 { (center - lo) / span } else { 0.5 };
        let mut new_lo = center - new_span * frac;
        new_lo = new_lo.clamp(0.0, 1.0 - new_span);
        self.view = (new_lo, new_lo + new_span);
    }

    fn pan(&mut self, fraction: f64) {
        let (lo, hi) = self.view;
        let span = hi - lo;
        let new_lo = (lo + span * fraction).clamp(0.0, 1.0 - span);
        self.view = (new_lo, new_lo + span);
    }
}
