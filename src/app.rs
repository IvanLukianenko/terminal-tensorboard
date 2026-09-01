//! Application state and key handling.

use std::collections::HashSet;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub const ZOOM_FACTOR: f64 = 0.7;
pub const PAN_FRACTION: f64 = 0.15;
pub const MIN_SPAN: f64 = 1e-4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    /// Runs seen at least once, so the default cap is applied to each run
    /// exactly once and never fights a choice the user has made since.
    known_runs: HashSet<String>,
    /// How many runs are enabled by default; 0 means "all of them".
    pub max_runs: usize,
    /// Set once, when the cap first hides something, to explain it.
    pub capped_notice: Option<usize>,
    pub filter_text: String,
    pub run_filter_text: String,
    pub filter_editing: bool,
    /// Which list `/` is filtering — Runs or Tags.
    pub filter_target: Focus,
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
    pub fn new(follow: bool, smoothing: f64, xmode: XMode, max_runs: usize) -> Self {
        App {
            focus: Focus::Tags,
            run_sel: 0,
            tag_sel: 0,
            run_scroll: 0,
            tag_scroll: 0,
            disabled: HashSet::new(),
            known_runs: HashSet::new(),
            max_runs,
            capped_notice: None,
            filter_text: String::new(),
            run_filter_text: String::new(),
            filter_editing: false,
            filter_target: Focus::Tags,
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

    /// Give newly discovered runs their default on/off state: the first
    /// `max_runs` are shown, the rest start hidden so 300 runs do not land on
    /// the chart at once. Applied to each run once, so it never overrides a
    /// choice the user made later.
    pub fn sync_runs(&mut self, run_names: &[String]) {
        if self.max_runs == 0 || run_names.len() == self.known_runs.len() {
            return;
        }
        let mut enabled = run_names
            .iter()
            .filter(|n| self.known_runs.contains(*n) && !self.disabled.contains(*n))
            .count();
        let mut hidden = 0;
        for name in run_names {
            if self.known_runs.contains(name) {
                continue;
            }
            self.known_runs.insert(name.clone());
            if enabled >= self.max_runs {
                self.disabled.insert(name.clone());
                hidden += 1;
            } else {
                enabled += 1;
            }
        }
        if hidden > 0 && self.capped_notice.is_none() {
            self.capped_notice = Some(run_names.len());
        }
    }

    /// Run names matching the current run filter, in list order.
    pub fn visible_runs(&self, all: &[String]) -> Vec<String> {
        if self.run_filter_text.is_empty() {
            return all.to_vec();
        }
        let needle = self.run_filter_text.to_lowercase();
        all.iter().filter(|n| n.to_lowercase().contains(&needle)).cloned().collect()
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
                // Scoped to the runs the filter is showing, so "/lr → a"
                // enables exactly that group.
                let listed = self.visible_runs(run_names);
                if listed.iter().any(|n| self.disabled.contains(n)) {
                    for n in &listed {
                        self.disabled.remove(n);
                    }
                    self.flash(&format!("showing {} run(s)", listed.len()));
                } else {
                    for n in &listed {
                        self.disabled.insert(n.clone());
                    }
                    self.flash(&format!("hid {} run(s)", listed.len()));
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
                if self.focus == Focus::Runs {
                    self.filter_target = Focus::Runs;
                    self.filter_backup = self.run_filter_text.clone();
                } else {
                    self.filter_target = Focus::Tags;
                    self.filter_backup = self.filter_text.clone();
                    self.focus = Focus::Tags;
                }
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
        let runs = self.filter_target == Focus::Runs;
        match key.code {
            KeyCode::Enter => {
                self.filter_editing = false;
                if runs {
                    self.run_sel = 0;
                } else {
                    self.tag_sel = 0;
                }
            }
            KeyCode::Esc => {
                if runs {
                    self.run_filter_text = self.filter_backup.clone();
                } else {
                    self.filter_text = self.filter_backup.clone();
                }
                self.filter_editing = false;
            }
            KeyCode::Backspace => {
                if runs {
                    self.run_filter_text.pop();
                } else {
                    self.filter_text.pop();
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if runs {
                    self.run_filter_text.push(c);
                    self.run_sel = 0;
                } else {
                    self.filter_text.push(c);
                    self.tag_sel = 0;
                }
            }
            _ => {}
        }
    }

    fn handle_runs_key(&mut self, key: KeyEvent, all_runs: &[String]) {
        let run_names = self.visible_runs(all_runs);
        let run_names = run_names.as_slice();
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
            KeyCode::Char(' ') => {
                let name = &run_names[self.run_sel];
                if !self.disabled.remove(name) {
                    self.disabled.insert(name.clone());
                }
            }
            // solo: show only this run
            KeyCode::Enter => {
                let name = run_names[self.run_sel].clone();
                self.disabled = all_runs.iter().filter(|n| **n != name).cloned().collect();
                self.flash(&format!("only {}", name));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("run{:03}", i)).collect()
    }

    fn app(max_runs: usize) -> App {
        App::new(true, 0.6, XMode::Step, max_runs)
    }

    #[test]
    fn only_the_first_n_runs_start_enabled() {
        let runs = names(300);
        let mut a = app(8);
        a.sync_runs(&runs);
        let enabled: Vec<&String> = runs.iter().filter(|n| !a.disabled.contains(*n)).collect();
        assert_eq!(enabled.len(), 8);
        assert_eq!(enabled[0], "run000");
        assert_eq!(enabled[7], "run007");
        assert_eq!(a.capped_notice, Some(300));
    }

    #[test]
    fn max_runs_zero_enables_everything() {
        let runs = names(300);
        let mut a = app(0);
        a.sync_runs(&runs);
        assert!(a.disabled.is_empty());
        assert_eq!(a.capped_notice, None);
    }

    #[test]
    fn the_cap_never_overrides_a_users_choice() {
        let runs = names(20);
        let mut a = app(3);
        a.sync_runs(&runs);
        assert_eq!(runs.iter().filter(|n| !a.disabled.contains(*n)).count(), 3);

        // user switches on five more by hand
        for n in runs.iter().skip(3).take(5) {
            a.disabled.remove(n);
        }
        a.sync_runs(&runs); // nothing new discovered
        assert_eq!(runs.iter().filter(|n| !a.disabled.contains(*n)).count(), 8);

        // a run appearing later starts hidden, and leaves the others alone
        let mut grown = runs.clone();
        grown.push("run999".into());
        a.sync_runs(&grown);
        assert!(a.disabled.contains("run999"));
        assert_eq!(grown.iter().filter(|n| !a.disabled.contains(*n)).count(), 8);
    }

    #[test]
    fn run_filter_narrows_the_list() {
        let runs = vec!["lr_high".to_string(), "lr_low".to_string(), "baseline".to_string()];
        let mut a = app(0);
        a.run_filter_text = "LR_".into(); // case-insensitive
        assert_eq!(a.visible_runs(&runs), vec!["lr_high", "lr_low"]);
        a.run_filter_text.clear();
        assert_eq!(a.visible_runs(&runs).len(), 3);
    }

    #[test]
    fn toggle_all_is_scoped_to_the_filtered_runs() {
        let runs = vec!["lr_high".to_string(), "lr_low".to_string(), "baseline".to_string()];
        let mut a = app(0);
        a.sync_runs(&runs);
        a.disabled = runs.iter().cloned().collect(); // start with all hidden
        a.run_filter_text = "lr_".into();
        a.focus = Focus::Runs;
        a.handle_key(KeyEvent::from(KeyCode::Char('a')), &runs, &[]);
        assert!(!a.disabled.contains("lr_high"));
        assert!(!a.disabled.contains("lr_low"));
        assert!(a.disabled.contains("baseline"), "a run outside the filter was touched");
    }

    #[test]
    fn enter_on_a_run_solos_it() {
        let runs = names(5);
        let mut a = app(0);
        a.sync_runs(&runs);
        a.focus = Focus::Runs;
        a.run_sel = 2;
        a.handle_key(KeyEvent::from(KeyCode::Enter), &runs, &[]);
        let enabled: Vec<&String> = runs.iter().filter(|n| !a.disabled.contains(*n)).collect();
        assert_eq!(enabled, vec!["run002"]);
    }

    #[test]
    fn solo_respects_the_filter_selection_not_the_full_list() {
        let runs = vec!["a1".to_string(), "b1".to_string(), "b2".to_string()];
        let mut a = app(0);
        a.sync_runs(&runs);
        a.focus = Focus::Runs;
        a.run_filter_text = "b".into();
        a.run_sel = 1; // "b2" within the filtered list
        a.handle_key(KeyEvent::from(KeyCode::Enter), &runs, &[]);
        let enabled: Vec<&String> = runs.iter().filter(|n| !a.disabled.contains(*n)).collect();
        assert_eq!(enabled, vec!["b2"]);
    }

    #[test]
    fn slash_filters_whichever_list_has_focus() {
        let runs = names(3);
        let mut a = app(0);
        a.focus = Focus::Runs;
        a.handle_key(KeyEvent::from(KeyCode::Char('/')), &runs, &[]);
        assert!(a.filter_editing);
        assert_eq!(a.filter_target, Focus::Runs);
        for c in "run0".chars() {
            a.handle_key(KeyEvent::from(KeyCode::Char(c)), &runs, &[]);
        }
        a.handle_key(KeyEvent::from(KeyCode::Enter), &runs, &[]);
        assert_eq!(a.run_filter_text, "run0");
        assert!(a.filter_text.is_empty(), "tag filter must be untouched");

        a.focus = Focus::Tags;
        a.handle_key(KeyEvent::from(KeyCode::Char('/')), &runs, &[]);
        assert_eq!(a.filter_target, Focus::Tags);
        a.handle_key(KeyEvent::from(KeyCode::Esc), &runs, &[]);
        assert_eq!(a.run_filter_text, "run0", "run filter must survive a tag-filter cancel");
    }
}
