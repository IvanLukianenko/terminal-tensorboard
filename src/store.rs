//! Run discovery and incremental scalar storage.
//!
//! A *run* is any directory under the log dir containing tfevents files
//! (TensorBoard's convention).  Every refresh reads only the bytes appended
//! since the previous one, so tailing a live training run costs microseconds.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::tfevents;

#[derive(Default)]
pub struct Series {
    pub steps: Vec<i64>,
    pub walls: Vec<f64>,
    pub vals: Vec<f64>,
    /// Set when an appended step went backwards; phrased so that the
    /// derived `Default` (an empty, trivially sorted series) is correct.
    unsorted: bool,
}

impl Series {
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    fn push(&mut self, step: i64, wall: f64, val: f64) {
        if let Some(&last) = self.steps.last() {
            if step < last {
                self.unsorted = true;
            }
        }
        self.steps.push(step);
        self.walls.push(wall);
        self.vals.push(val);
    }

    /// Append another series' columns onto this one.
    fn absorb(&mut self, mut other: Series) {
        if let (Some(&last), Some(&first)) = (self.steps.last(), other.steps.first()) {
            if first < last {
                self.unsorted = true;
            }
        }
        self.unsorted |= other.unsorted;
        self.steps.append(&mut other.steps);
        self.walls.append(&mut other.walls);
        self.vals.append(&mut other.vals);
    }

    /// Sort by step (rare path: only after a restarted/overlapping run).
    fn ensure_sorted(&mut self) {
        if !self.unsorted {
            return;
        }
        let mut order: Vec<usize> = (0..self.steps.len()).collect();
        order.sort_by_key(|&i| self.steps[i]);
        self.steps = order.iter().map(|&i| self.steps[i]).collect();
        self.walls = order.iter().map(|&i| self.walls[i]).collect();
        self.vals = order.iter().map(|&i| self.vals[i]).collect();
        self.unsorted = false;
    }
}

struct EventFile {
    offset: u64,
    dead: bool,
}

pub struct Run {
    files: BTreeMap<PathBuf, EventFile>,
    pub series: BTreeMap<String, Series>,
    pub first_wall: Option<f64>,
    /// Categorical colour slot, handed out once when the run is discovered.
    /// Never recomputed, so a run keeps its colour as other runs appear or
    /// are toggled off.
    pub color_slot: usize,
}

pub struct Store {
    pub logdir: PathBuf,
    pub runs: BTreeMap<String, Run>,
    /// Next colour slot to hand out; only ever increases.
    next_color_slot: usize,
    /// Bumped on every data change; the UI redraws when it moves.
    pub version: u64,
    pub total_points: u64,
    pub errors: Vec<String>,
}

/// A file with unread bytes, and where to resume.
pub struct Pending {
    run: String,
    path: PathBuf,
    offset: u64,
}

/// One file's worth of parsed points, ready to be merged.
pub struct Batch {
    run: String,
    path: PathBuf,
    new_offset: u64,
    corrupt: bool,
    error: Option<String>,
    series: BTreeMap<String, Series>,
    first_wall: Option<f64>,
    count: u64,
}

/// Read and parse the appended bytes of one file. Touches no shared state,
/// so it runs with the store unlocked.
pub fn load_file(pending: &Pending) -> Option<Batch> {
    let data = match read_from(&pending.path, pending.offset) {
        Ok(d) => d,
        Err(e) => {
            return Some(Batch {
                run: pending.run.clone(),
                path: pending.path.clone(),
                new_offset: pending.offset,
                corrupt: true, // unreadable: stop trying this file
                error: Some(format!("{}: {}", pending.path.display(), e)),
                series: BTreeMap::new(),
                first_wall: None,
                count: 0,
            })
        }
    };
    if data.is_empty() {
        return None;
    }
    let mut series: BTreeMap<String, Series> = BTreeMap::new();
    let mut first_wall: Option<f64> = None;
    let mut count = 0u64;
    let result = tfevents::parse_chunk(&data, &mut |tag, step, wall, val| {
        // Lookup by &str so a known tag costs no allocation.
        let tag_str = std::str::from_utf8(tag).unwrap_or("<invalid-utf8>");
        let s = match series.get_mut(tag_str) {
            Some(s) => s,
            None => series.entry(tag_str.to_string()).or_default(),
        };
        s.push(step, wall, val);
        if wall != 0.0 && first_wall.is_none_or(|w| wall < w) {
            first_wall = Some(wall);
        }
        count += 1;
    });
    Some(Batch {
        run: pending.run.clone(),
        path: pending.path.clone(),
        new_offset: pending.offset + result.consumed as u64,
        corrupt: result.corrupt,
        error: result.corrupt.then(|| format!("{}: corrupt record", pending.path.display())),
        series,
        first_wall,
        count,
    })
}

fn is_event_file(name: &str) -> bool {
    name.contains("tfevents") && !name.ends_with(".profile-empty")
}

impl Store {
    pub fn new(logdir: &Path) -> Self {
        Store {
            logdir: logdir.to_path_buf(),
            runs: BTreeMap::new(),
            next_color_slot: 0,
            version: 0,
            total_points: 0,
            errors: Vec::new(),
        }
    }

    fn discover_dir(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        // `read_dir` yields in filesystem order, which is arbitrary. Sort so
        // runs are discovered — and so colour slots are handed out — in name
        // order, the same order the sidebar lists them in.
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() || (ft.is_symlink() && path.is_dir()) {
                self.discover_dir(&path);
            } else if is_event_file(&entry.file_name().to_string_lossy()) {
                let rel = path
                    .parent()
                    .and_then(|p| p.strip_prefix(&self.logdir).ok())
                    .map(|p| p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/"))
                    .unwrap_or_default();
                let run_name = if rel.is_empty() { ".".to_string() } else { rel };
                if !self.runs.contains_key(&run_name) {
                    self.runs.insert(
                        run_name.clone(),
                        Run {
                            files: BTreeMap::new(),
                            series: BTreeMap::new(),
                            first_wall: None,
                            color_slot: self.next_color_slot,
                        },
                    );
                    self.next_color_slot += 1;
                }
                let run = self.runs.get_mut(&run_name).unwrap();
                run.files.entry(path).or_insert(EventFile { offset: 0, dead: false });
            }
        }
    }

    /// Discover runs and list the files with bytes left to read.
    ///
    /// This is the only part of a refresh that needs the store, so callers
    /// hold the lock for it and then parse each file with the lock released.
    pub fn pending_files(&mut self) -> Vec<Pending> {
        let logdir = self.logdir.clone();
        self.discover_dir(&logdir);
        let mut out = Vec::new();
        for (run_name, run) in &self.runs {
            for (path, file) in &run.files {
                if file.dead {
                    continue;
                }
                let Ok(meta) = std::fs::metadata(path) else { continue };
                if meta.len() <= file.offset {
                    continue;
                }
                out.push(Pending {
                    run: run_name.clone(),
                    path: path.clone(),
                    offset: file.offset,
                });
            }
        }
        out
    }

    /// Fold one parsed file into the store. Cheap: it moves whole columns
    /// for a new tag and extends them for a known one, so the lock is held
    /// for appends only, never for parsing.
    pub fn merge(&mut self, batch: Batch) -> bool {
        if let Some(e) = batch.error {
            self.errors.push(e);
        }
        let Some(run) = self.runs.get_mut(&batch.run) else { return false };
        for (tag, incoming) in batch.series {
            match run.series.get_mut(&tag) {
                Some(dst) => dst.absorb(incoming),
                None => {
                    run.series.insert(tag, incoming);
                }
            }
        }
        if let Some(w) = batch.first_wall {
            if run.first_wall.is_none_or(|cur| w < cur) {
                run.first_wall = Some(w);
            }
        }
        if let Some(file) = run.files.get_mut(&batch.path) {
            file.offset = batch.new_offset;
            if batch.corrupt {
                file.dead = true;
            }
        }
        if batch.count == 0 {
            return false;
        }
        for series in run.series.values_mut() {
            series.ensure_sorted();
        }
        self.total_points += batch.count;
        self.version += 1;
        true
    }

    /// Convenience wrapper that does a whole refresh in one call, holding
    /// nothing: used by `bench` and the tests, where there is no UI to
    /// starve.
    pub fn refresh(&mut self) -> bool {
        let mut changed = false;
        for pending in self.pending_files() {
            if let Some(batch) = load_file(&pending) {
                changed |= self.merge(batch);
            }
        }
        changed
    }

    pub fn run_names(&self) -> Vec<String> {
        self.runs.keys().cloned().collect()
    }

    pub fn tags(&self, enabled: &std::collections::HashSet<String>) -> Vec<String> {
        let mut tags: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (name, run) in &self.runs {
            if !enabled.contains(name) {
                continue;
            }
            for tag in run.series.keys() {
                if seen.insert(tag.clone()) {
                    tags.push(tag.clone());
                }
            }
        }
        tags.sort();
        tags
    }
}

fn read_from(path: &Path, offset: u64) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tfevents::{encode_scalar_event, frame_record};
    use std::io::Write;

    #[test]
    fn incremental_refresh() {
        let tmp = std::env::temp_dir().join(format!("ttb-test-{}", std::process::id()));
        let run_dir = tmp.join("run1");
        std::fs::create_dir_all(&run_dir).unwrap();
        let path = run_dir.join("events.out.tfevents.123.host");
        let mut f = File::create(&path).unwrap();
        f.write_all(&frame_record(&encode_scalar_event("loss", 1, 1.0, 0.9, false))).unwrap();
        f.flush().unwrap();

        let mut store = Store::new(&tmp);
        assert!(store.refresh());
        assert_eq!(store.run_names(), vec!["run1"]);
        assert_eq!(store.runs["run1"].series["loss"].steps, vec![1]);

        f.write_all(&frame_record(&encode_scalar_event("loss", 2, 2.0, 0.8, false))).unwrap();
        f.flush().unwrap();
        assert!(store.refresh());
        assert_eq!(store.runs["run1"].series["loss"].steps, vec![1, 2]);
        assert!(!store.refresh()); // nothing new

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_new_run_does_not_repaint_the_existing_ones() {
        let tmp = std::env::temp_dir().join(format!("ttb-color-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        // "zzz" sorts last now, but "aaa" appearing later must not take its slot
        for name in ["mmm", "zzz"] {
            std::fs::create_dir_all(tmp.join(name)).unwrap();
            let mut f = File::create(tmp.join(name).join("events.out.tfevents.1.h")).unwrap();
            f.write_all(&frame_record(&encode_scalar_event("loss", 1, 1.0, 0.5, false))).unwrap();
        }
        let mut store = Store::new(&tmp);
        store.refresh();
        let (mmm, zzz) = (store.runs["mmm"].color_slot, store.runs["zzz"].color_slot);
        assert_ne!(mmm, zzz, "runs discovered together get different slots");

        // a run appearing later sorts first alphabetically but takes a fresh slot
        std::fs::create_dir_all(tmp.join("aaa")).unwrap();
        let mut f = File::create(tmp.join("aaa").join("events.out.tfevents.1.h")).unwrap();
        f.write_all(&frame_record(&encode_scalar_event("loss", 1, 1.0, 0.5, false))).unwrap();
        f.flush().unwrap();
        store.refresh();
        assert_eq!(store.runs["mmm"].color_slot, mmm, "existing run was repainted");
        assert_eq!(store.runs["zzz"].color_slot, zzz, "existing run was repainted");
        let aaa = store.runs["aaa"].color_slot;
        assert!(aaa != mmm && aaa != zzz, "new run reused a taken slot");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn out_of_order_steps_sorted() {
        let mut s = Series::default();
        for (step, val) in [(5, 0.5), (1, 0.1), (3, 0.3)] {
            s.push(step, step as f64, val);
        }
        s.ensure_sorted();
        assert_eq!(s.steps, vec![1, 3, 5]);
        assert_eq!(s.vals, vec![0.1, 0.3, 0.5]);
    }
}
