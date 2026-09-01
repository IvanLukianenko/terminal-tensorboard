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
        for entry in entries.flatten() {
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

    /// Discover new runs/files and ingest appended bytes.  Returns true if
    /// any data changed.
    pub fn refresh(&mut self) -> bool {
        let logdir = self.logdir.clone();
        self.discover_dir(&logdir);

        let mut changed = false;
        let run_names: Vec<String> = self.runs.keys().cloned().collect();
        for run_name in run_names {
            let pending: Vec<(PathBuf, u64)> = {
                let run = &self.runs[&run_name];
                run.files
                    .iter()
                    .filter(|(_, f)| !f.dead)
                    .map(|(p, f)| (p.clone(), f.offset))
                    .collect()
            };
            for (path, offset) in pending {
                let Ok(meta) = std::fs::metadata(&path) else { continue };
                if meta.len() <= offset {
                    continue;
                }
                let data = match read_from(&path, offset) {
                    Ok(d) => d,
                    Err(e) => {
                        self.errors.push(format!("{}: {}", path.display(), e));
                        self.runs.get_mut(&run_name).unwrap().files.get_mut(&path).unwrap().dead =
                            true;
                        continue;
                    }
                };
                let run = self.runs.get_mut(&run_name).unwrap();
                let mut count = 0u64;
                let mut first_wall = run.first_wall;
                let series = &mut run.series;
                let result = tfevents::parse_chunk(&data, &mut |tag, step, wall, val| {
                    // BTreeMap<String> lookup by &str avoids allocating for
                    // existing tags (the overwhelmingly common case).
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
                run.first_wall = first_wall;
                let file = run.files.get_mut(&path).unwrap();
                file.offset = offset + result.consumed as u64;
                if result.corrupt {
                    file.dead = true;
                    self.errors.push(format!("{}: corrupt record", path.display()));
                }
                if count > 0 {
                    self.total_points += count;
                    self.version += 1;
                    changed = true;
                }
            }
        }
        if changed {
            for run in self.runs.values_mut() {
                for series in run.series.values_mut() {
                    series.ensure_sorted();
                }
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
