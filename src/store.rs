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
    /// Thinning: one point in every `1 << stride_shift` offered is stored.
    /// A shift, not a stride, so the derived `Default` means "keep all".
    stride_shift: u32,
    /// Points offered to this series so far, kept or dropped. Thinning keys
    /// off this running index rather than the stored length, which is what
    /// keeps the retained sample even across the whole series instead of
    /// leaving the old part sparser than the new.
    offered: u64,
}

impl Series {
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// One stored point per `stride()` recorded; 1 while nothing was thinned.
    pub fn stride(&self) -> u64 {
        1u64 << self.stride_shift
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

    /// Offer a point, storing it only if the current stride selects it, and
    /// thinning further the moment `cap` is passed. Applied while parsing, so
    /// a series never holds more than the cap even transiently.
    fn offer(&mut self, step: i64, wall: f64, val: f64, cap: usize) {
        if self.offered & (self.stride() - 1) == 0 {
            self.push(step, wall, val);
            if cap > 0 && self.steps.len() > cap {
                self.halve();
            }
        }
        self.offered += 1;
    }

    /// Append another series' points, thinning to stay under `cap`
    /// (0 = keep everything).
    /// Append a batch that continued this series' offered-numbering.
    ///
    /// The batch may have thinned further than this series did while it was
    /// being parsed, so bring the coarser stride to both sides first: both
    /// hold points at offered-indices that are multiples of their stride, so
    /// halving the finer one lines them up exactly.
    fn absorb(&mut self, mut other: Series, cap: usize) {
        if let (Some(&last), Some(&first)) = (self.steps.last(), other.steps.first()) {
            if first < last {
                self.unsorted = true;
            }
        }
        self.unsorted |= other.unsorted;
        while self.stride_shift < other.stride_shift {
            self.halve();
        }
        while other.stride_shift < self.stride_shift {
            other.halve();
        }
        self.steps.append(&mut other.steps);
        self.walls.append(&mut other.walls);
        self.vals.append(&mut other.vals);
        self.offered = self.offered.max(other.offered);
        if cap > 0 {
            while self.steps.len() > cap {
                self.halve();
            }
        }
    }

    /// Start a batch that carries on from this series' thinning state, so the
    /// points it keeps line up with the ones already stored.
    fn continuation(&self) -> Series {
        Series { stride_shift: self.stride_shift, offered: self.offered, ..Default::default() }
    }

    /// Drop every second stored point, doubling the stride.
    ///
    /// Stored points sit at offered-indices that are multiples of the old
    /// stride, so keeping the even ones leaves exactly the multiples of the
    /// new one — the sample stays uniform, and the first point (and with it
    /// the start of the x range) is always kept.
    fn halve(&mut self) {
        let mut keep = 0usize;
        for i in 0..self.steps.len() {
            if i % 2 == 0 {
                self.steps[keep] = self.steps[i];
                self.walls[keep] = self.walls[i];
                self.vals[keep] = self.vals[i];
                keep += 1;
            }
        }
        self.steps.truncate(keep);
        self.walls.truncate(keep);
        self.vals.truncate(keep);
        self.stride_shift += 1;
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
    /// Points kept per run+tag before thinning kicks in; 0 = keep all.
    max_points: usize,
    /// Coarsest thinning in force anywhere, so the UI can say the curves
    /// are a subsample. 1 while nothing has been thinned.
    pub max_stride: u64,
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
    /// Thinning state of the destination series, so the points this file
    /// keeps continue the same even sample rather than restarting it.
    seed: BTreeMap<String, Series>,
    cap: usize,
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

/// Bytes pulled from a file per parsing round. Bounds the read buffer so a
/// huge event file does not have to be resident to be read.
const READ_CHUNK: usize = 4 << 20;

/// Read and parse the appended bytes of one file. Touches no shared state,
/// so it runs with the store unlocked.
///
/// Reads in slices and thins as it goes, so neither the raw bytes nor the
/// parsed points of one file exceed their bounds however large the file is.
pub fn load_file(pending: &Pending) -> Option<Batch> {
    let fail = |e: std::io::Error| {
        Some(Batch {
            run: pending.run.clone(),
            path: pending.path.clone(),
            new_offset: pending.offset,
            corrupt: true, // unreadable: stop trying this file
            error: Some(format!("{}: {}", pending.path.display(), e)),
            series: BTreeMap::new(),
            first_wall: None,
            count: 0,
        })
    };
    let mut file = match File::open(&pending.path) {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    if let Err(e) = file.seek(SeekFrom::Start(pending.offset)) {
        return fail(e);
    }

    let cap = pending.cap;
    let mut series: BTreeMap<String, Series> = BTreeMap::new();
    let mut first_wall: Option<f64> = None;
    let mut count = 0u64;
    let mut consumed_total = 0usize;
    let mut corrupt = false;
    // Size the buffers to what is actually left to read: allocating (and
    // zeroing) a full-size chunk per file dominated the cost on a directory
    // of many small files.
    let remaining = file
        .metadata()
        .map(|m| m.len().saturating_sub(pending.offset) as usize)
        .unwrap_or(READ_CHUNK);
    let chunk_len = remaining.clamp(1, READ_CHUNK);
    let mut buf: Vec<u8> = Vec::with_capacity(chunk_len);
    let mut chunk = vec![0u8; chunk_len];

    loop {
        let n = match file.read(&mut chunk) {
            Ok(0) => 0,
            Ok(n) => n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return fail(e),
        };
        if n > 0 {
            buf.extend_from_slice(&chunk[..n]);
        }
        let result = tfevents::parse_chunk(&buf, &mut |tag, step, wall, val| {
            let tag_str = std::str::from_utf8(tag).unwrap_or("<invalid-utf8>");
            let s = match series.get_mut(tag_str) {
                Some(s) => s,
                None => series.entry(tag_str.to_string()).or_insert_with(|| {
                    pending.seed.get(tag_str).map_or_else(Series::default, |d| d.continuation())
                }),
            };
            s.offer(step, wall, val, cap);
            if wall != 0.0 && first_wall.is_none_or(|w| wall < w) {
                first_wall = Some(wall);
            }
            count += 1;
        });
        consumed_total += result.consumed;
        buf.drain(..result.consumed);
        if result.corrupt {
            corrupt = true;
            break;
        }
        if n == 0 {
            break; // end of file; anything left in buf is a partial record
        }
    }

    if consumed_total == 0 && !corrupt {
        return None;
    }
    Some(Batch {
        run: pending.run.clone(),
        path: pending.path.clone(),
        new_offset: pending.offset + consumed_total as u64,
        corrupt,
        error: corrupt.then(|| format!("{}: corrupt record", pending.path.display())),
        series,
        first_wall,
        count,
    })
}

fn is_event_file(name: &str) -> bool {
    name.contains("tfevents") && !name.ends_with(".profile-empty")
}

impl Store {
    pub fn new(logdir: &Path, max_points: usize) -> Self {
        Store {
            logdir: logdir.to_path_buf(),
            runs: BTreeMap::new(),
            max_points,
            max_stride: 1,
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
                    seed: run
                        .series
                        .iter()
                        .map(|(tag, s)| (tag.clone(), s.continuation()))
                        .collect(),
                    cap: self.max_points,
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
        let cap = self.max_points;
        let Some(run) = self.runs.get_mut(&batch.run) else { return false };
        let mut stride = 1u64;
        for (tag, incoming) in batch.series {
            match run.series.get_mut(&tag) {
                Some(dst) => dst.absorb(incoming, cap),
                None => {
                    run.series.insert(tag.clone(), incoming);
                }
            }
            stride = stride.max(run.series[&tag].stride());
        }
        self.max_stride = self.max_stride.max(stride);
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

        let mut store = Store::new(&tmp, 0);
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
        let mut store = Store::new(&tmp, 0);
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

    /// Feed `n` points through `absorb` the way merges do, in chunks, so the
    /// test exercises the same path a live run takes.
    fn fill(cap: usize, n: i64, chunk: i64) -> Series {
        let mut dst = Series::default();
        let mut step = 0;
        while step < n {
            // exactly what load_file does: the batch continues the
            // destination's thinning state instead of restarting it
            let mut batch = dst.continuation();
            for s in step..(step + chunk).min(n) {
                batch.offer(s, s as f64, s as f64 * 2.0, cap);
            }
            dst.absorb(batch, cap);
            step += chunk;
        }
        dst
    }

    #[test]
    fn a_series_never_exceeds_its_cap() {
        let s = fill(1000, 100_000, 997);
        assert!(s.len() <= 1000, "kept {} points, cap was 1000", s.len());
        assert!(s.len() > 500, "thinned too far: {}", s.len());
    }

    #[test]
    fn cap_zero_keeps_every_point() {
        let s = fill(0, 50_000, 1000);
        assert_eq!(s.len(), 50_000);
        assert_eq!(s.stride(), 1);
    }

    #[test]
    fn thinning_keeps_an_even_sample_over_the_whole_range() {
        let s = fill(1000, 64_000, 1000);
        // stride is a power of two and the kept points are exactly the steps
        // that are multiples of it — uniform from the first point to the last
        let stride = s.stride();
        assert!(stride > 1, "nothing was thinned");
        assert_eq!(s.steps[0], 0, "the start of the range must survive");
        for w in s.steps.windows(2) {
            assert_eq!(
                (w[1] - w[0]) as u64,
                stride,
                "gap {} != stride {} — the sample is not even",
                w[1] - w[0],
                stride
            );
        }
        assert!(s.steps[s.len() - 1] as u64 >= 64_000 - stride, "the end of the range was lost");
    }

    #[test]
    fn thinning_subsamples_and_never_invents_values() {
        let s = fill(500, 20_000, 512);
        for (i, (&step, &val)) in s.steps.iter().zip(s.vals.iter()).enumerate() {
            assert_eq!(val, step as f64 * 2.0, "value at {} was not an original", i);
            assert_eq!(s.walls[i], step as f64);
        }
    }

    #[test]
    fn thinning_survives_a_chunk_size_that_is_not_a_power_of_two() {
        // chunks land mid-stride, which is where an index-based scheme would
        // drift; the running offered-counter is what keeps it even
        let s = fill(300, 10_000, 37);
        let stride = s.stride();
        for w in s.steps.windows(2) {
            assert_eq!((w[1] - w[0]) as u64, stride);
        }
    }

    #[test]
    fn a_live_tail_stays_as_evenly_sampled_as_a_cold_load() {
        // tiny appends onto an already-thinned series are where a naive
        // scheme leaves the tail denser than the body
        let live = fill(500, 20_000, 7);
        let cold = fill(500, 20_000, 20_000);
        assert_eq!(live.stride(), cold.stride());
        for w in live.steps.windows(2) {
            assert_eq!((w[1] - w[0]) as u64, live.stride(), "the tail is denser than the body");
        }
        assert_eq!(live.steps, cold.steps, "append size changed which points were kept");
    }

    #[test]
    fn thinning_applies_while_parsing_not_only_afterwards() {
        // a batch parsed against an empty destination must cap itself, so a
        // huge file never lands in memory whole
        let mut batch = Series::default();
        for step in 0..100_000i64 {
            batch.offer(step, step as f64, 1.0, 1000);
        }
        assert!(batch.len() <= 1000, "batch grew to {}", batch.len());
        assert!(batch.stride() > 1);
    }

    #[test]
    fn store_reports_the_thinning_in_force() {
        let tmp = std::env::temp_dir().join(format!("ttb-thin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("r")).unwrap();
        let mut f = File::create(tmp.join("r").join("events.out.tfevents.1.h")).unwrap();
        for step in 0..5000 {
            f.write_all(&frame_record(&encode_scalar_event("loss", step, 1.0, 0.5, false)))
                .unwrap();
        }
        f.flush().unwrap();

        let mut store = Store::new(&tmp, 1000);
        store.refresh();
        assert_eq!(store.total_points, 5000, "every point is still read");
        assert!(store.runs["r"].series["loss"].len() <= 1000);
        assert!(store.max_stride > 1, "thinning was not reported");

        let mut unlimited = Store::new(&tmp, 0);
        unlimited.refresh();
        assert_eq!(unlimited.runs["r"].series["loss"].len(), 5000);
        assert_eq!(unlimited.max_stride, 1);

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
