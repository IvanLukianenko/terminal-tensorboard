//! ttb — an ultra-fast terminal UI for TensorBoard training logs.

mod app;
mod colors;
mod gen;
mod plot;
mod store;
mod tfevents;
mod ui;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use crossterm::event::{self, Event};
use ratatui::prelude::CrosstermBackend;
use ratatui::Terminal;

use app::{App, XMode};
use store::Store;

const USAGE: &str = "\
ttb — an ultra-fast terminal UI for TensorBoard scalar logs

USAGE:
    ttb LOGDIR [OPTIONS]
    ttb gen-demo DIR [--steps N] [--live]

OPTIONS:
    --refresh SEC     poll interval for live tailing (default: 2.0)
    --no-follow       do not tail files for new data
    --smoothing W     initial EMA smoothing weight, 0..0.99 (default: 0.6)
    --x MODE          initial x axis: step | reltime | wall (default: step)
    --max-runs N      runs shown by default; the rest start hidden and can be
                      switched on in the sidebar. 0 shows every run (default: 8)
    --max-points N    points kept per run+tag; past this the series is thinned
                      to an even subsample. 0 keeps every point (default: 100000)
    -h, --help        show this help
    -V, --version     show version
";

struct Args {
    logdir: PathBuf,
    refresh: f64,
    follow: bool,
    smoothing: f64,
    xmode: XMode,
    max_runs: usize,
    max_points: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut logdir: Option<PathBuf> = None;
    let mut refresh = 2.0f64;
    let mut follow = true;
    let mut smoothing = 0.6f64;
    let mut xmode = XMode::Step;
    // Eight is the number of hues the palette gives before they repeat, and
    // more than eight overlaid series is already hard to read.
    let mut max_runs = 8usize;
    // Far more than a terminal can resolve — a 400-column chart still gets
    // 250 points per column at this cap — so thinning stays invisible while
    // bounding what one very long run can cost.
    let mut max_points = 100_000usize;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{}", USAGE);
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("ttb {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--no-follow" => follow = false,
            "--refresh" => {
                refresh = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--refresh needs a number")?;
            }
            "--smoothing" => {
                smoothing = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--smoothing needs a number")?;
            }
            "--max-runs" => {
                max_runs = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--max-runs needs a whole number (0 = all)")?;
            }
            "--max-points" => {
                max_points = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--max-points needs a whole number (0 = keep all)")?;
            }
            "--x" => {
                xmode = match args.next().as_deref() {
                    Some("step") => XMode::Step,
                    Some("reltime") => XMode::RelTime,
                    Some("wall") => XMode::Wall,
                    _ => return Err("--x must be step, reltime or wall".into()),
                };
            }
            s if s.starts_with('-') => return Err(format!("unknown option: {}", s)),
            _ => {
                if logdir.is_some() {
                    return Err(format!("unexpected argument: {}", arg));
                }
                logdir = Some(PathBuf::from(arg));
            }
        }
    }
    let logdir = logdir.ok_or("missing LOGDIR")?;
    if !logdir.is_dir() {
        return Err(format!("not a directory: {}", logdir.display()));
    }
    Ok(Args { logdir, refresh: refresh.max(0.2), follow, smoothing, xmode, max_runs, max_points })
}

fn main() {
    // gen-demo subcommand
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("gen-demo") {
        let dir = PathBuf::from(argv.get(2).map(String::as_str).unwrap_or("demo_logs"));
        let mut steps = 5000i64;
        let mut live = false;
        let mut i = 3;
        while i < argv.len() {
            match argv[i].as_str() {
                "--steps" => {
                    steps = argv.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(steps);
                    i += 1;
                }
                "--live" => live = true,
                _ => {}
            }
            i += 1;
        }
        if let Err(e) = gen::run(&dir, steps, live) {
            eprintln!("gen-demo failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // bench subcommand: time a cold load + a steady-state refresh tick
    if argv.get(1).map(String::as_str) == Some("bench") {
        let dir = PathBuf::from(argv.get(2).map(String::as_str).unwrap_or("."));
        let cap = argv
            .iter()
            .position(|a| a == "--max-points")
            .and_then(|i| argv.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0usize);
        let t0 = std::time::Instant::now();
        let mut store = Store::new(&dir, cap);
        store.refresh();
        let cold = t0.elapsed();
        let t0 = std::time::Instant::now();
        store.refresh();
        let tick = t0.elapsed();
        let stored: usize =
            store.runs.values().flat_map(|r| r.series.values()).map(|s| s.len()).sum();
        println!(
            "cold load: {} points in {:.0?} ({:.1}M pts/s); steady-state tick: {:.0?}",
            store.total_points,
            cold,
            store.total_points as f64 / cold.as_secs_f64() / 1e6,
            tick,
        );
        println!(
            "stored: {} points (cap {}/series, thinning ÷{})",
            stored, cap, store.max_stride
        );
        // render prep (bucketize + smooth) for the largest series, to 400 pixel columns
        if let Some((len, dur)) = store
            .runs
            .values()
            .flat_map(|r| r.series.values())
            .max_by_key(|s| s.len())
            .map(|s| {
                let t0 = std::time::Instant::now();
                let reps = 100;
                for _ in 0..reps {
                    let mut pts = plot::bucketize(
                        s.len(),
                        |i| s.steps[i] as f64,
                        &s.vals,
                        s.steps[0] as f64,
                        s.steps[s.len() - 1] as f64,
                        400,
                    );
                    plot::ema_smooth(&mut pts, 0.6);
                    std::hint::black_box(&pts);
                }
                (s.len(), t0.elapsed() / reps)
            })
        {
            println!("render prep for {} pts -> 400 cols: {:.2?}/frame", len, dur);
        }
        return;
    }

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}\n\n{}", e, USAGE);
            std::process::exit(2);
        }
    };
    if let Err(e) = run_tui(args) {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn run_tui(args: Args) -> std::io::Result<()> {
    let store = Arc::new(Mutex::new(Store::new(&args.logdir, args.max_points)));
    let follow_flag = Arc::new(AtomicBool::new(args.follow));
    let loaded_flag = Arc::new(AtomicBool::new(false));
    let busy_flag = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let (wake_tx, wake_rx) = mpsc::channel::<()>();

    // background loader: initial ingest + live tailing
    let loader = {
        let store = Arc::clone(&store);
        let follow = Arc::clone(&follow_flag);
        let loaded = Arc::clone(&loaded_flag);
        let busy = Arc::clone(&busy_flag);
        let stop = Arc::clone(&stop_flag);
        let interval = Duration::from_secs_f64(args.refresh);
        std::thread::spawn(move || {
            let mut first = true;
            loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                if first || follow.load(Ordering::Relaxed) {
                    busy.store(true, Ordering::Relaxed);
                    // Discovery needs the store, so it takes the lock; each
                    // file is then read and parsed with the lock released and
                    // merged in a short critical section. The UI stays
                    // responsive through a long load, and the charts fill in
                    // run by run instead of appearing all at once.
                    let pending = store.lock().unwrap().pending_files();
                    if first && !pending.is_empty() {
                        // Runs and tags are known now; let the UI draw them
                        // while their points are still being read.
                        loaded.store(true, Ordering::Relaxed);
                        first = false;
                    }
                    for p in &pending {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        if let Some(batch) = store::load_file(p) {
                            store.lock().unwrap().merge(batch);
                        }
                    }
                    busy.store(false, Ordering::Relaxed);
                    if first {
                        loaded.store(true, Ordering::Relaxed);
                        first = false;
                    }
                }
                // sleep until the next poll or an explicit wake ('r' / follow toggle)
                match wake_rx.recv_timeout(interval) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
        })
    };

    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = event_loop(
        &mut terminal,
        &store,
        &follow_flag,
        &loaded_flag,
        &busy_flag,
        &wake_tx,
        args,
    );

    stop_flag.store(true, Ordering::Relaxed);
    let _ = wake_tx.send(());
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), crossterm::terminal::LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    drop(loader);
    result
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    store: &Arc<Mutex<Store>>,
    follow_flag: &AtomicBool,
    loaded_flag: &AtomicBool,
    busy_flag: &AtomicBool,
    wake_tx: &mpsc::Sender<()>,
    args: Args,
) -> std::io::Result<()> {
    let mut app = App::new(args.follow, args.smoothing, args.xmode, args.max_runs);
    let palette = colors::Palette::detect();
    let mut last_version = u64::MAX;
    let mut dirty = true;

    loop {
        app.loaded = loaded_flag.load(Ordering::Relaxed);
        {
            let s = store.lock().unwrap();
            if s.version != last_version {
                last_version = s.version;
                dirty = true;
            }
            let run_names = s.run_names();
            drop(s);
            let before = app.capped_notice;
            app.sync_runs(&run_names);
            if before.is_none() {
                if let Some(total) = app.capped_notice {
                    app.flash(&format!(
                        "{} runs found — showing the first {}. Shift-Tab for RUNS, then Space to add, / to filter",
                        total, app.max_runs
                    ));
                    dirty = true;
                }
            }
        }
        if app.loaded && dirty {
            let ui_state = ui::UiState { palette: palette.clone(), busy: busy_flag.load(Ordering::Relaxed) };
            let s = store.lock().unwrap();
            terminal.draw(|f| ui::draw(f, &mut app, &s, &ui_state))?;
            dirty = false;
        } else if !app.loaded {
            let ui_state = ui::UiState { palette: palette.clone(), busy: true };
            let s = store.lock().unwrap();
            terminal.draw(|f| ui::draw(f, &mut app, &s, &ui_state))?;
        }

        if !event::poll(Duration::from_millis(150))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != event::KeyEventKind::Release => {
                let (run_names, tags) = {
                    let s = store.lock().unwrap();
                    (s.run_names(), ui::visible_tags(&s, &app))
                };
                app.handle_key(key, &run_names, &tags);
                if app.reload_requested {
                    app.reload_requested = false;
                    let _ = wake_tx.send(());
                }
                follow_flag.store(app.follow, Ordering::Relaxed);
                if app.quit {
                    return Ok(());
                }
                dirty = true;
            }
            Event::Resize(_, _) => dirty = true,
            _ => {}
        }
    }
}
