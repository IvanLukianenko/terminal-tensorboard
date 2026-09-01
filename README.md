# terminal-tensorboard

An ultra-fast terminal UI for viewing TensorBoard training logs, written in
Rust. Point it at your log directory and watch your losses live — over SSH,
in tmux, anywhere you have a terminal. No TensorFlow, no protobuf, no
browser, no Python.

```
 ttb  ~/runs                        runs 3/3 │ tags 5 │ pts 1.1M │ live ● │ x:step │ y:lin │ smooth 0.60
 RUNS                    │ train/loss                                              ● baseline
  ▣ ● baseline           │  2.5┼⢣                                                  ● high_lr
  ▣ ● high_lr            │     │⠘⢆⡀                                                ● low_lr/warmup
  ▣ ● low_lr/warmup      │  1.5┼  ⠑⠢⢄⡀
 TAGS (5)                │     │      ⠉⠒⠒⠤⠤⣀⣀⡀
  ▶ train/loss           │  0.5┼              ⠉⠉⠉⠉⠒⠒⠒⠒⠤⠤⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀
    train/accuracy       │     └0        20k        40k       60k        80k
```

## Speed

Measured on 50 MB of logs — 3 runs × 5 tags × 80 000 steps, **970 000
scalar points** (`ttb bench LOGDIR`, warm page cache):

| | Rust | Python (v0.1) | speedup |
| --- | --- | --- | --- |
| Cold load, 970k points | **84 ms** (11.5M pts/s) | 3 027 ms (0.32M pts/s) | **36×** |
| Live refresh tick | **90 µs** | 250 µs | 2.8× |
| Render prep, 80k pts → 400 columns | **156 µs** | 850 µs | 5.4× |

At 156 µs of frame preparation, redraws are bounded by the terminal, not by
the data — panning and zooming through a million points is instant.

### Why it's fast

- **Zero-copy tfevents parser.** A hand-rolled TFRecord + protobuf-wire-format
  reader walks the mapped bytes and hands out `&[u8]` tag slices — no
  protobuf crate, no allocation per point, no intermediate message structs.
  Field keys are matched as whole bytes (`0x15` = `simple_value`) so the
  common path is a single jump table, and non-scalar payloads (images,
  histograms) are skipped without being decoded.
- **Incremental tailing.** Each file is read once; every refresh parses only
  the bytes appended since the last one. Partially-written trailing records
  are handled correctly and re-read once complete.
- **Compact columnar storage.** Points live in flat `Vec<i64>` / `Vec<f64>`
  columns — 24 bytes per point, cache-friendly for the bucketing scan.
- **Pixel-bucket rendering.** Before drawing, each series is reduced to one
  mean value per braille pixel column: a binary search for the visible range,
  then a single linear pass, so the per-frame cost depends on terminal width
  rather than on run length.
- **Background loading.** Parsing runs on a loader thread behind a mutex the
  UI holds only while drawing, so the interface stays responsive even while
  ingesting gigabytes on first start.
- **Release profile** with fat LTO and a single codegen unit.

## Install

```bash
cargo install --path .        # from a checkout
# or run it directly:
cargo run --release -- ~/runs
```

Requires Rust 1.75+. Linux, macOS and Windows (crossterm handles all three).

## Usage

```bash
ttb LOGDIR                  # scan LOGDIR recursively, follow for new data
ttb LOGDIR --no-follow      # one-shot view
ttb LOGDIR --refresh 0.5    # poll twice a second
ttb LOGDIR --smoothing 0.9  # heavier EMA smoothing
ttb LOGDIR --x reltime      # x axis: step | reltime | wall
ttb bench LOGDIR            # time a cold load, a refresh tick and a frame
```

Every subdirectory containing `tfevents` files becomes a run, exactly like
TensorBoard. Both classic `simple_value` scalars and TF2 tensor-encoded
scalars are understood, whether written by TensorFlow, PyTorch's
`SummaryWriter`, or anything else that speaks the format.

## Keys

| Key | Action |
| --- | --- |
| `Tab` / `Shift-Tab` | cycle focus: runs → tags → chart |
| `j` `k` / arrows | move in lists; prev/next tag when chart is focused |
| `Space` | toggle a run on/off (`a` toggles all) |
| `Enter` | open the selected tag in the chart |
| `/` | filter tags (Enter apply, Esc cancel) |
| `h` `l` / arrows | move the data cursor over the chart (`c`/Esc clears) |
| `+` `-` | zoom in / out (centred on the cursor) |
| `[` `]` | pan left / right, `0` resets the view |
| `s` / `S` | less / more smoothing (TensorBoard-style debiased EMA) |
| `L` | toggle log-scale Y |
| `x` | cycle X axis: step → relative time → wall clock |
| `g` | grid view: up to 4 charts at once |
| `f` | toggle live follow, `r` reload now |
| `b` | toggle the sidebar |
| `?` | help, `q` quit |

## Try it without a training run

A demo-log generator is built into the binary:

```bash
ttb gen-demo demo_logs                 # 3 runs, 5 tags, 5k steps
ttb gen-demo demo_logs --steps 80000   # ~1M points
ttb gen-demo demo_logs --live          # keeps appending, for live-follow
ttb demo_logs
```

## Run colors

Each run is drawn in its own color, from a fixed order of eight hues — blue,
orange, aqua, yellow, magenta, green, violet, red — validated for
colorblind separation and for contrast against the terminal background
(CVD ΔE ≥ 8 on adjacent pairs, all eight ≥ 3:1 contrast).

- **A run keeps its color.** The slot is assigned once, when the run is first
  discovered, and never recomputed — so switching a run off, or a new run
  appearing mid-training, never repaints the others.
- **Past eight runs the hues repeat, but the stroke changes**: runs 9–16 are
  drawn dashed, 17+ dotted, and their legend marker changes with them
  (`●` → `◆` → `▪`). Two runs never share both a hue and a stroke, so the
  chart itself stays readable without cross-checking the legend.
- **The palette follows the terminal.** Truecolor terminals get the exact
  hues; 256-color terminals get them snapped to the nearest xterm index that
  still clears the lightness and chroma gates; anything else falls back to the
  basic ANSI eight in the same order. Detection is from `COLORTERM`/`TERM`.

Every run's name sits beside its color in the sidebar and in the chart
legend, so identity is never carried by color alone.

## Notes

- Smoothing is applied to the per-column bucket means (each already the mean
  of the raw points in that column), which matches TensorBoard's look while
  staying O(width) per frame.
- Corrupt or truncated event files never crash the viewer — parsing stops at
  the first bad record and keeps everything before it.
- Record CRCs are not verified on read. Framing plus protobuf structure is
  validation enough here, and skipping the checksum is a large part of the
  read speed; CRCs *are* written correctly by `gen-demo`.

## Development

```bash
cargo test        # 17 unit tests: parser, store, plotting, colors
cargo clippy      # clean
```

Layout: `src/tfevents.rs` (parser/writer) · `src/store.rs` (run discovery,
incremental ingest) · `src/plot.rs` (braille canvas, bucketing, smoothing) ·
`src/colors.rs` (categorical palette, line styles) · `src/app.rs` (state, key
handling) · `src/ui.rs` (rendering) · `src/gen.rs` (demo logs) ·
`src/main.rs` (CLI, loader thread, event loop).

### Python version

The original dependency-free Python implementation lives in
[`python/`](python/) and is kept as a reference and a fallback for
environments without a Rust toolchain:

```bash
python -m terminal_tensorboard ~/runs     # from python/
python -m unittest discover -s tests      # 11 tests
```

It is feature-equivalent but roughly 36× slower to load and no longer the
recommended way to run this tool.
