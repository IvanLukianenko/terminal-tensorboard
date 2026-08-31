# terminal-tensorboard

A lightning-fast, zero-dependency terminal UI for viewing TensorBoard training
logs. Point it at your log directory and watch your losses live — over SSH, in
tmux, anywhere you have a terminal. No TensorFlow, no protobuf, no browser.

```
 ttb  ~/runs                        runs 3/3 │ tags 12 │ pts 1.2M │ live ● │ x:step │ y:lin │ smooth 0.60
 RUNS                    │ train/loss                                              ● baseline
  ▣ ● baseline           │  2.5┼⢣                                                  ● high_lr
  ▣ ● high_lr            │     │⠘⢆⡀                                                ● low_lr/warmup
  ▣ ● low_lr/warmup      │  1.5┼  ⠑⠢⢄⡀
 TAGS (5)  /              │     │      ⠉⠒⠒⠤⠤⣀⣀⡀
  ▶ train/loss           │  0.5┼              ⠉⠉⠉⠉⠒⠒⠒⠒⠤⠤⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀
    train/accuracy       │     └0        5k        10k        15k        20k
```

## Why it's fast

- **Own tfevents parser.** A hand-rolled TFRecord + protobuf-wire-format
  reader extracts scalars directly from the bytes — no TensorFlow, no protobuf
  package, no dependencies at all. Non-scalar payloads (images, histograms)
  are skipped without being decoded.
- **Incremental tailing.** Files are read once; every refresh only parses the
  bytes appended since the last one, so following a live training run costs
  microseconds per tick. Partially-written trailing records are handled
  correctly and re-read once complete.
- **Compact storage.** Points live in `array('d')` / `array('q')` buffers —
  about 24 bytes per point, millions of points without breaking a sweat.
- **Pixel-bucket rendering.** Before drawing, each series is reduced to one
  mean value per braille pixel column (O(points) done by C-level slicing, then
  everything is O(columns)), so redraws are instant even on huge runs.
- **Background loading.** Parsing happens on a loader thread; the UI never
  blocks, even while ingesting gigabytes on first start.

## Install

```bash
pip install .            # from a checkout
# or just run it in place — there are no dependencies:
python -m terminal_tensorboard ~/runs
```

Python ≥ 3.8, Linux/macOS (on Windows: `pip install windows-curses` first).

## Usage

```bash
ttb LOGDIR                  # scan LOGDIR recursively, follow for new data
ttb LOGDIR --no-follow      # one-shot view
ttb LOGDIR --refresh 0.5    # poll twice a second
ttb LOGDIR --smoothing 0.9  # heavier EMA smoothing
ttb LOGDIR --x reltime      # x axis: step | reltime | wall
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

A demo-log generator (also dependency-free) is included:

```bash
python scripts/generate_demo_logs.py demo_logs          # 3 runs, 5 tags
python scripts/generate_demo_logs.py demo_logs --live   # keeps appending
ttb demo_logs
```

## Notes

- Smoothing is applied to the per-column bucket means (each already the mean
  of the raw points in that column), which matches TensorBoard's look while
  staying O(width) per frame.
- Colors adapt to the terminal: a 256-color palette when available, the basic
  8 otherwise; the UI itself uses only default-theme-safe attributes.
- Corrupt or truncated event files never crash the viewer — parsing stops at
  the first bad record and keeps everything before it.

## Development

```bash
python -m unittest discover -s tests
```

Layout: `tfevents.py` (parser/writer) · `store.py` (run discovery, incremental
ingest) · `plot.py` (braille canvas, bucketing, smoothing) · `app.py` (curses
UI) · `cli.py`.
