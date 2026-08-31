#!/usr/bin/env python3
"""Generate synthetic TensorBoard logs for trying out the TUI.

No TensorFlow needed — uses the project's own tfevents writer.

    python scripts/generate_demo_logs.py demo_logs
    ttb demo_logs

With --live it keeps appending points every second so you can watch
the live-follow mode in action.
"""

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terminal_tensorboard.tfevents import encode_scalar_event, frame_record  # noqa: E402


def write_run(logdir, run, steps, lr, seed, tensor_format=False, start_step=0, fh=None):
    path = os.path.join(logdir, run)
    os.makedirs(path, exist_ok=True)
    if fh is None:
        fname = "events.out.tfevents.%d.demo" % int(time.time())
        fh = open(os.path.join(path, fname), "ab")
    rng = random.Random(seed)
    t0 = time.time() - steps * 0.05
    for i in range(start_step, start_step + steps):
        wall = t0 + i * 0.05
        progress = i / max(1, start_step + steps)
        loss = 2.5 * math.exp(-3.0 * lr * i / 100.0) + 0.08 + rng.gauss(0, 0.05)
        acc = 1.0 - 0.9 * math.exp(-2.5 * lr * i / 100.0) + rng.gauss(0, 0.01)
        lr_now = lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        grad = abs(rng.gauss(1.0, 0.4)) * math.exp(-i / 4000.0)
        for tag, value in (
            ("train/loss", loss),
            ("train/accuracy", min(1.0, max(0.0, acc))),
            ("train/lr", lr_now),
            ("train/grad_norm", grad),
        ):
            fh.write(frame_record(encode_scalar_event(tag, i, wall, value, tensor=tensor_format)))
        if i % 25 == 0:
            fh.write(
                frame_record(
                    encode_scalar_event("val/loss", i, wall, loss + 0.15 + rng.gauss(0, 0.03))
                )
            )
    fh.flush()
    return fh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", nargs="?", default="demo_logs")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--live", action="store_true", help="keep appending points forever")
    args = parser.parse_args()

    runs = [
        ("baseline", 1.0, 1),
        ("high_lr", 2.2, 2),
        ("low_lr/warmup", 0.4, 3),
    ]
    handles = {}
    for run, lr, seed in runs:
        handles[run] = write_run(
            args.logdir, run, args.steps, lr, seed, tensor_format=(seed == 3)
        )
    print("wrote %d runs x %d steps to %s" % (len(runs), args.steps, args.logdir))

    if args.live:
        print("appending 20 steps/second per run — Ctrl-C to stop")
        step = args.steps
        try:
            while True:
                for run, lr, seed in runs:
                    write_run(
                        args.logdir, run, 20, lr, seed + step,
                        tensor_format=(seed == 3), start_step=step, fh=handles[run],
                    )
                step += 20
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
