"""Command line entry point."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ttb",
        description="A lightning-fast terminal UI for TensorBoard scalar logs.",
    )
    parser.add_argument("logdir", help="directory containing tfevents files (searched recursively)")
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        metavar="SEC",
        help="poll interval for live tailing (default: 2.0)",
    )
    parser.add_argument("--no-follow", action="store_true", help="do not tail files for new data")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        metavar="W",
        help="initial EMA smoothing weight, 0..0.99 (default: 0.6)",
    )
    parser.add_argument(
        "--x",
        choices=("step", "reltime", "wall"),
        default="step",
        help="initial x axis (default: step)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = parser.parse_args(argv)

    if not os.path.isdir(args.logdir):
        parser.error("not a directory: %s" % args.logdir)

    try:
        import curses  # noqa: F401
    except ImportError:
        print(
            "curses is not available. On Windows install it with:\n"
            "    pip install windows-curses",
            file=sys.stderr,
        )
        return 1

    from .app import run_app

    try:
        run_app(
            args.logdir,
            refresh_interval=max(0.2, args.refresh),
            follow=not args.no_follow,
            smoothing=args.smoothing,
            xmode=args.x,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
