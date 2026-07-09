"""Sampler-only stress / smoke test for the file-backed CPU samplers.

This is a *standalone diagnostic script*, not a pytest test.  It lives next
to the pytest tests in ``tests/`` for tidiness, but pytest skips it
automatically (filename does not start with ``test_``).

Runs ``pysampler.sample(...)`` in a loop with the same batch size used by the
training driver, but does NOT touch tinycudann / PyTorch / CUDA at all.  Use
this to localize crashes during training:

  - If this script runs cleanly for the same number of iterations that
    ``train_file_backed.py`` crashes at, the segfault / bus error is on the
    training side (tcnn, CUDA, PyTorch), not in the sampler.
  - If this script crashes too, the bug is in the sampler / file reader and
    should be reproducible without GPU dependencies.

The script enables ``faulthandler`` at startup, so a fatal signal (SIGSEGV /
SIGBUS / SIGABRT / SIGFPE / SIGILL) will print a Python stack trace before the
process terminates -- no debugger required.

Usage (from the toolkit repo root):
  python tests/bench_file_backed_sampler.py \
      --filename /path/to/volume.raw --dims 640 220 229 \
      --dtype float32 --range 0 1 --mode out_of_core \
      --steps 32000 --batch-size 65536
"""

import argparse
import faulthandler
import os
import sys
import time

import numpy as np

# faulthandler.enable() already installs handlers for SIGSEGV / SIGFPE /
# SIGABRT / SIGBUS / SIGILL, so a fatal signal will dump a Python stack trace
# to stderr before the process terminates.
faulthandler.enable()

# Allow running directly from a source tree (e.g. ``python tests/bench_...``)
# even when ``inrtoolkit`` is not installed in the active environment.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inrtoolkit import create_sampler


SAMPLER_BACKENDS = ("virtual_memory", "out_of_core")


class _ParseDims(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1 and "x" in values[0].lower():
            dims = [int(d) for d in values[0].lower().split("x")]
        else:
            dims = [int(v) for v in values]
        if len(dims) != 3:
            parser.error(f"--dims requires exactly 3 values, got {len(dims)}")
        setattr(namespace, self.dest, dims)


def main():
    parser = argparse.ArgumentParser(
        description="Stress-test a file-backed sampler (no GPU / model involved)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--filename", required=True)
    parser.add_argument("--dims", nargs="+", default=[128, 128, 128], action=_ParseDims)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "uint8", "uint16", "int8", "int16",
                                 "uint32", "int32", "float64"])
    parser.add_argument("--range", type=float, nargs=2, required=True,
                        metavar=("VMIN", "VMAX"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mode", default="out_of_core",
                        choices=SAMPLER_BACKENDS)
    parser.add_argument("--steps", type=int, default=10_000,
                        help="number of sample() calls to issue")
    parser.add_argument("--batch-size", type=int, default=64 * 1024)
    parser.add_argument("--report-every", type=int, default=500,
                        help="print rate / progress every N steps")
    args = parser.parse_args()

    if args.range[0] >= args.range[1]:
        parser.error(f"--range must satisfy VMIN < VMAX, got {tuple(args.range)}")

    sampler = create_sampler(
        "structuredRegular", args.mode,
        dims=args.dims, dtype=args.dtype, n_channels=1,
        filename=args.filename, offset=args.offset,
        is_big_endian=False,
        range=list(args.range),
    )

    n_channels = sampler.n_channels()
    print(
        f"[bench] backend={args.mode}  steps={args.steps}  batch={args.batch_size}  "
        f"channels={n_channels}",
        flush=True,
    )

    # Allocate host buffers once and reuse, mimicking the training inner loop.
    coords = np.empty((args.batch_size, 3), dtype=np.float32)
    values = np.empty((n_channels, args.batch_size), dtype=np.float32)

    import pysampler as _ext

    # Light running statistics so we notice if values go NaN / out of range.
    finite_misses = 0
    out_of_range  = 0

    t_start = time.time()
    last_t  = t_start

    for step in range(1, args.steps + 1):
        _ext.sample(sampler, coords.ctypes.data, values.ctypes.data, args.batch_size)

        # Cheap sanity checks on a small subset (avoid scanning all 65 K samples
        # every step -- that itself can dominate the bench).
        v_first = values[0, :64]
        if not np.isfinite(v_first).all():
            finite_misses += 1
        if v_first.min() < -1e-4 or v_first.max() > 1.0 + 1e-4:
            out_of_range += 1

        if step % args.report_every == 0 or step == args.steps:
            now = time.time()
            window = now - last_t
            last_t = now
            rate = args.report_every / window if window > 0 else float("inf")
            elapsed = now - t_start
            print(
                f"[bench] step {step:>7d}/{args.steps}  "
                f"rate={rate:6.1f} it/s  elapsed={elapsed:6.1f}s  "
                f"non_finite={finite_misses}  out_of_range={out_of_range}",
                flush=True,
            )

    total = time.time() - t_start
    print(
        f"[bench] done.  total={total:.1f}s  avg_rate={args.steps / total:.1f} it/s  "
        f"non_finite={finite_misses}  out_of_range={out_of_range}",
        flush=True,
    )


if __name__ == "__main__":
    main()
