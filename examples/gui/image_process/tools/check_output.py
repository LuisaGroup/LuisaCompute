#!/usr/bin/env python3
"""Check the output of the image_process example against an expected result.

Two modes:
  * plain comparison:   check_output.py <source> <result> [--tolerance N]
  * operator check:     check_output.py <source> <result> --mul R G B --add R G B [--tolerance N]
                        (result must equal clamp(round(source * mul + add)) per channel,
                         which is what a single "mul" / "add" operator chain produces)

Requires Pillow and numpy (development tool only, not a build dependency).
"""

import argparse
import sys

import numpy as np


def parse_color(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float32)
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 3:
        raise SystemExit(f"expected three components, got {text!r}")
    return np.asarray(values, dtype=np.float32)


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required: python -m pip install pillow")
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("result")
    parser.add_argument("--mul", default=None, help="RGB multiplier, e.g. \"0.5 0.5 0.5\"")
    parser.add_argument("--add", default=None, help="RGB offset, e.g. \"0.1 0.0 0.0\"")
    parser.add_argument("--tolerance", type=int, default=1, help="allowed 8 bit difference")
    args = parser.parse_args()

    source = Image.open(args.source).convert("RGB")
    result = Image.open(args.result).convert("RGBA")
    if source.size != result.size:
        print(f"size mismatch: source {source.size} vs result {result.size}")
        return 1

    src = np.asarray(source, dtype=np.uint8).astype(np.float32)
    res = np.asarray(result, dtype=np.uint8)

    if args.mul is None and args.add is None:
        expected = src
    else:
        # the kernel quantizes the color argument to 8 bit per channel first
        mul = np.round(np.clip(parse_color(args.mul, [1.0, 1.0, 1.0]), 0.0, 1.0) * 255.0) / 255.0
        add = np.round(np.clip(parse_color(args.add, [0.0, 0.0, 0.0]), 0.0, 1.0) * 255.0) / 255.0
        expected = np.round(np.clip(src / 255.0 * mul + add, 0.0, 1.0) * 255.0)

    diff = np.abs(expected - res[:, :, :3].astype(np.float32))
    max_diff = int(diff.max()) if diff.size else 0
    differing = int(np.count_nonzero(diff > 0))
    print(f"{args.source} -> {args.result}: {source.size[0]}x{source.size[1]}, "
          f"{differing} differing pixels, max channel diff {max_diff} (tolerance {args.tolerance})")
    if max_diff > args.tolerance:
        print("FAILED")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
