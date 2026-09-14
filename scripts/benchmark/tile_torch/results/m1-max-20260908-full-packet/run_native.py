#!/usr/bin/env python3
"""Reuse the audited actual-object RMSNorm replay, independently per mapping/fusion."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw', required=True, type=Path)
args = parser.parse_args()
replay = Path(__file__).resolve().with_name('replay_native.py')
for rows, columns in ((64, 256), (1024, 4096)):
    for local in (1, 8):
        for fusion in (0, 1):
            stem = f'rmsnorm-{rows}x{columns}-r0-l{local}-f{fusion}'
            command = [sys.executable, str(replay),
                       '--baseline', str(args.raw / 'capture' / (stem + '-p0')),
                       '--candidate', str(args.raw / 'capture' / (stem + '-p1')),
                       '--output', str(args.raw / f'native-libc-{rows}-l{local}-f{fusion}')]
            subprocess.run(command, check=True)
