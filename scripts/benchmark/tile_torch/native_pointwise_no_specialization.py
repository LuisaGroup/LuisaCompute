#!/usr/bin/env python3
"""Independent pointwise off/on cohort with full-packet specialization disabled.

Reuse the frozen native_pointwise capture, ABI checks and common native replay
without editing that runner. Both variants retain identical fixed policies;
only pointwise fusion differs within this new cohort. Comparisons against a
previous specialization-enabled cohort are not paired causal measurements.

Use this entry point for capture, verify and replay. It records its own source
identity in addition to all original runner identities. Configuration changes
are scoped to the synchronous CLI invocation, not applied at module import.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import shutil

import native_pointwise as pointwise


# Keep ENABLE=1 as in the frozen control; the production DISABLE flag takes
# precedence. Adding this one flag is the entire fixed-environment change.
FIXED_ENV = dict(pointwise.FIXED_ENV, LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION='1')
_BASE_RUNNERS = pointwise.runners


def check_realization(measurement, enabled):
    # Match native_pointwise.check_realization, except require explicit zero
    # clone counters. Do not rewrite measurement fields to pass its positive
    # specialization check: captured metadata must remain authoritative.
    realization = measurement['realization']
    pointwise.require('W8, 32 workers/block, 1 CPU workers;' in realization, 'packet/block/worker mapping changed')
    fields = dict(re.findall(r'\b([a-z_]+)=([^;]+)', realization))
    for name, expected in dict(local_lanes='8', blocks_per_task='0', max_unrolled_tile_elements='64',
                               unordered_reduction_partitions='4', load_reduction_fusion='false',
                               expression_reduction_fusion='false', map_fusion='false', fast_math='false',
                               pointwise_fusion=str(enabled).lower(), custom_cost_policy='false').items():
        pointwise.require(fields.get(name) == expected, 'fixed realization policy changed: ' + name)
    for name in ('full_packet_specializations', 'full_packet_cloned_instructions'):
        pointwise.require(fields.get(name) == '0', 'disabled full-packet specialization requires explicit zero: ' + name)
    for name in ('private_workspace_bytes', 'fused_pointwise_regions', 'fused_pointwise_loads', 'fused_pointwise_stores'):
        pointwise.require(name in fields and fields[name].isdigit(), 'missing realization counter: ' + name)
    if not enabled:
        pointwise.require(int(fields['fused_pointwise_regions']) == 0, 'off control still has pointwise fusion')
    if enabled and measurement['operation'] == 'rope':
        for name, count in dict(fused_pointwise_regions=1, fused_pointwise_loads=4,
                                fused_pointwise_stores=2, pointwise_alias_checks=3).items():
            pointwise.require(fields.get(name) == str(count), 'RoPE did not realize the expected pointwise DAG: ' + name)
    return fields


def runners(directory):
    identities = _BASE_RUNNERS(directory)
    adapter = Path(__file__).resolve()
    shutil.copy2(adapter, directory / 'runner-sources' / adapter.name)
    identities.update(pointwise.hashes([adapter]))
    return identities


@contextmanager
def configured_runner():
    # The command is synchronous; do not use this context concurrently with
    # another command sharing the imported runner module.
    original = pointwise.FIXED_ENV, pointwise.check_realization, pointwise.runners
    pointwise.FIXED_ENV = FIXED_ENV
    pointwise.check_realization = check_realization
    pointwise.runners = runners
    try:
        yield
    finally:
        pointwise.FIXED_ENV, pointwise.check_realization, pointwise.runners = original


def main():
    with configured_runner():
        pointwise.main()


if __name__ == '__main__':
    main()
