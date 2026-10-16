"""Protocol/graph helpers derived from the published V1 stdlib replay; no execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics



def need(value, text):
    if not value:
        raise ValueError(text)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe(root, name):
    p = (root / name).resolve()
    need(p.is_relative_to(root.resolve()) and p.is_file(), 'invalid bundled path: ' + name)
    return p


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(values):
    need(isinstance(values, list) and len(values) == 7 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), 'seven positive finite raw samples required')
    return statistics.median(values)


def graph_protocol(raw, torch=False):
    g = raw['compiled_graph'] if torch else raw
    batch = g['batch'] if torch else g['graph_batch']
    replay = g['replays_per_sample'] if torch else g['graph_replays_per_sample']
    operations = g['operations_per_sample'] if torch else g['graph_operations_per_sample']
    target = g['sample_target_ms'] if torch else g['graph_sample_target_ms']
    warm = g['warmup_actual_ms'] if torch else g['graph_warmup_actual_ms']
    events = g['event_span_ms']['samples'] if torch else g['graph_event_span_ms']
    hosts = g['host_span_ms']['samples'] if torch else g['graph_host_span_ms']
    event_us = g['event_us_per_operation']['samples'] if torch else g['graph_event_stream_span_us_per_op']
    host_us = g['host_wall_us_per_operation']['samples'] if torch else g['graph_host_wall_us_per_op']
    need(g['graph_protocol'] == 'adaptive_replay_span_v2' and batch == 100 and type(replay) is int and replay > 0 and operations == replay * batch, 'graph protocol/denominator')
    need(target == 100 and warm >= 500 and g['calibration_target_reached' if torch else 'graph_calibration_target_reached'] is True, 'sample/warmup/calibration')
    for span, values in ((events, event_us), (hosts, host_us)):
        samples(span); samples(values)
        need(all(math.isclose(a * 1000 / operations, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(span, values)), 'raw span normalization')
    need(min(events) >= target * 0.8, 'achieved event span below validation floor')
    return event_us


def actual_graph(folder, stage, case, evidence):
    with (folder / 'diagnostic-graph.csv').open(newline='') as f:
        nodes = list(csv.DictReader(f))
    with (folder / 'diagnostic-resources.csv').open(newline='') as f:
        resources = list(csv.DictReader(f))
    obs = read(folder / 'diagnostic-observation.json')
    need(resources == evidence['observation']['resources'] and obs == evidence['observation']['observed'], 'actual observation/resource packet differs')
    manifest = read(folder / 'manifest.json')
    size = {'float32': 4, 'float16': 2, 'bfloat16': 2}
    descriptors = manifest['inputs'] + [manifest['output']]
    sizes = [math.prod(d['shape']) * size[d['storage_dtype']] for d in descriptors]
    need(len(nodes) == 100, 'all actual graph nodes required')
    r = case['dimensions'][0]
    subgroup = stage == 'subgroup'
    bindings = obs['device_slot_to_host_index'] if subgroup else list(range(4))
    if subgroup:
        proof = read(folder / 'diagnostic-ranges.json')
        need(proof == evidence['observation']['proof'] and proof['minimum_host_bytes'] == sizes and proof['actual_final_ranges_disjoint'] is True, 'actual subgroup spans')
        pointers = proof['final_host_pointers']
        need(len(bindings) == len(set(bindings)) and set(bindings) <= set(range(4)) and {0, 3} <= set(bindings), 'binding permutation')
        need(case['operation'] != 'rmsnorm' or 1 in bindings, 'RMS gamma binding')
        need(obs['threads_per_group'] == 128 and obs['programs_per_group'] == 2 and obs['lane_elements'] in (1, 8) and obs['cache_reduction_inputs'] is False, 'fixed mapper configuration')
        grid, block, entry = [(r + 1) // 2, 1, 1], [128, 1, 1], obs['entry']
    else:
        proof = read(folder / 'diagnostic-row-regroup.json')
        need(proof == evidence['observation']['proof'] and proof['actual_capture_tile'] == case['tile'], 'original native BR1 proof')
        pointers = [int(nodes[0]['arg' + str(i)], 16) for i in range(4)]
        grid, block, entry = [r, 1, 1], [1, 1, 1], 'luisa_tile_main'
    need(len(pointers) == 4 and all(type(p) is int and 0 < p < 2**64 - n for p, n in zip(pointers, sizes)), 'valid final whole spans')
    need(all(pointers[i] + sizes[i] <= pointers[3] or pointers[3] + sizes[3] <= pointers[i] for i in range(3)), 'disjoint whole ranges')
    functions = set()
    for node in nodes:
        need(node['entry'] == entry and int(node['function'], 16) > 0 and [int(node['grid_' + a]) for a in 'xyz'] == grid and [int(node['block_' + a]) for a in 'xyz'] == block, 'actual function/grid/block')
        need(int(node['dynamic_shared_bytes' if subgroup else 'shared_bytes']) == 0, 'dynamic shared bytes')
        for i in bindings:
            need(int(node[('host_arg' if subgroup else 'arg') + str(i)], 16) == pointers[i], 'actual final argument')
        functions.add(node['function'])
    need(len(functions) == 1 and len(resources) == 4 and {d['attribute'] for d in resources} == {'registers', 'static_shared_bytes', 'local_bytes', 'max_threads'}, 'actual function/resource set')
    need(len({d['module'] for d in resources}) == 1, 'same module')
    for d in resources:
        need(d['entry'] == entry and d['function'] in functions and int(d['module'], 16) > 0, 'resource function binding')
        need((d['known'] == '1' and d['cuda_status'] == '0' and int(d['value']) >= 0) or (d['known'] == '0' and d['value'] == 'unknown'), 'known/unknown resource')
    source = (folder / 'source.txt').read_text(encoding='utf-8')
    if subgroup:
        match = re.search(r'extern "C" __global__ void __launch_bounds__\(128\) ' + re.escape(entry) + r'\(([^\n]*)\) \{', source)
        need(match is not None and [int(x) for x in re.findall(r'arg(\d+)_ptr', match.group(1))] == bindings, 'source ABI/actual permutation')
        for op in ('add.rn.f32', 'min.f32', 'max.f32'):
            need(source.count('asm("' + op + ' ') == 1, 'v1 no-FTZ helper')
    return len(nodes)

