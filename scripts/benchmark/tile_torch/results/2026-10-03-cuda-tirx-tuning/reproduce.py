"""Replay three closed native-only experiments from bundled evidence. Stdlib only."""
import argparse
import json
import math
import ntpath
from pathlib import Path
import re
from replay_support import need, read, safe, sha, samples, graph_protocol, actual_graph

SPECS = {
    'unroll-v2': ('4137de744d8963bcdca09b4de75ca065fa6ef6886a6fdd18f2defb938687103d',
                  ('unroll1', 'unroll8', 'unroll1_recheck'), (1, 8, 1), (8, 8, 8),
                  ('unroll8_over_unroll1', 'unroll8_over_unroll1_recheck', 'unroll1_recheck_over_initial')),
    'lane-v1': ('5b2125d852cb6ec529ee2339d4d9c16ab731261ac692c5a1a1616e51d7d2f0e5',
                ('lane8', 'lane1', 'lane8_recheck'), (8, 8, 8), (8, 1, 8),
                ('lane1_over_lane8', 'lane1_over_lane8_recheck', 'lane8_recheck_over_initial')),
    'partial-tree-ab-v1': ('8268be5e080ce919d076d19bc50026ee4b52cbfea7bb284fae44bfdd13f22465',
                          ('A_initial', 'B', 'A_recheck'), (8, 8, 8), (8, 8, 8),
                          ('B_over_A_initial', 'B_over_A_recheck', 'A_recheck_over_initial')),
}

def path_key(value): return ntpath.normcase(ntpath.normpath(value))

def observed_host(folder, evidence, data, role):
    host = evidence['host_modules']; capture = data['bridge_captures'][role]
    need(host['role'] == role and host['actual_bridge_receipt']['sha256'] == capture['build']['dll_sha256'] ==
         capture['files']['luisa-tile-bridge-tirx.dll']['sha256'], 'actual bridge/capture identity')
    need(host['common_file_receipts'] == data['loaded_common'], 'common dependency receipts')
    records = []
    for phase in ('before', 'after'):
        raw = read(folder / f'diagnostic-host-modules-{phase}.json')
        need(raw == host['phases'][phase] and raw['phase'] == phase and raw['bridge_module_count'] == 1 and
             raw['executable_local_bridge'] is True and raw['bridge_base'] > 0 and
             raw['evidence'] == 'GetProcAddress_existing_compile_device_FROM_ADDRESS_and_Toolhelp', 'actual host observation')
        need(path_key(raw['bridge_path']) == path_key(host['actual_bridge_receipt']['path']) and
             path_key(raw['executable_path']) == path_key(host['actual_executable_receipt']['path']), 'actual bridge/executable path')
        need(path_key(ntpath.dirname(raw['bridge_path'])) == path_key(ntpath.dirname(raw['executable_path'])), 'local bridge directory')
        names = {}; bridge = []
        for item in raw['modules']:
            name = item['name'].lower()
            need(item['base'] > 0 and item['image_bytes'] > 0 and ntpath.basename(item['path']).lower() == name, 'module record')
            if name == 'luisa-tile-bridge-tirx.dll': bridge.append(item)
            elif name.startswith(('luisa-', 'tvm_')) or name in ('nvcuda.dll', 'nvcuda64.dll'):
                need(name not in names, 'duplicate module'); names[name] = item
        need(len(bridge) == 1 and bridge[0]['base'] == raw['bridge_base'], 'one actual bridge')
        need(names.keys() == data['loaded_common'].keys(), 'complete common module set')
        for name, item in names.items():
            need(path_key(item['path']) == path_key(data['loaded_common'][name]['path']), 'loaded common path')
        records.append((names, raw['bridge_base']))
    need(records[0] == records[1], 'module identity changed within process')

def verify(root):
    index = read(root / 'index.json')
    need(index['schema'] == 1 and index['closed_summaries'] == {k: v[0] for k, v in SPECS.items()}, 'fixed experiment provenance')
    for name, record in index['files'].items():
        file = safe(root, name)
        need(sha(file) == record['public_sha256'] and file.stat().st_size == record['public_bytes'], 'bundled bytes changed: ' + name)
        if record['transform'] == 'byte-exact':
            need(record['raw_sha256'] == record['public_sha256'] and record['raw_bytes'] == record['public_bytes'], 'exact-byte claim')

    def binds(path, receipt):
        record = index['files'][path.relative_to(root).as_posix()]
        need(record['raw_sha256'] == receipt['sha256'] and record['raw_bytes'] == receipt['bytes'], 'raw receipt binding')

    rows = []; total_samples = total_nodes = 0
    for key, (digest, stages, unrolls, lanes, ratio_names) in SPECS.items():
        base = root / key; data = read(base / 'results.json'); queue = read(base / 'provenance/queue.json')
        need(index['files'][key + '/results.json']['raw_sha256'] == digest, 'closed summary identity')
        need(data['status'] == 'completed_validated' and data['torch_status'] == 'not_requested' and len(data['cases']) == 6 and
             data['counts']['native_validations'] == 18 and data['counts']['torch_validations'] == 0 and
             data['counts']['primary_graph_samples'] == 126 and data['counts']['actual_native_graph_nodes'] == 1800, 'complete cohort counts')
        need(queue['status'] == 'passed' and [s['name'] for s in queue['stages']] == list(stages) and
             all(s['status'] == 'passed' for s in queue['stages']), 'closed original queue')
        binds(base/'provenance/queue.json', data['queue_receipt']); binds(base/'provenance/snapshot.json', data['snapshot_receipt'])
        ids = [c['base_case']['id'] for c in data['cases']]
        need(len(set(ids)) == 6, 'six unique complete cases')
        for item in data['cases']:
            case = item['base_case']; ident = case['id']; medians = []; canonical = None
            need(case['fast_math'] is True and case['tile'][0] == 1 and tuple(item['stages']) == stages, 'fixed math/BR/stages')
            for stage, unroll, lane in zip(stages, unrolls, lanes):
                evidence = item['stages'][stage]; folder = base / 'artifacts' / ident / stage
                raw = read(folder / 'results.json'); manifest = read(folder/'manifest.json')
                need(raw['status'] == 'passed' and raw['fast_math'] is True and raw['realization'] == evidence['realization'], 'actual result/math')
                for field in ('operation', 'precision', 'dimensions', 'tile', 'seed', 'pattern', 'fast_math'):
                    need(raw[field] == case[field], 'actual case ' + field)
                correct = raw['correctness']; saved = evidence['saved_output']; count = math.prod(case['dimensions'])
                need(correct == evidence['recorded_correctness'] and correct['checks'] == 4 and correct['errors'] == 0 and
                     correct['elements_per_check'] == count and all(correct[k] is True for k in ('inputs_unchanged','guards_unchanged','all_outputs_finite')),
                     'retained full numerical/guard checks')
                need(saved['failed_elements'] == 0 and saved['elements'] == count and
                     all(saved['receipt'][k] == evidence['output_receipt'][k] for k in ('sha256', 'bytes')), 'full saved-output closure record')
                values = graph_protocol(raw); median = samples(values)
                need(values == evidence['event_us']['samples'] and median == evidence['event_us']['p50'], 'samples/median')
                medians.append(median); total_samples += 7
                need(manifest == evidence['manifest'] and manifest['lowering'] == 'tirx', 'fixture manifest')
                work = read(folder/'diagnostic-collective-work.json')
                need(work == evidence['collective_work']['data'], 'actual IR work'); work.pop('analysis_wall_ms')
                identity = (manifest, evidence['fixture_sha256'], work)
                if canonical is None: canonical = identity
                else: need(canonical == identity, 'inputs/oracle/work changed across stage')
                obs = evidence['observation']['observed']
                need(obs['lane_elements'] == lane and obs['requested_unroll_factor'] == obs['actual_plan_unroll_factor'] ==
                     evidence['requested_unroll'] == unroll, 'requested/actual mapping')
                if key == 'lane-v1':
                    need(obs['requested_lane_elements'] == obs['actual_plan_lane_elements'] == evidence['requested_lane'] == lane, 'actual lane receipt')
                marker = f"cuda-subgroup-plan=threads128:programs2:warps2:lane-elements{lane}:reductions{1 if case['operation']=='rmsnorm' else 2}:unroll{unroll}"
                need(raw['realization'].count(marker) == 1 and 'cuda-subgroup-math=fast-elements-preserved-reductions-v1' in raw['realization'], 'actual plan/math marker')
                total_nodes += actual_graph(folder, 'subgroup', case, evidence)
                for name, receipt in [('source.txt', evidence['source_receipt']), ('results.json', evidence['result_receipt'])]: binds(folder/name, receipt)
                for name, receipt in evidence['observation']['receipts'].items(): binds(folder/name, receipt)
                if key == 'partial-tree-ab-v1':
                    role = 'B' if stage == 'B' else 'A'; observed_host(folder, evidence, data, role)
                    for phase, receipt in evidence['host_modules']['observation_receipts'].items(): binds(folder/f'diagnostic-host-modules-{phase}.json',receipt)
            paths = [base/'artifacts'/ident/stage/'source.txt' for stage in stages]
            need(paths[0].read_bytes() == paths[2].read_bytes(), 'control source drift')
            if key == 'partial-tree-ab-v1': need(paths[0].read_bytes() != paths[1].read_bytes(), 'A/B source identity')
            a,b,c = medians; ratios = (b/a,b/c,c/a)
            need(all(item[name] == value for name,value in zip(ratio_names,ratios)), 'ratios/drift')
            rows.append(dict(experiment=key,case=ident,medians_us=dict(zip(stages,medians)),ratios=dict(zip(ratio_names,ratios))))
        if key == 'partial-tree-ab-v1':
            inventories = []
            for role, capture in data['bridge_captures'].items():
                for name, receipt in capture['files'].items():
                    if not name.endswith('.dll'): binds(base/f'provenance/bridge-{role}'/name, receipt)
                need(capture['build']['source_diff_sha256'] == capture['files']['source.diff']['sha256'], 'bridge source/build capture')
                inventory = read(base/f'provenance/bridge-{role}/source-inventory.json'); inventories.append(inventory)
                for name in ('cuda_codegen.cpp','reduction.cpp'):
                    record = capture['files'][name]
                    need(inventory['src/tile/bridge/tirx/'+name] == {k:record[k] for k in ('sha256','bytes')}, 'archived source inventory')
            a,b=inventories
            need(a.keys()==b.keys() and {k for k in a if a[k]!=b[k]} == {'src/tile/bridge/tirx/cuda_codegen.cpp','src/tile/bridge/tirx/reduction.cpp'}, 'only authorized A/B source changes')
            binds(base/'provenance/installation.json', data['installation_receipt'])
    need(len(rows)==18 and total_samples==378 and total_nodes==5400, 'all three complete experiments')
    return dict(status='passed',native_validations=54,primary_samples=total_samples,actual_graph_nodes=total_nodes,cases=rows,
                scope='Bundled source/result/protocol/observation replay; omitted tensors and DLL binaries are not recomputed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().root.resolve()),indent=2,allow_nan=False))
