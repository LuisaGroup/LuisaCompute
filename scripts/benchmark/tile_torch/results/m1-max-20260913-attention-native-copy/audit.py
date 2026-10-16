"""Offline audit of frozen actual-ORC native copy off/on attention evidence.

Run only after replay finishes: python3 -B audit.py > audit.json
Requires NumPy. Prints deterministic JSON; writes no files and never imports
the experiment/runner, reads live build paths, or executes/loads native code.
Historical absolute paths are receipts mapped to archived files/fingerprints.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics as stats
import sys
import tarfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
import numpy as np

ROOT = Path(__file__).resolve().parent
MARKER = 'luisa-attention-native-copy.n6RmGZ'
# Filled only after the pre-capture source freeze; None deliberately fails closed.
PLAN_SHA256 = 'e530dc5e3076ab0f1254ff442fd7f8b35dd626baa1c0576820288dc6143fddd9'
SOURCE_SHA256 = '90581800bc76f80aad6b841e609db91605c0c4b209d86dd6c92f3e474ea85885'
CASES = [
    ('decode-mha-d64', (1, 8, 8, 1, 2048, 64, 64), (1, 16), 0, False),
    ('decode-gqa-d80', (1, 8, 2, 1, 2053, 80, 96), (1, 16), 8, False),
    ('decode-long-kv', (1, 16, 4, 1, 8193, 128, 128), (1, 16), 8, False),
    ('prefill-q4', (1, 4, 2, 32, 65, 32, 32), (4, 16), 0, True),
    ('prefill-q8', (1, 4, 2, 64, 129, 32, 48), (8, 16), 0, True),
    ('batch-gqa-q4', (2, 6, 2, 17, 67, 40, 48), (4, 16), 8, True),
]
VARIANTS = (('off', 0), ('on', 4))
TESTS = ('test_simd_schedule_ir', 'test_simd_llvm_schedule_codegen', 'test_simd_xir_to_schedule',
         'test_simd_warp_uniformity', 'test_tile_dsl', 'test_tile_xir_target_info',
         'test_tile_xir_runtime', 'test_tile_xir_llm', 'test_xir_verifier',
         'test_xir_interchange', 'test_xir_passes')
SOURCE_FILES = {
    'native_tile.py': 'scripts/benchmark/tile_torch/native_tile.py',
    'native_tile_replay.cpp': 'scripts/benchmark/tile_torch/native_tile_replay.cpp',
    'backends/simd/llvm/llvm_schedule_codegen.h': 'src/backends/simd/llvm/llvm_schedule_codegen.h',
}
PHYSICAL = ('static_snapshot_bytes_per_worker', 'static_snapshot_allocations',
            'interleaved_private_arrays', 'full_packet_specializations',
            'full_packet_cloned_instructions', 'native_mmas',
            'native_contraction_mmas', 'native_output_mmas', 'native_copies')
OBSERVED = PHYSICAL + ('snapshot_budget', 'private_workspace_bytes', 'blocked_mmas',
                       'two_dimensional_mmas', 'rolled_mmas', 'contiguous_private_reads',
                       'contiguous_private_writes', 'unordered_reduction_partitions')
BOUNDARY = ('Common C++ timer; native entries, launch resets, block traversal and compiler-emitted libc/allocations included. '
            'Runtime/Python/JIT/caller allocation/validation excluded.')
REPORT = dict(status='running', scope='six actual SIMD Tile lowering cases; native copy width 0 versus 4, fixed native MMA width 4',
              captures=[], comparisons=[dict(case=n, status='NotRun') for n, *_ in CASES],
              limitations=[
                  'Source archive is the actual selected export, not full HEAD, a reproducible-build proof, or a loader dependency closure.',
                  'The audit never opens historical producer/tool paths: binary and compiler identities are frozen fingerprint receipts, not loader attestation.',
                  'Temporary uv Python launcher paths are command receipts; their symlink targets/interpreter binaries were not independently archived or fingerprinted.',
                  'Copy width 4 preserves logical snapshot capacity and FP32 math; private interleaving, helper code, workspace bytes and clone admission may change. Ratios describe the entire lowering candidate, not isolated vector-load latency.',
                  'Each arm is checked against an independent dense FP64 oracle; captured A/B bits and each visit final output versus its own capture must be identical.',
                  'Runtime helper guards/oracles are checked after preflight, warmup, calibration and each sample batch, not after every timed native invocation. Bitwise equality is checked on each visit final output.',
                  'Full finite checks cover these archived inputs/outputs, not all FP32 inputs or untested shapes.',
                  'Resource counters are cross-checked between metadata, capture summaries and actual-entry manifests; no cycle calibration or post-optimization physical-memory proof is inferred.',
                  'Native-entry timing includes the common C++ launch traversal/reset and emitted code; synchronized Runtime capture timing is a different metric and is not mixed into ratios.',
                  'ABBA min/max describes observed spread, not a confidence interval; user/system background work was not controlled.',
              ])
HASHES = {}
COMMANDS = set()
PYTHON_LAUNCHERS = set()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(type(name) is str and name and not path.is_absolute() and '..' not in path.parts and
            '\\' not in name and '\x00' not in name, 'unsafe relative path: ' + repr(name))
    return path


def safe(base, name):
    parts = safe_name(name).parts
    for i in range(1, len(parts) + 1):
        require(not (base / Path(*parts[:i])).is_symlink(), 'symlink artifact: ' + name)
    path = base / name
    require(path.is_file() and path.resolve().is_relative_to(ROOT), 'missing/nonregular/escaped artifact: ' + name)
    return path


def sha(path):
    path = safe(ROOT, Path(path).relative_to(ROOT).as_posix())
    state = path.stat()
    key = (str(path), state.st_size, state.st_mtime_ns, state.st_ctime_ns)
    if key not in HASHES:
        with path.open('rb') as stream:
            HASHES[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return HASHES[key]


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(path):
    raw = safe(ROOT, Path(path).relative_to(ROOT).as_posix()).read_bytes()
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))
    def finite(item):
        if type(item) is float:
            require(math.isfinite(item), 'nonfinite JSON number')
        elif type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)
    finite(value)
    return value


def archive_relative(original):
    parts = PurePosixPath(original).parts
    require(parts.count(MARKER) == 1, 'unmapped historical experiment path: ' + original)
    return safe_name(PurePosixPath(*parts[parts.index(MARKER) + 1:]).as_posix()).as_posix()


def field(text, name, boolean=False):
    values = re.findall(r'\b' + re.escape(name) + '=' + ('(true|false)' if boolean else r'(\d+)') + r'\b', text)
    require(len(values) == 1, 'missing/duplicate realization field: ' + name)
    return values[0] == 'true' if boolean else int(values[0])


def python_launcher(name):
    path = PurePosixPath(name)
    require(path.is_absolute() and re.fullmatch(r'python(?:3(?:\.\d+)?)?', path.name), 'unexpected Python command launcher')
    PYTHON_LAUNCHERS.add(name)
    return name


def historical_tmp_alias(name):
    return '/tmp/' + name[len('/private/tmp/'):] if name.startswith('/private/tmp/') else name


def sources_and_gates():
    require(isinstance(PLAN_SHA256, str) and isinstance(SOURCE_SHA256, str) and
            re.fullmatch(r'[0-9a-f]{64}', PLAN_SHA256) and re.fullmatch(r'[0-9a-f]{64}', SOURCE_SHA256),
            'audit expected plan/source hashes have not been frozen yet')
    require(sha(ROOT / 'plan.json') == PLAN_SHA256, 'predeclared plan changed')
    plan, frozen = load(ROOT / 'plan.json'), load(ROOT / 'provenance.json')
    require(sha(ROOT / 'sources.tar.gz') == frozen['source_archive_sha256'] == SOURCE_SHA256, 'frozen source identity')
    require((ROOT / 'sources.tar.gz').stat().st_size == frozen['source_archive_bytes'], 'source archive extent')
    require(frozen['plan_sha256'] == PLAN_SHA256 and frozen['format'] == 'attention-native-copy-source-freeze-v1', 'provenance contract')
    inventory = frozen['source_inventory']
    require(inventory and frozen['source_sha256'] == {n: r['sha256'] for n, r in inventory.items()}, 'source inventory')
    seen, saved = set(), {}
    with tarfile.open(ROOT / 'sources.tar.gz', 'r:gz') as archive:
        for member in archive:
            safe_name(member.name)
            require(member.isfile() and member.name not in seen and member.name in inventory, 'unsafe/duplicate/unlisted tar member')
            seen.add(member.name)
            raw = archive.extractfile(member).read()
            require(len(raw) == member.size == inventory[member.name]['bytes'] and
                    hashlib.sha256(raw).hexdigest() == inventory[member.name]['sha256'], 'source member changed: ' + member.name)
            if member.name.startswith('_provenance/') or member.name == '_build/CMakeCache.txt':
                saved[member.name] = raw
    require(seen == set(inventory), 'incomplete source member set')
    owned = (ROOT / 'owned.txt').read_text().splitlines()
    require(len(owned) == len(set(owned)) == 42, 'owned source cardinality')
    hashes = {name: inventory[name]['sha256'] for name in owned}
    require(hashes == frozen['owned_source_sha256'] == plan['identities']['owned_source_sha256'] ==
            plan['identities']['selected_source_sha256'], 'owned selected/main/gate identity')
    require('src/xir' in frozen['selection']['isolated_paths'], 'whole XIR selection missing')
    for name in ('src/xir/metadata.cpp', 'src/xir/verifier.cpp', 'src/xir/translators/xir_interchange.cpp',
                 'src/xir/translators/xir2text.cpp', 'src/xir/passes/inline.cpp',
                 'include/luisa/xir/metadata/strided_mma.h', 'include/luisa/xir/metadata/contiguous_copy.h',
                 'src/backends/simd/schedule/contiguous_copy.h', 'src/backends/simd/schedule/contiguous_copy_types.h',
                 'src/tile/bridge/xir/native_copy.h', 'src/backends/simd/llvm/llvm_schedule_emitter_mma.cpp',
                 'src/tests/common/tile_llm_test_utils.h', 'src/tests/common/tile_llm_benchmark.h'):
        require(name in inventory, 'required implementation absent: ' + name)
    for name in ('run.py', 'gate.py', 'owned.txt', 'freeze.py', 'package.py', 'plan.json'):
        require(sha(ROOT / name) == inventory['_experiment/' + name]['sha256'], 'frozen experiment file changed: ' + name)
    declarations = {}
    for node in ast.parse((ROOT / 'run.py').read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('CASES', 'VARIANTS', 'TESTS'):
                    require(target.id not in declarations, 'duplicate literal declaration')
                    declarations[target.id] = ast.literal_eval(node.value)
    require(declarations == dict(CASES=CASES, VARIANTS=VARIANTS, TESTS=TESTS), 'runner literal cases/variants/test drift')
    require(plan['controls']['cases'] == [dict(case=n, dimensions=list(d), block=list(b), cap=c, two_dimensional=t)
                                         for n, d, b, c, t in CASES], 'planned shapes')
    require(plan['controls']['variants'] == [dict(name=n, native_copy_vector_width=w) for n, w in VARIANTS], 'planned arms')
    for key, value in dict(native_mma_vector_width=4, requested_mma_output_block=4, packet_width=8,
                           workers_per_block=32, local_lanes=1, max_unrolled_tile_elements=64,
                           max_unrolled_region_work=4096, full_packet_specialization=True,
                           attention_qk='mma', attention_pv='mma', precision='fp32', fast_math=False,
                           native_cycles=3, native_samples=7, native_warmup_ms=30, native_target_ms=15,
                           capture_samples=3, capture_sample_ms=1, capture_warmup_ms=1, capture_timeout_s=180,
                           replay_timeout_s=120, require_ab_root_order_equal=True,
                           require_ab_blocks_per_task_equal=True, require_equal_capacity=True,
                           require_ab_bitwise_output=True, require_own_capture_bitwise_replay=True,
                           metric='single_thread_native_entry_host_wall_us', atol=5e-5, rtol=5e-5,
                           expected_native_counts={'off': [2, 1, 1], 'on': [2, 1, 1]},
                           expected_native_copies={'off': 0, 'on': 3},
                           count_order=['native_mmas', 'native_contraction_mmas', 'native_output_mmas']).items():
        require(plan['controls'].get(key) == value, 'plan control changed: ' + key)
    receipts = {}
    for record in (*frozen['binary_fingerprints'].values(), *frozen['tool_fingerprints'].values()):
        for key in ('path', 'resolved_path'):
            require(record[key] not in receipts or receipts[record[key]] == record['sha256'], 'conflicting receipt identity')
            receipts[record[key]] = record['sha256']
    for original, expected in plan['identities']['files_sha256'].items():
        if MARKER in PurePosixPath(original).parts:
            actual = sha(safe(ROOT, archive_relative(original)))
        elif original in receipts:
            actual = receipts[original]
        else:
            names = [str(PurePosixPath(original).relative_to(base)) for base in (frozen['main_root'], frozen['isolated_source'])
                     if PurePosixPath(original).is_relative_to(base)]
            require(len(names) == 1 and names[0] in inventory, 'unmapped source receipt: ' + original)
            actual = inventory[names[0]]['sha256']
        require(actual == expected, 'plan/provenance receipt changed: ' + original)
    abi = frozen['main_isolated_abi_equality']
    require(abi['main_sha256'] == abi['isolated_sha256'] == inventory[abi['path']]['sha256'], 'ABI identity')
    head = saved['_provenance/main-head.raw']
    require(head == (frozen['main_head'] + '\n').encode() and re.fullmatch(rb'[0-9a-f]{40}\n', head), 'raw HEAD receipt')
    for row in frozen['git_records'].values():
        require(hashlib.sha256(saved[row['stdout_member']]).hexdigest() == row['stdout_sha256'], 'raw git bytes changed')
    cache_roots = re.findall(rb'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', saved['_build/CMakeCache.txt'], re.M)
    require(len(cache_roots) == 1 and historical_tmp_alias(cache_roots[0].decode()) ==
            historical_tmp_alias(frozen['isolated_source']), 'actual selected CMake root (Darwin /tmp alias)')
    # All terminal historical gate files must be present both loose and frozen.
    # Final attempt numbers are not guessed while the build is still running.
    actual_validation = {p.name for p in ROOT.iterdir()
                         if re.fullmatch(r'(full-build|regression|syntax)-.+\.(json|log)', p.name)}
    expected_validation = {str(PurePosixPath(n).relative_to('_validation'))
                           for n in frozen['validation_members']}
    require(actual_validation == expected_validation and actual_validation, 'validation attempts missing/extra')
    expected_attempts = {str(PurePosixPath(n).with_suffix('')) for n in actual_validation}
    require(actual_validation == {n + suffix for n in expected_attempts for suffix in ('.json', '.log')},
            'validation attempt lacks paired JSON/log')
    cpp = [(i, name) for i, name in enumerate(owned) if name.endswith('.cpp')]
    attempts = []
    final_syntax = {}
    for name in sorted(expected_attempts):
        gate = load(ROOT / (name + '.json'))
        for suffix in ('.json', '.log'):
            require(sha(ROOT / (name + suffix)) == inventory['_validation/' + name + suffix]['sha256'],
                    'validation bytes changed: ' + name)
        require(gate['log_sha256'] == sha(ROOT / (name + '.log')) and type(gate['returncode']) is int and
                gate['started'] <= gate['finished'] <= frozen['started_unix'], 'terminal validation evidence')
        require(set(gate['source_sha256']) == set(owned), 'historical gate owned coverage')
        require(historical_tmp_alias(gate['selected_source']) == historical_tmp_alias(frozen['isolated_source']) and
                historical_tmp_alias(gate['build']) == historical_tmp_alias(frozen['selected_build']), 'gate selected source/build')
        # A failed build can legitimately contain earlier bytes in these same
        # owned files. Preserve its actual status and its old hashes.
        attempts.append(dict(name=name, returncode=gate['returncode'], source_unchanged=gate['source_unchanged'],
                             owned_source_count=len(gate['source_sha256']),
                             source_matches_final=gate['source_sha256'] == hashes, log_sha256=gate['log_sha256']))
        syntax_match = re.fullmatch(r'syntax-(\d+)-(\d+)', name)
        if syntax_match:
            generation, index = map(int, syntax_match.groups())
            require((index, owned[index]) in cpp, 'syntax index is not an owned C++ TU')
            expected_command = [frozen['main_root'] + '/scripts/check_cpp_syntax.py',
                                frozen['isolated_source'] + '/' + owned[index],
                                '--project-root', frozen['isolated_source'], '--compile-commands-dir', frozen['selected_build'],
                                '--clangd', '/opt/homebrew/opt/llvm/bin/clangd']
            require(gate['command'][1:] == expected_command, 'syntax command coverage')
            if index not in final_syntax or generation > final_syntax[index][0]:
                final_syntax[index] = (generation, gate)
    require(set(final_syntax) == {i for i, _ in cpp}, 'final syntax TU coverage')
    require(len({version for version, _ in final_syntax.values()}) == 1, 'incomplete final syntax generation')
    for _, gate in final_syntax.values():
        require(gate['returncode'] == 0 and gate['source_unchanged'] is True and gate['source_sha256'] == hashes,
                'final syntax gate not successful/current')
    selected = [load(ROOT / plan[key]) for key in ('build_gate', 'test_gate')]
    commands = [['cmake', '--build', frozen['selected_build'], '-j', '6'],
                ['ctest', '--test-dir', frozen['selected_build'], '--output-on-failure', '-R', '^(' + '|'.join(TESTS) + ')$']]
    for gate, command in zip(selected, commands):
        require(gate['command'] == command and gate['returncode'] == 0 and gate['source_unchanged'] is True and
                gate['source_sha256'] == hashes and
                historical_tmp_alias(gate['selected_source']) == historical_tmp_alias(frozen['isolated_source']) and
                historical_tmp_alias(gate['build']) == historical_tmp_alias(frozen['selected_build']),
                'selected full gate failed or changed')
    ctest = (ROOT / Path(plan['test_gate']).with_suffix('.log')).read_text()
    require(re.search(r'100% tests passed(?:, 0 tests failed)? out of 11\b', ctest), 'eleven-test completion acknowledgement')
    for name in TESTS:
        require(len(re.findall(r'\b' + re.escape(name) + r'\s+\.+\s+Passed\b', ctest)) == 1,
                'missing/duplicate passing test: ' + name)
    require(selected[0]['finished'] <= selected[1]['started'] <= selected[1]['finished'] <= plan['frozen_unix'] <=
            frozen['started_unix'] <= frozen['frozen_unix'], 'build/test/plan/source chronology')
    REPORT['gates'] = dict(source_members=len(inventory), owned_sources=len(owned), ctests=11,
                           final_syntax_tus=len(cpp), historical_attempts=attempts,
                           selected_build=plan['build_gate'], selected_test=plan['test_gate'],
                           ctest_seconds=float(re.search(r'Total Test time \(real\) =\s*([\d.]+)', ctest)[1]))
    return plan, frozen, inventory


def command(base, stem, timeout):
    initial = load(base / (stem + '.command.json'))
    final = load(base / (stem + '.result.json'))
    require(all(final.get(k) == v for k, v in initial.items()), 'command/result initial receipt differs')
    require(final['returncode'] == 0 and 'error' not in final and final['process_group_isolated'] is True and
            final['timeout_s'] == timeout and final['started'] <= final['finished'], 'failed command/isolation/timeout')
    require(safe(base, stem + '.stderr').read_bytes() == b'', 'stderr requires review: ' + stem)
    safe(base, stem + '.stdout')
    for suffix in ('.command.json', '.result.json', '.stdout', '.stderr'):
        sha(base / (stem + suffix))
    COMMANDS.add((base.relative_to(ROOT) / (stem + '.command.json')).as_posix())
    return final


def shape_info(dims):
    b, h, kh, q, k, d, dv = dims
    require(all(type(x) is int and x > 0 for x in dims) and h % kh == 0 and q <= k, 'shape admission')
    return [(b, h, q, d), (b, kh, k, d), (b, kh, k, dv)], (b, h, q, dv)


def array(path, shape, dtype):
    require(math.prod(shape) <= 2**26 and safe(ROOT, path.relative_to(ROOT).as_posix()).stat().st_size ==
            math.prod(shape) * np.dtype(dtype).itemsize, 'tensor exact byte extent: ' + str(path))
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    require(np.isfinite(value).all(), 'nonfinite tensor: ' + str(path))
    return value


def reference(base, dims):
    shapes, output_shape = shape_info(dims)
    query, key, value = [array(base / f'input{i}.f32', shape, '<f4').astype(np.float64) for i, shape in enumerate(shapes)]
    b, h, kh, q, k, d, dv = dims
    result = np.empty(output_shape, dtype=np.float64)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    visible = np.arange(k)[None, :] <= np.arange(q)[:, None] + k - q
    for batch in range(b):
        for head in range(h):
            group = head // (h // kh)
            score = np.einsum('qd,kd->qk', query[batch, head], key[batch, group], optimize=False) * scale
            score[~visible] = -np.inf
            weight = np.exp(score - score.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            result[batch, head] = np.einsum('qk,kd->qd', weight, value[batch, group], optimize=False)
    require(np.isfinite(result).all(), 'nonfinite independent FP64 oracle')
    return result


def error_check(actual, expected, tolerance=5e-5):
    difference = np.abs(actual.astype(np.float64) - expected)
    require(np.all(difference <= tolerance * (1 + np.abs(expected))), 'full FP64 oracle failure')
    return float(difference.max())


def numeric_receipt(row, actual, stored):
    maximum = error_check(actual, stored)
    require(row['elements'] == stored.size and row['atol'] == row['rtol'] == 5e-5 and
            math.isclose(maximum, row['max_abs_error'], rel_tol=1e-12, abs_tol=1e-15), 'reported numerical check differs')
    return maximum


def realization(metadata, dims, block, cap, two_d, width):
    shapes, output_shape = shape_info(dims)
    required = dict(implementation='tile_xir_simd', backend='cpu', operation='attention', precision='fp32',
                    fast_math=False, relaxed_precision=False, attention_qk='mma', attention_pv='mma',
                    source_kind='tile_lowering_source', source_reduction_policy='unordered_tree',
                    dimensions=list(dims), attention_block=list(block), input_shapes=[list(x) for x in shapes],
                    output_shape=list(output_shape), reduction_tree=False, requested_input_views=False, requested_group_threads=0)
    require(all(metadata.get(k) == v for k, v in required.items()), 'capture mathematical contract')
    text = metadata['realization']
    require(re.findall(r'\bW(\d+), (\d+) workers/block\b', text) == [('8', '32')], 'packet/block realization')
    for key, value in dict(local_lanes=1, max_unrolled_tile_elements=64, max_unrolled_region_work=4096,
                           requested_mma_output_block=4, requested_max_unrolled_mma_terms=cap,
                           native_mma_vector_width=4, native_copy_vector_width=width).items():
        require(field(text, key) == value, 'requested control: ' + key)
    require(field(text, 'requested_mma_2d_blocking', True) is two_d and not field(text, 'fast_math', True), '2D/FP control')
    require([field(text, k) for k in ('native_mmas', 'native_contraction_mmas', 'native_output_mmas')] ==
            [2, 1, 1], 'actual QK/PV native admission')
    require(field(text, 'native_copies') == (3 if width else 0), 'three static Q/K/V copy sites')
    orders = re.findall(r'\broot order \[([0-9]+(?:\s*,\s*[0-9]+)*)\]', text)
    require(len(orders) == 1, 'root-order acknowledgement')
    order = [int(x) for x in orders[0].split(',')]
    require(sorted(order) == list(range(len(order))), 'root order permutation')
    resources = {k: field(text, k) for k in OBSERVED}
    require(resources['static_snapshot_bytes_per_worker'] <= resources['snapshot_budget'], 'snapshot budget exceeded')
    return dict(root_order=order, blocks_per_task=field(text, 'blocks_per_task')), resources


def capture_arm(case, variant, width, row, frozen, inventory, plan):
    name, dims, block, cap, two_d = case
    label = name + '-' + variant
    base, prepared_dir = ROOT / label, ROOT / ('prepared-' + label)
    capture = command(base, 'capture', 180)
    prepare = command(base, 'prepare', 180)
    require(frozen['frozen_unix'] <= capture['started'] <= capture['finished'] <= prepare['started'], 'capture source chronology')
    require(capture['command'][:-1] == [frozen['selected_build'] + '/bin/benchmark_tile_xir', 'llm', 'attention',
            ','.join(map(str, dims)), *map(str, block), '3', '1', '1'] and
            archive_relative(capture['command'][-1]) == label + '/output.f32', 'producer exact argv')
    env = capture['environment']
    required_env = dict(LUISA_TILE_BENCH_XIR_BACKEND='simd', LUISA_TILE_BENCH_ATTENTION_QK='mma', LUISA_TILE_BENCH_ATTENTION_PV='mma',
                        LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK='4', LUISA_TILE_BENCH_XIR_MMA_UNROLL_TERMS=str(cap),
                        LUISA_TILE_BENCH_XIR_MMA_2D_BLOCKING=str(int(two_d)), LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32',
                        LUISA_TILE_BENCH_XIR_LOCAL_LANES='1', LUISA_TILE_BENCH_XIR_REGION_WORK='4096',
                        LUISA_SIMD_NATIVE_MMA_VECTOR_WIDTH='4', LUISA_SIMD_NATIVE_COPY_VECTOR_WIDTH=str(width),
                        LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1',
                        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    require(set(env) == {*required_env, 'LUISA_TILE_BENCH_DUMP_SOURCE', 'LUISA_SIMD_DUMP_ASSEMBLY_DIR'} and
            all(env[k] == v for k, v in required_env.items()), 'capture environment drift')
    require(archive_relative(env['LUISA_TILE_BENCH_DUMP_SOURCE']) == label + '/output.f32.source.txt' and
            archive_relative(env['LUISA_SIMD_DUMP_ASSEMBLY_DIR']) == label + '/objects', 'capture artifact paths')
    require(prepare['environment'] == {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}, 'prepare environment')
    expected_argv = [python_launcher(prepare['command'][0]), frozen['main_root'] + '/scripts/benchmark/tile_torch/native_tile.py', 'prepare', '--capture-kind', 'llm',
                     '--llvm', plan['llvm'], '--prefix', capture['command'][-1], '--log', str(PurePosixPath(capture['command'][-1]).parent / 'capture.stdout'),
                     '--objects', env['LUISA_SIMD_DUMP_ASSEMBLY_DIR'], '--output', str(PurePosixPath(capture['command'][-1]).parent.parent / ('prepared-' + label)), '--name', variant]
    require(prepare['command'] == expected_argv, 'prepare exact argv')
    metadata, manifest = load(base / 'capture.stdout'), load(prepared_dir / 'prepared.json')
    execution, resources = realization(metadata, dims, block, cap, two_d, width)
    require(manifest['status'] == 'prepared' and manifest['format'] == 'native-tile-entry-v1' and manifest['capture_kind'] == 'llm' and
            manifest['name'] == variant and manifest['metadata'] == metadata, 'prepared metadata')
    require(prepare['started'] <= manifest['started_unix'] <= manifest['finished_unix'] <= prepare['finished'], 'prepare chronology')
    require(row == dict(label=label, prepared_manifest_sha256=sha(prepared_dir / 'prepared.json'), execution_controls=execution,
                        realization=metadata['realization']), 'capture summary receipt')
    actual_files = {p.relative_to(prepared_dir).as_posix() for p in prepared_dir.rglob('*') if p.is_file() and p.name != 'prepared.json'}
    require(actual_files == set(manifest['files']), 'prepared complete file inventory')
    for relative, expected in manifest['files'].items():
        require(sha(safe(prepared_dir, relative)) == expected, 'prepared artifact changed: ' + relative)
    for relative, source in SOURCE_FILES.items():
        require(manifest['files'][relative] == inventory[source]['sha256'], 'helper/runner/ABI source differs')
    originals = {}
    for original, expected in manifest['original_sha256'].items():
        if MARKER in PurePosixPath(original).parts:
            relative = archive_relative(original)
            require(relative.startswith(label + '/'), 'original from wrong capture')
            actual = sha(safe(ROOT, relative))
        else:
            require(PurePosixPath(original).is_relative_to(frozen['main_root']), 'unmapped original source receipt')
            relative = str(PurePosixPath(original).relative_to(frozen['main_root']))
            require(relative in SOURCE_FILES.values(), 'unexpected source dependency')
            actual = inventory[relative]['sha256']
        require(relative not in originals and actual == expected, 'original capture identity')
        originals[relative] = expected
    objects = list((base / 'objects').glob('*.o'))
    require(len(objects) == 1 and sha(objects[0]) == manifest['files']['kernel.o'] and
            sha(base / 'output.f32.source.txt') == manifest['files']['kernel.ll'] and
            sha(base / 'capture.stdout') == manifest['files']['capture.log'], 'actual ORC/LLVM/log identity')
    expected_originals = {label + '/capture.stdout', label + '/output.f32.source.txt', label + '/output.f32',
                          label + '/output.f32.expected.f64', *[label + f'/output.f32.input{i}.f32' for i in range(3)],
                          objects[0].relative_to(ROOT).as_posix(), *SOURCE_FILES.values()}
    require(set(originals) == expected_originals, 'complete original capture identities')
    llvm = (prepared_dir / 'kernel.ll').read_text()
    symbols = re.findall(r'^define dso_local void @([-a-zA-Z$._0-9]+\.packet_batch(?:\.blocks)?)\(', llvm, re.M)
    blocks = [symbol for symbol in symbols if symbol.endswith('.blocks')]
    symbols = blocks if blocks else symbols
    require(symbols == [manifest['symbol']] and manifest['abi'] == (0 if blocks else 2) and
            not re.search(r'cooperative[._]|@llvm\.coro\.', llvm), 'actual entry ABI admission: ' + label)
    # Derive the ABI from this arm's actual LLVM/export receipts. Do not reuse
    # the previous experiment's clone/entry inventory as a guessed invariant.
    require(manifest['symbol'] == 'llm_attention.packet_batch' + ('.blocks' if blocks else ''), 'actual attention entry')
    require(manifest['packet_width'] == 8 and manifest['block'] == [32, 1, 1] and manifest['dispatch'] == metadata['dispatch'] and
            len(manifest['dispatch']) == 3 and all(type(x) is int and 0 < x <= 0xffffffff for x in manifest['dispatch']) and
            manifest['workspace_bytes'] == resources['private_workspace_bytes'], 'prepared launch/resource contract')
    tools = {PurePosixPath(p).name: (p, h) for p, h in manifest['tool_sha256'].items()}
    require(set(tools) == {'clang++', 'llvm-nm'} and len(tools) == len(manifest['tool_sha256']), 'prepared compiler identity coverage')
    for name, key in (('clang++', 'llvm_clangxx'), ('llvm-nm', 'llvm_nm')):
        receipt = frozen['tool_fingerprints'][key]
        require(tools[name][1] == receipt['sha256'] and PurePosixPath(tools[name][0]).name == name and
                PurePosixPath(tools[name][0]).parent in (PurePosixPath(receipt['path']).parent,
                                                     PurePosixPath(receipt['resolved_path']).parent), 'actual compiler/tool receipt')
    historical_dir = str(PurePosixPath(next(p for p in manifest['original_sha256'] if p.endswith('/capture.stdout'))).parent.parent / ('prepared-' + label))
    cc, nm = tools['clang++'][0], tools['llvm-nm'][0]
    commands = dict(imports=[nm, '--undefined-only', '--just-symbol-name', historical_dir + '/kernel.o'],
                    exports=[nm, '--defined-only', '--extern-only', '--just-symbol-name', historical_dir + '/kernel.o'],
                    compiler=[cc, '--version'], link=[cc, '-dynamiclib', historical_dir + '/kernel.o', '-o', historical_dir + '/kernel.dylib'],
                    helper=[cc, '-std=c++20', '-O3', '-dynamiclib', '-I' + historical_dir, historical_dir + '/native_tile_replay.cpp', '-o', historical_dir + '/replay.dylib'])
    for stem, argv in commands.items():
        receipt = load(prepared_dir / (stem + '.command.json'))
        require(receipt == dict(argv=argv, returncode=0) and (prepared_dir / (stem + '.stderr')).read_bytes() == b'', 'preparation compiler/link command')
        COMMANDS.add((prepared_dir.relative_to(ROOT) / (stem + '.command.json')).as_posix())
    imports = (prepared_dir / 'imports.stdout').read_text().split()
    require(imports == manifest['imports'] and set(imports) <= {'_memcpy', '_memset', '_bzero', '___chkstk_darwin'}, 'uninspected actual ORC imports')
    require('_' + manifest['symbol'] in (prepared_dir / 'exports.stdout').read_text().split(), 'actual ORC exported entry missing')
    require(manifest['input_files'] == ['input0.f32', 'input1.f32', 'input2.f32'] and manifest['atol'] == manifest['rtol'] == 5e-5, 'full payload contract')
    for source, target in [('output.f32', 'captured.f32'), ('output.f32.expected.f64', 'expected.f64'),
                           *[(f'output.f32.input{i}.f32', f'input{i}.f32') for i in range(3)]]:
        require(sha(base / source) == manifest['files'][target], 'raw/prepared payload differs')
    return manifest, execution, resources, prepare['finished']


def replay_case(case, row, manifests, expected, stored, capture_hashes, check, plan, frozen, earliest):
    name = case[0]
    base = ROOT / ('replay-' + name)
    execution = command(ROOT, 'replay-' + name, 120)
    result = load(base / 'results.json')
    require(row['status'] == 'OK' and row['results_sha256'] == sha(base / 'results.json'), 'replay matrix receipt')
    require(execution['environment'] == {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}, 'replay environment')
    argv = execution['command']
    require(argv[:3] == [python_launcher(argv[0]), frozen['main_root'] + '/scripts/benchmark/tile_torch/native_tile.py', 'replay'] and
            argv[3::2] == ['--prepared', '--prepared', '--output', '--cycles', '--samples', '--warmup-ms', '--target-ms'] and
            [archive_relative(x) for x in argv[4:9:2]] == [f'prepared-{name}-off/prepared.json', f'prepared-{name}-on/prepared.json', 'replay-' + name] and
            argv[10::2] == ['3', '7', '30', '15'], 'replay exact argv')
    require(earliest <= execution['started'] <= result['started_unix'] <= result['finished_unix'] <= execution['finished'], 'replay chronology/order')
    require(result['status'] == 'passed' and result['artifacts_unchanged'] is True and result['metric'] == 'single_thread_native_entry_host_wall_us' and
            result['cpu_threads'] == 1 and result['boundary'] == BOUNDARY and result['order_policy'] == 'ABBA per cycle; two matched pairs per cycle', 'native timing boundary')
    require(result['runner_sha256'] == manifests[0]['files']['native_tile.py'], 'executing runner identity')
    required_prepared = {f'prepared-{name}-{variant}/prepared.json': sha(ROOT / f'prepared-{name}-{variant}/prepared.json') for variant, _ in VARIANTS}
    require({archive_relative(p): h for p, h in result['prepared_sha256'].items()} == required_prepared, 'replay prepared identity')
    options = result['options']
    require(options == dict(command='replay', prepared=argv[4:7:2], output=argv[8], cycles=3, samples=7,
                            warmup_ms=30, target_ms=15, validate_only=False), 'timing options')
    require(len(result['visits']) == 12, 'ABBA visit coverage')
    errors, outputs = [], set()
    for index, visit in enumerate(result['visits']):
        variant = ('off', 'on', 'on', 'off')[index % 4]
        require((visit['cycle'], visit['position'], visit['variant']) == (index // 4, index % 4, variant), 'ABBA order')
        require(visit['valid'] is True and visit['returncode'] == 0 and visit['all_guards_passed'] is True and visit['inputs_unchanged'] is True, 'native helper visit preflight/batch guard/readonly check')
        require(type(visit['repetitions']) is int and visit['repetitions'] > 0 and len(visit['samples_us']) == 7 and
                all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in visit['samples_us']), 'timing sample/repetition admission')
        require(stats.median(visit['samples_us']) == visit['median_us'], 'visit median')
        require(visit['output'] == f'visit-{index // 4}-{index % 4}.f32' and visit['output'] not in outputs, 'unique complete visit outputs')
        outputs.add(visit['output'])
        output = safe(base, visit['output'])
        require(sha(output) == visit['output_sha256'] == capture_hashes[variant], 'replay is not bitwise equal to its own capture')
        actual = array(output, expected.shape, '<f4')
        errors.append(error_check(actual, expected))
        maximum = numeric_receipt(visit['correctness'], actual, stored)
        require(math.isclose(maximum, visit['helper_max_abs_error'], abs_tol=1e-15, rel_tol=1e-12), 'C++ helper/Python oracle error')
    require({p.name for p in base.iterdir()} == outputs | {'results.json'}, 'replay output inventory')
    medians = {v: stats.median(x['median_us'] for x in result['visits'] if x['variant'] == v) for v, _ in VARIANTS}
    samples = {v: [t for x in result['visits'] if x['variant'] == v for t in x['samples_us']] for v, _ in VARIANTS}
    require(all(len(x) == 42 for x in samples.values()), 'all-sample arm coverage')
    pairs = [result['visits'][a]['median_us'] / result['visits'][b]['median_us'] for c in range(3) for a, b in ((4*c+1, 4*c), (4*c+2, 4*c+3))]
    ratio = dict(baseline='off', candidate='on', pairs=pairs, median=stats.median(pairs), minimum=min(pairs), maximum=max(pairs))
    require(result['summary_us'] == medians and result['candidate_over_baseline'] == ratio, 'reported ABBA summary')
    require(row['correctness'] == check, 'replay/capture correctness receipts differ')
    summary = dict(status='passed', all_sample_median_us={v: stats.median(s) for v, s in samples.items()},
                   all_sample_on_over_off=stats.median(samples['on']) / stats.median(samples['off']),
                   all_sample_off_over_on=stats.median(samples['off']) / stats.median(samples['on']),
                   visit_median_us=medians, paired_on_over_off=ratio, visits=12, samples=84,
                   max_abs_error=max(errors), results_sha256=sha(base / 'results.json'))
    return summary, execution['finished']


def audit():
    plan, frozen, inventory = sources_and_gates()
    captures, replays = load(ROOT / 'captures.json'), load(ROOT / 'replays.json')
    REPORT['raw_statuses'] = dict(capture=[dict(case=r['case'], status=r['status']) for r in captures['cases']],
                                  replay=[dict(case=r['case'], status=r['status']) for r in replays['experiments']])
    require(captures['plan_sha256'] == replays['plan_sha256'] == PLAN_SHA256 and captures['status'] == replays['status'] == 'complete', 'incomplete/changed matrix')
    require([r['case'] for r in captures['cases']] == [r['case'] for r in replays['experiments']] == [n for n, *_ in CASES], 'complete six-case order')
    require(all(r['status'] == 'OK' for r in (*captures['cases'], *replays['experiments'])), 'Error/NotRun cohort cannot pass audit')
    require(frozen['frozen_unix'] <= captures['started'] <= captures['updated'] <= replays['started'] <= replays['updated'], 'capture/replay chronology')
    REPORT['chronology'] = dict(plan=plan['frozen_unix'], source=frozen['frozen_unix'], capture_started=captures['started'],
                                capture_finished=captures['updated'], replay_started=replays['started'], replay_finished=replays['updated'])
    previous_capture, previous_replay = captures['started'], replays['started']
    for ci, case in enumerate(CASES):
        name, dims, block, cap, two_d = case
        row = captures['cases'][ci]
        require([v['label'] for v in row['variants']] == [name + '-' + v for v, _ in VARIANTS], 'two capture arms')
        manifests, executions, resources = [], [], []
        for (variant, width), entry in zip(VARIANTS, row['variants']):
            capture_start = load(ROOT / (name + '-' + variant) / 'capture.command.json')['started']
            require(previous_capture <= capture_start, 'capture order overlap')
            manifest, execution, physical, previous_capture = capture_arm(case, variant, width, entry, frozen, inventory, plan)
            require(previous_capture <= captures['updated'] <= replays['started'], 'all preparation before replay')
            manifests.append(manifest); executions.append(execution); resources.append(physical)
        require(executions[0] == executions[1], 'A/B root-order/task-grain changed')
        for key in ('abi', 'symbol', 'packet_width', 'block', 'dispatch', 'tool_sha256', 'atol', 'rtol', 'input_files', 'output_elements'):
            require(manifests[0][key] == manifests[1][key], 'A/B entry/source contract: ' + key)
        for relative in ('input0.f32', 'input1.f32', 'input2.f32', 'expected.f64', *SOURCE_FILES):
            require(manifests[0]['files'][relative] == manifests[1]['files'][relative], 'A/B input/oracle/helper difference')
        expected = reference(ROOT / ('prepared-' + name + '-off'), dims)
        stored = array(ROOT / ('prepared-' + name + '-off') / 'expected.f64', expected.shape, '<f8')
        oracle_error = error_check(stored, expected, 1e-12)
        checks, capture_hashes = {}, {}
        for vi, ((variant, _), manifest) in enumerate(zip(VARIANTS, manifests)):
            base = ROOT / ('prepared-' + name + '-' + variant)
            own_expected = array(base / 'expected.f64', expected.shape, '<f8')
            error_check(own_expected, expected, 1e-12)
            actual = array(base / 'captured.f32', expected.shape, '<f4')
            error = error_check(actual, expected)
            numeric_receipt(manifest['capture_correctness'], actual, own_expected)
            require(manifest['output_elements'] == expected.size and manifest['metadata']['correctness']['elements_per_check'] == expected.size and
                    manifest['metadata']['correctness']['checks'] == 2 and manifest['metadata']['correctness']['guard_elements_per_check'] == 34 and
                    manifest['metadata']['correctness']['atol'] == manifest['metadata']['correctness']['rtol'] == 5e-5, 'capture check extent/guards')
            capture_hashes[variant] = manifest['files']['captured.f32']
            checks[variant] = dict(captured_output_sha256=capture_hashes[variant], max_abs_error=error,
                                   workspace_bytes=manifest['workspace_bytes'], physical={k: resources[vi][k] for k in PHYSICAL})
            REPORT['captures'].append(dict(case=name, variant=variant, status='passed', max_abs_error=error,
                                            captured_sha256=capture_hashes[variant], actual_object_sha256=manifest['files']['kernel.o'],
                                            llvm_sha256=manifest['files']['kernel.ll'], imports=manifest['imports'], resources=resources[vi],
                                            entry={k: manifest[k] for k in ('symbol', 'abi', 'dispatch', 'block', 'packet_width', 'workspace_bytes')}))
        require(capture_hashes['off'] == capture_hashes['on'], 'copy changed captured FP32 output bits')
        for key in ('static_snapshot_bytes_per_worker', 'static_snapshot_allocations'):
            require(resources[0][key] == resources[1][key], 'copy changed snapshot capacity: ' + key)
        check = row['correctness']
        require(check['independent_fp64'] is True and check['elements'] == expected.size and check['atol'] == check['rtol'] == 5e-5 and
                check['ab_bitwise_equal'] == (capture_hashes['off'] == capture_hashes['on']) and check['execution_controls'] == executions[0], 'pair numerical/exec receipt')
        require(set(check['variants']) == {'off', 'on'}, 'pair correctness arms')
        for variant, _ in VARIANTS:
            recorded = check['variants'][variant]
            calculated = checks[variant]
            require(all(recorded[k] == calculated[k] for k in calculated if k != 'max_abs_error') and
                    math.isclose(recorded['max_abs_error'], calculated['max_abs_error'], abs_tol=1e-12), 'independent pair oracle receipt')
        summary, previous_replay = replay_case(case, replays['experiments'][ci], manifests, expected, stored, capture_hashes,
                                               check, plan, frozen, previous_replay)
        REPORT['comparisons'][ci].update(summary, dimensions=list(dims), attention_block=list(block), cap=cap,
                                         two_dimensional=two_d, ab_bitwise_equal=check['ab_bitwise_equal'], execution_controls=executions[0],
                                         oracle_einsum_vs_stored_max_abs_error=oracle_error,
                                         resource_delta_on_minus_off={k: resources[1][k] - resources[0][k] for k in OBSERVED})
    require(previous_replay <= replays['updated'], 'final replay chronology')
    all_commands = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.command.json')}
    require(all_commands == COMMANDS and len(COMMANDS) == 90, 'raw command receipt coverage')
    REPORT.update(status='passed', captures_checked=12, visits=72, samples=504, commands_checked=len(COMMANDS),
                  source_archive_sha256=SOURCE_SHA256, plan_sha256=PLAN_SHA256, audit_sha256=sha(ROOT / 'audit.py'),
                  recorded_python_launchers=sorted(PYTHON_LAUNCHERS),
                  metric='single_thread_native_entry_host_wall_us',
                  table_convention='all_sample_median_us uses all 42 samples per arm; on/off < 1 is faster. visit_median_us is the median of six visit medians. Paired ratios separately use the six adjacent ABBA visit-median pairs.')


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        REPORT.update(status='error', error=str(error))
    print(json.dumps(REPORT, indent=2, allow_nan=False))
    raise SystemExit(0 if REPORT['status'] == 'passed' else 1)
