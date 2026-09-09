#!/usr/bin/env python3
"""Capture and replay actual SIMD/Inductor row-kernel entries, without Runtime.

Preparation and timing are separate commands. The generated Inductor wrapper
is parsed, not executed, to recover argument order, scratch and aliased views.
Only one static FP32 C++ entry is admitted. Compiler-emitted allocations and
libc calls remain inside native timing; host/Python allocations do not.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

import numpy as np

from compare_llm import check_metadata, parse_case, reference, shapes_for, validate_output

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OPS = ('rmsnorm', 'layernorm', 'masked_softmax', 'swiglu', 'gelu_residual', 'rope')
SHAPES = ((17, 65), (129, 768), (257, 1538), (1024, 4097))
LLVM_BIN = Path('/opt/homebrew/opt/llvm@21/bin')
PREDECESSOR = HERE / 'results/m1-max-20260909-cohort-private/provenance.json'
MAX_ELEMENTS = 2**26


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def snapshot_sources(directory):
    sources = directory / 'runner-sources'
    sources.mkdir()
    for path in (Path(__file__), HERE / 'native_rows_replay.cpp', HERE / 'compare_llm.py'):
        shutil.copy2(path, sources / path.name)


def command(argv, directory, stem, **kwargs):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, **kwargs)
    (directory / (stem + '.stdout.log')).write_text(result.stdout)
    (directory / (stem + '.stderr.log')).write_text(result.stderr)
    save(directory / (stem + '.command.json'), dict(argv=list(map(str, argv)), returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f'{stem}: exit {result.returncode}: {result.stderr[-2000:]}')
    return result.stdout


def case_name(op, dimensions):
    return op + '-' + 'x'.join(map(str, dimensions))


def contiguous_strides(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


def storage_size(shape, strides):
    if (len(shape) != len(strides) or not shape or
            any(type(x) is not int or x <= 0 for x in shape) or
            any(type(x) is not int or x < 0 for x in strides)):
        raise ValueError('expected positive static shapes and nonnegative strides')
    size = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    if size > MAX_ELEMENTS:
        raise ValueError('allocation exceeds benchmark bound')
    return size


def parse_inductor_wrapper(source, graph_inputs, input_shapes):
    """Fail-closed static ABI extraction; no eval/exec or operator-name rules."""
    tree = ast.parse(source)
    bindings = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and
                ast.unparse(node.value.func) == 'async_compile.cpp_pybinding'):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or len(node.value.args) != 2 or node.value.keywords:
                raise ValueError('unsupported C++ binding')
            bindings.append((node.targets[0].id, ast.literal_eval(node.value.args[0]), ast.literal_eval(node.value.args[1])))
    if len(bindings) != 1:
        raise ValueError('expected exactly one native C++ binding')
    name, types, cpp = bindings[0]
    if not isinstance(types, list) or not 3 <= len(types) <= 6 or any(t not in ('const float*', 'float*') for t in types):
        raise ValueError('unsupported native argument types')
    signatures = re.findall(r'extern\s+"C"\s+void\s+kernel\(([^)]*)\)', cpp)
    if len(signatures) != 1 or '#pragma omp parallel' in cpp:
        raise ValueError('expected one serial native entry')
    declarations = signatures[0].split(',')
    actual_types = []
    for declaration in declarations:
        match = re.fullmatch(r'\s*(const\s+)?float\s*\*\s*[a-zA-Z_][a-zA-Z_0-9]*\s*', declaration)
        if not match:
            raise ValueError('unrecognized C++ signature')
        actual_types.append('const float*' if match[1] else 'float*')
    if actual_types != types:
        raise ValueError('binding and actual C++ signature disagree')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'call']
    if len(calls) != 1:
        raise ValueError('expected one generated call wrapper')
    body = calls[0].body
    first = body[0]
    if not (isinstance(first, ast.Assign) and len(first.targets) == 1 and
            isinstance(first.targets[0], ast.Tuple) and ast.unparse(first.value) == 'args'):
        raise ValueError('unsupported argument unpack')
    unpack = first.targets[0].elts
    if len(unpack) != len(graph_inputs) or any(not isinstance(n, ast.Name) for n in unpack):
        raise ValueError('graph/wrapper input count differs')
    source_input = {'L_x_': 0, 'L_u_': 1, 'L_v_': 2}
    environment, allocations = {}, {}
    for variable, graph_input in zip(unpack, graph_inputs):
        if graph_input not in source_input:
            raise ValueError('unrecognized FX input: ' + graph_input)
        index = source_input[graph_input]
        shape = tuple(input_shapes[index])
        environment[variable.id] = dict(storage=f'input{index}', offset=0, shape=shape, strides=contiguous_strides(shape))
    arguments = output = None
    checked_inputs = set()

    def view(node):
        if not isinstance(node, ast.Name) or node.id not in environment:
            raise ValueError('unknown buffer expression')
        return dict(environment[node.id])

    def check_size(node, shape, strides):
        descriptor = view(node)
        if tuple(shape) != tuple(descriptor['shape']) or tuple(strides) != tuple(descriptor['strides']):
            raise ValueError('input shape/stride guard mismatch')
        checked_inputs.add(node.id)

    for node in body[1:]:
        if isinstance(node, ast.Assign):
            if arguments is not None or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError('unsupported allocation/assignment phase')
            target, value = node.targets[0].id, node.value
            if isinstance(value, ast.Name):
                environment[target] = view(value)
            elif isinstance(value, ast.Call) and not value.keywords:
                function = ast.unparse(value.func)
                if function == 'empty_strided_cpu' and len(value.args) == 3 and ast.unparse(value.args[2]) == 'torch.float32':
                    shape, strides = map(ast.literal_eval, value.args[:2])
                    if target in allocations or target in environment:
                        raise ValueError('duplicate allocation')
                    size = storage_size(shape, strides)
                    allocations[target] = dict(elements=size, shape=shape, strides=strides)
                    environment[target] = dict(storage=target, offset=0, shape=shape, strides=strides)
                elif function == 'reinterpret_tensor' and len(value.args) == 4:
                    parent = view(value.args[0])
                    shape, strides, offset = map(ast.literal_eval, value.args[1:])
                    size = storage_size(shape, strides)
                    if parent['offset'] != 0 or parent['storage'] not in allocations or type(offset) is not int or offset < 0 or offset + size > allocations[parent['storage']]['elements']:
                        raise ValueError('unsupported or out-of-bounds alias')
                    environment[target] = dict(storage=parent['storage'], offset=offset, shape=shape, strides=strides)
                else:
                    raise ValueError('unsupported wrapper allocation: ' + function)
            else:
                raise ValueError('unsupported wrapper assignment')
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id not in environment:
                    raise ValueError('unsupported deletion')
                del environment[target.id]
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            function = ast.unparse(call.func)
            if call.keywords:
                raise ValueError('keyword effects are not admitted')
            if function == 'args.clear' and not call.args and arguments is None:
                continue
            if function == 'assert_size_stride' and len(call.args) == 4 and ast.literal_eval(call.args[-1]) == 'input':
                check_size(call.args[0], ast.literal_eval(call.args[1]), ast.literal_eval(call.args[2]))
            elif function == 'assert_size_stride_grouped' and len(call.args) == 4 and ast.literal_eval(call.args[-1]) == 'input':
                names, shapes, strides = call.args[0], ast.literal_eval(call.args[1]), ast.literal_eval(call.args[2])
                if not isinstance(names, ast.Tuple) or len(names.elts) != len(shapes) or len(shapes) != len(strides):
                    raise ValueError('bad grouped input guards')
                for item, shape, stride in zip(names.elts, shapes, strides):
                    check_size(item, shape, stride)
            elif function == name and len(call.args) == len(types) and arguments is None:
                if checked_inputs != {n.id for n in unpack}:
                    raise ValueError('missing input shape/stride guards')
                arguments = [view(arg) for arg in call.args]
                for kind, descriptor in zip(types, arguments):
                    if kind == 'float*' and descriptor['storage'].startswith('input'):
                        raise ValueError('native replay may not mutate source inputs')
            else:
                raise ValueError('unsupported wrapper effect: ' + function)
        elif isinstance(node, ast.Return) and arguments is not None and output is None:
            if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 1:
                raise ValueError('expected one returned tensor')
            output = view(node.value.elts[0])
        else:
            raise ValueError('unsupported wrapper statement: ' + ast.dump(node))
    if arguments is None or output is None or output['storage'] not in allocations:
        raise ValueError('missing native call or allocated output')
    return dict(binding=name, types=types, const_mask=sum(1 << i for i, t in enumerate(types) if t == 'const float*'),
                arguments=arguments, allocations=allocations, output=output, cpp=cpp,
                compiler_internal_allocation_sites=len(re.findall(r'std::make_unique|\bmalloc\s*\(|\bcalloc\s*\(', cpp)))


def torch_program(torch, op, rows, columns):
    def invoke(x, u, v):
        if op == 'rmsnorm':
            return torch.nn.functional.rms_norm(x, (columns,), u[0], 1e-5)
        if op == 'layernorm':
            return torch.nn.functional.layer_norm(x, (columns,), u[0], v[0], 1e-5)
        if op == 'masked_softmax':
            mask = torch.arange(columns)[None, :] <= torch.arange(rows)[:, None] % columns
            return torch.softmax(torch.where(mask, x, -1e30), dim=-1)
        if op == 'swiglu':
            return torch.nn.functional.silu(x) * u
        if op == 'gelu_residual':
            return torch.nn.functional.gelu(x, approximate='tanh') + u
        if op == 'rope':
            left, right = x.chunk(2, dim=-1)
            return torch.cat((left * u - right * v, left * v + right * u), dim=-1)
        raise ValueError(op)
    return invoke


def load_inputs(directory, shapes):
    return [np.fromfile(directory / f'output.f32.input{i}.f32', dtype=np.float32).reshape(shape) for i, shape in enumerate(shapes)]


def capture(args):
    binary = args.binary.resolve()
    build = binary.parent.parent
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory)
    command(['caffeinate', '-i', 'cmake', '--build', build, '--parallel', '8'], directory, 'full-build', timeout=1800)
    predecessor = json.loads(PREDECESSOR.read_text())
    for key in ('tested_source_sha256', 'inherited_source_sha256'):
        for relative, sha in predecessor[key].items():
            if digest(REPOSITORY / relative) != sha or digest(build.parent / 'source' / relative) != sha:
                raise ValueError('source changed since verified checkpoint: ' + relative)
    closure = {str(p): digest(p) for p in [binary, *sorted(binary.parent.glob('libluisa-*'))] if p.is_file()}
    if closure != predecessor['binary_closure_sha256']:
        raise ValueError('binary closure differs from verified predecessor')
    selected = args.case or [(op, (m, n + (n % 2 if op == 'rope' else 0))) for op in OPS for m, n in SHAPES]
    if len(set((op, tuple(dims)) for op, dims in selected)) != len(selected) or any(op not in OPS for op, _ in selected):
        raise ValueError('unsupported or repeated row case')
    report = dict(started_unix=time.time(), cases=selected, mappings=[1, 8], packet_width=8, block_size=32,
                  source_sha256=digest(__file__), predecessor=str(PREDECESSOR), predecessor_sha256=digest(PREDECESSOR),
                  binary_closure_sha256=closure, capture_timing_not_comparative=True, results=[])
    save(directory / 'results.json', report)
    for op, dims in selected:
        for local in (1, 8):
            target = directory / (case_name(op, dims) + f'-l{local}')
            target.mkdir()
            output = target / 'output.f32'
            env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
            env.update(LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1',
                       LUISA_TILE_BENCH_XIR_LOCAL_LANES=str(local), LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32',
                       LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK='0', LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
                       LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1', LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1',
                       LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1', LUISA_TILE_BENCH_DUMP_SOURCE=str(target / 'kernel.ll'),
                       LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(target / 'object'))
            row = dict(operation=op, dimensions=dims, local_lanes=local, directory=str(target), valid=False,
                       environment={k: v for k, v in env.items() if k.startswith('LUISA_')})
            try:
                text = command([binary, 'llm', op, ','.join(map(str, dims)), '1', '1', '3', '1', '1', output], target, 'capture', env=env, timeout=300)
                measurement = json.loads(text)
                check_metadata(measurement, 'cpu', op, dims, (1, 1), 3)
                shapes, out_shape = shapes_for(op, dims)
                arrays = load_inputs(target, shapes)
                check = validate_output(np.fromfile(output, dtype=np.float32).reshape(out_shape), reference(op, dims, arrays))
                save(target / 'measurement.json', measurement)
                row.update(valid=True, correctness=check, input_sha256=[array_digest(a) for a in arrays], output_sha256=digest(output))
            except Exception as error:
                row['error'] = str(error)
            report['results'].append(row)
            save(directory / 'results.json', report)
            print(target.name, 'valid' if row['valid'] else row['error'], flush=True)
    report.update(finished_unix=time.time(), closure_unchanged=closure == {p: digest(p) for p in closure},
                  runner_unchanged=report['source_sha256'] == digest(__file__))
    save(directory / 'results.json', report)
    if not report['closure_unchanged'] or not report['runner_unchanged'] or not all(r['valid'] for r in report['results']):
        raise RuntimeError('capture failures retained')


def prepare(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory)
    capture_root = args.capture.resolve()
    captured = json.loads((capture_root / 'results.json').read_text())
    if not captured.get('closure_unchanged') or not all(r['valid'] for r in captured['results']):
        raise ValueError('incomplete capture')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(directory / 'inductor-cache')
    import torch
    from torch._inductor.utils import run_and_get_code
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = dict(started_unix=time.time(), source_sha256=digest(__file__), capture=str(capture_root),
                  capture_sha256=digest(capture_root / 'results.json'), torch_version=torch.__version__,
                  torch_git_version=torch.version.git_version, torch_config=torch.__config__.show(),
                  cpu_threads=1, cases=[])
    save(directory / 'manifest.json', report)
    for op, dims in captured['cases']:
        target = directory / case_name(op, dims)
        target.mkdir()
        shapes, out_shape = shapes_for(op, dims)
        first = capture_root / (case_name(op, dims) + '-l1')
        arrays = load_inputs(first, shapes)
        expected = reference(op, dims, arrays)
        entries = {}
        for local in (1, 8):
            source = capture_root / (case_name(op, dims) + f'-l{local}')
            other = load_inputs(source, shapes)
            if any(array_digest(a) != array_digest(b) for a, b in zip(arrays, other)):
                raise ValueError('mapping input bits differ')
            measurement = json.loads((source / 'measurement.json').read_text())
            llvm = (source / 'kernel.ll').read_text()
            symbol, abi = 'llm_rows.packet_batch.blocks', 0
            if f'define dso_local void @{symbol}(' not in llvm:
                symbol, abi = 'llm_rows.packet_batch', 2
            if f'define dso_local void @{symbol}(' not in llvm or 'W8, 32 workers/block' not in measurement['realization']:
                raise ValueError('unsupported emitted entry')
            match = re.search(r'local_lanes=(\d+)', measurement['realization'])
            if (int(match[1]) if match else 1) != local:
                raise ValueError('local mapping not acknowledged')
            workspace = re.search(r'private_workspace_bytes=(\d+)', measurement['realization'])
            objects = list((source / 'object').glob('*.o'))
            if len(objects) != 1:
                raise ValueError('expected one actual ORC object')
            variant = 'whole' if local == 1 else 'local'
            imports = command([LLVM_BIN / 'llvm-nm', '--undefined-only', '--just-symbol-name', objects[0]], target, variant + '-imports').split()
            if platform.system() != 'Darwin' or set(imports) - {'_memcpy', '_memset', '_bzero', '___chkstk_darwin'}:
                raise ValueError('uninspected ORC imports: ' + str(imports))
            library = target / (variant + '.dylib')
            command(['clang++', '-dynamiclib', objects[0], '-o', library], target, variant + '-link')
            command([LLVM_BIN / 'llvm-objdump', '--disassemble', '--no-show-raw-insn', objects[0]], target, variant + '-assembly')
            entries[variant] = dict(library=str(library), library_sha256=digest(library), symbol=symbol, abi=abi,
                                    local_lanes=local, block_size=32, packet_width=8,
                                    workspace_bytes=int(workspace[1]) if workspace else 0,
                                    capture=str(source), object_sha256=digest(objects[0]), llvm_sha256=digest(source / 'kernel.ll'),
                                    system_imports=imports, realization=measurement['realization'])
        graphs = []

        def backend(graph, inputs):
            graphs.append(dict(source=graph.code, inputs=[str(n.target) for n in graph.graph.nodes if n.op == 'placeholder']))
            return torch._inductor.compile(graph, inputs)

        output, codes = run_and_get_code(torch.compile(torch_program(torch, op, *dims), backend=backend, fullgraph=True, dynamic=False),
                                        *(torch.from_numpy(a) for a in arrays))
        validation = validate_output(output.numpy(), expected)
        if len(codes) != 1 or len(graphs) != 1:
            raise ValueError('expected one compiled graph/wrapper')
        (target / 'inductor-wrapper.py').write_text(codes[0])
        save(target / 'inductor-graph.json', graphs[0])
        plan = parse_inductor_wrapper(codes[0], graphs[0]['inputs'], shapes)
        if tuple(plan['output']['shape']) != tuple(out_shape):
            raise ValueError('unexpected returned tensor shape')
        cpp = plan.pop('cpp')
        sources = [p for p in (directory / 'inductor-cache').rglob('*.cpp') if cpp.strip() in p.read_text()]
        if len(sources) != 1 or not sources[0].with_suffix('.so').is_file():
            raise ValueError('actual generated source/library not uniquely found')
        shutil.copy2(sources[0], target / 'inductor.cpp')
        shutil.copy2(sources[0].with_suffix('.so'), target / 'inductor.so')
        command([LLVM_BIN / 'llvm-objdump', '--disassemble-symbols=_kernel', '--no-show-raw-insn', target / 'inductor.so'], target, 'inductor-assembly')
        entries['inductor'] = dict(library=str(target / 'inductor.so'), library_sha256=digest(target / 'inductor.so'),
                                   source_sha256=digest(target / 'inductor.cpp'), symbol='kernel', abi=1, plan=plan,
                                   source=str(sources[0]), graph=graphs[0], wrapper_sha256=digest(target / 'inductor-wrapper.py'))
        case = dict(operation=op, dimensions=dims, input_shapes=shapes, output_shape=out_shape,
                    inputs=str(first), input_sha256=[array_digest(a) for a in arrays], entries=entries,
                    compiled_wrapper_correctness=validation)
        report['cases'].append(case)
        save(directory / 'manifest.json', report)
        print(case_name(op, dims), plan['types'], 'prepared', flush=True)
    helper = directory / 'native_rows_replay.dylib'
    command(['clang++', '-std=c++20', '-O3', '-dynamiclib', '-I' + str(REPOSITORY / 'src'),
             HERE / 'native_rows_replay.cpp', '-o', helper], directory, 'helper-build')
    report.update(finished_unix=time.time(), helper=str(helper), helper_sha256=digest(helper),
                  helper_source_sha256=digest(HERE / 'native_rows_replay.cpp'),
                  abi_header_sha256=digest(REPOSITORY / 'src/backends/simd/llvm/llvm_schedule_codegen.h'),
                  runner_unchanged=report['source_sha256'] == digest(__file__))
    save(directory / 'manifest.json', report)
    if not report['runner_unchanged']:
        raise ValueError('runner changed during preparation')


class Guarded:
    """64-byte aligned payload, at least 32 guard floats at either end."""
    def __init__(self, count, data=None):
        self.owner = np.full(count + 128, -731.25, np.float32)
        self.start = 32 + (-self.owner.ctypes.data // 4) % 16
        self.data = self.owner[self.start:self.start + count]
        self.data[:] = np.nan if data is None else np.asarray(data).reshape(-1)
        assert self.data.ctypes.data % 64 == 0

    def check(self):
        if not np.all(self.owner[:self.start] == -731.25) or not np.all(self.owner[self.start + self.data.size:] == -731.25):
            raise ValueError('native entry overwrote guard')


def view_array(descriptor, owners):
    return np.ndarray(tuple(descriptor['shape']), dtype=np.float32, buffer=owners[descriptor['storage']].data,
                      offset=descriptor['offset'] * 4, strides=tuple(s * 4 for s in descriptor['strides']))


def replay(args):
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if not manifest.get('runner_unchanged') or digest(manifest['helper']) != manifest['helper_sha256']:
        raise ValueError('incomplete or changed preparation')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory)
    # Imports resolve the actual Inductor module's Torch/C10 symbols. No Torch
    # operation or Python call executes within the common C++ timer loop.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.__version__ != manifest['torch_version'] or torch.version.git_version != manifest['torch_git_version']:
        raise ValueError('Torch build changed')
    helper_library = ctypes.CDLL(manifest['helper'])
    helper = helper_library.replay_native_rows
    ptr, u32, size = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
    helper.argtypes = [ptr, u32, u32, u32, ctypes.POINTER(ptr), ctypes.POINTER(size), u32, u32, u32,
                       u32, u32, size, u32, u32, u32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64)]
    helper.restype = ctypes.c_int
    report = dict(started_unix=time.time(), manifest=str(prepared / 'manifest.json'), manifest_sha256=digest(prepared / 'manifest.json'),
                  source_sha256=digest(__file__), cpu_threads=1, metric='single_thread_native_entry_host_wall_us',
                  samples=7, warmup_ms=100, target_ms=30, aligned_payload_bytes=64, per_allocation_guard_elements=128,
                  boundary='Common C++ callback timer; actual emitted entries, block traversal, launch resets, compiler-emitted libc/allocations included. Runtime dispatch, Python, JIT and caller allocations excluded.',
                  cases=[])
    save(directory / 'results.json', report)
    for case in manifest['cases']:
        op, dims = case['operation'], case['dimensions']
        target = directory / case_name(op, dims)
        target.mkdir()
        arrays = load_inputs(Path(case['inputs']), case['input_shapes'])
        if [array_digest(a) for a in arrays] != case['input_sha256']:
            raise ValueError('input bits changed')
        expected = reference(op, dims, arrays)
        inputs = {f'input{i}': Guarded(a.size, a) for i, a in enumerate(arrays)}
        variants, libraries = {}, []
        for name, entry in case['entries'].items():
            if digest(entry['library']) != entry['library_sha256']:
                raise ValueError('entry library changed')
            library = ctypes.CDLL(entry['library'])
            libraries.append(library)
            address = ctypes.cast(getattr(library, entry['symbol']), ptr)
            owners = dict(inputs)
            if entry['abi'] == 1:
                plan = entry['plan']
                owners.update({name: Guarded(value['elements']) for name, value in plan['allocations'].items()})
                arguments = [view_array(v, owners) for v in plan['arguments']]
                output = view_array(plan['output'], owners)
                const_mask, block, local, workspace = plan['const_mask'], 32, 1, 0
            else:
                owners['output'] = Guarded(math.prod(case['output_shape']))
                output = owners['output'].data.reshape(case['output_shape'])
                arguments = [owners[f'input{i}'].data for i in range(3)] + [output]
                const_mask, block, local, workspace = 7, entry['block_size'], entry['local_lanes'], entry['workspace_bytes']
            pointers = (ptr * len(arguments))(*(a.ctypes.data for a in arguments))
            sizes = (size * len(arguments))(*(a.nbytes for a in arguments))
            variants[name] = (address, entry['abi'], const_mask, block, local, workspace, pointers, sizes, owners, output)
        case_result = dict(operation=op, dimensions=dims, results=[], output_sha256={})
        report['cases'].append(case_result)
        for round_id, order in enumerate(itertools.permutations(variants)):
            for name in order:
                address, abi, mask, block, local, workspace, pointers, sizes, owners, output = variants[name]
                for key, owner in owners.items():
                    if not key.startswith('input'):
                        owner.data.fill(np.nan)
                samples = np.empty(report['samples'], np.float64)
                repetitions = ctypes.c_uint64()
                code = helper(address, abi, len(pointers), mask, pointers, sizes, *dims, 8, block, local, workspace,
                              samples.size, report['warmup_ms'], report['target_ms'],
                              samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(repetitions))
                if code or not np.isfinite(samples).all() or np.any(samples <= 0):
                    raise ValueError(f'native replay failed with {code}')
                validation = validate_output(output, expected)
                for owner in owners.values():
                    owner.check()
                if [array_digest(inputs[f'input{i}'].data) for i in range(3)] != case['input_sha256']:
                    raise ValueError('native replay mutated an input')
                output_hash = array_digest(output)
                if name not in case_result['output_sha256']:
                    output.tofile(target / (name + '.f32'))
                    case_result['output_sha256'][name] = output_hash
                if case_result['output_sha256'][name] != output_hash:
                    raise ValueError('fixed native entry output changed between rounds')
                row = dict(round=round_id, order=order, variant=name, samples_us=samples.tolist(),
                           median_us=float(statistics.median(samples)), repetitions=repetitions.value, correctness=validation,
                           output_sha256=output_hash, guard_elements=128 * len(owners), workspace_guards_passed=True,
                           inputs_unchanged=True)
                case_result['results'].append(row)
                save(directory / 'results.json', report)
                print(case_name(op, dims), round_id, name, f'{row["median_us"]:.6f}', 'us', flush=True)
        case_result['summary_us'] = {name: statistics.median(r['median_us'] for r in case_result['results'] if r['variant'] == name) for name in variants}
        case_result['paired_ratios'] = {}
        for numerator, denominator in (('local', 'inductor'), ('whole', 'inductor'), ('local', 'whole')):
            values = []
            for round_id in range(6):
                pair = {r['variant']: r['median_us'] for r in case_result['results'] if r['round'] == round_id}
                values.append(pair[numerator] / pair[denominator])
            case_result['paired_ratios'][numerator + '/' + denominator] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values), wins=int(sum(v < 1 for v in values)), rounds=values)
        save(directory / 'results.json', report)
    report.update(finished_unix=time.time(), runner_unchanged=report['source_sha256'] == digest(__file__))
    save(directory / 'results.json', report)
    if not report['runner_unchanged']:
        raise ValueError('runner changed during replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    cap = commands.add_parser('capture')
    cap.add_argument('--binary', type=Path, required=True)
    cap.add_argument('--case', type=parse_case, action='append')
    prep = commands.add_parser('prepare')
    prep.add_argument('--capture', type=Path, required=True)
    run = commands.add_parser('replay')
    run.add_argument('--prepared', type=Path, required=True)
    for command_parser in (cap, prep, run):
        command_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'capture': capture, 'prepare': prepare, 'replay': replay}[args.mode](args)


if __name__ == '__main__':
    main()
