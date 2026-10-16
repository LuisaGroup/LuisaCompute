"""Pure Python probe admission/evidence tests; never loads native code or Torch.

Native numerical/guard integration is the explicit runner `selftest` command,
which is separately gated by the full build and executes no timer loops.
"""
from __future__ import annotations

import array
import contextlib
import copy
import ctypes as c
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import native_attention_probe as probe


class ContractTests(unittest.TestCase):
    def baseline(self):
        shape = probe.dimensions([1, 4, 2, 3, 7, 5, 9])
        metadata = dict(implementation='tile_xir_simd', operation='attention', precision='fp32',
                        source_kind='tile_lowering_source', fast_math=False, relaxed_precision=False,
                        attention_qk='mma', attention_pv='mma', source_reduction_policy='unordered_tree',
                        dimensions=shape['dimensions'], input_shapes=shape['input_shapes'],
                        output_shape=shape['output_shape'], dispatch=[32, 1, 1],
                        realization='W8, 32 workers/block; private_workspace_bytes=4096',
                        correctness=dict(checks=2, elements_per_check=shape['output_elements'], atol=5e-5, rtol=5e-5))
        return dict(metadata=metadata, capture_kind='llm', input_files=['input0.f32', 'input1.f32', 'input2.f32'],
                    output_elements=shape['output_elements'], atol=5e-5, rtol=5e-5, abi=0,
                    symbol='llm_attention.packet_batch.blocks', dispatch=[32, 1, 1], block=[32, 1, 1],
                    packet_width=8, workspace_bytes=4096)

    def test_good_shapes_and_scratch(self):
        for shape in probe.SELFTEST_SHAPES:
            value = probe.dimensions(shape)
            self.assertEqual(value['workspace_bytes'], max(shape[3] * shape[4], 16) * 4)
        self.assertEqual(probe.baseline_contract(self.baseline())['dimensions'], [1, 4, 2, 3, 7, 5, 9])

    def test_dimensions_are_strict_positive_bounded_integers(self):
        for invalid in (None, [], [1] * 6, [1] * 8):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probe.dimensions(invalid)
        for index in range(7):
            for invalid in (True, 1.0, '1', 0, -1, 1 << 27):
                shape = [1, 4, 2, 3, 7, 5, 9]
                shape[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    probe.dimensions(shape)

    def test_gqa_causal_tensor_and_scratch_bounds(self):
        for shape in ([1, 3, 2, 1, 7, 5, 9], [1, 4, 2, 8, 7, 5, 9],
                      [1 << 20, 8, 2, 1, 17, 8, 8], [1, 1, 1, 2049, 2049, 1, 1]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                probe.dimensions(shape)

    def test_metadata_fail_closed(self):
        for field in ('implementation', 'operation', 'precision', 'source_kind', 'fast_math',
                      'relaxed_precision', 'attention_qk', 'attention_pv', 'source_reduction_policy'):
            for invalid in (None, True, 0, 'reduce'):
                baseline = self.baseline()
                baseline['metadata'][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    probe.baseline_contract(baseline)

    def test_payload_metadata_checks_and_tolerance_cannot_weaken(self):
        changes = ((('capture_kind',), 'migrated'), (('input_files',), ['input0.f32']),
                   (('output_elements',), True), (('metadata', 'correctness', 'checks'), True),
                   (('metadata', 'correctness', 'elements_per_check'), 1), (('atol',), 1e-3),
                   (('metadata', 'correctness', 'rtol'), 1e-3), (('metadata', 'output_shape'), [1, 4, 3, 8]),
                   (('metadata', 'input_shapes'), [[True, 4, 3, 5], [1, 2, 7, 5], [1, 2, 7, 9]]))
        for keys, value in changes:
            baseline = self.baseline()
            destination = baseline
            for key in keys[:-1]:
                destination = destination[key]
            destination[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                probe.baseline_contract(baseline)

    def test_launch_must_be_actual_abi_and_metadata(self):
        for field, value in (('abi', True), ('abi', 1), ('packet_width', 3), ('packet_width', 64),
                             ('block', [0, 1, 1]), ('block', [16, 1, 1]), ('dispatch', [1, 1, 1]),
                             ('dispatch', [0xffffffff] * 3), ('workspace_bytes', probe.MAX_WORKSPACE + 1),
                             ('workspace_bytes', 0), ('symbol', 'bad symbol')):
            baseline = self.baseline()
            baseline[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe.baseline_contract(baseline)

    def test_probe_abi_algorithm_and_math_are_explicit(self):
        shape = probe.dimensions([1, 4, 2, 3, 7, 5, 9])
        for variant in probe.VARIANTS:
            entry = probe.probe_entry(variant, shape)
            self.assertEqual(entry['abi'], 0)
            self.assertEqual(entry['dispatch'], [1, 1, 1])
            self.assertEqual(entry['block'], [1, 1, 1])
            self.assertEqual(entry['packet_width'], 1)
            self.assertEqual(entry['source_kind'], 'handwritten_probe_not_tile_lowering')
        self.assertFalse(probe.MATH['reduced_precision'])
        self.assertFalse(probe.MATH['cross_implementation_bitwise_required'])
        with self.assertRaises(ValueError):
            probe.probe_entry('torch', shape)


class OracleTests(unittest.TestCase):
    def test_bottom_right_gqa_every_output(self):
        shape = probe.dimensions([1, 4, 2, 2, 3, 1, 2])
        inputs = [array.array('f', [0] * 8).tobytes(), array.array('f', [0] * 6).tobytes(),
                  array.array('f', [2, 4, 4, 8, 12, 24, 20, 40, 40, 80, 120, 240]).tobytes()]
        raw, receipt = probe.oracle(inputs, shape)
        expected = [3, 6, 6, 12] * 2 + [30, 60, 60, 120] * 2
        self.assertEqual(list(array.array('d', raw)), expected)
        self.assertEqual(receipt['causal_alignment'], 'bottom_right')

    def test_nonfinite_and_wrong_extents_rejected(self):
        shape = probe.dimensions([1] * 7)
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                probe.oracle([struct.pack('<f', value), bytes(4), bytes(4)], shape)
        with self.assertRaises(ValueError):
            probe.oracle([bytes(8), bytes(4), bytes(4)], shape)
        with self.assertRaises(ValueError):
            probe.oracle([bytes(4), bytes(4)], shape)

    def test_every_output_and_finite_contract(self):
        reference = array.array('d', [0, 1, -1]).tobytes()
        valid = array.array('f', [0, 1, -1]).tobytes()
        self.assertEqual(probe.check_bytes(valid, reference)['elements'], 3)
        for index in range(3):
            for value in (10, float('nan'), float('inf')):
                output = array.array('f', valid)
                output[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    probe.check_bytes(output.tobytes(), reference)
        with self.assertRaises(ValueError):
            probe.check_bytes(valid[:-1], reference)
        with self.assertRaises(ValueError):
            probe.check_bytes(valid[:4], reference)

    def test_patterns_are_repeatable_finite_and_different(self):
        shape = probe.dimensions(probe.SELFTEST_SHAPES[0])
        patterns = [probe.pattern_inputs(shape, name) for name in probe.PATTERNS]
        self.assertEqual(patterns[0], probe.pattern_inputs(shape, 'random'))
        self.assertEqual(len({tuple(x) for x in patterns}), 3)
        for inputs in patterns:
            expected, _ = probe.oracle(inputs, shape)
            self.assertEqual(len(expected), shape['output_elements'] * 8)


class ArtifactTests(unittest.TestCase):
    def test_inventory_missing_mutation_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            file = directory / 'file'
            file.write_bytes(b'content')
            frozen = probe.inventory(directory)
            probe.verify_files(directory, frozen)
            for name in ('../escape', '/absolute', '', None):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    probe.safe_file(directory, name)
            (directory / 'link').symlink_to(file)
            with self.assertRaises(ValueError):
                probe.safe_file(directory, 'link')
            file.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                probe.verify_files(directory, frozen)
            file.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                probe.verify_files(directory, frozen)

    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            (directory / 'real').mkdir()
            (directory / 'real/file').write_bytes(b'x')
            (directory / 'alias').symlink_to(directory / 'real', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                probe.safe_file(directory, 'alias/file')

    def test_compiler_flags_are_strict(self):
        flags = probe.compile_flags(Path('/tmp/example'))
        for flag in ('-std=c++20', '-O3', '-fno-fast-math', '-ffp-contract=off', '-dynamiclib', '-mmacosx-version-min=15.0'):
            self.assertIn(flag, flags)
        self.assertNotIn('-ffast-math', flags)

    def test_command_failure_retains_raw_evidence(self):
        cases = [subprocess.CompletedProcess(['fake'], 2, b'output', b'failure'),
                 subprocess.TimeoutExpired(['fake'], 300, output=b'partial', stderr=b'partial-error'),
                 FileNotFoundError('missing compiler')]
        for outcome in cases:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                mock = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
                with patch.object(probe.subprocess, 'run', mock), self.assertRaises(Exception):
                    probe.command(['fake'], directory, 'compile')
                record = json.loads((directory / 'compile.command.json').read_text())
                self.assertEqual(record['argv'], ['fake'])
                self.assertTrue((directory / 'compile.stdout').is_file())
                self.assertTrue((directory / 'compile.stderr').is_file())
                if isinstance(outcome, subprocess.TimeoutExpired):
                    self.assertTrue(record['timeout'])
                    self.assertEqual((directory / 'compile.stderr').read_bytes(), b'partial-error')
                elif isinstance(outcome, FileNotFoundError):
                    self.assertIn('missing compiler', record['error'])
                else:
                    self.assertEqual(record['returncode'], 2)
                    self.assertEqual((directory / 'compile.stdout').read_bytes(), b'output')


class NativeProtocolTests(unittest.TestCase):
    def library(self, version=1, workspace=64, status=0, mode=1):
        return SimpleNamespace(attention_probe_contract_version=Mock(return_value=version),
                               attention_probe_workspace_bytes=Mock(return_value=workspace),
                               attention_probe_set_single_threaded=Mock(return_value=status),
                               attention_probe_threading=Mock(return_value=mode))

    def test_threading_is_set_and_queried_not_environment_guess(self):
        library = self.library()
        receipt = probe.configure_probe(library, 64)
        self.assertEqual(receipt['observed_enum'], 1)
        library.attention_probe_set_single_threaded.assert_called_once_with()
        library.attention_probe_threading.assert_called_once_with()
        for changes in (dict(version=0), dict(workspace=128), dict(status=-1), dict(mode=0), dict(mode=2)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                probe.configure_probe(self.library(**changes), 64)

    def visit(self, directory, samples=0, returncode=0, corrupt=False):
        shape = probe.dimensions([1] * 7)
        entry = probe.probe_entry('online_neon', shape)
        inputs = [struct.pack('<f', x) for x in (1, 2, 3)]
        expected = struct.pack('<d', 3)
        # The callback is an address placeholder; the Python fake helper never
        # invokes it, so these protocol tests execute no native operator.
        callback = c.CFUNCTYPE(None)()
        library = SimpleNamespace(attention_online_neon=callback)
        seen = []

        def helper(*args):
            seen.append(args)
            c.cast(args[3][3], c.POINTER(c.c_float))[0] = 4 if corrupt else 3
            for i in range(args[11]):
                args[16][i] = 10 + i
            c.cast(args[17], c.POINTER(c.c_uint64))[0] = 4 if args[11] else 0
            c.cast(args[18], c.POINTER(c.c_double))[0] = 0
            return returncode

        result = probe.native_visit(helper, library, entry, inputs, expected, directory, 'output.f32',
                                    samples=samples, warmup_ms=0 if not samples else 30,
                                    target_ms=0 if not samples else 15)
        return result, seen[0]

    def test_validation_only_and_timing_protocol_share_exact_helper(self):
        for samples in (0, 7):
            with tempfile.TemporaryDirectory() as temporary:
                result, call = self.visit(Path(temporary), samples)
                self.assertTrue(result['valid'])
                self.assertEqual(call[1:3], (0, 4))
                self.assertEqual(list(call[7]), [1, 1, 1])
                self.assertEqual(list(call[8]), [1, 1, 1])
                self.assertEqual(call[9:11], (1, 64))
                self.assertEqual(call[11:14], (samples, 30, 15) if samples else (0, 0, 0))
                self.assertEqual(result['median_us'], 13 if samples else None)

    def test_native_error_and_full_mismatch_keep_output_and_receipt(self):
        for options in (dict(returncode=4), dict(returncode=5), dict(returncode=6), dict(corrupt=True)):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with self.assertRaises(ValueError):
                    self.visit(directory, **options)
                self.assertTrue((directory / 'output.f32').is_file())
                receipt = json.loads((directory / 'output.f32.json').read_text())
                self.assertFalse(receipt['valid'])


class ReplayControlsTests(unittest.TestCase):
    def test_cli_defaults_and_bounds(self):
        base = ['replay', '--bundle', '/tmp/probe.json', '--variant', 'online_neon', '--output', '/tmp/result']
        args = probe.parse_args(base)
        self.assertEqual((args.cycles, args.samples, args.warmup_ms, args.target_ms), (3, 7, 30, 15))
        for option, value in (('--cycles', '0'), ('--cycles', '21'), ('--samples', '101'), ('--samples', '0'),
                              ('--target-ms', '0'), ('--warmup-ms', '-1'), ('--target-ms', '10001'),
                              ('--warmup-ms', '10001'), ('--target-ms', '15.5'), ('--variant', 'torch')):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_args(base + [option, value])

    def visits(self):
        return [dict(cycle=cycle, position=position, variant=variant, valid=True, median_us=value)
                for cycle in range(3) for position, (variant, value) in
                enumerate(zip(('tile', 'online_neon', 'online_neon', 'tile'), (10, 8, 9, 12)))]

    def test_abba_keeps_all_pairs_exact_medians_and_negative_results(self):
        rows = self.visits()
        result = probe.summarize(rows, 3, 'online_neon')
        self.assertEqual(result['summary_us'], dict(tile=11, online_neon=8.5))
        self.assertEqual(result['candidate_over_baseline']['pairs'], [0.8, 0.75] * 3)
        self.assertEqual(result['candidate_over_baseline']['median'], 0.775)
        for row in rows:
            if row['variant'] == 'online_neon':
                row['median_us'] *= 10
        self.assertGreater(probe.summarize(rows, 3, 'online_neon')['candidate_over_baseline']['minimum'], 1)

    def test_missing_reordered_invalid_visits_fail(self):
        rows = self.visits()
        for mutation in (lambda x: x.pop(), lambda x: x[0].update(valid=False),
                         lambda x: x[0].update(position=1), lambda x: x[0].update(variant='online_neon'),
                         lambda x: x[0].update(median_us=float('nan')), lambda x: x[0].update(median_us=0)):
            changed = copy.deepcopy(rows)
            mutation(changed)
            with self.assertRaises(ValueError):
                probe.summarize(changed, 3, 'online_neon')

    def test_no_torch_arm_and_fixed_selftest_matrix(self):
        self.assertEqual(len(probe.SELFTEST_SHAPES) * len(probe.PATTERNS) * len(probe.VARIANTS), 24)
        self.assertNotIn('torch', probe.VARIANTS)
        self.assertIn('BLAS-internal packing/allocations included', probe.BOUNDARY)


if __name__ == '__main__':
    unittest.main()
