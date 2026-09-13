"""Pure adapter contracts: no captures, native libraries, kernels or timing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import native_pointwise as pointwise
import native_pointwise_no_specialization as adapter
from test_native_pointwise import captured_manifest, measurement


def no_specialization_measurement(enabled=True, **overrides):
    counters = dict(full_packet_specializations='0', full_packet_cloned_instructions='0')
    counters.update(overrides)
    return measurement(enabled, **counters)


class NativePointwiseNoSpecializationTests(unittest.TestCase):
    def setUp(self):
        for target in ('native_pointwise.nr.command', 'native_pointwise.subprocess.run',
                       'native_pointwise.ctypes.CDLL', 'native_pointwise.capture', 'native_pointwise.replay'):
            patcher = mock.patch(target, side_effect=AssertionError('native execution is forbidden in these unit tests'))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_environment_adds_only_explicit_disable(self):
        self.assertEqual(adapter.FIXED_ENV, dict(pointwise.FIXED_ENV,
                                               LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION='1'))
        self.assertNotIn('LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION', pointwise.FIXED_ENV)
        self.assertEqual(adapter.FIXED_ENV['LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION'], '1')
        self.assertFalse(any('POINTWISE_FUSION' in key for key in adapter.FIXED_ENV))

    def test_valid_zero_clone_realizations_preserve_actual_metadata(self):
        for enabled in (False, True):
            row = no_specialization_measurement(enabled)
            before = dict(row)
            fields = adapter.check_realization(row, enabled)
            with self.subTest(enabled=enabled):
                self.assertEqual(fields['full_packet_specializations'], '0')
                self.assertEqual(fields['full_packet_cloned_instructions'], '0')
                self.assertEqual(fields['pointwise_fusion'], str(enabled).lower())
                self.assertEqual(row, before)

    def test_both_clone_counters_must_be_present_and_exactly_zero(self):
        for enabled in (False, True):
            for name in ('full_packet_specializations', 'full_packet_cloned_instructions'):
                for value in (None, '1', '-1', '0.0', '00', '', 'false'):
                    with self.subTest(enabled=enabled, name=name, value=value), self.assertRaisesRegex(ValueError, name):
                        adapter.check_realization(no_specialization_measurement(enabled, **{name: value}), enabled)

    def test_other_policy_and_mapping_drift_still_rejected(self):
        mutations = dict(local_lanes='1', blocks_per_task='1', max_unrolled_tile_elements='32',
                         unordered_reduction_partitions='1', load_reduction_fusion='true',
                         expression_reduction_fusion='true', map_fusion='true', fast_math='true',
                         pointwise_fusion='false', custom_cost_policy='true')
        for name, value in mutations.items():
            for changed in (value, None):
                with self.subTest(name=name, value=changed), self.assertRaisesRegex(ValueError, name):
                    adapter.check_realization(no_specialization_measurement(**{name: changed}), True)
        for original, changed in (('W8,', 'W16,'), ('32 workers/block', '64 workers/block'),
                                  ('1 CPU workers;', '8 CPU workers;')):
            row = no_specialization_measurement()
            row['realization'] = row['realization'].replace(original, changed)
            with self.subTest(mapping=changed), self.assertRaisesRegex(ValueError, 'mapping changed'):
                adapter.check_realization(row, True)

    def test_missing_or_invalid_resource_and_fusion_counters_rejected(self):
        for name in ('private_workspace_bytes', 'fused_pointwise_regions', 'fused_pointwise_loads', 'fused_pointwise_stores'):
            for value in (None, '-1', '1.5'):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    adapter.check_realization(no_specialization_measurement(**{name: value}), True)

    def test_rope_expected_dag_and_alias_guards_remain_exact(self):
        expected = dict(fused_pointwise_regions=1, fused_pointwise_loads=4,
                        fused_pointwise_stores=2, pointwise_alias_checks=3)
        for name, correct in expected.items():
            for value in (None, '0', str(correct + 1), '-1', '1.5'):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    adapter.check_realization(no_specialization_measurement(**{name: value}), True)
        with self.assertRaisesRegex(ValueError, 'off control still has pointwise fusion'):
            adapter.check_realization(no_specialization_measurement(False, fused_pointwise_regions='1'), False)

    def test_other_operators_do_not_inherit_rope_only_counts(self):
        row = no_specialization_measurement(operation='gelu_residual', fused_pointwise_regions='2',
                                           fused_pointwise_loads='3', fused_pointwise_stores='1', pointwise_alias_checks='2')
        self.assertEqual(adapter.check_realization(row, True)['fused_pointwise_regions'], '2')

    def test_scoped_configuration_restores_runner_even_after_error(self):
        original = pointwise.FIXED_ENV, pointwise.check_realization, pointwise.runners
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with adapter.configured_runner():
                self.assertIs(pointwise.FIXED_ENV, adapter.FIXED_ENV)
                self.assertIs(pointwise.check_realization, adapter.check_realization)
                self.assertIs(pointwise.runners, adapter.runners)
                raise ValueError('sentinel')
        for current, expected in zip((pointwise.FIXED_ENV, pointwise.check_realization, pointwise.runners), original):
            self.assertIs(current, expected)

    def test_capture_verify_and_replay_share_adapter_configuration(self):
        for mode in ('capture', 'verify', 'replay'):
            argv = [str(Path(adapter.__file__)), mode]
            if mode == 'capture':
                for name in ('binary', 'source-root', 'baseline-manifest', 'full-build-log', 'output'):
                    argv.extend(('--' + name, '/mock/' + name))
                argv.extend(('--cases', 'rope-17x66'))
            else:
                argv.extend(('--prepared', '/mock/capture', '--output', '/mock/result'))

            def inspect_only(args):
                self.assertEqual(args.mode, mode)
                self.assertIs(pointwise.FIXED_ENV, adapter.FIXED_ENV)
                self.assertIs(pointwise.check_realization, adapter.check_realization)
                self.assertIs(pointwise.runners, adapter.runners)

            target = 'capture' if mode == 'capture' else 'replay'
            with self.subTest(mode=mode), mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(pointwise, target, side_effect=inspect_only) as entry:
                adapter.main()
                entry.assert_called_once()
        self.assertNotIn('LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION', pointwise.FIXED_ENV)

    def verify_mock_manifest(self, manifest):
        path = mock.Mock(spec=Path)
        path.read_text.return_value = json.dumps(manifest)
        with mock.patch.object(pointwise, 'unchanged') as unchanged, \
             mock.patch.object(pointwise, 'source_hashes', return_value={}), \
             mock.patch.object(pointwise, 'binary_hashes', return_value={}):
            result = pointwise.verify_manifest(path)
            self.assertEqual(unchanged.call_count, 5)
            return result

    def test_verify_requires_no_specialization_environment_without_weakening_original(self):
        original = captured_manifest()
        disabled = dict(original, fixed_environment=dict(adapter.FIXED_ENV))
        with adapter.configured_runner():
            self.assertEqual(self.verify_mock_manifest(disabled), disabled)
            with self.assertRaisesRegex(ValueError, 'fixed option contract changed'):
                self.verify_mock_manifest(original)
            for key, value in (('LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION', None),
                               ('LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION', '0'),
                               ('LUISA_SIMD_WORKER_COUNT', '8')):
                changed = dict(adapter.FIXED_ENV)
                if value is None:
                    del changed[key]
                else:
                    changed[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'fixed option contract changed'):
                    self.verify_mock_manifest(dict(disabled, fixed_environment=changed))
        self.assertEqual(self.verify_mock_manifest(original), original)
        with self.assertRaisesRegex(ValueError, 'fixed option contract changed'):
            self.verify_mock_manifest(disabled)

    def test_source_snapshot_and_hash_include_adapter_and_every_original_runner(self):
        with tempfile.TemporaryDirectory(prefix='native-pointwise-no-specialization-contract-') as temporary:
            directory = Path(temporary)
            identities = adapter.runners(directory)
            sources = [Path(pointwise.__file__).resolve(), Path(pointwise.nr.__file__).resolve(),
                       pointwise.HERE / 'native_rows_replay.cpp', pointwise.HERE / 'compare_llm.py',
                       Path(adapter.__file__).resolve()]
            self.assertEqual(identities, pointwise.hashes(sources))
            for source in sources:
                with self.subTest(source=source.name):
                    snapshot = directory / 'runner-sources' / source.name
                    self.assertEqual(pointwise.nr.digest(snapshot), identities[str(source)])


if __name__ == '__main__':
    unittest.main()
