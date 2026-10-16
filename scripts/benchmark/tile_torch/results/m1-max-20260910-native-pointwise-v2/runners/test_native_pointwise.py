"""Pure contract tests: never capture, load native libraries, or execute kernels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import native_pointwise as pointwise


def baseline_case(op='rope', dimensions=(17, 66)):
    shapes, output = pointwise.nr.shapes_for(op, dimensions)
    return dict(operation=op, dimensions=list(dimensions),
                input_shapes=[list(shape) for shape in shapes], output_shape=list(output))


def measurement(enabled=True, operation='rope', **overrides):
    fields = dict(local_lanes='8', blocks_per_task='0', max_unrolled_tile_elements='64',
                  unordered_reduction_partitions='4', load_reduction_fusion='false',
                  expression_reduction_fusion='false', map_fusion='false', fast_math='false',
                  pointwise_fusion=str(enabled).lower(), custom_cost_policy='false',
                  full_packet_specializations='1', private_workspace_bytes='0',
                  fused_pointwise_regions='1' if enabled else '0',
                  fused_pointwise_loads='4' if enabled else '0',
                  fused_pointwise_stores='2' if enabled else '0', pointwise_alias_checks='3' if enabled else '0')
    fields.update(overrides)
    text = 'TileIR -> XIR SSA -> SIMD Schedule -> LLVM; W8, 32 workers/block, 1 CPU workers; '
    text += '; '.join(f'{name}={value}' for name, value in fields.items() if value is not None) + ';'
    return dict(operation=operation, realization=text)


def captured_manifest():
    case = baseline_case()
    case['entries'] = {name: {} for name in pointwise.VARIANTS}
    return dict(format='native-pointwise-v1', status='captured',
                source_unchanged=True, closure_unchanged=True, runner_unchanged=True,
                artifact_sha256={}, runner_sha256={}, inherited_sha256={},
                configuration_sha256={}, llvm_tools_sha256={},
                source_root='/mock/source', source_sha256={}, binary='/mock/build/bin/kernel',
                binary_closure_sha256={}, fixed_environment=dict(pointwise.FIXED_ENV),
                selected_cases=['rope-17x66'], cases=[case])


class NativePointwiseTests(unittest.TestCase):
    def setUp(self):
        # A regression in a tested helper must not accidentally become a
        # benchmark or dlopen a native entry while running these unit tests.
        for target in ('native_pointwise.nr.command', 'native_pointwise.subprocess.run',
                       'native_pointwise.ctypes.CDLL', 'native_pointwise.capture', 'native_pointwise.replay'):
            patcher = mock.patch(target, side_effect=AssertionError('native execution is forbidden in these unit tests'))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fixed_environment_contract(self):
        expected = dict(
            LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1',
            LUISA_TILE_BENCH_XIR_BACKEND='simd', LUISA_TILE_BENCH_XIR_LOCAL_LANES='8',
            LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32', LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK='0',
            LUISA_TILE_BENCH_XIR_SEARCH_TASK_GRAIN='0',
            LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
            LUISA_SIMD_DISABLE_EXPRESSION_REDUCTION_FUSION='1', LUISA_SIMD_DISABLE_MAP_FUSION='1',
            LUISA_SIMD_DISABLE_LINEAR_1D_BLOCK_COALESCING='1',
            LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1',
            LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1', LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        self.assertEqual(pointwise.FIXED_ENV, expected)
        self.assertFalse(any('POINTWISE_FUSION' in name for name in pointwise.FIXED_ENV))

    def test_valid_fixed_realizations(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                fields = pointwise.check_realization(measurement(enabled), enabled)
                self.assertEqual(fields['pointwise_fusion'], str(enabled).lower())

    def test_changed_fixed_metadata_rejected(self):
        mutations = dict(local_lanes='1', blocks_per_task='1', max_unrolled_tile_elements='32',
                         unordered_reduction_partitions='1', load_reduction_fusion='true',
                         expression_reduction_fusion='true', map_fusion='true', fast_math='true',
                         pointwise_fusion='false', custom_cost_policy='true', full_packet_specializations='0',
                         private_workspace_bytes=None)
        for key, value in mutations.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                pointwise.check_realization(measurement(**{key: value}), True)
        for original, changed in (('W8,', 'W16,'), ('32 workers/block', '64 workers/block'),
                                  ('1 CPU workers;', '8 CPU workers;')):
            row = measurement()
            row['realization'] = row['realization'].replace(original, changed)
            with self.subTest(mapping=changed), self.assertRaises(ValueError):
                pointwise.check_realization(row, True)

    def test_rope_expected_fusion_counts_rejected_when_missing_or_wrong(self):
        expected = dict(fused_pointwise_regions='1', fused_pointwise_loads='4',
                        fused_pointwise_stores='2', pointwise_alias_checks='3')
        for name, correct in expected.items():
            for value in (None, '0', str(int(correct) + 1), '-1', '1.5'):
                with self.subTest(counter=name, value=value), self.assertRaises(ValueError):
                    pointwise.check_realization(measurement(**{name: value}), True)

    def test_off_cannot_acknowledge_fused_region(self):
        with self.assertRaises(ValueError):
            pointwise.check_realization(measurement(False, fused_pointwise_regions='1'), False)

    def test_rope_counts_are_not_imposed_on_other_operators(self):
        row = measurement(operation='gelu_residual', fused_pointwise_regions='2',
                          fused_pointwise_loads='3', fused_pointwise_stores='1', pointwise_alias_checks='2')
        self.assertEqual(pointwise.check_realization(row, True)['fused_pointwise_regions'], '2')

    def test_case_selection_preserves_explicit_order(self):
        small, large = baseline_case(), baseline_case(dimensions=(1024, 4098))
        baseline = dict(cases=[small, large])
        self.assertEqual(pointwise.select_cases(baseline, ['rope-1024x4098', 'rope-17x66']), [large, small])
        self.assertEqual(pointwise.select_cases(baseline, ['all']), [small, large])

    def test_missing_unknown_and_duplicate_case_requests_rejected(self):
        baseline = dict(cases=[baseline_case()])
        for names in ([], ['unknown-17x66'], ['rope-1024x4098'], ['rope-17x66', 'rope-17x66'],
                      ['all', 'rope-17x66']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                pointwise.select_cases(baseline, names)
        with self.assertRaises(ValueError):
            pointwise.select_cases(dict(cases=[]), ['all'])

    def test_duplicate_or_invalid_baseline_cases_rejected(self):
        case = baseline_case()
        with self.assertRaises(ValueError):
            pointwise.select_cases(dict(cases=[case, copy.deepcopy(case)]), ['all'])
        for changes in (dict(operation='unknown'), dict(dimensions=[0, 66]), dict(dimensions=[17, 65]),
                        dict(dimensions=[17, 66.0]), dict(dimensions=[17, 66, 1]),
                        dict(dimensions=[2, pointwise.nr.MAX_ELEMENTS]), dict(output_shape=[17, 65]),
                        dict(input_shapes=[[17, 66], [17, 66], [17, 33]])):
            changed = dict(case, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pointwise.select_cases(dict(cases=[changed]), ['all'])

    def verify_mock_manifest(self, manifest):
        path = mock.Mock(spec=Path)
        path.read_text.return_value = json.dumps(manifest)
        with mock.patch.object(pointwise, 'unchanged') as unchanged, \
             mock.patch.object(pointwise, 'source_hashes', return_value={}), \
             mock.patch.object(pointwise, 'binary_hashes', return_value={}):
            result = pointwise.verify_manifest(path)
            self.assertEqual(unchanged.call_count, 5)
            return result

    def test_complete_manifest_contract(self):
        manifest = captured_manifest()
        self.assertEqual(self.verify_mock_manifest(manifest), manifest)

    def test_partial_failed_unknown_manifests_rejected_before_file_checks(self):
        for changes in (dict(status='capturing'), dict(status='error'), dict(format='unknown'),
                        dict(format=None), dict(status=None)):
            path = mock.Mock(spec=Path)
            path.read_text.return_value = json.dumps(dict(captured_manifest(), **changes))
            with self.subTest(changes=changes), mock.patch.object(pointwise, 'unchanged') as unchanged:
                with self.assertRaises(ValueError):
                    pointwise.verify_manifest(path)
                unchanged.assert_not_called()

    def test_changed_identity_flags_and_environment_rejected(self):
        for field in ('source_unchanged', 'closure_unchanged', 'runner_unchanged'):
            for value in (False, 1, 'true'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.verify_mock_manifest(dict(captured_manifest(), **{field: value}))
        manifest = captured_manifest()
        manifest['fixed_environment']['LUISA_SIMD_WORKER_COUNT'] = '8'
        with self.assertRaises(ValueError):
            self.verify_mock_manifest(manifest)

    def test_missing_variant_or_reordered_capture_cohort_rejected(self):
        for name in pointwise.VARIANTS:
            manifest = captured_manifest()
            del manifest['cases'][0]['entries'][name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.verify_mock_manifest(manifest)
        manifest = captured_manifest()
        manifest['selected_cases'] = ['rope-1024x4098']
        with self.assertRaises(ValueError):
            self.verify_mock_manifest(manifest)

    def test_source_inventory_excludes_unrelated_payloads(self):
        with tempfile.TemporaryDirectory(prefix='native-pointwise-contract-') as temporary:
            root = Path(temporary)
            included = ['CMakeLists.txt', 'include/luisa/tile/ir.h', 'src/tile/lower.cpp',
                        'src/ext/dependency/include/detail.hpp', '.cmake/config.cmake']
            excluded = ['docs/generated.xml', 'docs/irrelevant.cpp', 'src/results/payload.cpp',
                        'src/ext/.git/hidden.h', 'scripts/benchmark/results/payload.json', 'src/image.png']
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            actual = pointwise.source_hashes(root)
            self.assertEqual(set(actual), {str(root / name) for name in included})


if __name__ == '__main__':
    unittest.main()
