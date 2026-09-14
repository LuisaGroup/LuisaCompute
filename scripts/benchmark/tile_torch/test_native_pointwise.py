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


def tail_ir(enabled=False):
    tail_group = '#1' if enabled else '#0'
    return '''define internal void @llm_rows(ptr %args, ptr %returns, ptr %launch, i32 %count) #0 {
  ret void
}
define internal void @llm_rows.full_packet(ptr %args, ptr %returns, ptr %launch) #0 {
  ret void
}
define dso_local void @llm_rows.packet_batch(ptr %args, ptr %returns, ptr %launch) {
  call void @llm_rows.full_packet(ptr %args, ptr %returns, ptr %launch) #0
  call void @llm_rows.full_packet(ptr %args, ptr %returns, ptr %launch) #0
  call void @llm_rows(ptr %args, ptr %returns, ptr %launch, i32 %packet.tail.lane.count) ''' + tail_group + '''
  ret void
}
attributes #0 = { nounwind }
attributes #1 = { noinline }
'''


def outlined_manifest():
    manifest = dict(captured_manifest(), **pointwise.comparison_contract('outlined-tail'))
    for variant in ('off', 'on'):
        row = measurement(True)
        manifest['cases'][0]['entries'][variant] = dict(
            capture='/mock/' + variant, environment=pointwise.variant_environment(variant, 'outlined-tail', {}),
            realization=row['realization'], realization_fields=pointwise.check_realization(row, True),
            codegen_checks=pointwise.check_outlined_tail_ir(tail_ir(variant == 'on'), variant == 'on'))
    return manifest


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

    def test_pointwise_environment_remains_the_default_and_scrubs_conflicts(self):
        inherited = dict(PATH='/fixture/bin', LUISA_SIMD_ENABLE_POINTWISE_FUSION='1',
                         LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION='1',
                         LUISA_SIMD_ENABLE_OUTLINED_PACKET_TAIL='1', DYLD_INSERT_LIBRARIES='/bad')
        for variant, prefix in (('off', 'DISABLE'), ('on', 'ENABLE')):
            env = pointwise.variant_environment(variant, inherited=inherited)
            self.assertEqual(env, dict(pointwise.FIXED_ENV, PATH='/fixture/bin',
                LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL='1',
                **{'LUISA_SIMD_' + prefix + '_POINTWISE_FUSION': '1'}))
            self.assertEqual(pointwise.variant_environment(variant, inherited={}),
                             pointwise.variant_environment(variant, 'pointwise', {}))

    def test_outlined_tail_changes_only_the_disable_precedence_control(self):
        off = pointwise.variant_environment('off', 'outlined-tail', {})
        on = pointwise.variant_environment('on', 'outlined-tail', {})
        self.assertEqual(off, dict(on, LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL='1'))
        self.assertEqual(on['LUISA_SIMD_ENABLE_POINTWISE_FUSION'], '1')
        self.assertEqual(on['LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION'], '1')
        self.assertEqual(on['LUISA_SIMD_ENABLE_OUTLINED_PACKET_TAIL'], '1')
        for variant in ('off', 'on'):
            self.assertTrue(pointwise.comparison_contract('outlined-tail')['variant_semantics'][variant]['pointwise_fusion'])
            pointwise.check_realization(measurement(True), True)

    def test_comparison_and_variant_validation(self):
        for comparison in ('unknown', None, True, 1):
            with self.subTest(comparison=comparison), self.assertRaises(ValueError):
                pointwise.comparison_contract(comparison)
        for variant in ('inductor', 'unknown', None):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                pointwise.variant_environment(variant)

    def test_capture_cli_comparison_default_and_explicit_choices(self):
        base = ['native_pointwise.py', 'capture', '--binary', '/mock/kernel', '--source-root', '/mock/source',
                '--baseline-manifest', '/mock/baseline.json', '--full-build-log', '/mock/build.log',
                '--output', '/mock/output', '--cases', 'rope-17x66']
        for option in (None, 'pointwise', 'outlined-tail'):
            argv = base if option is None else base + ['--comparison', option]
            with self.subTest(option=option), mock.patch.object(pointwise.sys, 'argv', argv), \
                 mock.patch.object(pointwise, 'capture') as capture:
                pointwise.main()
                self.assertEqual(capture.call_args.args[0].comparison, option or 'pointwise')

    def test_no_specialization_adapter_keeps_default_but_rejects_outlined_tail(self):
        fixed = dict(pointwise.FIXED_ENV, LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION='1')
        with mock.patch.object(pointwise, 'FIXED_ENV', fixed):
            self.assertEqual(pointwise.variant_environment('on')['LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION'], '1')
            with self.assertRaises(ValueError):
                pointwise.comparison_contract('outlined-tail')
        with mock.patch.object(pointwise, 'FIXED_ENV', {}), self.assertRaises(ValueError):
            pointwise.comparison_contract('outlined-tail')

    def test_preopt_tail_callsite_attributes_are_checked(self):
        for enabled in (False, True):
            checks = pointwise.check_outlined_tail_ir(tail_ir(enabled), enabled)
            self.assertEqual(checks, dict(stage='pre_optimization_llvm', tail_call_count=1,
                tail_call_noinline=enabled, full_packet_call_count=2, full_packet_noinline_calls=0,
                body_function_noinline=False, full_packet_function_noinline=False))
            with self.assertRaises(ValueError):
                pointwise.check_outlined_tail_ir(tail_ir(enabled), not enabled)

    def test_noinline_cannot_leak_to_full_calls_or_function_definitions(self):
        source = tail_ir(True)
        changes = [
            ('call void @llm_rows.full_packet(ptr %args, ptr %returns, ptr %launch) #0',
             'call void @llm_rows.full_packet(ptr %args, ptr %returns, ptr %launch) #1'),
            ('i32 %count) #0 {', 'i32 %count) #1 {'),
            ('ptr %launch) #0 {', 'ptr %launch) #1 {')]
        for old, new in changes:
            with self.subTest(target=old), self.assertRaises(ValueError):
                pointwise.check_outlined_tail_ir(source.replace(old, new), True)

    def test_preopt_shape_and_attribute_errors_fail_closed(self):
        source = tail_ir(True)
        call = 'call void @llm_rows(ptr %args, ptr %returns, ptr %launch, i32 %packet.tail.lane.count) #1'
        mutations = [source.replace(call, ''), source.replace(call, call + '\n  ' + call),
                     source.replace('%packet.tail.lane.count)', '8)'),
                     source.replace('@llm_rows.full_packet', '@unrelated'),
                     source.replace(call, call.replace('#1', '#999')),
                     source.replace('{ noinline }', '{ noinline alwaysinline }'),
                     source + 'attributes #1 = { noinline }\n']
        for altered in mutations:
            with self.subTest(source=altered), self.assertRaises(ValueError):
                pointwise.check_outlined_tail_ir(altered, True)

    def test_noinline_in_comments_and_string_attributes_is_not_an_attribute(self):
        source = tail_ir(False).replace('{ nounwind }', '{ nounwind "note"="noinline" }')
        source += '; call void @llm_rows(ptr %a, ptr %b, ptr %c, i32 %packet.tail.lane.count) #1\n'
        self.assertFalse(pointwise.check_outlined_tail_ir(source, False)['tail_call_noinline'])

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

    def test_new_pointwise_manifest_and_legacy_default(self):
        old = captured_manifest()
        self.assertNotIn('comparison', self.verify_mock_manifest(old))
        current = dict(old, **pointwise.comparison_contract())
        self.assertEqual(self.verify_mock_manifest(current), current)

    def test_invalid_comparison_metadata_rejected_before_file_checks(self):
        non_boolean = pointwise.comparison_contract()['variant_semantics']
        non_boolean['off']['pointwise_fusion'] = 0
        changes = [dict(comparison='unknown'), dict(comparison=None), dict(comparison='pointwise'),
                   dict(comparison='pointwise', variant_semantics={}),
                   dict(comparison='pointwise', variant_semantics=non_boolean),
                   dict(comparison='outlined-tail', variant_semantics=pointwise.comparison_contract()['variant_semantics'])]
        for change in changes:
            path = mock.Mock(spec=Path)
            path.read_text.return_value = json.dumps(dict(captured_manifest(), **change))
            with self.subTest(change=change), mock.patch.object(pointwise, 'unchanged') as unchanged:
                with self.assertRaises(ValueError):
                    pointwise.verify_manifest(path)
                unchanged.assert_not_called()

    def test_outlined_manifest_rechecks_recorded_environment_realization_and_ir(self):
        manifest = outlined_manifest()
        with mock.patch.object(Path, 'read_text', side_effect=[tail_ir(False), tail_ir(True)]):
            self.assertEqual(self.verify_mock_manifest(manifest), manifest)
        for variant, flag, value in [('off', 'LUISA_SIMD_ENABLE_POINTWISE_FUSION', '0'),
                                    ('off', 'LUISA_SIMD_DISABLE_POINTWISE_FUSION', '1'),
                                    ('off', 'LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION', '1'),
                                    ('on', 'LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL', '1')]:
            altered = copy.deepcopy(manifest)
            altered['cases'][0]['entries'][variant]['environment'][flag] = value
            with self.subTest(flag=flag), mock.patch.object(Path, 'read_text', return_value=tail_ir(False)), self.assertRaises(ValueError):
                self.verify_mock_manifest(altered)
        altered = copy.deepcopy(manifest)
        altered['cases'][0]['entries']['off']['codegen_checks']['tail_call_noinline'] = True
        with mock.patch.object(Path, 'read_text', return_value=tail_ir(False)), self.assertRaises(ValueError):
            self.verify_mock_manifest(altered)
        altered = copy.deepcopy(manifest)
        altered['cases'][0]['entries']['off']['codegen_checks']['tail_call_noinline'] = 0
        with mock.patch.object(Path, 'read_text', return_value=tail_ir(False)), self.assertRaises(ValueError):
            self.verify_mock_manifest(altered)
        altered = copy.deepcopy(manifest)
        altered['cases'][0]['entries']['off']['realization'] = measurement(False)['realization']
        with self.assertRaises(ValueError):
            self.verify_mock_manifest(altered)

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
