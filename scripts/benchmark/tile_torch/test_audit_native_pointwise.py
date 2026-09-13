"""Pure independent-auditor contracts; no captures, native calls or benchmarks."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest import mock

import audit_native_pointwise as audit


def llvm_source(enabled):
    group = '#1' if enabled else '#0'
    return '''define internal void @llm_rows(ptr %a, ptr %r, ptr %c, i32 %n) #0 {
  ret void
}
define internal void @llm_rows.full_packet(ptr %a, ptr %r, ptr %c) #0 {
  ret void
}
define dso_local void @llm_rows.packet_batch(ptr %a, ptr %r, ptr %c) {
  call void @llm_rows.full_packet(ptr %a, ptr %r, ptr %c) #0
  call void @llm_rows.full_packet(ptr %a, ptr %r, ptr %c) #0
  call void @llm_rows(ptr %a, ptr %r, ptr %c, i32 %packet.tail.lane.count) ''' + group + '''
  ret void
}
attributes #0 = { nounwind }
attributes #1 = { noinline }
'''


def cohort(comparison='outlined-tail', full=True, legacy=False):
    policies = {'off': {'pointwise_fusion': comparison == 'outlined-tail', 'outlined_packet_tail': False},
                'on': {'pointwise_fusion': True, 'outlined_packet_tail': comparison == 'outlined-tail'}}
    fixed = {'LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION': '1', 'LUISA_SIMD_WORKER_COUNT': '1'}
    if not full:
        fixed['LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION'] = '1'
    manifest = dict(fixed_environment=fixed, cases=[], artifact_sha256={})
    report, payloads = {}, {}
    if not legacy:
        for document in (manifest, report):
            document.update(comparison=comparison, variant_semantics=copy.deepcopy(policies))
    case = dict(operation='rope', dimensions=[17, 66], entries={'inductor': {}})
    manifest['cases'].append(case)
    for variant in ('off', 'on'):
        env = dict(fixed)
        prefix = 'ENABLE' if policies[variant]['pointwise_fusion'] else 'DISABLE'
        env['LUISA_SIMD_' + prefix + '_POINTWISE_FUSION'] = '1'
        if comparison == 'outlined-tail':
            env['LUISA_SIMD_ENABLE_OUTLINED_PACKET_TAIL'] = '1'
            if variant == 'off':
                env['LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL'] = '1'
        env['LUISA_TILE_BENCH_DUMP_SOURCE'] = '/fixture/' + variant + '/kernel.ll'
        env['LUISA_SIMD_DUMP_ASSEMBLY_DIR'] = '/fixture/' + variant + '/object'
        fields = dict(local_lanes='8', blocks_per_task='0', max_unrolled_tile_elements='64',
                      unordered_reduction_partitions='4', load_reduction_fusion='false',
                      expression_reduction_fusion='false', map_fusion='false', fast_math='false',
                      custom_cost_policy='false', pointwise_fusion=str(policies[variant]['pointwise_fusion']).lower(),
                      full_packet_specializations='1' if full else '0')
        realization = 'W8, 32 workers/block, 1 CPU workers; ' + '; '.join(k + '=' + v for k, v in fields.items()) + ';'
        code = llvm_source(variant == 'on').encode()
        digest = hashlib.sha256(code).hexdigest()
        path = '/fixture/' + variant + '/kernel.ll'
        payloads[path] = code
        manifest['artifact_sha256'][path] = digest
        case['entries'][variant] = dict(environment=env, realization=realization, realization_fields=fields,
            capture='/fixture/' + variant, llvm_sha256=digest,
            codegen_checks=dict(stage='pre_optimization_llvm', tail_call_count=1, tail_call_noinline=variant == 'on',
                                full_packet_call_count=2, full_packet_noinline_calls=0,
                                body_function_noinline=False, full_packet_function_noinline=False))
    return manifest, report, payloads


def replace_llvm(manifest, payloads, variant, source):
    path = '/fixture/' + variant + '/kernel.ll'
    payloads[path] = source.encode()
    digest = hashlib.sha256(payloads[path]).hexdigest()
    manifest['artifact_sha256'][path] = digest
    manifest['cases'][0]['entries'][variant]['llvm_sha256'] = digest


class IndependentComparisonTests(unittest.TestCase):
    def setUp(self):
        for target in ('nr.command', 'nr.ctypes.CDLL', 'nr.subprocess.run'):
            patcher = mock.patch('audit_native_pointwise.' + target,
                                 side_effect=AssertionError('native execution is forbidden'))
            patcher.start()
            self.addCleanup(patcher.stop)

    def check(self, manifest, report, payloads):
        with mock.patch.object(Path, 'read_bytes', autospec=True, side_effect=lambda path: payloads[str(path)]) as reader:
            result = audit.comparison_checks(manifest, report)
            return result, reader.call_count

    def test_outlined_tail_reads_and_validates_actual_llvm(self):
        m, r, payloads = cohort()
        result, reads = self.check(m, r, payloads)
        self.assertEqual(reads, 2)
        self.assertEqual(result['comparison'], 'outlined-tail')
        checks = result['comparison_checks'][0]['codegen_checks']
        self.assertIs(checks['off']['tail_call_noinline'], False)
        self.assertIs(checks['on']['tail_call_noinline'], True)

    def test_legacy_and_explicit_pointwise_allow_full_specialization_off(self):
        for legacy in (False, True):
            for full in (False, True):
                m, r, payloads = cohort('pointwise', full, legacy)
                before = copy.deepcopy((m, r))
                with self.subTest(legacy=legacy, full=full):
                    result, reads = self.check(m, r, payloads)
                    self.assertEqual(result['comparison'], 'pointwise')
                    self.assertEqual(reads, 0)
                    self.assertEqual((m, r), before)

    def test_labels_must_be_known_matched_and_explicit_on_both_sides(self):
        for value in ('unknown', None, False, 0, 'pointwise'):
            m, r, payloads = cohort()
            r['comparison'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(m, r, payloads)
        for which in ('manifest', 'report'):
            m, r, payloads = cohort()
            (m if which == 'manifest' else r).pop('comparison')
            with self.subTest(which=which), self.assertRaises(ValueError):
                self.check(m, r, payloads)
        m, r, payloads = cohort('pointwise', legacy=True)
        r['variant_semantics'] = {}
        with self.assertRaises(ValueError):
            self.check(m, r, payloads)

    def test_integer_booleans_and_wrong_variant_claims_are_rejected(self):
        for value in (0, 1, 'true', None):
            m, r, payloads = cohort()
            r['variant_semantics']['on']['pointwise_fusion'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(m, r, payloads)
        m, r, payloads = cohort()
        r['variant_semantics']['off']['pointwise_fusion'] = False
        with self.assertRaises(ValueError):
            self.check(m, r, payloads)

    def test_outlined_tail_rejects_full_off_and_noncanonical_flags(self):
        with self.assertRaises(ValueError):
            self.check(*cohort(full=False))
        for key, value in [('LUISA_SIMD_DISABLE_POINTWISE_FUSION', '1'),
                           ('LUISA_SIMD_ENABLE_OUTLINED_PACKET_TAIL', True),
                           ('LUISA_SIMD_ENABLE_OUTLINED_PACKET_TAIL', 'true'),
                           ('LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL', '1')]:
            m, r, payloads = cohort()
            m['cases'][0]['entries']['on']['environment'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(m, r, payloads)

    def test_uncontrolled_environment_or_realization_change_is_rejected(self):
        m, r, payloads = cohort()
        m['cases'][0]['entries']['on']['environment']['LUISA_SIMD_UNRELATED_OPTION'] = '1'
        with self.assertRaises(ValueError):
            self.check(m, r, payloads)
        for original, changed in [('W8,', 'W16,'), ('fast_math=false', 'fast_math=true'),
                                  ('full_packet_specializations=1', 'full_packet_specializations=0'),
                                  ('pointwise_fusion=true', 'pointwise_fusion=false')]:
            m, r, payloads = cohort()
            entry = m['cases'][0]['entries']['off']
            entry['realization'] = entry['realization'].replace(original, changed)
            with self.subTest(field=original), self.assertRaises(ValueError):
                self.check(m, r, payloads)

    def test_llvm_hash_must_match_both_entry_and_artifact_inventory(self):
        for target in ('entry', 'inventory', 'payload'):
            m, r, payloads = cohort()
            if target == 'entry':
                m['cases'][0]['entries']['on']['llvm_sha256'] = '0' * 64
            elif target == 'inventory':
                del m['artifact_sha256']['/fixture/on/kernel.ll']
            else:
                payloads['/fixture/on/kernel.ll'] += b'\n'
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.check(m, r, payloads)

    def test_recorded_codegen_checks_cannot_substitute_for_actual_attributes(self):
        for value in (True, 0):
            m, r, payloads = cohort()
            m['cases'][0]['entries']['off']['codegen_checks']['tail_call_noinline'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(m, r, payloads)
        m, r, payloads = cohort()
        replace_llvm(m, payloads, 'on', llvm_source(False))
        with self.assertRaises(ValueError):
            self.check(m, r, payloads)

    def test_actual_llvm_mislabeling_fails_even_with_consistent_hashes(self):
        source = llvm_source(True)
        tail = 'call void @llm_rows(ptr %a, ptr %r, ptr %c, i32 %packet.tail.lane.count) #1'
        mutations = [source.replace(tail, ''), source.replace(tail, tail + '\n  ' + tail),
            source.replace('%packet.tail.lane.count)', '8)'),
            source.replace('i32 %n) #0 {', 'i32 %n) #1 {'),
            source.replace('ptr %c) #0 {', 'ptr %c) #1 {'),
            source.replace('call void @llm_rows.full_packet(ptr %a, ptr %r, ptr %c) #0',
                           'call void @llm_rows.full_packet(ptr %a, ptr %r, ptr %c) #1'),
            source.replace('@llm_rows.full_packet', '@other'),
            source.replace('attributes #1 = { noinline }', ''),
            source.replace('{ noinline }', '{ noinline alwaysinline }'),
            source + 'attributes #1 = { noinline }\n']
        for altered in mutations:
            m, r, payloads = cohort()
            replace_llvm(m, payloads, 'on', altered)
            with self.subTest(source=altered), self.assertRaises(ValueError):
                self.check(m, r, payloads)

    def test_comments_and_quoted_attributes_are_not_keywords(self):
        source = llvm_source(False).replace('{ nounwind }', '{ nounwind "note"="noinline; #999" }')
        source += '; call void @llm_rows(ptr %a, ptr %r, ptr %c, i32 %packet.tail.lane.count) #1\n'
        result = audit._tail_attributes(source, False)
        self.assertIs(result['tail_call_noinline'], False)

    def test_new_comparison_mutations_are_separate_and_do_not_rewrite_inputs(self):
        for mode, legacy in [('outlined-tail', False), ('pointwise', False), ('pointwise', True)]:
            m, r, payloads = cohort(mode, legacy=legacy)
            before = copy.deepcopy((m, r))
            with mock.patch.object(Path, 'read_bytes', autospec=True, side_effect=lambda path: payloads[str(path)]):
                rejected = audit.comparison_mutation_checks(m, r)
            self.assertEqual(len(rejected), 8 if mode == 'outlined-tail' else 5)
            self.assertEqual((m, r), before)


if __name__ == '__main__':
    unittest.main()
