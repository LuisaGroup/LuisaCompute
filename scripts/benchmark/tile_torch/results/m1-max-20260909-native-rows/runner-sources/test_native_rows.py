"""Fail-closed ABI extraction, alias extents and guard tests for native replay."""
import unittest

import numpy as np

from native_rows import Guarded, parse_inductor_wrapper, storage_size, view_array


def wrapper(types, body):
    declarations = ', '.join(t + ' a' + str(i) for i, t in enumerate(types))
    return (f"native = async_compile.cpp_pybinding({types!r}, '''extern \"C\" void kernel({declarations}) {{}}''')\n"
            + 'class Runner:\n    def call(self, args):\n'
            + '\n'.join('        ' + line for line in body.splitlines()) + '\n')


TYPES = ['const float*', 'const float*', 'float*']
BODY = '''arg0_1, arg1_1 = args
args.clear()
assert_size_stride_grouped((arg0_1, arg1_1), ((2, 4), (2, 4)), ((4, 1), (4, 1)), 'input')
buf0 = empty_strided_cpu((2, 4), (4, 1), torch.float32)
native(arg0_1, arg1_1, buf0)
del arg0_1
del arg1_1
return (buf0, )'''
INPUTS = ['L_x_', 'L_u_']
SHAPES = [(2, 4), (2, 4), (2, 4)]


class NativeWrapperTests(unittest.TestCase):
    def parse(self, source=None, inputs=INPUTS, shapes=SHAPES):
        return parse_inductor_wrapper(source or wrapper(TYPES, BODY), inputs, shapes)

    def test_plain_entry(self):
        plan = self.parse()
        self.assertEqual(plan['const_mask'], 3)
        self.assertEqual([v['storage'] for v in plan['arguments']], ['input0', 'input1', 'buf0'])
        self.assertEqual(plan['allocations']['buf0']['elements'], 8)

    def test_fx_order_controls_pointer_order(self):
        plan = self.parse(inputs=['L_u_', 'L_x_'])
        self.assertEqual([v['storage'] for v in plan['arguments'][:2]], ['input1', 'input0'])

    def test_reused_mutable_scratch(self):
        types = ['float*', 'const float*', 'const float*', 'float*']
        body = BODY.replace('native(arg0_1, arg1_1, buf0)', '''scratch0 = empty_strided_cpu((2, 1), (1, 2), torch.float32)
scratch1 = scratch0
del scratch0
native(scratch1, arg0_1, arg1_1, buf0)''')
        plan = self.parse(wrapper(types, body))
        self.assertEqual(plan['const_mask'], 6)
        self.assertEqual(plan['arguments'][0]['storage'], 'scratch0')

    def test_partitioned_output_aliases(self):
        types = ['const float*', 'const float*', 'float*', 'float*']
        body = BODY.replace('native(arg0_1, arg1_1, buf0)', '''lo = reinterpret_tensor(buf0, (2, 2), (4, 1), 0)
hi = reinterpret_tensor(buf0, (2, 2), (4, 1), 2)
native(arg0_1, arg1_1, lo, hi)''')
        plan = self.parse(wrapper(types, body))
        self.assertEqual(plan['arguments'][3]['offset'], 2)
        self.assertEqual(plan['arguments'][2]['storage'], plan['arguments'][3]['storage'])
        owners = {'buf0': Guarded(8)}
        view_array(plan['arguments'][2], owners)[:] = 3
        view_array(plan['arguments'][3], owners)[:] = 7
        np.testing.assert_array_equal(view_array(plan['output'], owners), [[3, 3, 7, 7], [3, 3, 7, 7]])
        owners['buf0'].check()

    def test_unsupported_effects_and_abis_rejected(self):
        base = wrapper(TYPES, BODY)
        mutations = {
            'signature': base.replace('const float* a0', 'float* a0'),
            'parallel': base.replace('{}', '{\n#pragma omp parallel\n}'),
            'shape_guard': base.replace('((2, 4), (2, 4))', '((2, 4), (2, 5))'),
            'stride_guard': base.replace('((4, 1), (4, 1))', '((4, 1), (5, 1))'),
            'missing_guard': base.replace("assert_size_stride_grouped((arg0_1, arg1_1), ((2, 4), (2, 4)), ((4, 1), (4, 1)), 'input')", 'args.clear()'),
            'extra_effect': base.replace('native(arg0_1, arg1_1, buf0)', 'torch.add(arg0_1, arg1_1)'),
            'double_call': base.replace('native(arg0_1, arg1_1, buf0)', 'native(arg0_1, arg1_1, buf0)\n        native(arg0_1, arg1_1, buf0)'),
            'input_mutation': base.replace('native(arg0_1, arg1_1, buf0)', 'native(arg0_1, arg1_1, arg0_1)'),
            'dynamic_shape': base.replace('empty_strided_cpu((2, 4)', 'empty_strided_cpu((2, n)'),
            'unknown_argument': base.replace('native(arg0_1, arg1_1, buf0)', 'native(arg0_1, unknown, buf0)'),
            'return_input': base.replace('return (buf0, )', 'return (arg0_1, )'),
            'oversized': base.replace('empty_strided_cpu((2, 4), (4, 1)', 'empty_strided_cpu((2, 67108864), (67108864, 1)'),
        }
        for name, source in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.parse(source)

    def test_alias_boundaries_rejected(self):
        for offset in (-1, 3, 100):
            body = BODY.replace('native(arg0_1, arg1_1, buf0)', f'alias = reinterpret_tensor(buf0, (2, 2), (4, 1), {offset})\nnative(arg0_1, arg1_1, alias)')
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.parse(wrapper(TYPES, body))

    def test_unknown_fx_input_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(inputs=['L_x_', 's0'])

    def test_storage_shape_rules(self):
        self.assertEqual(storage_size((17, 1), (1, 17)), 17)
        for shape, strides in (((2, 4), (4, -1)), ((0, 4), (4, 1)), ((2, 4), (4,))):
            with self.assertRaises(ValueError):
                storage_size(shape, strides)

    def test_alignment_and_guard_detection(self):
        for count in (1, 7, 65, 4097):
            owner = Guarded(count)
            self.assertEqual(owner.data.ctypes.data % 64, 0)
            self.assertGreaterEqual(owner.start, 32)
            self.assertGreaterEqual(owner.owner.size - owner.start - count, 32)
            owner.check()
            owner.owner[owner.start - 1] = 0
            with self.assertRaises(ValueError):
                owner.check()


if __name__ == '__main__':
    unittest.main()
