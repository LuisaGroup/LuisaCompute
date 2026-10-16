"""Validation-only tests of the native Tile replay ABI and evidence guards.

The temporary C++ fixture contains fake packet entries, not timed kernels. Every
helper call has sample_count=0: one invocation, no warmup/calibration/timer loop.
Run directly with the standard library: python3 test_native_tile_replay.py.
"""
from __future__ import annotations

import ctypes as c
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURE = r'''
#include <cstddef>
#include <cstdint>
#include "backends/simd/llvm/llvm_schedule_codegen.h"
using namespace luisa::compute::simd;
namespace {
uint32_t calls, violations, output_index, mode, width;
size_t workspace_size;
void fill_block(const SIMDHostBufferView *args, SIMDPacketLaunchConfig *launch) {
    const auto *input = static_cast<const float *>(args[0].data);
    const auto *other = output_index == 2u ? static_cast<const float *>(args[1].data) : nullptr;
    auto *output = static_cast<float *>(args[output_index].data);
    auto count = launch->block_size[0] * launch->block_size[1] * launch->block_size[2];
    for (uint32_t thread = 0u; thread < count; thread++) {
        auto x = launch->block_id[0] * launch->block_size[0] + thread % launch->block_size[0];
        auto remaining = thread / launch->block_size[0];
        auto y = launch->block_id[1] * launch->block_size[1] + remaining % launch->block_size[1];
        auto z = launch->block_id[2] * launch->block_size[2] + remaining / launch->block_size[1];
        if (x < launch->dispatch_size[0] && y < launch->dispatch_size[1] && z < launch->dispatch_size[2]) {
            auto flat = x + launch->dispatch_size[0] * (y + launch->dispatch_size[1] * z);
            output[flat] = input[flat] + (other ? other[flat] : 3.0f);
        }
    }
}
void corrupt(const SIMDHostBufferView *args, SIMDPacketLaunchConfig *launch) {
    auto *output = static_cast<std::byte *>(args[output_index].data);
    auto *workspace = static_cast<std::byte *>(launch->private_workspace);
    switch (mode) {
        case 1u: static_cast<std::byte *>(args[0].data)[0] ^= std::byte{1}; break;
        case 2u: output[-1] = std::byte{0}; break;
        case 3u: output[args[output_index].size_bytes] = std::byte{0}; break;
        case 4u: workspace[-1] = std::byte{0}; break;
        case 5u: workspace[workspace_size] = std::byte{0}; break;
        case 6u: launch->dispatch_size[0]++; break;
        case 7u: static_cast<float *>(args[output_index].data)[args[output_index].size_bytes / sizeof(float) - 1u] += 1000.0f; break;
        case 8u: launch->private_workspace = nullptr; break;
    }
}
}
extern "C" void fixture_reset(uint32_t index, uint32_t operation, uint32_t packet_width, size_t workspace_bytes) {
    calls = violations = 0u;
    output_index = index;
    mode = operation;
    width = packet_width;
    workspace_size = workspace_bytes;
}
extern "C" uint32_t fixture_calls() { return calls; }
extern "C" uint32_t fixture_violations() { return violations; }
extern "C" void fixture_packets(const void *opaque, void *returns, SIMDPacketLaunchConfig *launch, uint32_t packets) {
    auto total = launch->block_size[0] * launch->block_size[1] * launch->block_size[2];
    auto flat = launch->block_id[0] + launch->grid_size[0] * (launch->block_id[1] + launch->grid_size[1] * launch->block_id[2]);
    violations += returns != nullptr || launch->thread_index != 0u || flat != calls || packets * width != total;
    calls++;
    fill_block(static_cast<const SIMDHostBufferView *>(opaque), launch);
    // Mutating these is legal; the adapter must reset them before the next block.
    launch->thread_index = 1234u;
    launch->block_id[0] = launch->block_id[1] = launch->block_id[2] = UINT32_MAX;
    if (calls == launch->grid_size[0] * launch->grid_size[1] * launch->grid_size[2]) {
        corrupt(static_cast<const SIMDHostBufferView *>(opaque), launch);
    }
}
extern "C" void fixture_blocks(const void *opaque, void *returns, SIMDPacketLaunchConfig *launch, uint32_t blocks) {
    violations += returns != nullptr || launch->thread_index != 0u || launch->block_id[0] != 0u ||
                  launch->block_id[1] != 0u || launch->block_id[2] != 0u ||
                  blocks != launch->grid_size[0] * launch->grid_size[1] * launch->grid_size[2];
    calls++;
    for (uint32_t block = 0u; block < blocks; block++) {
        launch->block_id[0] = block % launch->grid_size[0];
        auto remaining = block / launch->grid_size[0];
        launch->block_id[1] = remaining % launch->grid_size[1];
        launch->block_id[2] = remaining / launch->grid_size[1];
        fill_block(static_cast<const SIMDHostBufferView *>(opaque), launch);
    }
    launch->thread_index = 1234u;
    launch->block_id[0] = launch->block_id[1] = launch->block_id[2] = UINT32_MAX;
    corrupt(static_cast<const SIMDHostBufferView *>(opaque), launch);
}
'''


class NativeTileReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='luisa-native-tile-replay-test-')
        cls.addClassCleanup(cls.directory.cleanup)
        directory = Path(cls.directory.name)
        fixture = directory / 'fixture.cpp'
        fixture.write_text(FIXTURE)
        library = directory / 'fixture.dylib'
        subprocess.run(['clang++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-dynamiclib', '-I' + str(ROOT / 'src'),
                        str(HERE / 'native_tile_replay.cpp'), str(fixture), '-o', str(library)], check=True)
        cls.library = c.CDLL(str(library))
        cls.helper = cls.library.replay_native_tile
        p, u32, size, d = c.c_void_p, c.c_uint32, c.c_size_t, c.c_double
        cls.helper.argtypes = [p, u32, u32, c.POINTER(p), c.POINTER(size), c.POINTER(u32),
                              c.POINTER(c.POINTER(d)), c.POINTER(u32), c.POINTER(u32), u32, size,
                              u32, u32, u32, d, d, c.POINTER(d), c.POINTER(c.c_uint64), c.POINTER(d)]
        cls.helper.restype = c.c_int
        cls.library.fixture_reset.argtypes = [u32, u32, u32, size]
        cls.library.fixture_calls.restype = u32
        cls.library.fixture_violations.restype = u32

    def fixture(self, count=2):
        n = 30
        inputs = [(c.c_float * n)(*(i + j / 2 for i in range(n))) for j in range(count - 1)]
        output = (c.c_float * n)(*([float('nan')] * n))
        values = [*inputs, output]
        oracle = (c.c_double * n)(*(inputs[0][i] + (inputs[1][i] if count == 3 else 3) for i in range(n)))
        return dict(values=values, oracle=oracle,
                    arguments=(c.c_void_p * count)(*(c.addressof(v) for v in values)),
                    sizes=(c.c_size_t * count)(*([n * 4] * count)),
                    writable=(c.c_uint32 * count)(*([0] * (count - 1) + [1])),
                    expected=(c.POINTER(c.c_double) * count)(*([None] * (count - 1) + [oracle])),
                    dispatch=(c.c_uint32 * 3)(5, 3, 2), block=(c.c_uint32 * 3)(4, 2, 1),
                    count=count, abi=2, mode=0, width=4, workspace=16)

    def invoke(self, fixture):
        self.library.fixture_reset(fixture['count'] - 1, fixture['mode'], fixture['width'], fixture['workspace'])
        address = c.cast(self.library.fixture_blocks if fixture['abi'] == 0 else self.library.fixture_packets, c.c_void_p)
        repetitions, error = c.c_uint64(999), c.c_double(-1)
        result = self.helper(address, fixture['abi'], fixture['count'], fixture['arguments'], fixture['sizes'],
                             fixture['writable'], fixture['expected'], fixture['dispatch'], fixture['block'],
                             fixture['width'], fixture['workspace'], 0, 0, 0, 0, 0, None,
                             c.byref(repetitions), c.byref(error))
        return result, repetitions.value, error.value

    def test_two_and_three_buffers_xyz_and_abi_resets(self):
        for count in (2, 3):
            for abi in (0, 2):
                with self.subTest(count=count, abi=abi):
                    fixture = self.fixture(count)
                    fixture['abi'] = abi
                    before = [bytes(v) for v in fixture['values'][:-1]]
                    self.assertEqual(self.invoke(fixture), (0, 0, 0))
                    self.assertEqual(list(fixture['values'][-1]), list(fixture['oracle']))
                    self.assertEqual(before, [bytes(v) for v in fixture['values'][:-1]])
                    self.assertEqual(self.library.fixture_calls(), 1 if abi == 0 else 8)
                    self.assertEqual(self.library.fixture_violations(), 0)

    def test_corruption_is_rejected(self):
        for mode, expected_code in ((1, 6), (2, 4), (3, 4), (4, 5), (5, 5), (6, 8), (7, 7), (8, 8)):
            for abi in (0, 2):
                with self.subTest(mode=mode, abi=abi):
                    fixture = self.fixture()
                    fixture.update(mode=mode, abi=abi)
                    before = bytes(fixture['values'][0])
                    result, repetitions, _ = self.invoke(fixture)
                    self.assertEqual(result, expected_code)
                    self.assertEqual(repetitions, 0)
                    self.assertEqual(bytes(fixture['values'][0]), before)

    def test_zero_workspace_guard(self):
        fixture = self.fixture()
        fixture.update(workspace=0, mode=5)
        self.assertEqual(self.invoke(fixture)[0], 5)

    def test_multiple_writable_buffers_have_separate_oracles(self):
        fixture = self.fixture(3)
        second_oracle = (c.c_double * 30)(*fixture['values'][1])
        fixture['writable'][1] = 1
        fixture['expected'][1] = second_oracle
        self.assertEqual(self.invoke(fixture), (0, 0, 0))
        second_oracle[29] += 1
        self.assertEqual(self.invoke(fixture)[0], 7)

    def test_every_output_element_is_checked(self):
        for bad_index in range(30):
            with self.subTest(index=bad_index):
                fixture = self.fixture()
                fixture['oracle'][bad_index] += 1
                self.assertEqual(self.invoke(fixture)[0], 7)

    def test_invalid_metadata_is_rejected_without_invocation(self):
        mutations = (
            (lambda f: f.update(abi=1), 1),
            (lambda f: f.update(width=3), 1),
            (lambda f: f.update(width=16), 2),
            (lambda f: f.update(workspace=16 * 1024 * 1024 + 1), 1),
            (lambda f: f['dispatch'].__setitem__(1, 0), 2),
            (lambda f: f['block'].__setitem__(2, 0), 2),
            (lambda f: f.update(dispatch=(c.c_uint32 * 3)(0xffffffff, 0xffffffff, 2)), 2),
            (lambda f: f.update(block=(c.c_uint32 * 3)(0xffffffff, 0xffffffff, 2)), 2),
            (lambda f: f['sizes'].__setitem__(0, 0), 2),
            (lambda f: f['sizes'].__setitem__(1, 119), 3),
            (lambda f: f['writable'].__setitem__(1, 2), 2),
            (lambda f: f['expected'].__setitem__(1, None), 3),
            (lambda f: f['oracle'].__setitem__(0, float('nan')), 3),
            (lambda f: f['oracle'].__setitem__(0, float('inf')), 3),
            (lambda f: f['arguments'].__setitem__(0, None), 2),
            (lambda f: f['arguments'].__setitem__(1, f['arguments'][0]), 2),
            (lambda f: f['arguments'].__setitem__(1, f['arguments'][0] + 4), 2),
        )
        for index, (mutate, expected_code) in enumerate(mutations):
            with self.subTest(index=index):
                fixture = self.fixture()
                mutate(fixture)
                self.assertEqual(self.invoke(fixture)[0], expected_code)
                self.assertEqual(self.library.fixture_calls(), 0)


if __name__ == '__main__':
    unittest.main()
