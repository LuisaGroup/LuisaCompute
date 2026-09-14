#include "ut/ut.hpp"
#include "tile_migrated_test_utils.h"
#include <luisa/tile/runtime.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
namespace migrated = luisa::test::tile_migrated;

int main(int argc, char *argv[]) {
    "migrated_complete_oracle_rejects_corruption"_test = [] {
        vector<double> expected{1.0, 2.0, 3.0};
        vector<float> seed(expected.size() + 2u * migrated::padding, migrated::canary);
        std::copy(expected.begin(), expected.end(), seed.begin() + migrated::padding);
        expect(migrated::validate(seed, expected).passed);
        for (auto index : {size_t{0}, migrated::padding - 1u, migrated::padding,
                           migrated::padding + expected.size() - 1u, seed.size() - 1u}) {
            auto broken = seed;
            broken[index] += 1.0f;
            expect(!migrated::validate(broken, expected).passed);
        }
        for (auto bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            auto broken = seed;
            broken[migrated::padding + 1u] = bad;
            expect(!migrated::validate(broken, expected).passed);
        }
        expect(!migrated::validate(span{seed}.first(seed.size() - 1u), expected).passed);
        expect(!migrated::parse("unknown"));
    };
    "migrated_periodic_gemm_oracle_matches_independent_dot_products"_test = [] {
        for (auto dimensions : {std::array<int64_t, 3>{7, 13, 19}, {19, 23, 107}}) {
            auto [m, n, k] = dimensions;
            auto fixture = migrated::make(migrated::Operation::GEMM, m, n, k, {4, 5, 8});
            expect(fixture.kernel.valid());
            for (auto row = int64_t{0}; row < m; row++) {
                for (auto column = int64_t{0}; column < n; column++) {
                    auto expected = 0.0;
                    for (auto j = int64_t{0}; j < k; j++) {
                        expected += static_cast<double>(fixture.inputs[0][row * k + j]) * fixture.inputs[1][j * n + column];
                    }
                    expect(eq(fixture.expected[row * n + column], expected));
                }
            }
        }
    };
    "migrated_execution_first_captures"_test = [] {
        for (auto name : migrated::names) {
            auto op = *migrated::parse(name);
            auto fixture = migrated::make(op, 17, 65, op == migrated::Operation::GEMM ? 23 : 1, {4, 8, 8});
            expect(fixture.kernel.valid()) << name;
        }
    };
    if (argc == 1) { return 0; }
    auto backend = string_view{argv[1]};
    if (backend != "metal" && backend != "simd") { return 2; }
    if (argc > 3 || (argc == 3 && !migrated::parse(argv[2]))) { return 2; }
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    "migrated_kernels_full_outputs_and_tail_guards"_test = [&] {
        for (auto name : migrated::names) {
            // Avoid Boost.UT's eager overloaded logical/comparison operators
            // evaluating a nonexistent argv[2] in the all-operations mode.
            if (argc == 3) {
                if (string_view{name}.compare(argv[2]) != 0) { continue; }
            }
            auto op = *migrated::parse(name);
            for (auto dimensions : {std::array<int64_t, 2>{1, 1}, {17, 65}, {19, 32}}) {
                auto [rows, columns] = dimensions;
                LUISA_INFO("Migrated {} {}x{}: capture/compile", name, rows, columns);
                // Keep routine correctness runs bounded. The 4x8x8 GEMM
                // machine-scheduler compile-time cliff is retained in the
                // migration report/benchmark, not hidden as a timing pass.
                auto block = op == migrated::Operation::GEMM ? example::tile::Block{2, 2, 4} : example::tile::Block{4, 8, 8};
                auto fixture = migrated::make(op, rows, columns, op == migrated::Operation::GEMM ? 19 : 1, block);
                expect(fixture.kernel.valid()) << name;
                if (!fixture.kernel.valid()) { continue; }
                tile::CompileOptions options;
                options.lowering = backend == "metal" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
                auto shader = tile::compile(device, fixture.kernel, options, {.enable_fast_math = false});
                expect(static_cast<bool>(shader)) << name << shader.metadata().error;
                if (!shader) { continue; }
                LUISA_INFO("Migrated {} {}x{}: execute/check", name, rows, columns);
                vector<Buffer<float>> inputs;
                for (auto &&data : fixture.inputs) {
                    inputs.emplace_back(device.create_buffer<float>(data.size()));
                    stream << inputs.back().copy_from(span{data});
                }
                vector<float> output(fixture.expected.size() + 2u * migrated::padding, migrated::canary);
                std::fill(output.begin() + migrated::padding, output.end() - migrated::padding, std::numeric_limits<float>::quiet_NaN());
                auto destination = device.create_buffer<float>(output.size());
                auto view = destination.view(migrated::padding, fixture.expected.size());
                stream << destination.copy_from(span{output});
                if (inputs.size() == 2u) {
                    stream << shader(inputs[0], inputs[1], view).dispatch();
                } else {
                    stream << shader(inputs[0], view).dispatch();
                }
                stream << destination.copy_to(span{output}) << synchronize();
                auto checked = migrated::validate(output, fixture.expected);
                expect(checked.passed) << name << rows << columns << checked.bad_index << checked.max_abs_error;
            }
        }
    };
}
