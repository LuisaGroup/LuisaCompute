// Test for multi-threaded DSL kernel compilation
// This test verifies that the DSL can safely compile kernels
// from multiple threads concurrently without race conditions.
//
// Features tested:
// - Thread-safe kernel compilation
// - Callable usage across threads
// - Buffer and constant access from multiple threads
// - DSL syntax operations in multi-threaded context

#include <array>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/context.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test structure for DSL struct handling
struct Test {
    int3 something;
    float a;
};

// Register the structure with the DSL
LUISA_STRUCT(Test, something, a) {};

[[nodiscard]] int test_dsl_multithread(Device &device) {
    constexpr auto worker_count = 8u;
    constexpr auto element_count = 32u;

    // Create constant vector
    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    // Callable function that performs arithmetic operations
    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    using CompiledShader = Shader1D<Buffer<float>, Buffer<float>, uint>;
    std::array<CompiledShader, worker_count> shaders;
    std::array<double, worker_count> compile_times{};

    // Create worker threads for concurrent AST construction and compilation.
    // Every worker calls the same Callable, which exercises its thread-safe
    // capture into independent function builders.
    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    for (auto worker = 0u; worker < worker_count; ++worker) {
        threads.emplace_back([&, worker] {
            Clock clock;

            Kernel1D kernel_def = [&, worker](BufferVar<float> input,
                                              BufferVar<float> output,
                                              Var<uint> output_offset) noexcept {
                set_block_size(element_count, 1u, 1u);
                Shared<float> shared_values{element_count};

                auto index = dispatch_x();
                shared_values[thread_x()] = input.read(index);
                sync_block();

                Int table_index = cast<int>((index + worker) % 8u);
                Float value = callable(3, table_index, shared_values[thread_x()]);
                for (auto i : dynamic_range(3)) {
                    value += cast<float>(i + 1);
                }
                if_(index % 2u == 0u, [&] {
                    value += 10.0f;
                }).else_([&] {
                    value -= 5.0f;
                });
                switch_(cast<int>(index % 3u))
                    .case_(0, [&] { value += 100.0f; })
                    .case_(1, [&] { value += 200.0f; })
                    .default_([&] { value -= 300.0f; });

                Var<Test> result{
                    make_int3(table_index,
                              static_cast<int>(worker),
                              cast<int>(index)),
                    value};
                output.write(output_offset + index,
                             result.a + cast<float>(result.something.x + result.something.y));
            };

            clock.tic();
            shaders[worker] = device.compile(kernel_def);
            compile_times[worker] = clock.toc();
        });
    }

    // Wait for all threads to complete
    for (std::thread &t : threads) { t.join(); }

    auto all_compiled = true;
    for (auto worker = 0u; worker < worker_count; ++worker) {
        all_compiled &= static_cast<bool>(shaders[worker]);
        LUISA_INFO("Worker {} compile: {:.3f} ms", worker, compile_times[worker]);
    }
    expect(all_compiled) << "all concurrently compiled shaders must be valid";
    if (!all_compiled) { return 1; }

    luisa::vector<float> host_input(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        host_input[i] = 1.0f + static_cast<float>(i) * 0.25f;
    }
    auto input = device.create_buffer<float>(element_count);
    auto output = device.create_buffer<float>(worker_count * element_count);
    luisa::vector<float> host_output(worker_count * element_count);
    auto stream = device.create_stream();
    stream << input.copy_from(luisa::span{host_input});
    for (auto worker = 0u; worker < worker_count; ++worker) {
        stream << shaders[worker](input, output, worker * element_count).dispatch(element_count);
    }
    stream << output.copy_to(luisa::span{host_output}) << synchronize();

    auto all_correct = true;
    for (auto worker = 0u; worker < worker_count && all_correct; ++worker) {
        for (auto i = 0u; i < element_count; ++i) {
            auto table_index = static_cast<int>((i + worker) % 8u);
            auto expected = 3.0f + static_cast<float>(table_index) * host_input[i] + 6.0f;
            expected += i % 2u == 0u ? 10.0f : -5.0f;
            switch (i % 3u) {
                case 0u: expected += 100.0f; break;
                case 1u: expected += 200.0f; break;
                default: expected -= 300.0f; break;
            }
            expected += static_cast<float>(table_index + static_cast<int>(worker));
            auto actual = host_output[worker * element_count + i];
            if (std::abs(actual - expected) > 1e-5f) {
                LUISA_WARNING(
                    "Multithreaded DSL mismatch for worker {}, lane {}: got {}, expected {}.",
                    worker, i, actual, expected);
                all_correct = false;
                break;
            }
        }
    }
    expect(all_correct) << "every concurrently compiled shader must execute according to the host oracle";
    return all_correct ? 0 : 1;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    return test_dsl_multithread(device);
}
