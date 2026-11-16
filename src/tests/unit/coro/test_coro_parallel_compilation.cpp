// Exercise the real CUDA LLVM compiler and captured resources with 1/2/3 workers.
#include <luisa/luisa-compute.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
#ifdef _WIN32
    _putenv_s("LUISA_EXPERIMENTAL_LLVM_CODEGEN", "1");
    _putenv_s("LUISA_EXPERIMENTAL_XIR_CODEGEN", "0");
#else
    setenv("LUISA_EXPERIMENTAL_LLVM_CODEGEN", "1", 1);
    setenv("LUISA_EXPERIMENTAL_XIR_CODEGEN", "0", 1);
#endif
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "cuda");
    auto stream = device.create_stream();
    constexpr uint count = 65573u;
    constexpr uint untouched = ~0u;
    std::vector<uint> actual(count), visits(count), zero(count, 0u), sentinel(count, untouched);
    for (uint compiler_threads : {1u, 2u, 3u}) {
        auto output = device.create_buffer<uint>(count);
        auto completed = device.create_buffer<uint>(count);
        // Distinct literals produce distinct shaders for each compiler batch,
        // rather than only exercising parallel binary-cache lookups.
        auto seed = 0x12345678u + compiler_threads;
        Coroutine<void()> coroutine{[&] {
            auto i = dispatch_x();
            UInt value = i ^ seed;
            $for(round, 1u + i % 5u) {
                $switch((i >> (round % 3u)) & 3u) {
                    $case(0u) { $suspend("queue_a"); value ^= 0xabcdef01u; };
                    $case(1u) { $suspend("queue_b"); value += 0x543210feu; };
                    $case(2u) { $suspend("queue_c"); value *= 17u; };
                    $default { value += round; };
                };
                value = value * 1664525u + 1013904223u;
            };
            output->write(i, value);
            completed->atomic(i).fetch_add(1u);
        }};
        coro::WavefrontCoroSchedulerConfig config;
        config.thread_count = 257u;
        config.shader_compile_threads = compiler_threads;
        config.global_memory_soa = true;
        config.largest_continuation_first = true;
        config.incremental_continuation_counts = true;
        coro::WavefrontCoroScheduler<> task{device, coroutine, config};
        for (uint extent : {1u, 17u, 31u, 32u, 33u, 65u, count}) {
            stream << output.copy_from(sentinel.data()) << completed.copy_from(zero.data());
            auto begin = std::chrono::steady_clock::now();
            stream << task().dispatch(extent);
            stream << output.copy_to(actual.data()) << completed.copy_to(visits.data()) << synchronize();
            auto elapsed = std::chrono::duration<double, std::milli>{
                std::chrono::steady_clock::now() - begin}.count();
            for (uint i = 0u; i < count; ++i) {
                uint expected = untouched;
                if (i < extent) {
                    expected = i ^ seed;
                    for (uint round = 0u; round < 1u + i % 5u; ++round) {
                        switch ((i >> (round % 3u)) & 3u) {
                            case 0u: expected ^= 0xabcdef01u; break;
                            case 1u: expected += 0x543210feu; break;
                            case 2u: expected *= 17u; break;
                            default: expected += round; break;
                        }
                        expected = expected * 1664525u + 1013904223u;
                    }
                }
                LUISA_ASSERT(actual[i] == expected && visits[i] == uint(i < extent),
                             "Worker count {} extent {} index {}: output {} / {}, visits {}.",
                             compiler_threads, extent, i, actual[i], expected, visits[i]);
            }
            std::cout << "CORO COMPILE ORACLE workers=" << compiler_threads
                      << " extent=" << extent << " dispatch_ms=" << elapsed << '\n';
        }
    }
    std::cout << "CORO PARALLEL COMPILATION PASS\n";
}
