#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_soa_layout(luisa::test::coro_test::Options options) {

    "runtime_soa_layout_is_linear_and_capacity_invariant"_test = [] {
        auto coro = Coroutine<void(Buffer<float4>)>([](BufferFloat4 output) {
            auto i = dispatch_x();
            auto scalar = def(cast<float>(i) + 0.5f);
            auto vector = make_float3(scalar, scalar + 1.0f, scalar + 2.0f);
            auto tag = def(i * 7u + 3u);
            $suspend("mixed_alignment");
            output.write(i, make_float4(vector * scalar, cast<float>(tag)));
        });

        constexpr size_t small_capacity = 17u;
        constexpr size_t large_capacity = 257u;
        auto small = CoroFrameStorageLayout::make_runtime_soa(
            coro.frame(), small_capacity);
        auto large = CoroFrameStorageLayout::make_runtime_soa(
            coro.frame(), large_capacity);
        expect(small.has_runtime_capacity());
        expect(large.has_runtime_capacity());
        expect(small.frame_stride == large.frame_stride);
        expect(small.field_strides == large.field_strides);
        expect(small.field_capacity_strides ==
               large.field_capacity_strides);
        expect(small.size_bytes == small.frame_stride * small_capacity);
        expect(large.size_bytes == large.frame_stride * large_capacity);

        for (auto capacity : {small_capacity, large_capacity}) {
            luisa::vector<std::pair<size_t, size_t>> ranges;
            ranges.reserve(coro.frame().frame_field_count());
            for (auto i = 0u; i < coro.frame().frame_field_count(); i++) {
                auto alignment = std::max<size_t>(
                    coro.frame().frame_field_type(i)->alignment(), 4u);
                auto begin = small.field_offsets[i] +
                             capacity * small.field_capacity_strides[i];
                auto end = begin + capacity * small.field_strides[i];
                expect(begin % alignment == 0u)
                    << "every field array base must satisfy its ABI alignment";
                ranges.emplace_back(begin, end);
            }
            std::sort(ranges.begin(), ranges.end());
            auto cursor = size_t{0u};
            for (auto [begin, end] : ranges) {
                expect(begin == cursor)
                    << "runtime SoA field arrays must form one disjoint partition";
                expect(end >= begin);
                cursor = end;
            }
            expect(cursor == small.frame_stride * capacity);
        }
    };

    "wavefront_shader_structure_does_not_hash_pool_capacity"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto value = def(dispatch_x() * 11u + 5u);
            $suspend("live_value");
            output.write(dispatch_x(), value);
        });
        auto make_config = [](uint capacity) noexcept {
            return WavefrontCoroSchedulerConfig{
                .thread_count = capacity,
                .global_memory_soa = true,
                .gather_by_sorting = false,
                .frame_buffer_compaction = true};
        };
        WavefrontCoroScheduler<Buffer<uint>> small{
            device, coro, make_config(17u)};
        WavefrontCoroScheduler<Buffer<uint>> large{
            device, coro, make_config(257u)};
        auto small_hashes = small.shader_structure_hashes();
        auto large_hashes = large.shader_structure_hashes();
        expect(!small_hashes.empty());
        expect(small_hashes.size() == large_hashes.size());
        expect(std::equal(small_hashes.begin(), small_hashes.end(),
                          large_hashes.begin(), large_hashes.end()))
            << "frame-pool capacity is an allocation/runtime parameter and "
               "must not invalidate scheduler shader caches";
    };

    "soa_layout_constructs_and_has_correct_config"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        WavefrontCoroScheduler<> aos_scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = false}};
        expect(aos_scheduler.config().global_memory_soa == false);

        WavefrontCoroScheduler<> soa_scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        expect(soa_scheduler.config().global_memory_soa == true);
    };

    "soa_layout_compiles_and_runs"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(scheduler.config().global_memory_soa == true);
        expect(scheduler.config().thread_count >= N);
    };

    "soa_layout_1suspend_with_buffer"_test = [options] {
        constexpr uint N = 128u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            $suspend("s1");
            buf.write(tid, tid + 42u);
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("SoA dispatch complete");
        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 42u) {
                LUISA_WARNING("soa_layout_1suspend_with_buffer mismatch at {}: got {}, expected {}",
                              i, host[i], i + 42u);
                ok = false;
                break;
            }
        }
        expect(ok) << "all wavefront SoA coroutine instances should write expected values";
    };

    "soa_layout_3suspend_smoke"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(int)>([](Var<int> unused) {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<int> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(scheduler.config().global_memory_soa == true);
        expect(scheduler.config().thread_count >= N);
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_soa_layout(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
