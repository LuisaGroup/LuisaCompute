// Full AST -> XIR -> coroutine -> backend regression. Immutable uniform
// aggregates forwarded through nested calls must not become per-path state.
#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

struct ReadonlyUniforms {
    float4x4 a, b, c, d;
};
LUISA_STRUCT(ReadonlyUniforms, a, b, c, d) {};

int main(int argc, char **argv) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    "nested_readonly_uniforms_are_replayed_not_spilled"_test = [options] {
        Callable leaf = [](Var<ReadonlyUniforms> &parameters, Float &value) {
            value += parameters.a[0u][0u] + parameters.b[1u][1u] +
                     parameters.c[2u][2u] + parameters.d[3u][3u];
        };
        Callable wrapper = [leaf](Var<ReadonlyUniforms> &parameters, Float &value) {
            leaf(parameters, value);
        };
        Coroutine coroutine = [wrapper](BufferFloat output,
                                        Var<ReadonlyUniforms> parameters) {
            Float value = dispatch_x().cast<float>();
            $for (step, 3u) {
                $suspend("nested-readonly-step");
                wrapper(parameters, value);
            };
            $suspend("nested-readonly-finish");
            output.write(dispatch_x(), value);
        };
        auto &frame = coroutine.frame_desc();
        expect(frame.total_size() <= 16u)
            << "only the mutable scalar and loop state may persist; "
               "the 256-byte immutable argument is available in every stage";
        for (size_t i = 0u; i < frame.field_count(); ++i) {
            expect(!frame.field(i).type->is_matrix());
        }
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        constexpr uint count = 257u;
        auto output = device.create_buffer<float>(count);
        vector<float> actual(count);
        for (bool soa : {false, true}) {
            WavefrontCoroSchedulerConfig config{
                .thread_count = 64u,
                .global_memory_soa = soa,
                .shader_option = {.enable_cache = false, .enable_fast_math = true}};
            WavefrontCoroScheduler scheduler{device, coroutine, config};
            for (auto scale : {1.0f, 3.0f}) {
                const ReadonlyUniforms parameters{
                    make_float4x4(scale), make_float4x4(2.0f * scale),
                    make_float4x4(3.0f * scale), make_float4x4(4.0f * scale)};
                stream << scheduler(output, parameters).dispatch(count)
                       << output.copy_to(actual.data()) << synchronize();
                for (uint i = 0u; i < count; ++i) {
                    expect(actual[i] == static_cast<float>(i) + 30.0f * scale);
                }
            }
        }
    };
    return luisa::test::coro_test::run_tests(argc, argv);
}
