#include "ut/ut.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <luisa/dsl/local.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
    Context context{argv[0]};
    auto device = context.create_device(backend);

    "packed callable pointers preserve reads and writes to every local"_test = [&] {
        constexpr auto local_count = 20u;
        constexpr auto extent = 8u;
        constexpr auto thread_count = 64u;
        Kernel1D kernel = [=](BufferUInt input, BufferUInt output) noexcept {
            const auto tid = dispatch_x();
            const auto value = input.read(tid);
            const auto index = value % extent;
            std::vector<Local<uint>> locals;
            locals.reserve(local_count);
            for (auto i = 0u; i < local_count; ++i) {
                locals.emplace_back(extent);
            }
            // Force a boundary only in this regression: ordinary IPO can
            // retain the same wide pointer ABI for a sufficiently large body.
            $outline_noinline_with_name("packed_pointer_effects") {
                for (auto i = 0u; i < local_count; ++i) {
                    locals[i].write(index, 100u * i + value);
                }
            };
            for (auto i = 0u; i < local_count; ++i) {
                output.write(tid * local_count + i, locals[i].read(0u));
            }
        };
        const auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false});
        std::array<uint, thread_count> inputs{};
        for (auto i = 0u; i < thread_count; ++i) {
            inputs[i] = extent * (17u * i + 3u);
        }
        std::array<uint, thread_count * local_count> values{};
        auto input = device.create_buffer<uint>(inputs.size());
        auto output = device.create_buffer<uint>(values.size());
        auto stream = device.create_stream();
        stream << input.copy_from(inputs.data())
               << shader(input, output).dispatch(thread_count)
               << output.copy_to(values.data())
               << synchronize();
        for (auto tid = 0u; tid < thread_count; ++tid) {
            for (auto i = 0u; i < local_count; ++i) {
                const auto expected = 100u * i + inputs[tid];
                expect(values[tid * local_count + i] == expected)
                    << "thread" << tid << "local" << i
                    << "actual" << values[tid * local_count + i]
                    << "expected" << expected;
            }
        }
    };
}
