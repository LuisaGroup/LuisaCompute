#include "ut/ut.hpp"

#include <cmath>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    auto stream = device.create_stream();

    constexpr auto capacity = 12u;
    constexpr auto thread_count = 257u;
    constexpr auto output_stride = 4u;
    auto output = device.create_buffer<float>(
        thread_count * output_stride);

    Kernel1D kernel = [](BufferFloat output) noexcept {
        set_block_size(64u);
        auto gid = dispatch_x();
        const auto diagonal = [](Float value) noexcept {
            return make_float4x4(
                make_float4(value, 0.0f, 0.0f, 0.0f),
                make_float4(0.0f, value, 0.0f, 0.0f),
                make_float4(0.0f, 0.0f, value, 0.0f),
                make_float4(0.0f, 0.0f, 0.0f, value));
        };

        Local<float4x4> block_0{capacity};
        Local<float4x4> block_1{capacity};
        Local<float4x4> block_2{capacity};
        block_0.write(0u, make_float4x4(0.0f));
        block_1.write(0u, make_float4x4(0.0f));
        block_2.write(0u, make_float4x4(0.0f));

        UInt count = 0u;
        for (auto source = 0u; source < capacity; ++source) {
            auto retained = ((source + gid) % 3u) != 0u;
            $if (retained) {
                auto value = cast<float>(gid * 100u + source + 1u);
                block_0.write(count, diagonal(value));
                block_1.write(count, diagonal(value + 1000.0f));
                block_2.write(count, diagonal(value + 2000.0f));
                count += 1u;
            };
        }

        Float sum_0 = 0.0f;
        Float sum_1 = 0.0f;
        Float sum_2 = 0.0f;
        UInt index = 0u;
        $while (index < count) {
            auto value_0 = block_0.read(index);
            auto value_1 = block_1.read(index);
            auto value_2 = block_2.read(index);
            sum_0 += value_0[0u].x + value_0[1u].y +
                     value_0[2u].z + value_0[3u].w;
            sum_1 += value_1[0u].x + value_1[1u].y +
                     value_1[2u].z + value_1[3u].w;
            sum_2 += value_2[0u].x + value_2[1u].y +
                     value_2[2u].z + value_2[3u].w;
            index += 1u;
        };

        auto requested = (gid * 7u) % (capacity + 3u);
        auto valid = requested < count;
        auto safe_index = select(0u, requested, valid);
        auto selected = block_2.read(safe_index)[0u].x;
        selected = select(0.0f, selected, valid);

        auto base = gid * 4u;
        output.write(base + 0u, sum_0);
        output.write(base + 1u, sum_1);
        output.write(base + 2u, sum_2);
        output.write(base + 3u, selected);
    };

    auto shader = device.compile(kernel);
    luisa::vector<float> actual(
        thread_count * output_stride, 0.0f);
    stream << shader(output).dispatch(thread_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto gid = 0u; gid < thread_count; ++gid) {
        luisa::vector<float> retained;
        for (auto source = 0u; source < capacity; ++source) {
            if (((source + gid) % 3u) != 0u) {
                retained.emplace_back(
                    static_cast<float>(gid * 100u + source + 1u));
            }
        }
        auto sum = 0.0f;
        for (auto value : retained) { sum += 4.0f * value; }
        const auto base = gid * output_stride;
        expect(std::abs(actual[base + 0u] - sum) < 1e-3f);
        expect(std::abs(actual[base + 1u] -
                        (sum + 4000.0f * retained.size())) < 1e-3f);
        expect(std::abs(actual[base + 2u] -
                        (sum + 8000.0f * retained.size())) < 1e-3f);
        const auto requested = (gid * 7u) % (capacity + 3u);
        const auto selected = requested < retained.size() ?
                                  retained[requested] + 2000.0f :
                                  0.0f;
        expect(std::abs(actual[base + 3u] - selected) < 1e-3f);
    }
}
