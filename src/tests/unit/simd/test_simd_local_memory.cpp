#include "ut/ut.hpp"

#include <array>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};
    auto device = context.create_device("simd");

    constexpr auto thread_count = 77u;
    constexpr auto values_per_thread = 3u;
    auto output = device.create_buffer<uint>(
        thread_count * values_per_thread);

    Kernel1D kernel = [](BufferUInt result) noexcept {
        set_block_size(32u, 1u, 1u);
        auto gid = dispatch_x();
        auto tid = thread_x();

        $array<uint, 4u> values;
        values[0u] = gid * 16u + 1u;
        values[1u] = gid * 16u + 2u;
        values[2u] = gid * 16u + 3u;
        values[3u] = gid * 16u + 4u;
        auto index = (tid * 3u + block_x()) & 3u;
        auto before = def(values[index]);
        $if ((tid & 1u) == 0u) {
            values[index] = before + 1000u;
        };
        auto after = def(values[index]);

        $array<uint2, 2u> pairs;
        pairs[0u] = make_uint2(gid + 10u, gid + 20u);
        pairs[1u] = make_uint2(gid + 30u, gid + 40u);
        auto pair_index = tid & 1u;
        auto component_index = (tid >> 1u) & 1u;
        auto nested = def(pairs[pair_index][component_index]);

        result.write(gid * 3u + 0u, before);
        result.write(gid * 3u + 1u, after);
        result.write(gid * 3u + 2u, nested);
    };

    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    luisa::vector<uint> host(thread_count * values_per_thread, 0u);
    stream << shader(output).dispatch(thread_count)
           << output.copy_to(luisa::span{host})
           << synchronize();

    for (auto gid = 0u; gid < thread_count; gid++) {
        auto tid = gid % 32u;
        auto block = gid / 32u;
        auto index = (tid * 3u + block) & 3u;
        auto before = gid * 16u + index + 1u;
        auto after = (tid & 1u) == 0u ? before + 1000u : before;
        auto pair_index = tid & 1u;
        auto component_index = (tid >> 1u) & 1u;
        auto nested = gid + 10u + pair_index * 20u +
                      component_index * 10u;
        expect(host[gid * values_per_thread + 0u] == before)
            << "dynamic local-array load must remain lane-private";
        expect(host[gid * values_per_thread + 1u] == after)
            << "divergent local-array store must update active lanes only";
        expect(host[gid * values_per_thread + 2u] == nested)
            << "nested dynamic local indexing must preserve ABI layout";
    }
}
