#include <luisa/luisa-compute.h>

#include <array>
#include <iostream>
#include <utility>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    auto device = context.create_device("fallback");
    auto stream = device.create_stream();
    bool passed = true;
    // An LLVM barrier frame is per GPU lane; all lanes remain live until
    // the block completes. Check overflow of the former 4 MiB arena,
    // different frame sizes/block widths, and reuse after the block finishes.
    for (const auto [block, count] : std::array{
             std::pair{64u, 256u}, std::pair{64u, 18432u},
             std::pair{128u, 9216u}, std::pair{64u, 18432u},
             std::pair{64u, 256u}}) {
        const auto dispatch_count = block * 3u;
        Kernel1D kernel = [block, count](BufferUInt2 output, UInt seed) {
            set_block_size(block);
            Local<uint> values{count};
            const auto id = dispatch_x();
            $for(i, count) { values.write(i, (id * 3u + i) ^ seed); };
            sync_block();
            UInt sum = 0u;
            $for(i, count) { sum += values.read((i + seed) % count); };
            sync_block();
            output.write(id, make_uint2(sum, values.read((seed * 17u) % count)));
        };
        auto shader = device.compile(kernel, ShaderOption{
            .enable_cache = false, .enable_fast_math = true});
        auto output = device.create_buffer<uint2>(dispatch_count);
        std::vector<uint2> actual(dispatch_count);
        for (auto seed : {7u, 91u, 7u}) {
            stream << shader(output, seed).dispatch(dispatch_count)
                   << output.copy_to(actual.data()) << synchronize();
            for (auto id = 0u; id < actual.size(); ++id) {
                uint sum = 0u;
                for (auto i = 0u; i < count; ++i) { sum += (id * 3u + i) ^ seed; }
                const auto selected = (id * 3u + (seed * 17u) % count) ^ seed;
                if (actual[id].x != sum || actual[id].y != selected) {
                    std::cerr << "block=" << block << " locals=" << count
                              << " seed=" << seed << " lane=" << id << '\n';
                    passed = false;
                }
            }
        }
        std::cout << "Fallback barrier frames: block=" << block
                  << " local_bytes_per_lane=" << count * sizeof(uint) << std::endl;
    }
    return passed ? 0 : 1;
}
