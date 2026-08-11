#include "ut/ut.hpp"

#include <array>

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};

    for (auto width : std::array{1u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        expect(device.compute_warp_size() == width)
            << "SIMD device must expose its configured warp width";

        auto output = device.create_buffer<uint>(width);
        Kernel1D kernel = [width](BufferUInt result) noexcept {
            set_block_size(width, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto lane = warp_lane_id();
            auto sum = warp_active_sum(lane + 1u);
            result.write(dispatch_x(), sum + lane);
        };
        auto shader = device.compile(kernel);
        auto stream = device.create_stream();
        luisa::vector<uint> host(width, 0u);
        stream << shader(output).dispatch(width)
               << output.copy_to(luisa::span{host})
               << synchronize();

        auto sum = width * (width + 1u) / 2u;
        for (auto lane = 0u; lane < width; lane++) {
            expect(host[lane] == sum + lane)
                << "SIMD fixed-width runtime result mismatch";
        }
    }
}
