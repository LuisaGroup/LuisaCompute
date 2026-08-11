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

    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        expect(device.compute_warp_size() == width)
            << "SIMD device must expose its configured warp width";

        constexpr auto block_threads = 32u;
        auto output = device.create_buffer<uint>(block_threads);
        Kernel1D kernel = [width](BufferUInt result) noexcept {
            // Block size and logical warp size are separate contracts. Like
            // fallback's scalar thread loop, SIMD partitions this 32-thread
            // block into packets of the configured width.
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto lane = warp_lane_id();
            auto sum = warp_active_sum(lane + 1u);
            result.write(dispatch_x(), sum + lane);
        };
        auto shader = device.compile(kernel);
        auto stream = device.create_stream();
        luisa::vector<uint> host(block_threads, 0u);
        stream << shader(output).dispatch(block_threads)
               << output.copy_to(luisa::span{host})
               << synchronize();

        auto sum = width * (width + 1u) / 2u;
        for (auto thread = 0u; thread < block_threads; thread++) {
            auto lane = thread % width;
            expect(host[thread] == sum + lane)
                << "SIMD thread-block packet partition mismatch";
        }

        // Regression: LLVM integer div/rem may lower to trapping scalar
        // instructions even when a zero divisor exists only in an inactive
        // tail lane. Sanitization therefore has to happen before the vector
        // operation; masking the result afterwards is too late.
        auto tail_threads = width == 1u ? 1u : width - 1u;
        auto tail_input = device.create_buffer<uint>(tail_threads);
        auto tail_output = device.create_buffer<uint>(tail_threads);
        Kernel1D tail_kernel = [width](
                                   BufferUInt input,
                                   BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto divisor = input.read(dispatch_x());
            result.write(dispatch_x(), 17u % divisor);
        };
        auto tail_shader = device.compile(tail_kernel);
        luisa::vector<uint> tail_divisors(tail_threads, 3u);
        luisa::vector<uint> tail_host(tail_threads, 0u);
        stream << tail_input.copy_from(luisa::span{tail_divisors})
               << tail_shader(tail_input, tail_output).dispatch(tail_threads)
               << tail_output.copy_to(luisa::span{tail_host})
               << synchronize();
        for (auto value : tail_host) {
            expect(value == 2u)
                << "inactive SIMD tail lane reached integer remainder";
        }

        // Exercise the runtime's fixed-width texture packet callback rather
        // than only the generic Schedule-to-LLVM packet ABI. The non-multiple
        // dispatch shape covers sparse edge masks and inactive tail lanes.
        constexpr auto image_size = make_uint2(7u, 3u);
        auto image = device.create_image<float>(
            PixelStorage::FLOAT4, image_size);
        luisa::vector<float4> image_input(image_size.x * image_size.y);
        luisa::vector<float4> image_output(image_input.size());
        for (auto y = 0u; y < image_size.y; y++) {
            for (auto x = 0u; x < image_size.x; x++) {
                image_input[y * image_size.x + x] = make_float4(
                    static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(x + y), 1.0f);
            }
        }
        Kernel2D image_kernel = [width](ImageFloat target) noexcept {
            set_block_size(8u, 4u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto value = target.read(coordinate);
            target.write(
                coordinate,
                value + make_float4(1.0f, 2.0f, 3.0f, 4.0f));
        };
        auto image_shader = device.compile(image_kernel);
        stream << image.copy_from(luisa::span{image_input})
               << image_shader(image).dispatch(image_size)
               << image.copy_to(luisa::span{image_output})
               << synchronize();
        for (auto i = size_t{0u}; i < image_input.size(); i++) {
            expect(all(image_output[i] ==
                       image_input[i] +
                           make_float4(1.0f, 2.0f, 3.0f, 4.0f)))
                << "SIMD fixed-width texture packet mismatch";
        }
    }
}
