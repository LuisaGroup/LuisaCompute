#include "ut/ut.hpp"
#include "test_device.h"
// Test for mipmap generation using compute shaders.
//
// This test implements parallel mipmap generation using the
// recursive box filter algorithm. Each level is computed by
// averaging 2x2 pixel blocks from the previous level.
//
// Implementation features:
// - Single-pass multi-level generation using shared memory
// - Hierarchical reduction within thread blocks
// - Support for up to 6 mipmap levels
//
// Mipmaps are used for:
// - Texture minification filtering
// - Level-of-detail (LOD) selection
// - Faster texture sampling at distance

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_mipmap(Device &device) {
    Stream stream = device.create_stream();

    // Load input image
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("logo.png", &image_width, &image_height, &image_channels, 4);

    // Create texture with 6 mipmap levels
    auto texture = device.create_image<float>(PixelStorage::BYTE4, uint2(image_width, image_height), 6u);

    // Block size for compute kernel
    constexpr uint32_t block_size = 32;

    // Helper to write to specific mipmap level using switch
    auto WriteTex = [&](ImageVar<float> **levels, UInt2 pixel, Float4 value, UInt index) {
        switch_(index)
            .case_(0, [&] {
                levels[0]->write(pixel, value);
            })
            .case_(1, [&] {
                levels[1]->write(pixel, value);
            })
            .case_(2, [&] {
                levels[2]->write(pixel, value);
            })
            .case_(3, [&] {
                levels[3]->write(pixel, value);
            })
            .case_(4, [&] {
                levels[4]->write(pixel, value);
            })
            .case_(5, [&] {
                levels[5]->write(pixel, value);
            });
    };

    // Mipmap generation kernel
    // Uses shared memory for efficient 2x2 averaging across multiple levels
    Kernel2D generate_mip_levels =
        [&](ImageVar<float> level0,
            ImageVar<float> level1,
            ImageVar<float> level2,
            ImageVar<float> level3,
            ImageVar<float> level4,
            ImageVar<float> level5) {
            set_block_size(block_size, block_size, 1);

            // Shared memory for hierarchical reduction
            // Stores intermediate results during 2x2 averaging
            Shared<float3> shared_array{block_size * block_size};

            // Array of mipmap level views
            ImageVar<float> *levels[] = {
                std::addressof(level0),
                std::addressof(level1),
                std::addressof(level2),
                std::addressof(level3),
                std::addressof(level4),
                std::addressof(level5)};

            Var block_coord = block_id().xy();
            Var local_coord = thread_id().xy();
            Var tex_size = dispatch_size().xy();

            // Read source pixel with clamping to handle borders
            Var col = level0.read(clamp(dispatch_id().xy(), make_uint2(0u), tex_size - 1u)).xyz();

            Var lefted_block = block_size;
            Var ite = 0u;

            // Hierarchical reduction loop
            // Each iteration halves the resolution by averaging 2x2 blocks
            $while (lefted_block > 0) {
                Var next_block = lefted_block / 2;

                // Store current values to shared memory
                $if (all(local_coord < make_uint2(lefted_block))) {
                    shared_array[lefted_block * local_coord.y + local_coord.x] = col;
                };
                sync_block();

                // Average 2x2 blocks for next mipmap level
                $if (all(local_coord < make_uint2(next_block))) {
                    Var last_coord = local_coord * make_uint2(2);
                    // Box filter: average 4 neighboring pixels
                    col = shared_array[lefted_block * last_coord.y + last_coord.x] +
                          shared_array[lefted_block * (last_coord.y + 1) + last_coord.x] +
                          shared_array[lefted_block * last_coord.y + (last_coord.x + 1)] +
                          shared_array[lefted_block * (last_coord.y + 1) + (last_coord.x + 1)];
                    col *= 0.25f;// Divide by 4

                    ite += 1u;
                    Var level_coord = block_coord * next_block + local_coord;

                    // Write to appropriate mipmap level if within bounds
                    $if (all(level_coord < tex_size)) {
                        WriteTex(levels, level_coord, make_float4(col, 1.0f), ite);
                    };
                };

                // Prepare for next iteration (half resolution)
                lefted_block = next_block;
                tex_size /= 2u;
            };
        };

    // Compile and execute kernel
    auto shader = device.compile(generate_mip_levels);
    stream << texture.copy_from(luisa::span{image_pixels, static_cast<size_t>(image_width * image_height * 4)})
           << shader(
                  texture.view(0),
                  texture.view(1),
                  texture.view(2),
                  texture.view(3),
                  texture.view(4),
                  texture.view(5))
                  .dispatch(texture.size());

    // Save generated mipmap levels
    for (int i = 1; i < 6; ++i) {
        auto view = texture.view(i);
        luisa::vector<std::byte> host_image(view.size_bytes());
        stream << view.copy_to(luisa::span{host_image}) << synchronize();
        auto name = luisa::string{"logo_mip"}.append(std::to_string(i)).append(".png");
        auto size = view.size();
        stbi_write_png(name.c_str(), size.x, size.y, 4, host_image.data(), 0);
    }
    expect(true) << "mipmap test completed";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_mipmap(device);
}
