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

int main(int argc, char *argv[]) {
    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("logo.png", &image_width, &image_height, &image_channels, 4);
    auto texture = device.create_image<float>(PixelStorage::BYTE4, uint2(image_width, image_height), 6u);
    constexpr uint32_t block_size = 32;
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
    Kernel2D generate_mip_levels =
        [&](ImageVar<float> level0,
            ImageVar<float> level1,
            ImageVar<float> level2,
            ImageVar<float> level3,
            ImageVar<float> level4,
            ImageVar<float> level5) {
            set_block_size(block_size, block_size, 1);
            Shared<float3> shared_array{block_size * block_size};
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
            Var col = level0.read(clamp(dispatch_id().xy(), make_uint2(0u), tex_size - 1u)).xyz();
            Var lefted_block = block_size;
            Var ite = 0u;
            $while(lefted_block > 0) {
                Var next_block = lefted_block / 2;
                $if(all(local_coord < make_uint2(lefted_block))) {
                    shared_array[lefted_block * local_coord.y + local_coord.x] = col;
                };
                sync_block();
                $if(all(local_coord < make_uint2(next_block))) {
                    Var last_coord = local_coord * make_uint2(2);
                    col = shared_array[lefted_block * last_coord.y + last_coord.x] +
                          shared_array[lefted_block * (last_coord.y + 1) + last_coord.x] +
                          shared_array[lefted_block * last_coord.y + (last_coord.x + 1)] +
                          shared_array[lefted_block * (last_coord.y + 1) + (last_coord.x + 1)];
                    col *= 0.25f;
                    ite += 1u;
                    Var level_coord = block_coord * next_block + local_coord;
                    $if(all(level_coord < tex_size)) {
                        WriteTex(levels, level_coord, make_float4(col, 1.0f), ite);
                    };
                };
                lefted_block = next_block;
                tex_size /= 2u;
            };
        };
    auto shader = device.compile(generate_mip_levels);
    stream << texture.copy_from(image_pixels)
           << shader(
                  texture.view(0),
                  texture.view(1),
                  texture.view(2),
                  texture.view(3),
                  texture.view(4),
                  texture.view(5))
                  .dispatch(texture.size());
    for (int i = 1; i < 6; ++i) {
        auto view = texture.view(i);
        luisa::vector<std::byte> host_image(view.size_bytes());
        stream << view.copy_to(host_image.data()) << synchronize();
        auto name = luisa::string{"logo_mip"}.append(std::to_string(i)).append(".png");
        auto size = view.size();
        stbi_write_png(name.c_str(), size.x, size.y, 4, host_image.data(), 0);
    }
    return 0;
}
