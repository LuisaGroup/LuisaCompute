// Demonstrates bindless mip-level read and explicit-LOD sampling.

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] auto make_level_pixels(uint2 size, uint8_t value) noexcept {
    luisa::vector<std::byte> pixels(size.x * size.y * 4u);
    for (auto i = 0u; i < size.x * size.y; i++) {
        pixels[i * 4u + 0u] = static_cast<std::byte>(value);
        pixels[i * 4u + 1u] = static_cast<std::byte>(value + 1u);
        pixels[i * 4u + 2u] = static_cast<std::byte>(value + 2u);
        pixels[i * 4u + 3u] = static_cast<std::byte>(255u);
    }
    return pixels;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>", argv[0]);
        return 0;
    }

    static constexpr auto level_count = 4u;
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    auto texture = device.create_image<float>(PixelStorage::BYTE4, make_uint2(8u, 8u), level_count);
    auto heap = device.create_bindless_array(1u);
    auto out = device.create_buffer<float4>(level_count * 2u);
    luisa::vector<float4> host(out.size());

    for (auto level = 0u; level < level_count; level++) {
        auto view = texture.view(level);
        auto pixels = make_level_pixels(view.size(), static_cast<uint8_t>(32u + level * 48u));
        stream << view.copy_from(luisa::span{pixels});
    }
    stream << heap.emplace_on_update(0u, texture, Sampler::point_edge()).update();

    Kernel1D read_and_sample = [&](BindlessVar bindless, BufferVar<float4> output) noexcept {
        auto level = dispatch_id().x;
        auto tex = bindless.tex2d(0u);
        auto read_value = tex.read(make_uint2(0u), level);
        auto sample_value = tex.sample(make_float2(0.5f), cast<float>(level));
        output.write(level * 2u, read_value);
        output.write(level * 2u + 1u, sample_value);
    };

    auto shader = device.compile(read_and_sample);
    stream << shader(heap, out).dispatch(level_count)
           << out.copy_to(luisa::span{host})
           << synchronize();

    for (auto level = 0u; level < level_count; level++) {
        auto read_value = host[level * 2u];
        auto sample_value = host[level * 2u + 1u];
        LUISA_INFO("level {}: read=({}, {}, {}, {}), sample=({}, {}, {}, {})",
                   level,
                   read_value.x, read_value.y, read_value.z, read_value.w,
                   sample_value.x, sample_value.y, sample_value.z, sample_value.w);
    }
}
