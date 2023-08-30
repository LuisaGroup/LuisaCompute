#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/sparse_buffer.h>
#include <luisa/runtime/sparse_image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>
#include <luisa/core/clock.h>
#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/runtime/sparse_command_list.h>

using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1], nullptr, true);
    auto stream = device.create_stream();
    auto dstorage_ext = device.extension<DStorageExt>();
    Stream dstorage_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::MemorySource});
    SparseCommandList sparse_cmdlist;
    // Test sparse buffer
    {
        Kernel1D write_sparse_kernel = [&](BufferVar<float> buffer) {
            UInt coord = dispatch_id().x;
            buffer.write(coord, coord.cast<float>() + 0.5f);
        };
        auto write_sparse_shader = device.compile(write_sparse_kernel);
        // An extremely huge virtual buffer
        constexpr size_t buffer_size = 1024ull * 1024ull * 1024ull * 32ull;
        auto sparse_buffer = device.create_sparse_buffer<float>(buffer_size);
        auto one_tile_size = sparse_buffer.tile_size() / sizeof(float);
        luisa::vector<float> result(one_tile_size);
        SparseBufferHeap buffer_heap = device.allocate_sparse_buffer_heap(sparse_buffer.tile_size());
        sparse_cmdlist << sparse_buffer.map_tile(1, 1, buffer_heap);
        auto buffer_view = sparse_buffer.view(one_tile_size, one_tile_size);
        stream
            << sparse_cmdlist.commit()
            << write_sparse_shader(buffer_view).dispatch(one_tile_size)
            << buffer_view.copy_to(result.data())
            << synchronize();
    }
    // Test sparse texture
    {
        // maximum virtual texture
        constexpr uint2 virtual_resolution = make_uint2(16384, 16384);
        constexpr uint2 resolution = make_uint2(1024, 1024);
        constexpr uint2 pixel_offset = resolution * make_uint2(2u);
        auto sparse_image = device.create_sparse_image<float>(PixelStorage::BYTE4, virtual_resolution.x, virtual_resolution.y);
        auto bindless_arr = device.create_bindless_array();
        bindless_arr.emplace_on_update(0, sparse_image, Sampler::linear_point_edge());
        Kernel2D write_sparse_kernel = [&](ImageVar<float> img, ImageVar<float> out, Float2 uv_offset, Float2 uv_scale) {
            Var coord = dispatch_id().xy();
            Var size = dispatch_size().xy();
            Var uv = (make_float2(coord) + 0.5f) / make_float2(size);
            uv = (uv * uv_scale) + uv_offset;
            out.write(coord, bindless_arr->tex2d(0).sample(uv));
        };
        Kernel2D write_kernel = [&](ImageVar<float> img, UInt2 offset) {
            Var coord = dispatch_id().xy();
            Var size = dispatch_size().xy();
            Var uv = (make_float2(coord) + 0.5f) / make_float2(size);
            img.write(coord + offset, make_float4(uv, 1.f, 1.0f));
        };
        auto event = device.create_event();
        auto write_sparse_shader = device.compile(write_sparse_kernel);
        auto write_shader = device.compile(write_kernel);
        auto buffer = device.create_buffer<uint>(resolution.x * resolution.y);
        Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
        luisa::vector<std::byte> pinned(image.view().size_bytes());
        luisa::vector<std::byte> result(image.view().size_bytes());
        SparseTextureHeap tex_heap = device.allocate_sparse_texture_heap(pixel_storage_size(image.storage(), make_uint3(resolution, 1u)), false);
        sparse_cmdlist << sparse_image.map_tile(pixel_offset / sparse_image.tile_size(), resolution / sparse_image.tile_size(), 0, tex_heap);
        stream
            << bindless_arr.update()
            << write_shader(image, make_uint2(0)).dispatch(resolution)
            << image.copy_to(pinned.data()) << sparse_cmdlist.commit()
            << [&]() {
                   auto file = dstorage_ext->pin_memory(pinned.data(), pinned.size_bytes());
                   dstorage_stream << file.copy_to(sparse_image, pixel_offset / sparse_image.tile_size(), resolution / sparse_image.tile_size(), 0) << event.signal() << [f = std::move(file)] {};
               }
            << synchronize() << event.wait() << write_sparse_shader(sparse_image.view(), image, make_float2(pixel_offset) / make_float2(virtual_resolution), make_float2(resolution) / make_float2(virtual_resolution)).dispatch(resolution) << image.copy_to(result.data()) << synchronize();
        stbi_write_png("test_sparse.png", resolution.x, resolution.y, 4, result.data(), 0);
    }
}
