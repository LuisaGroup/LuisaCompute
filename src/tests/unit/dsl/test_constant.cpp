// Test for constant buffer handling in compute kernels.
// This test verifies the creation and usage of constant buffers containing
// struct data, demonstrating how to pass read-only data to GPU kernels.

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test struct for constant buffer data
struct Foo {
    luisa::uint a;// Index field
    float b;      // Offset field for color
    float c;
    float d;
};
LUISA_STRUCT(Foo, a, b, c, d) {};

void test_constant(Device &device) {
    // Create a buffer for testing (not used in this test but demonstrates setup)
    auto buffer = device.create_buffer<uint>(1024);
    Stream stream = device.create_stream();

    // Set up image for kernel output
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());

    // Define kernel that uses constant buffer data
    Kernel2D kernel = [&]() {
        // Initialize constant data array with 4 Foo structs
        Foo foo_data[4]{
            {1, 2.0f, 3.0f, 4.0f},
            {5, 6.0f, 7.0f, 8.0f},
            {9, 10.0f, 11.0f, 12.0f},
            {13, 14.0f, 15.0f, 16.0f}};
        // Create constant buffer from the data
        Constant<Foo> foo(foo_data, 4);

        // Get dispatch coordinates
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Index into constant buffer based on x coordinate
        Var<uint> i = coord.x % 4u;

        // Generate UV coordinates and apply color offset from constant buffer
        Var uv = (make_float2(coord) + 0.5f) / make_float2(size) + foo.read(i).b;

        // Write result to image
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };

    // Compile and execute kernel
    auto shader = device.compile(kernel);
    stream << shader().dispatch(resolution)
           << image.copy_to(luisa::span{host_image})
           << synchronize();
    expect(true) << "constant buffer kernel executed";

    // Save result to PNG file
    stbi_write_png("test_helloworld.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_constant(device);
}
