// Test for manual AST (Abstract Syntax Tree) construction using FunctionBuilder.
// This test demonstrates low-level kernel construction without using the DSL syntax sugar.
// It manually builds a kernel that writes a color gradient to an image using raw AST nodes.

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/func.h>
#include <exception>
#include <stb/stb_image_write.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_manual_ast(Device &device) {
    Stream stream = device.create_stream();

    // Set up image dimensions
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());

    using FuncBuilder = luisa::compute::detail::FunctionBuilder;

    // Define a callable that writes to a texture: callable(tex: tex2d, coord: inout uint2, color: float3)
    shared_ptr<FuncBuilder const> callable_builder = FuncBuilder::define_callable([&]() {
        auto &cur = *FuncBuilder::current();
        // Define texture parameter
        auto arg0 = cur.texture(Type::of<Image<float>>());
        // Define coordinate reference parameter (read-write)
        auto arg1 = cur.reference(Type::of<uint2>());
        cur.mark_variable_usage(arg1->variable().uid(), Usage::READ_WRITE);
        // Define color parameter
        auto arg2 = cur.argument(Type::of<float3>());
        // Create float4 value = make_float4(arg2, 1.0f)
        auto value = cur.call(
            Type::of<float4>(),
            CallOp::MAKE_FLOAT4,
            {arg2,
             cur.literal(Type::of<float>(), 1.0f)});
        // Call texture.write(coord, value)
        cur.call(
            CallOp::TEXTURE_WRITE,
            {arg0,
             arg1,
             value});
    });

    // Generate AST for kernel that creates a gradient image
    auto generate_ast = [&](Expression const *arg0) {
        auto &cur = *FuncBuilder::current();
        // Set kernel block size
        cur.set_block_size(uint3(16, 16, 1));
        // Get dispatch ID
        auto coord_uint3 = cur.dispatch_id();
        // Create local uint2 coord variable
        auto coord = cur.local(Type::of<uint2>());
        // coord = dispatch_id().xy (swizzle first 2 components)
        uint64_t swizzle_code = (0ull) | (1ull << 4ull);
        cur.assign(coord, cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_code));
        // Get dispatch size
        auto size = cur.dispatch_size();
        // Convert coord to float2: coord_float = make_float2(coord)
        Expression const *coord_float = cur.call(
            Type::of<float2>(),
            CallOp::MAKE_FLOAT2,
            {coord});
        // coord_float = coord_float + 0.5f (pixel center offset)
        coord_float = cur.binary(Type::of<float2>(), BinaryOp::ADD, coord_float, cur.literal(Type::of<float2>(), float2(0.5f)));
        // Convert size to float2: size_float = float2(size)
        auto size_float_expr = cur.call(
            Type::of<float2>(),
            CallOp::MAKE_FLOAT2,
            {size});
        // Compute UV coordinates: uv = coord_float / size_float
        auto uv_expr = cur.binary(Type::of<float2>(), BinaryOp::DIV, coord_float, size_float_expr);
        auto uv_var = cur.local(Type::of<float2>());
        cur.assign(uv_var, uv_expr);
        // Create color = make_float3(uv, 0.5f)
        auto color = cur.call(
            Type::of<float3>(),
            CallOp::MAKE_FLOAT3,
            {uv_var,
             cur.literal(Type::of<float>(), 1.0f)});
        // Call the callable with texture, coord, and color
        cur.call(Function(callable_builder.get()), {arg0,
                                                    coord,
                                                    color});
    };

    // Define kernel using DSL syntax that wraps the manual AST generation
    Kernel2D<Image<float>> kernel{[&](ImageVar<float> img) {
        auto arg0 = img.expression();
        generate_ast(arg0);
    }};
    auto shader = device.compile(kernel);

    // Alternative approach using pure manual AST construction (commented out):
    // shared_ptr<FuncBuilder const> kernel = FuncBuilder::define_kernel([&]() {
    //     // manually define arguments
    //     auto &cur = *FuncBuilder::current();
    //     auto arg0 = cur.texture(Type::of<Image<float>>());
    //     generate_ast(arg0);
    // });
    // auto shader = Shader2D<Image<float>>(
    //     device.impl(),
    //     Function(kernel.get()),
    //     ShaderOption{});

    // Execute kernel and save result to PNG
    stream << shader(image).dispatch(resolution)
           << image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("test_manual_ast.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_manual_ast(device);
    test_manual_ast(device);
}
