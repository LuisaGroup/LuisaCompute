#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/ast/function_builder.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    // callable(tex: tex2d, coord: inout uint2, color: float3)
    shared_ptr<FuncBuilder const> callable_builder = FuncBuilder::define_callable([&]() {
        auto &cur = *FuncBuilder::current();
        auto arg0 = cur.texture(Type::of<Image<float>>());
        auto arg1 = cur.reference(Type::of<uint2>());
        cur.mark_variable_usage(arg1->variable().uid(), Usage::READ_WRITE);
        auto arg2 = cur.argument(Type::of<float3>());
        // float4 value = make_float4(arg2, 1.0f);
        auto value = cur.call(
            Type::of<float4>(),
            CallOp::MAKE_FLOAT4,
            {arg2,
             cur.literal(Type::of<float>(), 1.0f)});
        // texture.write(coord, value)
        cur.call(
            CallOp::TEXTURE_WRITE,
            {arg0,
             arg1,
             value});
    });
    // kernel(tex: tex2d)
    shared_ptr<FuncBuilder const> kernel = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        cur.set_block_size(uint3(16, 16, 1));
        auto arg0 = cur.texture(Type::of<Image<float>>());
        auto coord_uint3 = cur.dispatch_id();
        // uint2 coord;
        auto coord = cur.local(Type::of<uint2>());
        // coord = dispatch_id().xy;
        uint64_t swizzle_code = (0ull) | (1ull << 4ull);
        cur.assign(coord, cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_code));
        auto size = cur.dispatch_size();
        // float2 coord_float = make_float2(coord);
        Expression const *coord_float = cur.call(
            Type::of<float2>(),
            CallOp::MAKE_FLOAT2,
            {coord});
        // coord_float = coord_float + 0.5f
        coord_float = cur.binary(Type::of<float2>(), BinaryOp::ADD, coord_float, cur.literal(Type::of<float2>(), float2(0.5f)));
        // size_float = float2(size)
        auto size_float_expr = cur.call(
            Type::of<float2>(),
            CallOp::MAKE_FLOAT2,
            {size});
        // uv = coord_float / size_float
        auto uv_expr = cur.binary(Type::of<float2>(), BinaryOp::DIV, coord_float, size_float_expr);
        auto uv_var = cur.local(Type::of<float2>());
        cur.assign(uv_var, uv_expr);
        // color = make_float3(uv, 0.5f)
        auto color = cur.call(
            Type::of<float3>(),
            CallOp::MAKE_FLOAT3,
            {uv_var,
             cur.literal(Type::of<float>(), 1.0f)});
        cur.call(Function(callable_builder.get()), {arg0,
                                                    coord,
                                                    color});
    });
    // save shader to test_manual_ast.bytes
    auto invalid_shader = device.impl()->create_shader(
        {.compile_only = true,
         .name = "test_manual_ast.bytes"},
        Function(kernel.get()));
    // load shader from disk
    auto shader = device.load_shader<2, Image<float>>("test_manual_ast.bytes");

    // Kernel2D kernel = [&]() {
    //     Var coord = dispatch_id().xy();
    //     Var size = dispatch_size().xy();
    //     Var uv = (make_float2(coord) + 0.5f) / make_float2(size);
    //     image->write(coord, make_float4(uv, 0.5f, 1.0f));
    // };
    // auto shader = device.compile(Function(kernel.get()));
    stream << shader(image).dispatch(resolution)
           << image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_manual_ast.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    return 0;
}
