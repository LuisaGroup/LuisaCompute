#include <luisa/runtime/builtin_kernel.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/runtime/device.h>

namespace luisa::compute {

using FunctionBuilder = detail::FunctionBuilder;

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_buffer() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create buffer reference (uint buffer)
        auto buf_ref = cur.buffer(Type::of<Buffer<uint>>());

        // Get dispatch_id.x as index (1D dispatch)
        auto dispatch = cur.dispatch_id();
        auto idx = cur.swizzle(Type::of<uint>(), dispatch, 1, 0ull);// x component only

        // Create literal value (will be passed as argument)
        auto val = cur.argument(Type::of<uint>());

        // Call BUFFER_WRITE
        cur.call(CallOp::BUFFER_WRITE, {buf_ref, idx, val});

        // Set block size for 1D dispatch
        cur.set_block_size(uint3{128, 1, 1});
    });
}

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_image_uint() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create image reference (Image<uint>)
        auto img_ref = cur.texture(Type::of<Image<uint>>());

        // Get dispatch_id and swizzle to uint2 for 2D coordinates
        auto dispatch = cur.dispatch_id();
        // swizzle_code: 0 = x, 1 = y (each index in 4 bits)
        uint64_t swizzle_code = 0ull | (1ull << 4ull);
        auto coord = cur.swizzle(Type::of<uint2>(), dispatch, 2, swizzle_code);// xy components

        // Create literal value
        auto val = cur.argument(Type::of<uint>());
        auto texel = cur.call(Type::of<uint4>(), CallOp::MAKE_UINT4, {val});

        // Call TEXTURE_WRITE
        cur.call(CallOp::TEXTURE_WRITE, {img_ref, coord, texel});

        // Set block size for 2D dispatch
        cur.set_block_size(uint3{16, 8, 1});
    });
}

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_image_int() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create image reference (Image<int>)
        auto img_ref = cur.texture(Type::of<Image<int>>());

        // Get dispatch_id and swizzle to uint2 for 2D coordinates
        auto dispatch = cur.dispatch_id();
        // swizzle_code: 0 = x, 1 = y (each index in 4 bits)
        uint64_t swizzle_code = 0ull | (1ull << 4ull);
        auto coord = cur.swizzle(Type::of<uint2>(), dispatch, 2, swizzle_code);// xy components

        // Create literal value
        auto val = cur.argument(Type::of<int>());
        auto texel = cur.call(Type::of<int4>(), CallOp::MAKE_INT4, {val});

        // Call TEXTURE_WRITE
        cur.call(CallOp::TEXTURE_WRITE, {img_ref, coord, texel});

        // Set block size for 2D dispatch
        cur.set_block_size(uint3{16, 8, 1});
    });
}

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_image_float() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create image reference (Image<float>)
        auto img_ref = cur.texture(Type::of<Image<float>>());

        // Get dispatch_id and swizzle to uint2 for 2D coordinates
        auto dispatch = cur.dispatch_id();
        // swizzle_code: 0 = x, 1 = y (each index in 4 bits)
        uint64_t swizzle_code = 0ull | (1ull << 4ull);
        auto coord = cur.swizzle(Type::of<uint2>(), dispatch, 2, swizzle_code);// xy components

        // Create literal value
        auto val = cur.argument(Type::of<float>());
        auto texel = cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {val});

        // Call TEXTURE_WRITE
        cur.call(CallOp::TEXTURE_WRITE, {img_ref, coord, texel});

        // Set block size for 2D dispatch
        cur.set_block_size(uint3{16, 8, 1});
    });
}

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_volume_uint() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create volume reference (Volume<uint>)
        auto vol_ref = cur.texture(Type::of<Volume<uint>>());

        // Get dispatch_id as uint3 for 3D coordinates
        auto coord = cur.dispatch_id();

        // Create literal value
        auto val = cur.argument(Type::of<uint>());
        auto texel = cur.call(Type::of<uint4>(), CallOp::MAKE_UINT4, {val});

        // Call TEXTURE_WRITE
        cur.call(CallOp::TEXTURE_WRITE, {vol_ref, coord, texel});

        // Set block size for 3D dispatch
        cur.set_block_size(uint3{8, 8, 8});
    });
}

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_volume_int() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create volume reference (Volume<int>)
        auto vol_ref = cur.texture(Type::of<Volume<int>>());

        // Get dispatch_id as uint3 for 3D coordinates
        auto coord = cur.dispatch_id();

        // Create literal value
        auto val = cur.argument(Type::of<int>());
        auto texel = cur.call(Type::of<int4>(), CallOp::MAKE_INT4, {val});

        // Call TEXTURE_WRITE
        cur.call(CallOp::TEXTURE_WRITE, {vol_ref, coord, texel});

        // Set block size for 3D dispatch
        cur.set_block_size(uint3{8, 8, 8});
    });
}

luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_volume_float() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create volume reference (Volume<float>)
        auto vol_ref = cur.texture(Type::of<Volume<float>>());

        // Get dispatch_id as uint3 for 3D coordinates
        auto coord = cur.dispatch_id();

        // Create literal value
        auto val = cur.argument(Type::of<float>());
        auto texel = cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {val});

        // Call TEXTURE_WRITE
        cur.call(CallOp::TEXTURE_WRITE, {vol_ref, coord, texel});

        // Set block size for 3D dispatch
        cur.set_block_size(uint3{8, 8, 8});
    });
}
luisa::shared_ptr<const detail::FunctionBuilder> BuiltinKernel::fill_buffer_from_first() noexcept {
    return FunctionBuilder::define_kernel([&]() {
        auto &cur = *FunctionBuilder::current();

        // Create buffer reference (uint buffer)
        auto buf_ref = cur.buffer(Type::of<Buffer<uint>>());

        // Get dispatch_id.x as index
        auto dispatch = cur.dispatch_id();
        auto idx = cur.swizzle(Type::of<uint>(), dispatch, 1, 0ull);

        // argument_1 = uint (offset/element_size)
        auto offset = cur.argument(Type::of<uint>());

        // source_index = dispatch_id().x % argument_1
        auto source_index = cur.binary(Type::of<uint>(), BinaryOp::MOD, idx, offset);

        // dest_index = dispatch_id().x + argument_1
        auto dest_index = cur.binary(Type::of<uint>(), BinaryOp::ADD, idx, offset);

        // buffer[dest_index] = buffer[source_index]
        auto value = cur.call(Type::of<uint>(), CallOp::BUFFER_READ, {buf_ref, source_index});
        cur.call(CallOp::BUFFER_WRITE, {buf_ref, dest_index, value});

        // Set block size for 1D dispatch
        cur.set_block_size(uint3{128, 1, 1});
    });
}

void BuiltinKernel::compile_all(Device &device) {
    _fill_buffer_from_first = compile<1, Buffer<uint>, uint>(fill_buffer_from_first());
    _fill_buffer_uint = compile<1, Buffer<uint>, uint>(fill_buffer());
    _fill_image_uint = compile<2, Image<uint>, uint>(fill_image_uint());
    _fill_image_int = compile<2, Image<int>, int>(fill_image_int());
    _fill_image_float = compile<2, Image<float>, float>(fill_image_float());
    _fill_volume_uint = compile<3, Volume<uint>, uint>(fill_volume_uint());
    _fill_volume_int = compile<3, Volume<int>, int>(fill_volume_int());
    _fill_volume_float = compile<3, Volume<float>, float>(fill_volume_float());
}

// Constructor
BuiltinKernel::BuiltinKernel(Device const &device) noexcept
    : _device{device} {}

}// namespace luisa::compute
