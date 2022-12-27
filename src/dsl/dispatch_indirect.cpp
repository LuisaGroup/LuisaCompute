#include <dsl/dispatch_indirect.h>
#include <runtime/device.h>
#include <runtime/buffer.h>
namespace luisa::compute {
/*
Buffer<DrawIndirectArgs> Device::create_draw_buffer(const MeshFormat &mesh_format, size_t capacity) noexcept {
    Buffer<DrawIndirectArgs> v;
    // Resource
    v._device = _impl;
    auto ptr = _impl.get();
    auto buffer = ptr->create_draw_buffer(mesh_format, false, capacity);
    v._handle = buffer.handle;
    v._tag = Resource::Tag::BUFFER;
    // Buffer
    v._size = buffer.size / custom_struct_size;
    return v;
}
Buffer<DrawIndexedIndirectArgs> Device::create_indexed_draw_buffer(const MeshFormat &mesh_format, size_t capacity) noexcept {
    Buffer<DrawIndexedIndirectArgs> v;
    // Resource
    v._device = _impl;
    auto ptr = _impl.get();
    auto buffer = ptr->create_draw_buffer(mesh_format, true, capacity);
    v._handle = buffer.handle;
    v._tag = Resource::Tag::BUFFER;
    // Buffer
    v._size = buffer.size / custom_struct_size;
    return v;
}
*/
void clear_dispatch_buffer(Expr<Buffer<DispatchArgs1D>> buffer) {
    detail::FunctionBuilder::current()->call(CallOp::INDIRECT_CLEAR_DISPATCH_BUFFER, {buffer.expression()});
}
void clear_dispatch_buffer(Expr<Buffer<DispatchArgs2D>> buffer) {
    detail::FunctionBuilder::current()->call(CallOp::INDIRECT_CLEAR_DISPATCH_BUFFER, {buffer.expression()});
}
void clear_dispatch_buffer(Expr<Buffer<DispatchArgs3D>> buffer) {
    detail::FunctionBuilder::current()->call(CallOp::INDIRECT_CLEAR_DISPATCH_BUFFER, {buffer.expression()});
}
void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs1D>> buffer,
    Expr<uint> block_size,
    Expr<uint> dispatch_size,
    Expr<uint> kernel_id) {
    detail::FunctionBuilder::current()->call(CallOp::INDIRECT_EMPLACE_DISPATCH_KERNEL, {buffer.expression(), block_size.expression(), dispatch_size.expression(), kernel_id.expression()});
}
void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs2D>> buffer,
    Expr<uint2> block_size,
    Expr<uint2> dispatch_size,
    Expr<uint> kernel_id) {
    detail::FunctionBuilder::current()->call(CallOp::INDIRECT_EMPLACE_DISPATCH_KERNEL, {buffer.expression(), block_size.expression(), dispatch_size.expression(), kernel_id.expression()});
}
void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs3D>> buffer,
    Expr<uint3> block_size,
    Expr<uint3> dispatch_size,
    Expr<uint> kernel_id) {
    detail::FunctionBuilder::current()->call(CallOp::INDIRECT_EMPLACE_DISPATCH_KERNEL, {buffer.expression(), block_size.expression(), dispatch_size.expression(), kernel_id.expression()});
}
}// namespace luisa::compute