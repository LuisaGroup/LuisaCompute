//
// Created by Mike Smith on 2021/6/24.
//

#include <ast/function_builder.h>
#include <rtx/accel.h>

namespace luisa::compute {
/*
detail::Expr<Hit> Accel::trace_closest(detail::Expr<Ray> ray) const noexcept {
//    auto f = FunctionBuilder::current();
//    f->mark_raytracing();
//    return detail::Expr<Hit>{
//        f->call(Type::of<Hit>(), CallOp::TRACE_CLOSEST,
//                {detail::extract_expression(_handle), ray.expression()})};
}

detail::Expr<bool> Accel::trace_any(detail::Expr<Ray> ray) const noexcept {
//    auto f = FunctionBuilder::current();
//    f->mark_raytracing();
//    return detail::Expr<bool>{
//        f->call(Type::of<bool>(), CallOp::TRACE_ANY,
//                {detail::extract_expression(_handle), ray.expression()})};
}
*/
CommandHandle Accel::trace_closest(BufferView<Ray> rays, BufferView<Hit> hits) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_closest(BufferView<Ray> rays, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_any(BufferView<Ray> rays, BufferView<bool> hits) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_any(BufferView<Ray> rays, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept {
    return luisa::compute::CommandHandle();
}
CommandHandle Accel::update() const noexcept {
    return luisa::compute::CommandHandle();
}
}// namespace luisa::compute
