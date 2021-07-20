//
// Created by Mike Smith on 2021/6/24.
//

#include <ast/function_builder.h>
#include <rtx/geometry.h>

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
Command *Geometry::trace_closest(BufferView<Ray> rays, BufferView<Hit> hits) const noexcept {
    return nullptr;
}
Command *Geometry::trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits) const noexcept {
    return nullptr;
}

Command *Geometry::trace_closest(BufferView<Ray> rays, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept {
    return nullptr;
}

Command *Geometry::trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept {
    return nullptr;
}

Command *Geometry::trace_any(BufferView<Ray> rays, BufferView<bool> hits) const noexcept {
    return nullptr;
}

Command *Geometry::trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits) const noexcept {
    return nullptr;
}

Command *Geometry::trace_any(BufferView<Ray> rays, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept {
    return nullptr;
}

Command *Geometry::trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept {
    return nullptr;
}

Command *Geometry::update() noexcept {
    _dirty = false;
    return nullptr;
}

Command *Geometry::build() noexcept {
    Command *command = nullptr;
    Command *tail = nullptr;
    for (auto i = 0u; i < _mesh_built.size(); i++) {
        if (!_mesh_built[i]) {
            LUISA_WARNING_WITH_LOCATION(
                "Mesh #{} at index {} in geometry #{} is not built; building it first.",
                _mesh_handles[i], i, _handle);
            // TODO...
        }
    }
    return nullptr;
}

void Geometry::_mark_mesh_built(uint mesh_index) noexcept {
    _mesh_built[mesh_index] = true;
}

void Geometry::_mark_dirty() noexcept {
    _dirty = true;
}

Command *detail::Mesh::build() const noexcept {
    _geometry->_mark_mesh_built(_index);
    return nullptr;
}

Command *detail::Mesh::update() const noexcept {
    _geometry->_mark_dirty();
    return nullptr;
}

uint64_t detail::Mesh::handle() const noexcept {
    return _geometry->_mesh_handles[_index];
}

}// namespace luisa::compute
