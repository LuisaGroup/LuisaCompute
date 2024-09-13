#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/accel.h>

namespace luisa::compute {

luisa::unique_ptr<Command> Mesh::build(Mesh::BuildRequest request) noexcept {
    _check_is_valid();
    return luisa::make_unique<MeshBuildCommand>(
        handle(), request,
        _v_buffer, _v_buffer_offset_bytes, _v_buffer_size_bytes, _v_stride,
        _t_buffer, _t_buffer_offset_bytes, _t_buffer_size_bytes);
}

Mesh::~Mesh() noexcept {
    if (*this) { device()->destroy_mesh(handle()); }
}
namespace detail {
LC_RUNTIME_API void check_mesh_vert_align(size_t v_stride, size_t dst) {
    if ((v_stride % dst) == 0) [[likely]]
        return;
    LUISA_ERROR("Require mesh vertex buffer with stride {} can not be aligned to v_stride {}", dst, v_stride);
}
}// namespace detail
}// namespace luisa::compute
