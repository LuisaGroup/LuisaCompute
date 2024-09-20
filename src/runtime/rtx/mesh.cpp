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

void check_mesh_vert_align(size_t v_stride, size_t dst) {
    LUISA_ASSERT(v_stride % dst == 0,
                 "Required mesh vertex buffer with stride {} can not be aligned to v_stride {}",
                 dst, v_stride);
}

void check_mesh_vert_buffer_motion_keyframe_count(size_t total_vertex_count, uint motion_keyframe_count) {
    LUISA_ASSERT(motion_keyframe_count <= 1u || total_vertex_count % motion_keyframe_count == 0,
                 "Required vertex count {} can not be aligned to motion keyframe count {}",
                 total_vertex_count, motion_keyframe_count);
}

void check_mesh_triangle_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes) {
    LUISA_ASSERT(offset_bytes % sizeof(uint) == 0,
                 "Required triangle buffer offset {} can not be aligned to triangle index size {}",
                 offset_bytes, sizeof(uint));
    LUISA_ASSERT(size_bytes % sizeof(Triangle) == 0,
                 "Required triangle buffer size {} can not be aligned to triangle size {}",
                 size_bytes, sizeof(Triangle));
}

void check_mesh_vertex_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes, size_t v_stride) {
    LUISA_ASSERT(v_stride % 16u == 0u,
                 "Vertex stride must be aligned to 16 bytes.");
    LUISA_ASSERT(offset_bytes % v_stride == 0,
                 "Required vertex buffer offset {} can not be aligned to vertex stride {}",
                 offset_bytes, v_stride);
    LUISA_ASSERT(size_bytes % v_stride == 0,
                 "Required vertex buffer size {} can not be aligned to vertex stride {}",
                 size_bytes, v_stride);
}

}// namespace detail

}// namespace luisa::compute
