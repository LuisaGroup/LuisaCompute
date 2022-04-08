//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <embree3/rtcore.h>

namespace luisa::compute::ispc {

/**
 * @brief Mesh object of ispc
 * 
 */
class ISPCMesh {

private:
    RTCScene _handle;
    RTCGeometry _geometry;
    uint64_t _v_buffer;
    uint64_t _v_offset;
    uint64_t _v_stride;
    uint64_t _v_count;
    uint64_t _t_buffer;
    uint64_t _t_offset;
    uint64_t _t_count;
    AccelUsageHint _hint;
    std::atomic_bool _buffers_already_set{false};

public:
    /**
     * @brief Construct a new ISPCMesh object
     * 
     * @param device RTCdeivce
     * @param hint build hint
     * @param v_buffer handle of vertex buffer
     * @param v_offset offset of vertex buffer
     * @param v_stride stride of vertex buffer
     * @param v_count count of vertices
     * @param t_buffer handle of triangle buffer
     * @param t_offset offset of triangle buffer
     * @param t_count count of triangles
     */
    ISPCMesh(
        RTCDevice device, AccelUsageHint hint,
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count) noexcept;
    ~ISPCMesh() noexcept;
    /**
     * @brief Return handle of vertex buffer
     * 
     * @return handle of vertex buffer
     */
    [[nodiscard]] auto vertex_buffer() const noexcept { return _v_buffer; }
    /**
     * @brief Return handle of triangle buffer
     * 
     * @return handle of triangle buffer
     */
    [[nodiscard]] auto triangle_buffer() const noexcept { return _t_buffer; }
    /**
     * @brief Return handle of mesh's singleton scene
     * 
     * @return handle of scene
     */
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    /**
     * @brief commit change of mesh
     * 
     */
    void commit() noexcept;
};

}// namespace luisa::compute::ispc
