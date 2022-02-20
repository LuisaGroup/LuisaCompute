//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <embree3/rtcore.h>

namespace luisa::compute::ispc {

class ISPCMesh {

private:
    RTCScene _handle;
    RTCGeometry _geometry;
    uint64_t _v_buffer;
    uint64_t _t_buffer;

public:
    ISPCMesh(
        RTCDevice device, AccelBuildHint hint,
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count) noexcept;
    ~ISPCMesh() noexcept;
    [[nodiscard]] auto vertex_buffer() const noexcept { return _v_buffer; }
    [[nodiscard]] auto triangle_buffer() const noexcept { return _t_buffer; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void commit() noexcept;
};

}// namespace luisa::compute::ispc
