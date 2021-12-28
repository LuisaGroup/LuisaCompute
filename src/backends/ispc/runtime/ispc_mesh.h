#pragma once

#include <rtx/mesh.h>
#include <embree3/rtcore.h>

using namespace luisa::compute;

namespace lc::ispc {

class ISPCMesh {
private:
    uint64_t _v_buffer;
    size_t _v_offset;
    size_t _v_stride;
    size_t _v_count;
    uint64_t _t_buffer;
    size_t _t_offset;
    size_t _t_count;
    AccelBuildHint _hint;
    RTCGeometry geometry;
public:
    ISPCMesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint, RTCDevice device) noexcept;
    ~ISPCMesh() noexcept;

    [[nodiscard]] auto getVBufferHandle() const noexcept { return _v_buffer; }
    [[nodiscard]] auto getTBufferHandle() const noexcept { return _t_buffer; }
    [[nodiscard]] auto getRTCGeometry() const noexcept { return geometry; }
    void build() noexcept;
    void update() noexcept;

    friend class ISPCAccel;
};

}