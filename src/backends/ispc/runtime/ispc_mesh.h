#pragma once

#include <rtx/mesh.h>

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
public:
    ISPCMesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept;

    [[nodiscard]] auto getVBufferHandle() const noexcept;
    [[nodiscard]] auto getTBufferHandle() const noexcept;

    friend class ISPCAccel;
};

}