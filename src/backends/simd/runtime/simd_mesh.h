#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>

#include <luisa/runtime/rtx/mesh.h>

#include "simd_primitive.h"

namespace luisa::compute::simd {

[[nodiscard]] bool simd_direct_static_mesh_enabled() noexcept;

class SIMDMesh final : public SIMDPrimitive {

private:
    RTCScene _scene{nullptr};
    RTCGeometry _geometry{nullptr};
    AccelOption _option{};
    const std::byte *_v_buffer{nullptr};
    size_t _v_offset{0u};
    size_t _v_stride{0u};
    size_t _v_count{0u};
    const std::byte *_t_buffer{nullptr};
    size_t _t_offset{0u};
    size_t _t_count{0u};
    std::atomic<uint64_t> _build_revision{0u};
    mutable std::mutex _blas_mutex;
    mutable bool _blas_dirty{false};

public:
    SIMDMesh(RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDMesh() noexcept;

    void build(const MeshBuildCommand &command) noexcept;
    [[nodiscard]] const AccelOption &option() const noexcept { return _option; }
    [[nodiscard]] uint64_t build_revision() const noexcept {
        return _build_revision.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool has_shared_buffers() const noexcept {
        return _v_buffer != nullptr && _t_buffer != nullptr;
    }
    void bind_shared_buffers(RTCGeometry dst_geometry) const noexcept;
    [[nodiscard]] RTCScene handle() const noexcept override;
};

}// namespace luisa::compute::simd
