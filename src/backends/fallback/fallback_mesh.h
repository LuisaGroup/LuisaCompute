#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>

#include <luisa/runtime/rtx/mesh.h>
#include "fallback_prim.h"

namespace luisa::compute::fallback {

[[nodiscard]] bool luisa_fallback_direct_static_mesh_enabled() noexcept;

class FallbackMesh final : public FallbackPrim {

private:
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
    FallbackMesh(RTCDevice device, const AccelOption &option) noexcept;
    void build(luisa::unique_ptr<MeshBuildCommand> cmd) noexcept;
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] uint64_t build_revision() const noexcept override {
        return _build_revision.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool has_shared_buffers() const noexcept {
        return _v_buffer != nullptr && _t_buffer != nullptr;
    }
    void bind_shared_buffers(RTCGeometry dst_geom) const noexcept;
    [[nodiscard]] RTCScene committed_handle() const noexcept override;
};

}// namespace luisa::compute::fallback
