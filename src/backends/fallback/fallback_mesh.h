#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>

#include <luisa/runtime/rtx/mesh.h>
#include "../common/embree_accel_archive.h"
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
    std::atomic<bool> _is_imported_blas{false};
    mutable std::mutex _blas_mutex;
    mutable bool _blas_dirty{false};
    bool _built{false};
    EmbreeImportedBVHHandle _imported_bvh{};

public:
    FallbackMesh(RTCDevice device, const AccelOption &option) noexcept;
    ~FallbackMesh() noexcept override;
    void build(luisa::unique_ptr<MeshBuildCommand> cmd) noexcept;
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool is_built() const noexcept {
        std::scoped_lock lock{_blas_mutex};
        return _built;
    }
    [[nodiscard]] uint64_t last_triangle_count() const noexcept {
        std::scoped_lock lock{_blas_mutex};
        return static_cast<uint64_t>(_t_count);
    }
    [[nodiscard]] uint64_t last_vertex_count() const noexcept {
        std::scoped_lock lock{_blas_mutex};
        return static_cast<uint64_t>(_v_count);
    }
    [[nodiscard]] uint64_t last_vertex_stride() const noexcept {
        std::scoped_lock lock{_blas_mutex};
        return static_cast<uint64_t>(_v_stride);
    }
    [[nodiscard]] bool is_imported_blas() const noexcept {
        return _is_imported_blas.load(std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t build_revision() const noexcept override {
        return _build_revision.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool has_shared_buffers() const noexcept {
        return _v_buffer != nullptr && _t_buffer != nullptr;
    }
    void bind_shared_buffers(RTCGeometry dst_geom) const noexcept;
    [[nodiscard]] bool import_from_archive(
        const EmbreeResolvedTriangleGeometry &geom,
        luisa::span<const std::byte> bvh_payload,
        const EmbreeOutOfTreeABI &abi,
        size_t &out_unpacked_bytes,
        EmbreeArchiveStatus &out_status) noexcept;
    [[nodiscard]] RTCScene committed_handle() const noexcept override;
};

}// namespace luisa::compute::fallback
