#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>

#include <luisa/runtime/rtx/mesh.h>

#include "../../common/embree_accel_archive.h"
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
    std::atomic<bool> _is_imported_blas{false};
    mutable std::mutex _blas_mutex;
    mutable bool _blas_dirty{false};
    bool _built{false};

public:
    SIMDMesh(RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDMesh() noexcept;

    void build(const MeshBuildCommand &command) noexcept;
    [[nodiscard]] const AccelOption &option() const noexcept { return _option; }
    [[nodiscard]] bool is_built() const noexcept {
        std::scoped_lock lock{_blas_mutex};
        return _built;
    }
    [[nodiscard]] uint64_t last_triangle_count() const noexcept {
        std::scoped_lock lock{_blas_mutex};
        return static_cast<uint64_t>(_t_count);
    }
    [[nodiscard]] bool is_imported_blas() const noexcept {
        return _is_imported_blas.load(std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t build_revision() const noexcept {
        return _build_revision.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool has_shared_buffers() const noexcept {
        return _v_buffer != nullptr && _t_buffer != nullptr;
    }
    void bind_shared_buffers(RTCGeometry dst_geometry) const noexcept;
    [[nodiscard]] bool import_from_archive(
        const EmbreeResolvedTriangleGeometry &geom,
        luisa::span<const std::byte> bvh_payload,
        EmbreeArchiveImportFn import_fn,
        size_t &out_unpacked_bytes,
        EmbreeArchiveStatus &out_status) noexcept;
    [[nodiscard]] RTCScene handle() const noexcept override;
};

}// namespace luisa::compute::simd
