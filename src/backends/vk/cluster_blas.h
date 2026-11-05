#pragma once
#include <volk.h>
#include "primitive_base.h"
#include "default_buffer.h"
#include "blas.h"
namespace lc::vk {
class CommandBuffer;
using namespace luisa;
using namespace luisa::compute;
// A ray-traceable cluster (mega-geometry) bottom-level acceleration structure,
// built through VK_NV_cluster_acceleration_structure.
//
// The extension creates no VkAccelerationStructureKHR: a CLAS and a Cluster
// BLAS are opaque byte regions reachable only through a VkDeviceAddress, and
// the only way to produce them is vkCmdBuildClusterAccelerationStructureIndirectNV.
// ClusterBlas is therefore a sibling of Blas, never a subclass: it owns the byte
// pools and caches the traceable address instead of an acceleration-structure
// handle (LC_CLUSTER_IMPL_SPEC §1/§3.5).
//
// Build shape (both host-sized, no GPU->CPU readback):
//   1. BUILD_TRIANGLE_CLUSTER_NV + IMPLICIT_DESTINATIONS
//        -> CLAS bodies into _clas_pool, one VkDeviceAddress each into
//           _clas_addresses (written on the device).
//   2. BUILD_CLUSTERS_BOTTOM_LEVEL_NV + EXPLICIT_DESTINATIONS
//        -> clusterReferences is _clas_addresses' device address (stride 8) and
//           the single destination is a host-known pool address, so the
//           resulting traceable BLAS address is known on the host at record
//           time and cached in _blas_address.
//
// Pass contract for _blas_address (do not move it back into the execute pass):
// the traceable address is published by pre_build_clusters, i.e. during
// CommandBuffer::execute's *preprocess* pass, exactly like Blas::_pre_build
// publishes its VkAccelerationStructureKHR (blas.cpp `sync = recreate &&
// _accel != VK_NULL_HANDLE` / `if (sync) { _sync_tlas(); }`). Tlas::pre_build
// bakes the instance's accelerationStructureReference from
// get_accel_device_address() and always runs in a strictly later reorder layer
// than any mesh build, so it observes the published value; build_clusters (the
// execute pass) only reads it.
class ClusterBlas : public PrimitiveBase {
    friend class Tlas;

private:
    // One device buffer plus the byte offset of its alignment-corrected base.
    // VkPhysicalDeviceClusterAccelerationStructurePropertiesNV reports the
    // alignments every cluster destination and build-input array must satisfy;
    // the buffer allocator only guarantees its own alignment, so each block is
    // over-allocated by one alignment unit and its base rounded up.
    //
    // Implementation detail of the pool management: not part of the public
    // surface of ClusterBlas.
    struct AlignedBlock {
        vstd::unique_ptr<DefaultBuffer> buffer;
        uint64_t offset{0};
        [[nodiscard]] bool empty() const noexcept { return buffer == nullptr; }
        [[nodiscard]] uint64_t device_address() const noexcept {
            return buffer == nullptr ? 0u : buffer->get_device_address() + offset;
        }
    };

    luisa::spin_mutex _handle_mtx;
    AccelOption _option;
    // step 1: CLAS bodies (dstImplicitData) and the per-CLAS address/size arrays
    AlignedBlock _clas_pool;
    AlignedBlock _clas_addresses;
    AlignedBlock _clas_sizes;
    // step 2: the traceable Cluster BLAS body (explicit destination), its single
    // address/size slot and the single BuildClustersBottomLevelInfoNV record
    AlignedBlock _blas_pool;
    AlignedBlock _blas_addresses;
    AlignedBlock _blas_sizes;
    AlignedBlock _blas_info;
    // Traceable device address of the Cluster BLAS, and the byte size the
    // EXPLICIT_DESTINATIONS build reads back out of dstSizesArray.
    // Written only by pre_build_clusters (preprocess pass), read by
    // get_accel_device_address() (Tlas::pre_build) and read by build_clusters.
    uint64_t _blas_address{0};
    uint64_t _blas_byte_size{0};
    uint32_t _cluster_count{0};
    // Per-build inputs, cached by value. pre_build_clusters derives them from
    // the physical-device properties and the build's cluster count;
    // build_clusters re-derives the two VkClusterAccelerationStructureInputInfoNV
    // records from these fields instead of caching a pointer into
    // cmdbuffer.temp_desc, which is per command buffer (building the same
    // ClusterBlas twice before the first build executes, or from two streams,
    // would otherwise make the earlier build observe the later pre_build's
    // InputInfoNV, including its maxAccelerationStructureCount — a
    // VUID-...-10445 violation).
    VkClusterAccelerationStructureTriangleClusterInputNV _triangle_cluster_input{};
    VkClusterAccelerationStructureClustersBottomLevelInputNV _clusters_bottom_level_input{};
    VkBuildAccelerationStructureFlagsKHR _build_flags{0};
    // Result cache for the two vkGetClusterAccelerationStructureBuildSizesNV
    // queries: apart from the cluster count and the build flags every input of
    // those calls is a physical-device property, so the results only change when
    // this key changes.
    uint32_t _sizes_cluster_count{~0u};
    VkBuildAccelerationStructureFlagsKHR _sizes_build_flags{0};
    VkAccelerationStructureBuildSizesInfoKHR _clas_build_sizes{};
    VkAccelerationStructureBuildSizesInfoKHR _blas_build_sizes{};
    const Buffer *_scratch_buffer{nullptr};
    uint64_t _scratch_buffer_offset{0};
    vstd::fixed_vector<MeshHandle *, 2> _handles;

    // TLAS back-reference bookkeeping, mirroring Blas's _add_accel_ref /
    // _remove_accel_ref / _sync_tlas. MeshHandle::mesh is typed `Blas *` and
    // blas.h must not be modified, so a cluster handle leaves that field null
    // and Tlas tracks the owning primitive in its Instance slot instead.
    MeshHandle *_add_accel_ref(Tlas *accel, uint32_t index);
    void _remove_accel_ref(MeshHandle *handle);
    void _sync_tlas();
    void _ensure_block(
        CommandBuffer &cmdbuffer, AlignedBlock &block, size_t byte_size,
        size_t alignment, bool acceleration_structure_storage);

public:
    ClusterBlas(Device *device, AccelOption const &option);
    ~ClusterBlas() override;
    void pre_build_clusters(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void build_clusters(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    [[nodiscard]] uint64_t get_accel_device_address() const override;
    // Cluster geometry has no motion path in this backend: motion blur is a
    // per-BLAS VK_NV_ray_tracing_motion_blur feature of the KHR build command,
    // which cannot target a cluster acceleration structure.
    [[nodiscard]] bool has_motion() const noexcept override { return false; }
    // Traceable address of the Cluster BLAS produced by the last dispatched
    // build, or 0 before the first build.
    [[nodiscard]] uint64_t cluster_blas_address() const noexcept { return _blas_address; }
    [[nodiscard]] uint32_t cluster_count() const noexcept { return _cluster_count; }
    // Buffer holding the Cluster BLAS bytes; a TLAS build has to read-barrier it.
    [[nodiscard]] Buffer const *blas_pool() const noexcept { return _blas_pool.buffer.get(); }
};
}// namespace lc::vk
