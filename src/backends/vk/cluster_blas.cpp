#include "cluster_blas.h"
#include "device.h"
#include "stream.h"
#include "tlas.h"
#include "command_buffer_sync.h"
#include <cstddef>
#include <luisa/runtime/rtx/mesh.h>
namespace lc::vk {
namespace cluster_blas_detail {
// ClusterBuildInfo is handed to the driver verbatim, so its layout has to be
// byte-identical to VkClusterAccelerationStructureBuildTriangleClusterInfoNV.
static_assert(
    sizeof(luisa::compute::ClusterBuildInfo) ==
        sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
    "ClusterBuildInfo must match VkClusterAccelerationStructureBuildTriangleClusterInfoNV");
static_assert(
    alignof(luisa::compute::ClusterBuildInfo) ==
        alignof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
    "ClusterBuildInfo must match VkClusterAccelerationStructureBuildTriangleClusterInfoNV");
static_assert(
    sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) == 16u,
    "the Cluster BLAS build record is 16 bytes");
static_assert(
    sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV) == 64u,
    "the CLAS build record is 64 bytes");
// sizeof/alignof above only pin the total size; the driver reads the record by
// offset, so pin the field offsets the Vulkan struct dictates as well. A field
// reorder inside ClusterBuildInfo keeps sizeof == 64 and would otherwise be
// silent.
static_assert(
    offsetof(luisa::compute::ClusterBuildInfo, packed0) == 8u,
    "ClusterBuildInfo::packed0 must sit at offset 8 (triangleCount/vertexCount/"
    "positionTruncateBitCount/indexType/opacityMicromapIndexType)");
static_assert(
    offsetof(luisa::compute::ClusterBuildInfo, index_buffer) == 24u,
    "ClusterBuildInfo::index_buffer must sit at offset 24");
static_assert(
    offsetof(luisa::compute::ClusterBuildInfo, vertex_buffer) == 32u,
    "ClusterBuildInfo::vertex_buffer must sit at offset 32");
static_assert(
    offsetof(luisa::compute::ClusterBuildInfo, opacity_micromap_index_buffer) == 56u,
    "ClusterBuildInfo::opacity_micromap_index_buffer must sit at offset 56");

// The byte alignments in VkPhysicalDeviceClusterAccelerationStructurePropertiesNV
// (clusterByteAlignment, clusterBottomLevelByteAlignment,
// clusterScratchByteAlignment) are implementation-defined and carry no
// power-of-two guarantee — NVIDIA reports 128/256/256 today, but nothing in the
// spec forbids e.g. 384. Every alignment in this file therefore uses the modulus
// fix-up below instead of a mask: rounding 384 up to 512 does *not* produce a
// multiple of 384, and the assertions in pre_build_clusters would compare against
// the rounded value and pass while VUID-...-input-12317 / VUID-...-opMode-12316
// were violated. Only vstd::StackAllocator, whose allocate(size, align) uses mask
// arithmetic, is given a power of two (chunk_alignment_for).
[[nodiscard]] size_t driver_alignment(uint32_t reported) noexcept {
    return reported == 0u ? 1u : static_cast<size_t>(reported);
}
// Byte offset that has to be added to `address` to reach the next
// `alignment`-aligned address. Correct for any `alignment`, not only powers of
// two.
[[nodiscard]] size_t alignment_fixup(uint64_t address, size_t alignment) noexcept {
    auto misalign = static_cast<size_t>(address % alignment);
    return misalign == 0u ? 0u : alignment - misalign;
}
// `size` rounded up to a multiple of `alignment`, for any `alignment`.
[[nodiscard]] size_t round_up_to(size_t size, size_t alignment) noexcept {
    auto padded = size + alignment - 1u;
    return padded - padded % alignment;
}
// Smallest power of two >= `requested`. Only the stack-allocator chunk
// alignment needs this property; never used for a spec-mandated alignment.
[[nodiscard]] size_t chunk_alignment_for(size_t requested) noexcept {
    size_t power_of_two = 1u;
    while (power_of_two < requested) { power_of_two <<= 1u; }
    return power_of_two;
}
// `count * per_cluster` for the maxTotal* fields of the triangle cluster input.
// These are upper bounds the implementation only uses to size the build, and
// over-estimating is always legal (research §C.2/§E.1), so saturating is safe.
[[nodiscard]] uint32_t saturated_product(uint32_t count, uint32_t per_cluster) noexcept {
    auto product = static_cast<uint64_t>(count) * static_cast<uint64_t>(per_cluster);
    return product > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(product);
}
// `value + 1` without wrapping around; used for the "count" form of a maximum
// index (maxClusterGeometryIndex -> number of distinct geometry values).
[[nodiscard]] uint32_t saturating_increment(uint32_t value) noexcept {
    return value == 0xffffffffu ? value : value + 1u;
}
// The scalar fields every VkClusterAccelerationStructureInputInfoNV of this file
// needs. The union member (opInput) is set by the caller because it differs per
// operation; initialising the union through a common helper would write the
// wrong member.
[[nodiscard]] VkClusterAccelerationStructureInputInfoNV make_input_info(
    uint32_t max_acceleration_structure_count,
    VkBuildAccelerationStructureFlagsKHR flags,
    VkClusterAccelerationStructureOpTypeNV op_type,
    VkClusterAccelerationStructureOpModeNV op_mode) noexcept {
    VkClusterAccelerationStructureInputInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    info.pNext = nullptr;
    info.maxAccelerationStructureCount = max_acceleration_structure_count;
    info.flags = flags;
    info.opType = op_type;
    info.opMode = op_mode;
    return info;
}
// Device-address dependency between two cluster build commands.
//
// Vulkan's synchronization tracks VkBuffer/VkImage *objects*; every cluster
// destination and every cluster build input of this file is a raw
// VkDeviceAddress, so resource state cannot express these hazards and they have
// to be recorded explicitly as a global memory dependency
// (VULKAN_CLUSTER_API_RECIPE §I.2 and §I.3 — "barrier the pools, not the
// objects"; a vkCmdPipelineBarrier2-style VkMemoryBarrier2 keeps the 1.2
// dispatch through detail::cmd_pipeline_barrier).
void cmd_device_address_barrier(
    CommandBuffer &cmdbuffer,
    VkPipelineStageFlags2 src_stage,
    VkAccessFlags2 src_access,
    VkPipelineStageFlags2 dst_stage,
    VkAccessFlags2 dst_access) noexcept {
    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access};
    VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0u,
        .memoryBarrierCount = 1u,
        .pMemoryBarriers = &barrier,
        .bufferMemoryBarrierCount = 0u,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 0u,
        .pImageMemoryBarriers = nullptr};
    detail::cmd_pipeline_barrier(
        cmdbuffer.cmdbuffer(), cmdbuffer.device(), &dependency);
}
}// namespace cluster_blas_detail

ClusterBlas::ClusterBlas(Device *device, AccelOption const &option)
    : PrimitiveBase(device, PrimitiveBase::PrimTag::CLUSTER_BLAS),
      _option(option) {
    if (!device->enable_cluster_acceleration_structure()) [[unlikely]] {
        LUISA_ERROR(
            "Cluster (mega-geometry) acceleration structures require "
            "VK_NV_cluster_acceleration_structure, which is not enabled on this "
            "device.");
    }
    if (_option.motion.is_enabled()) [[unlikely]] {
        LUISA_WARNING(
            "Cluster (mega-geometry) meshes do not support motion blur; the "
            "motion option is ignored and every build is a full rebuild.");
    }
}

// TODO(dx): the D3D12 counterpart must free its cluster BLAS allocation
// (NVAPI cluster build / DXR clusters) and unlink it from every TLAS the same
// way; see the RTXMG research report §7 for the API divergence.
ClusterBlas::~ClusterBlas() {
    for (auto &&i : _handles) {
        auto accel = i->accel;
        accel->_all_instance[i->accel_index].handle = nullptr;
        // A mesh-refresh entry queued by the last Cluster BLAS rebuild
        // (_sync_tlas) may still reference this primitive; drop it so a later
        // TLAS build never bakes the address of a destroyed Cluster BLAS into an
        // instance descriptor.
        accel->_all_instance[i->accel_index].primitive = nullptr;
        accel->_drop_refresh(i->accel_index, this);
        MeshHandle::destroy_handle(i);
    }
    // There is no VkAccelerationStructureKHR to destroy: the pools are ordinary
    // buffers, so releasing them is the whole teardown.
}

MeshHandle *ClusterBlas::_add_accel_ref(Tlas *accel, uint32_t index) {
    auto meshHandle = MeshHandle::allocate_handle();
    // MeshHandle::mesh is typed `Blas *`; a cluster BLAS is not a Blas, and
    // blas.h must not be modified. Leave it null and let Tlas track the owner.
    meshHandle->mesh = nullptr;
    meshHandle->accel = accel;
    meshHandle->accel_index = index;
    {
        std::lock_guard lck(_handle_mtx);
        meshHandle->mesh_index = _handles.size();
        _handles.emplace_back(meshHandle);
    }
    return meshHandle;
}

void ClusterBlas::_remove_accel_ref(MeshHandle *handle) {
    {
        std::lock_guard lck(_handle_mtx);
        auto last = _handles.back();
        _handles.pop_back();
        if (last != handle) {
            last->mesh_index = handle->mesh_index;
            _handles[handle->mesh_index] = last;
        }
    }
    MeshHandle::destroy_handle(handle);
}

void ClusterBlas::_sync_tlas() {
    std::lock_guard lck(_handle_mtx);
    for (auto &&i : _handles) {
        i->accel->_update_mesh(i);
    }
}

void ClusterBlas::_ensure_block(
    CommandBuffer &cmdbuffer, AlignedBlock &block, size_t byte_size,
    size_t alignment, bool acceleration_structure_storage) {
    LUISA_ASSERT(alignment != 0u && byte_size != 0u,
                 "cluster acceleration structure block needs a non-zero size "
                 "and alignment");
    // Room for both the requested bytes and the alignment fix-up of the base.
    auto required = byte_size + alignment;
    if (!block.empty() && block.buffer->byte_size() < required) {
        // The old block and every device address inside it die with it. The
        // traceable address is *not* cleared here: pre_build_clusters republishes
        // _blas_address (and refreshes the TLASes) right after the pools are
        // (re)allocated, in this same preprocess pass.
        cmdbuffer.states()->dispose_after_flush(std::move(block.buffer));
        block = {};
    }
    if (block.empty()) {
        block.buffer = vstd::make_unique<DefaultBuffer>(
            device(), required, true,
            acceleration_structure_storage ?
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR :
                static_cast<VkBufferUsageFlagBits>(0));
    }
    block.offset = cluster_blas_detail::alignment_fixup(
        block.buffer->get_device_address(), alignment);
}

void ClusterBlas::pre_build_clusters(
    CommandBuffer &cmdbuffer, MeshBuildCommand const *cmd) {
    using namespace cluster_blas_detail;
    auto count = static_cast<uint32_t>(cmd->cluster_count());
    LUISA_ASSERT(count > 0u,
                 "A cluster (mega-geometry) build needs at least one "
                 "ClusterBuildInfo record.");
    // cluster_count() truncates, so a record array that is not a whole number of
    // 64-byte records would silently build fewer clusters than the caller
    // uploaded.
    LUISA_ASSERT(cmd->cluster_buffer_size() % sizeof(luisa::compute::ClusterBuildInfo) == 0u,
                 "Cluster build record array of {} bytes is not a multiple of the "
                 "{} byte ClusterBuildInfo record.",
                 cmd->cluster_buffer_size(), sizeof(luisa::compute::ClusterBuildInfo));
    // Both entry points come from VK_NV_cluster_acceleration_structure, which
    // the device only enables when the physical device supports it; volk leaves
    // them null otherwise.
    LUISA_ASSERT(vkGetClusterAccelerationStructureBuildSizesNV != nullptr &&
                     vkCmdBuildClusterAccelerationStructureIndirectNV != nullptr,
                 "VK_NV_cluster_acceleration_structure entry points are not "
                 "loaded on this device.");
    _cluster_count = count;
    auto *dev = device();
    auto const &prop = dev->cluster_acceleration_structure_properties();
    LUISA_ASSERT(prop.maxTrianglesPerCluster > 0u && prop.maxVerticesPerCluster > 0u,
                 "VK_NV_cluster_acceleration_structure reports no per-cluster "
                 "limits (maxTrianglesPerCluster={}, maxVerticesPerCluster={}).",
                 prop.maxTrianglesPerCluster, prop.maxVerticesPerCluster);
    // Cluster geometry has no update entry point (vkCmdBuildAccelerationStructuresKHR
    // cannot target a CLAS or a Cluster BLAS), so AccelBuildRequest::PREFER_UPDATE
    // is ignored and every build is a full rebuild (LC_CLUSTER_IMPL_SPEC §3.6).
    // TODO(dx): DXR clusters have no update path either; the D3D12 counterpart
    // must rebuild unconditionally (RTXMG research report §5/§7).
    auto flags = _option.hint == AccelOption::UsageHint::FAST_BUILD ?
                     VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR :
                     VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    // ---- step 1: describe the CLAS build (host side) -----------------------
    // Kept in a member (not in cmdbuffer.temp_desc) so build_clusters can
    // re-derive the InputInfoNV without depending on per-command-buffer host
    // memory; see the comment in cluster_blas.h.
    auto &triangle_input = _triangle_cluster_input;
    triangle_input = {};
    triangle_input.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
    triangle_input.pNext = nullptr;
    // ClusterBuildInfo carries positions only (float3), tightly packed.
    triangle_input.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangle_input.maxGeometryIndexValue = prop.maxClusterGeometryIndex;
    // Note the two micromap models are unrelated: VK_NV_cluster_acceleration
    // _structure takes its own opacity micromap *index buffer* inside the cluster
    // record (no VkMicromapEXT, no KHR micromap object), while
    // VK_KHR_opacity_micromap attaches a micromap object to a classic triangle
    // BLAS through VkAccelerationStructureTrianglesOpacityMicromapEXT - see
    // opacity_micromap.h and rt_extension_plan.h. A cluster build can therefore
    // never be the attach target of the KHR path.
    // Per-triangle geometry-index buffers and opacity micromaps are out of scope
    // (LC_CLUSTER_IMPL_SPEC §3.6): ClusterBuildInfo exposes
    // geometry_index_and_flags_buffer / opacity_micromap_array, but the records
    // live in device memory, so this backend cannot inspect them on the host and
    // cannot reject a caller that fills them in. Declare the honest upper bound
    // instead of one value per cluster so such a caller is not silently
    // under-declared: maxGeometryIndexValue already states the full legal index
    // range (VUID-...-baseGeometryIndex-10496), and over-estimating these bounds
    // is legal (recipe §C.2/§E.1), so this adds no worst case the driver was not
    // already told about. The assertion below pins the invariant that the
    // declaration covers that whole range.
    triangle_input.maxClusterUniqueGeometryCount =
        std::max(1u, saturating_increment(prop.maxClusterGeometryIndex));
    LUISA_ASSERT(triangle_input.maxClusterUniqueGeometryCount >= 1u &&
                     triangle_input.maxClusterUniqueGeometryCount >=
                         saturating_increment(triangle_input.maxGeometryIndexValue),
                 "maxClusterUniqueGeometryCount ({}) must cover every geometry "
                 "index the declared maxGeometryIndexValue ({}) allows; a "
                 "per-triangle geometry-index buffer cannot be checked on the "
                 "host because the ClusterBuildInfo records are device memory.",
                 triangle_input.maxClusterUniqueGeometryCount,
                 triangle_input.maxGeometryIndexValue);
    triangle_input.maxClusterTriangleCount = prop.maxTrianglesPerCluster;
    triangle_input.maxClusterVertexCount = prop.maxVerticesPerCluster;
    triangle_input.maxTotalTriangleCount =
        saturated_product(count, prop.maxTrianglesPerCluster);
    triangle_input.maxTotalVertexCount =
        saturated_product(count, prop.maxVerticesPerCluster);
    triangle_input.minPositionTruncateBitCount = 0u;

    // ---- step 2: describe the Cluster BLAS build (host side) ---------------
    auto &bottom_level_input = _clusters_bottom_level_input;
    bottom_level_input = {};
    bottom_level_input.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
    bottom_level_input.pNext = nullptr;
    // One input argument covering every cluster: the total and the per-argument
    // maximum are both `count` (recipe §C.2).
    bottom_level_input.maxTotalClusterCount = count;
    bottom_level_input.maxClusterCountPerAccelerationStructure = count;
    _build_flags = flags;

    // ---- host-side size queries (recipe §E.1) ------------------------------
    // Not commands: no command buffer, no submission, no synchronisation. They
    // only depend on (cluster count, build flags) — every other input above is a
    // physical-device property — so the two results are cached and re-queried
    // only when that key changes.
    if (_sizes_cluster_count != count || _sizes_build_flags != flags) {
        _sizes_cluster_count = count;
        _sizes_build_flags = flags;
        _clas_build_sizes = VkAccelerationStructureBuildSizesInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        _blas_build_sizes = VkAccelerationStructureBuildSizesInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        // IMPLICIT_DESTINATIONS returns the total pool size for all `count`
        // CLASes plus the scratch for the whole op.
        auto clas_query = make_input_info(
            count, flags,
            VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
            VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV);
        clas_query.opInput.pTriangleClusters = &_triangle_cluster_input;
        vkGetClusterAccelerationStructureBuildSizesNV(
            dev->logic_device(), &clas_query, &_clas_build_sizes);
        // Exactly one Cluster BLAS: one input argument == one acceleration
        // structure. Queried with the mode the build actually uses, so the
        // returned accelerationStructureSize is exactly one Cluster BLAS (also
        // the value the EXPLICIT_DESTINATIONS build reads back out of
        // dstSizesArray — recipe §B.3) and buildScratchSize matches the recorded
        // operation.
        auto blas_query = make_input_info(
            1u, flags,
            VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
            VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV);
        blas_query.opInput.pClustersBottomLevel = &_clusters_bottom_level_input;
        vkGetClusterAccelerationStructureBuildSizesNV(
            dev->logic_device(), &blas_query, &_blas_build_sizes);
    }
    auto const &clas_sizes = _clas_build_sizes;
    auto const &blas_sizes = _blas_build_sizes;

    LUISA_ASSERT(clas_sizes.accelerationStructureSize > 0u &&
                     blas_sizes.accelerationStructureSize > 0u,
                 "vkGetClusterAccelerationStructureBuildSizesNV reported an "
                 "empty cluster acceleration structure pool (CLAS={} bytes, "
                 "Cluster BLAS={} bytes).",
                 clas_sizes.accelerationStructureSize,
                 blas_sizes.accelerationStructureSize);
    // dstSizesArray carries 32-bit sizes (VUID-...-dstSizesArray-10475), so the
    // per-element Cluster BLAS size has to fit in 32 bits.
    LUISA_ASSERT(blas_sizes.accelerationStructureSize <= 0xffffffffull,
                 "Cluster BLAS needs {} bytes, which does not fit the 32-bit "
                 "dstSizesArray element the EXPLICIT_DESTINATIONS build reads.",
                 blas_sizes.accelerationStructureSize);

    // ---- (re)allocate the pools and helper arrays -------------------------
    auto clas_alignment = driver_alignment(prop.clusterByteAlignment);
    auto blas_alignment = driver_alignment(prop.clusterBottomLevelByteAlignment);
    _ensure_block(cmdbuffer, _clas_pool,
                  static_cast<size_t>(clas_sizes.accelerationStructureSize),
                  clas_alignment, true);
    auto *clas_addresses_before = _clas_addresses.buffer.get();
    _ensure_block(cmdbuffer, _clas_addresses,
                  static_cast<size_t>(count) * sizeof(VkDeviceAddress),
                  alignof(VkDeviceAddress), false);
    if (_clas_addresses.buffer.get() != clas_addresses_before) {
        // A freshly allocated address array is uninitialized device memory. In
        // IMPLICIT_DESTINATIONS mode it is an output (the driver writes one
        // VkDeviceAddress per CLAS), so a correct driver overwrites all `count`
        // entries — but VU 12382/VU 10484 make a short or partial write
        // undetectable host-side, and a driver or validation layer that *reads*
        // the array would see random bytes. Zero it once per (re)allocation so
        // that such a case degrades into an observable null address instead.
        // VK_BUFFER_USAGE_TRANSFER_DST_BIT is always part of DefaultBuffer's
        // usage set, and the transfer write is inside the source scope of the
        // first build barrier in build_clusters (TRANSFER / TRANSFER_WRITE).
        // vkCmdFillBuffer needs a 4-byte-aligned dstOffset
        // (VUID-vkCmdFillBuffer-dstOffset-00025); the block offset is an
        // alignment_fixup() of the buffer's device address, so it inherits the
        // alignment of that address, which is an allocator property rather than a
        // structural guarantee of this file.
        LUISA_ASSERT(_clas_addresses.offset % 4u == 0u,
                     "vkCmdFillBuffer destination offset {} must be 4-byte "
                     "aligned (the address array's device address is not).",
                     _clas_addresses.offset);
        vkCmdFillBuffer(
            cmdbuffer.cmdbuffer(), _clas_addresses.buffer->vk_buffer(),
            _clas_addresses.offset, VK_WHOLE_SIZE, 0u);
    }
    _ensure_block(cmdbuffer, _clas_sizes,
                  static_cast<size_t>(count) * sizeof(uint32_t),
                  alignof(uint32_t), false);
    _ensure_block(cmdbuffer, _blas_pool,
                  static_cast<size_t>(blas_sizes.accelerationStructureSize),
                  blas_alignment, true);
    _ensure_block(cmdbuffer, _blas_addresses,
                  sizeof(VkDeviceAddress), alignof(VkDeviceAddress), false);
    _ensure_block(cmdbuffer, _blas_sizes,
                  sizeof(uint32_t), alignof(uint32_t), false);
    _ensure_block(cmdbuffer, _blas_info,
                  sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                  alignof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                  false);
    _blas_byte_size = blas_sizes.accelerationStructureSize;

    // Verify the alignment the two VUs demand (VUID-...-input-12317 for the
    // CLAS pool, the generic VUID-...-opMode-12316 for the Cluster BLAS
    // destination) instead of trusting the allocator. The modulus is the
    // *driver-reported* value, not a normalized one, so a non-power-of-two
    // alignment is checked correctly.
    LUISA_ASSERT(_clas_pool.device_address() % clas_alignment == 0u,
                 "CLAS pool base 0x{:x} is not aligned to clusterByteAlignment {}.",
                 _clas_pool.device_address(), clas_alignment);
    LUISA_ASSERT(_blas_pool.device_address() % blas_alignment == 0u,
                 "Cluster BLAS pool base 0x{:x} is not aligned to "
                 "clusterBottomLevelByteAlignment {}.",
                 _blas_pool.device_address(), blas_alignment);

    // ---- scratch ----------------------------------------------------------
    // One scratch region serves both operations: they are serialized by the
    // CLAS->BLAS barrier recorded in build_clusters, so max() is enough
    // (LC_CLUSTER_IMPL_SPEC §3.5.6). clusterScratchByteAlignment has no
    // power-of-two guarantee, so the address fix-up uses the reported value and
    // keeps one extra unit of room for it; the chunk handed to the stack
    // allocator is aligned to the next power of two, which is all that
    // allocator's mask arithmetic requires.
    auto scratch_alignment = driver_alignment(prop.clusterScratchByteAlignment);
    auto scratch_chunk_alignment = chunk_alignment_for(std::max<size_t>(scratch_alignment, 256u));
    auto scratch_size = round_up_to(
        std::max<size_t>(clas_sizes.buildScratchSize, blas_sizes.buildScratchSize),
        scratch_alignment);
    scratch_size += scratch_alignment;// extra padding for address misalignment
    auto scratch_chunk = cmdbuffer.scratch_buffer_alloc->allocate(scratch_size, scratch_chunk_alignment);
    _scratch_buffer = reinterpret_cast<Buffer const *>(scratch_chunk.handle);
    _scratch_buffer_offset = scratch_chunk.offset +
                             alignment_fixup(
                                 _scratch_buffer->get_device_address() + scratch_chunk.offset,
                                 scratch_alignment);

    // ---- resource-state bookkeeping --------------------------------------
    // These buffers are ordinary resources as far as the tracker is concerned,
    // so their before/after states are recorded here; the raw-address hazards
    // between the two builds are covered by the explicit memory barriers in
    // build_clusters.
    cmdbuffer.resource_barrier->record(_clas_pool.buffer.get(), ResourceBarrier::Usage::kBuildAccel);
    cmdbuffer.resource_barrier->record(_clas_addresses.buffer.get(), ResourceBarrier::Usage::kBuildAccel);
    cmdbuffer.resource_barrier->record(_clas_sizes.buffer.get(), ResourceBarrier::Usage::kBuildAccel);
    cmdbuffer.resource_barrier->record(_blas_pool.buffer.get(), ResourceBarrier::Usage::kBuildAccel);
    // The three host-known constants below are rewritten with vkCmdUpdateBuffer
    // in build_clusters, i.e. by a transfer operation the tracker cannot see, so
    // their tracked state is declared as a copy destination: the next rebuild's
    // upload then gets a proper write-after-write dependency from the previous
    // build.
    cmdbuffer.resource_barrier->record(_blas_addresses.buffer.get(), ResourceBarrier::Usage::kCopyDest);
    cmdbuffer.resource_barrier->record(_blas_sizes.buffer.get(), ResourceBarrier::Usage::kCopyDest);
    cmdbuffer.resource_barrier->record(_blas_info.buffer.get(), ResourceBarrier::Usage::kCopyDest);
    cmdbuffer.resource_barrier->record(_scratch_buffer, ResourceBarrier::Usage::kComputeUAV);
    // srcInfosArray is read as an *indirect command* by the acceleration
    // structure build stage, which no existing Usage maps to; state the native
    // contract explicitly instead of inventing an enum value.
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->cluster_buffer()),
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        VK_IMAGE_LAYOUT_GENERAL);

    // ---- publish the traceable address (preprocess pass!) ------------------
    // EXPLICIT_DESTINATIONS makes the Cluster BLAS destination host-known, so
    // the traceable address exists here already — no GPU->CPU readback is
    // involved (recipe §K.5). It must be published *now*, during preprocessing,
    // not in build_clusters:
    //
    //  * CommandBuffer::execute preprocesses and then executes the reordered
    //    layers, and Tlas::pre_build bakes every instance's
    //    accelerationStructureReference from get_accel_device_address().
    //    Tlas::pre_build is always placed in a strictly later reorder layer than
    //    any mesh build (command_reorder_visitor.h, `get_last_layer_write` for
    //    ResourceType::Accel: `layer = std::max(layer, _max_mesh_level)`), so it
    //    runs after this function and before the first build_clusters of the
    //    flush - the exact contract Blas::_pre_build follows for a triangle BLAS
    //    (blas.cpp:84 `auto sync = recreate && _accel != VK_NULL_HANDLE;`,
    //    blas.cpp:107-109 `if (sync) { _sync_tlas(); }`).
    //  * Publishing in the execute pass instead would abort on the canonical
    //    `stream << records.copy_from(...) << mesh.build() << accel.build()` via
    //    get_accel_device_address()'s LUISA_ERROR, and a pool-growing rebuild
    //    would leave the already-disposed pool address in the TLAS for a frame.
    auto const pool_address = _blas_pool.device_address();
    LUISA_ASSERT(pool_address != 0u,
                 "Cluster BLAS pool has no device address; the traceable address "
                 "cannot be published during preprocessing.");
    auto const address_changed = _blas_address != pool_address;
    _blas_address = pool_address;
    if (address_changed) {
        // Queue a mesh refresh for every TLAS that instances this Cluster BLAS,
        // so a TLAS built later in this same flush (or in a later one) folds the
        // new address instead of the disposed pool's. Mirrors Blas::_pre_build,
        // which syncs only when the acceleration structure was recreated.
        _sync_tlas();
    }
}

void ClusterBlas::build_clusters(
    CommandBuffer &cmdbuffer, MeshBuildCommand const *cmd) {
    using namespace cluster_blas_detail;
    auto count = _cluster_count;
    LUISA_ASSERT(count > 0u && !_clas_pool.empty() && !_blas_pool.empty() &&
                     _blas_address != 0u,
                 "Cluster BLAS build without a preceding pre_build_clusters.");
    // The per-build state above is cached in this object by pre_build_clusters,
    // so a second build of the same ClusterBlas that is preprocessed before the
    // first one executes (two builds in one flush, two streams, or re-recording
    // after a reset) would make the earlier build record the later build's
    // counts — a VUID-...-10445 violation (the size query and the recorded
    // maxAccelerationStructureCount have to agree). One build per ClusterBlas per
    // flush is therefore a hard requirement; detect the cheapest symptom of a
    // violation instead of silently mis-tracing.
    LUISA_ASSERT(cmd->cluster_count() == count,
                 "Cluster BLAS was pre-built for {} clusters but is being built "
                 "with {}: the same Cluster BLAS cannot be built twice before the "
                 "first build executes.",
                 count, cmd->cluster_count());
    auto *cmd_buffer = cmdbuffer.cmdbuffer();
    auto scratch_address =
        _scratch_buffer->get_device_address() + _scratch_buffer_offset;

    // ---- the three host-known values the Cluster BLAS step consumes --------
    // dstAddressesArray (EXPLICIT mode) and dstSizesArray are read, and
    // srcInfosArray is always read, so all three have to live in device memory.
    // They are small, host-known constants -> a plain vkCmdUpdateBuffer each,
    // ordered before the build by the CLAS->BLAS barrier below.
    // VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV has no
    // sType/pNext: it is the build record, not a pNext-chained structure.
    auto bottom_level_info = VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV{
        .clusterReferencesCount = count,
        .clusterReferencesStride = static_cast<uint32_t>(sizeof(VkDeviceAddress)),
        // The CLAS build wrote one VkDeviceAddress per cluster here, on device:
        // this is the whole "no readback" handoff (recipe §C.1/§K.5).
        .clusterReferences = _clas_addresses.device_address()};
    // vkCmdUpdateBuffer requires a 4-byte-aligned dstOffset
    // (VUID-vkCmdUpdateBuffer-dstOffset-00036) and a 4-byte-multiple dataSize
    // (VUID-vkCmdUpdateBuffer-dataSize-00037). Every block base is carved out of
    // its allocation at an alignment_fixup() of the buffer's device address, and
    // these three blocks are aligned to 8/8/4, so the offsets are multiples of 4
    // — assert it rather than assuming the offset is zero.
    LUISA_ASSERT(_blas_info.offset % 4u == 0u &&
                     _blas_addresses.offset % 4u == 0u &&
                     _blas_sizes.offset % 4u == 0u,
                 "vkCmdUpdateBuffer destination offsets must be 4-byte aligned, "
                 "got ({}, {}, {}).",
                 _blas_info.offset, _blas_addresses.offset, _blas_sizes.offset);
    vkCmdUpdateBuffer(
        cmd_buffer, _blas_info.buffer->vk_buffer(), _blas_info.offset,
        sizeof(bottom_level_info), &bottom_level_info);
    // Published by pre_build_clusters during preprocessing; identical to the
    // pool base because EXPLICIT_DESTINATIONS puts the Cluster BLAS at offset 0
    // of the block we handed over.
    auto blas_address = _blas_address;
    LUISA_ASSERT(blas_address == _blas_pool.device_address(),
                 "Cluster BLAS address 0x{:x} diverged from its pool base 0x{:x}.",
                 blas_address, _blas_pool.device_address());
    vkCmdUpdateBuffer(
        cmd_buffer, _blas_addresses.buffer->vk_buffer(), _blas_addresses.offset,
        sizeof(blas_address), &blas_address);
    auto blas_byte_size = static_cast<uint32_t>(_blas_byte_size);
    vkCmdUpdateBuffer(
        cmd_buffer, _blas_sizes.buffer->vk_buffer(), _blas_sizes.offset,
        sizeof(blas_byte_size), &blas_byte_size);

    // ---- step 1: CLAS bodies ----------------------------------------------
    VkClusterAccelerationStructureCommandsInfoNV clas_cmd{};
    clas_cmd.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    clas_cmd.pNext = nullptr;
    // Re-derived from the cached scalars instead of copying a pointer that
    // belongs to some command buffer's temp_desc (see cluster_blas.h).
    clas_cmd.input = make_input_info(
        count, _build_flags,
        VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV);
    clas_cmd.input.opInput.pTriangleClusters = &_triangle_cluster_input;
    clas_cmd.dstImplicitData = _clas_pool.device_address();
    clas_cmd.scratchData = scratch_address;
    // One VkDeviceAddress per CLAS, stride 8 (VUID-...-dstAddressesArray-10474)
    clas_cmd.dstAddressesArray = VkStridedDeviceAddressRegionKHR{
        _clas_addresses.device_address(),
        sizeof(VkDeviceAddress),
        static_cast<VkDeviceSize>(count) * sizeof(VkDeviceAddress)};
    // The implementation writes precise per-CLAS sizes here in IMPLICIT mode.
    clas_cmd.dstSizesArray = VkStridedDeviceAddressRegionKHR{
        _clas_sizes.device_address(),
        sizeof(uint32_t),
        static_cast<VkDeviceSize>(count) * sizeof(uint32_t)};
    clas_cmd.srcInfosArray = VkStridedDeviceAddressRegionKHR{
        reinterpret_cast<Buffer const *>(cmd->cluster_buffer())->get_device_address() +
            cmd->cluster_buffer_offset(),
        sizeof(luisa::compute::ClusterBuildInfo),
        static_cast<VkDeviceSize>(cmd->cluster_buffer_size())};
    // 0 => the operation count is input.maxAccelerationStructureCount (== count)
    clas_cmd.srcInfosCount = 0u;
    clas_cmd.addressResolutionFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_NONE_NV;

    // Uploads/compute writes of the cluster records -> the build, plus the WAR
    // edge from the previous frame's acceleration-structure work. The CLAS and
    // Cluster BLAS pools are rewritten every frame and the last frame *read* them
    // at ACCELERATION_STRUCTURE_BUILD (a TLAS build) and RAY_TRACING_SHADER, so
    // those stages and accesses are part of the source scope as well: the tracker
    // cannot see raw device addresses (recipe §I.3 - "barrier the pools, not the
    // objects"), and two submissions on one queue are ordered in execution start,
    // not completion. This makes the barrier a full fence at the top of
    // build_clusters.
    // The vertex/index buffers the records point at are reachable only through
    // raw device addresses, so this is a memory barrier, not a buffer barrier
    // (recipe §I.2 (i); the access mask includes INDIRECT_COMMAND_READ as the
    // spec's synchronisation section requires, which the reference sample omits).
    cmd_device_address_barrier(
        cmdbuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_2_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd_buffer, &clas_cmd);

    // ---- step 1 -> step 2 --------------------------------------------------
    // Two things cross this barrier: the CLAS bodies (read by the Cluster BLAS
    // build) and the CLAS-address array (written by the CLAS build, read as
    // `clusterReferences`). The transfer edge covers the vkCmdUpdateBuffer
    // writes of the three host-known values above. Recipe §I.2 (ii).
    cmd_device_address_barrier(
        cmdbuffer,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

    // ---- step 2: the traceable Cluster BLAS -------------------------------
    VkClusterAccelerationStructureCommandsInfoNV blas_cmd{};
    blas_cmd.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    blas_cmd.pNext = nullptr;
    blas_cmd.input = make_input_info(
        1u, _build_flags,
        VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
        VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV);
    blas_cmd.input.opInput.pClustersBottomLevel = &_clusters_bottom_level_input;
    // Ignored in EXPLICIT_DESTINATIONS mode.
    blas_cmd.dstImplicitData = 0u;
    blas_cmd.scratchData = scratch_address;
    blas_cmd.dstAddressesArray = VkStridedDeviceAddressRegionKHR{
        _blas_addresses.device_address(),
        sizeof(VkDeviceAddress),
        sizeof(VkDeviceAddress)};
    blas_cmd.dstSizesArray = VkStridedDeviceAddressRegionKHR{
        _blas_sizes.device_address(),
        sizeof(uint32_t),
        sizeof(uint32_t)};
    blas_cmd.srcInfosArray = VkStridedDeviceAddressRegionKHR{
        _blas_info.device_address(),
        sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
        sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV)};
    blas_cmd.srcInfosCount = 0u;// => 1 operation
    blas_cmd.addressResolutionFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_NONE_NV;

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd_buffer, &blas_cmd);

    // ---- Cluster BLAS build -> TLAS build / ray tracing -------------------
    // The TLAS build dereferences this BLAS through a raw device address in its
    // instance buffer, and ray-tracing shaders read the same bytes; both
    // dependents must observe the freshly written pool. Recipe §I.2 (iii)/(iv).
    cmd_device_address_barrier(
        cmdbuffer,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_SHADER_READ_BIT);

    // _blas_address is *not* written here: it was published by
    // pre_build_clusters in the preprocess pass, because Tlas::pre_build (which
    // bakes it into the instance buffer) runs before this execute pass and must
    // already see it. See the comment at the end of pre_build_clusters.
}

uint64_t ClusterBlas::get_accel_device_address() const {
    if (_blas_address == 0u) [[unlikely]] {
        LUISA_ERROR(
            "Cluster BLAS not initialized: dispatch a build before using it as "
            "a TLAS instance.");
    }
    return _blas_address;
}
}// namespace lc::vk
