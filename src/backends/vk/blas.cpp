#include "blas.h"
#include "device.h"
#include "log.h"
#include "tlas.h"
#include "stream.h"
#include <luisa/runtime/rtx/aabb.h>
#include "opacity_micromap.h"
#include "command_buffer_sync.h"
#include "rt_extension_plan.h"
namespace lc::vk {

namespace blas_detail {

// `detail::plan_geometry_flags` keeps its bits Vulkan-free so the unit test can
// prove the decision without a Vulkan header. These two assertions are the other
// half of that contract: if VkGeometryFlagBitsKHR ever moves, the plan's byte
// stops matching the enum the build passes on and this fails instead of silently
// setting the wrong bit.
static_assert(detail::geometry_flag_opaque == VK_GEOMETRY_OPAQUE_BIT_KHR,
              "the plan's opaque bit must stay VK_GEOMETRY_OPAQUE_BIT_KHR");
static_assert(detail::geometry_flag_no_duplicate_any_hit ==
                  VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
              "the plan's no-duplicate-any-hit bit must stay the Vk enum value");
// The attach struct's index stride has to be a multiple of the index size
// (VUID-VkAccelerationStructureTrianglesOpacityMicromapKHR-indexStride-11573);
// this build indexes triangles with uint32, so the natural stride is 4.
static_assert(sizeof(uint32_t) == 4u,
              "the BLAS triangle index buffer is VK_INDEX_TYPE_UINT32");

// Attach request of the opacity micromap of a BLAS.
//
// Only a mesh created with a cut-out mask can give a BLAS one, and such a mesh can
// only exist if the public API asked for it, so `public_api_request` is what the
// plan requires before it answers ATTACH.
//
// Everything the plan can refuse is refused here as well, before a single Vulkan
// structure is filled: a geometry that already chains
// VkAccelerationStructureGeometryMotionTrianglesDataNV, a device that resolved only
// the KHR spelling while no codegen route emits the required `OpacityMicromapIdKHR`
// execution mode, an index format this backend cannot express, and a shader-side
// contract that admits `OpaqueKHR`.
[[nodiscard]] detail::OpacityMicromapAttachRequest micromap_attach_request(
    Device *device, bool micromap_created, bool motion_triangles,
    MeshMicromapBinding const &binding, uint32_t triangle_count) noexcept {
    detail::OpacityMicromapAttachRequest request;
    request.extension_enabled = device->optional_extension_enabled(
        detail::OptionalExtension::OPACITY_MICROMAP);
    request.public_api_request = true;
    request.micromap_available = micromap_created;
    request.triangle_geometry = true;
    request.motion_triangles = motion_triangles;
    request.ext_opacity_micromap_enabled =
        OpacityMicromap::ext_alias_negotiated(device);
    // With only VK_KHR_opacity_micromap negotiated the ray query has to declare the
    // OpacityMicromapIdKHR execution mode; the SPIR-V codegen emits it when the
    // device enabled that target feature.
    request.ray_query_execution_mode_emitted =
        device->opacity_micromap_execution_mode_available();
    // The ray flags belong to the shader that traces the structure, not to the
    // build; the mesh records what its caller promised (D7).
    request.opaque_ray_flags = !binding.assumes_non_opaque_traversal;
    request.instance_disables_micromaps = false;
    // VK_KHR_index_type_uint8 is not tracked by this backend, so a one-byte index
    // stride is refused by the plan (INDEX_TYPE_UNSUPPORTED) instead of being fed to
    // the driver without its extension.
    request.index_type_uint8_supported = false;
    switch (binding.index_stride) {
        case 0u:
            request.index_type = detail::OpacityMicromapIndexType::NONE;
            request.index_stride = 0u;
            break;
        case 2u:
            request.index_type = detail::OpacityMicromapIndexType::UINT16;
            request.index_stride = 2u;
            break;
        case 4u:
            request.index_type = detail::OpacityMicromapIndexType::UINT32;
            request.index_stride = 4u;
            break;
        default:
            request.index_type = detail::OpacityMicromapIndexType::UINT8;
            request.index_stride = binding.index_stride;
            break;
    }
    request.base_triangle = 0u;
    request.usage_count = 1u;
    request.triangle_count = triangle_count;
    return request;
}

}// namespace blas_detail

Blas::Blas(Device *device, AccelOption const &option)
    : PrimitiveBase(device, PrimitiveBase::PrimTag::BLAS), _option(option), _acceleration_build_geometry_info(nullptr) {
    if (!device->enable_raytracing()) [[unlikely]] {
        LUISA_ERROR("Raytracing not enabled, BLAS can not be loaded.");
    }
}
void Blas::_pre_build(
    CommandBuffer &cmdbuffer,
    VkAccelerationStructureGeometryKHR *acceleration_structure_geometry,
    uint32_t primitive_count,
    AccelBuildRequest request) {

    _acceleration_build_geometry_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildGeometryInfoKHR>();
    _acceleration_build_geometry_info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    _acceleration_build_geometry_info->type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    _acceleration_build_geometry_info->flags = _option.hint == AccelOption::UsageHint::FAST_BUILD ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (_option.allow_update) {
        _acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    // Enable BLAS-level vertex motion blur when the mesh has motion keyframes.
    // VK_NV_ray_tracing_motion_blur is NVIDIA-only and unavailable on most
    // Android GPUs; fail cleanly instead of referencing NV structs/flags with
    // the extension disabled.
    if (_option.motion.is_enabled() && !device()->enable_motion_blur()) [[unlikely]] {
        LUISA_ERROR("BLAS vertex motion blur requires VK_NV_ray_tracing_motion_blur, "
                    "which is not enabled on this device.");
    }
    if (_option.motion.is_enabled()) {
        _acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
    }
    _acceleration_build_geometry_info->geometryCount = 1;
    _acceleration_build_geometry_info->pGeometries = acceleration_structure_geometry;
    // Vulkan updates operate in-place on an existing acceleration-structure
    // handle and require the build geometry/primitive counts to match the
    // original build. A preference for update on the first build, or after a
    // primitive-count change, must therefore fall back to a full build.
    bool update = _option.allow_update &&
                  request == AccelBuildRequest::PREFER_UPDATE &&
                  _accel != VK_NULL_HANDLE &&
                  primitive_count == _last_primitive_count;
    _last_primitive_count = primitive_count;

    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
    acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    _acceleration_build_geometry_info->mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        device()->logic_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        _acceleration_build_geometry_info,
        &primitive_count,
        &acceleration_structure_build_sizes_info);
    uint scratch_buffer_size = update ? acceleration_structure_build_sizes_info.updateScratchSize : acceleration_structure_build_sizes_info.buildScratchSize;
    if (_accel_buffer && _accel_buffer->byte_size() < acceleration_structure_build_sizes_info.accelerationStructureSize) {
        cmdbuffer.states()->dispose_after_flush(std::move(_accel_buffer));
    }
    if (!_accel_buffer) {
        update = false;
        _accel_buffer = vstd::make_unique<DefaultBuffer>(
            device(),
            (acceleration_structure_build_sizes_info.accelerationStructureSize + 65535u) & (~65535u),
            false, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    }
    _acceleration_build_geometry_info->mode = update ?
                                                  VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
                                                  VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    cmdbuffer.resource_barrier->record(
        _accel_buffer.get(),
        ResourceBarrier::Usage::kBuildAccel);
    VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
    acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    acceleration_structure_create_info.buffer = _accel_buffer->vk_buffer();
    acceleration_structure_create_info.size = acceleration_structure_build_sizes_info.accelerationStructureSize;
    acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (_option.motion.is_enabled()) {
        acceleration_structure_create_info.createFlags = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
    }
    auto recreate = !update;
    auto sync = recreate && _accel != VK_NULL_HANDLE;
    if (recreate) {
        if (_accel) {
            cmdbuffer.states()->callbacks.emplace_back([a = _accel, device = device()]() {
                vkDestroyAccelerationStructureKHR(device->logic_device(), a, Device::alloc_callbacks());
            });
            _accel = VK_NULL_HANDLE;
        }
        VK_CHECK_RESULT(vkCreateAccelerationStructureKHR(
            device()->logic_device(), &acceleration_structure_create_info,
            Device::alloc_callbacks(), &_accel));
    }
    scratch_buffer_size = (scratch_buffer_size + 255) & (~(255u));
    scratch_buffer_size += 256u; // extra padding for GPU buffer address misalignment
    auto scratch_chunk = cmdbuffer.scratch_buffer_alloc->allocate(scratch_buffer_size, 256u);

    _scratch_buffer = reinterpret_cast<Buffer const *>(scratch_chunk.handle);
    auto addr = _scratch_buffer->get_device_address() + scratch_chunk.offset;
    auto misalign = addr & 255u;
    _scratch_buffer_offset = scratch_chunk.offset + (misalign ? (256u - misalign) : 0u);
    cmdbuffer.resource_barrier->record(
        _scratch_buffer,
        ResourceBarrier::Usage::kComputeUAV);
    if (sync) {
        _sync_tlas();
    }
}
void Blas::pre_build(
    CommandBuffer &cmdbuffer,
    ProceduralPrimitiveBuildCommand const *cmd) {
    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    // Same single source of truth as the triangle path below: an AABB geometry
    // never carries a geometry flag, and no optional extension can change that
    // (an opacity micromap only exists for triangles).
    acceleration_structure_geometry->flags = detail::plan_geometry_flags({
        .triangle_geometry = false,
        .opaque_requested = false,
        .micromap_chained = false})
                                                .flags;
    VkDeviceOrHostAddressConstKHR aabb_data_device_address{};
    aabb_data_device_address.deviceAddress = reinterpret_cast<Buffer const *>(cmd->aabb_buffer())->get_device_address() + cmd->aabb_buffer_offset();

    auto &aabbs = acceleration_structure_geometry->geometry.aabbs;
    aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data = aabb_data_device_address;
    aabbs.stride = sizeof(luisa::compute::AABB);
    _pre_build(cmdbuffer, acceleration_structure_geometry, cmd->aabb_buffer_size() / sizeof(luisa::compute::AABB), cmd->request());
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->aabb_buffer()),
        ResourceBarrier::Usage::kAccelInstanceBuffer);
}
void Blas::pre_build(
    CommandBuffer &cmdbuffer,
    MeshBuildCommand const *cmd) {
    VkDeviceOrHostAddressConstKHR vertex_data_device_address{};
    VkDeviceOrHostAddressConstKHR index_data_device_address{};
    auto vertex_buffer_base = reinterpret_cast<Buffer const *>(cmd->vertex_buffer())->get_device_address() + cmd->vertex_buffer_offset();
    vertex_data_device_address.deviceAddress = vertex_buffer_base;
    index_data_device_address.deviceAddress = reinterpret_cast<Buffer const *>(cmd->triangle_buffer())->get_device_address() + cmd->triangle_buffer_offset();

    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    auto &triangles = acceleration_structure_geometry->geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.pNext = nullptr;

    auto keyframe_count = _option.motion.is_enabled() ? _option.motion.keyframe_count : 1u;
    auto total_vertex_count = cmd->vertex_buffer_size() / cmd->vertex_stride();
    auto vertex_count_per_keyframe = total_vertex_count / keyframe_count;
    // Opacity micromap (VK_KHR_opacity_micromap).
    //
    // `_micromap` is null for every mesh built without a cut-out mask, so this block
    // is skipped and the build stays exactly what it was. A refused attach is an
    // *error*, not a silent fallback: a user who asked for a cut-out must not get an
    // opaque mesh by accident (D6/D7).
    //
    // The packed bits and the index buffer are declared as build inputs, so the
    // command-reorder pass orders the bake (or the upload) of the bits before this
    // build; without that the copy below could read uninitialised memory.
    VkAccelerationStructureTrianglesOpacityMicromapEXT *opacity_micromap = nullptr;
    // Always assigned: a build without a micromap detaches a stale one.
    _micromap = cmd->has_micromap() ?
                    reinterpret_cast<OpacityMicromap *>(cmd->micromap().micromap) :
                    nullptr;
    _micromap_data_buffer = VK_NULL_HANDLE;
    _micromap_data_offset = 0u;
    if (_micromap != nullptr) {
        auto const &binding = cmd->micromap();
        auto attach_request = blas_detail::micromap_attach_request(
            device(), true,
            _option.motion.is_enabled() && keyframe_count == 2u,
            binding,
            static_cast<uint32_t>(cmd->triangle_buffer_size() / 12));
        auto attach_plan = detail::plan_opacity_micromap_attach(attach_request);
        if (!attach_plan.chain) [[unlikely]] {
            LUISA_ERROR(
                "The opacity micromap of this mesh cannot be attached: {} "
                "(index stride {}, {} triangles, opaque-traversal-contract {}).",
                detail::opacity_micromap_attach_status_name(attach_plan.status),
                binding.index_stride, binding.triangle_count,
                binding.assumes_non_opaque_traversal ? "assumed" : "refused");
        }
        auto index_buffer_device_address = 0u;
        if (binding.index_stride != 0u) {
            index_buffer_device_address =
                reinterpret_cast<Buffer const *>(binding.index_buffer)
                    ->get_device_address() +
                binding.index_offset;
        }
        auto attach_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureTrianglesOpacityMicromapEXT>();
        if (!_micromap->fill_attach_info(*attach_info, attach_request,
                                         index_buffer_device_address)) [[unlikely]] {
            LUISA_ERROR("The opacity micromap of this mesh could not be described "
                        "for the BLAS build.");
        }
        opacity_micromap = attach_info;
        // Remember the source of the packed bits for build(): the copy into the
        // micromap's own usage-correct buffer happens right before the build.
        _micromap_data_buffer =
            reinterpret_cast<Buffer const *>(binding.data_buffer)->vk_buffer();
        _micromap_data_offset = binding.data_offset;
        cmdbuffer.resource_barrier->record(
            reinterpret_cast<Buffer const *>(binding.data_buffer),
            ResourceBarrier::Usage::kAccelInstanceBuffer);
        if (binding.index_stride != 0u) {
            cmdbuffer.resource_barrier->record(
                reinterpret_cast<Buffer const *>(binding.index_buffer),
                ResourceBarrier::Usage::kAccelInstanceBuffer);
        }
    }
    // Geometry flags: `plan_geometry_flags` reproduces today's
    // VK_GEOMETRY_OPAQUE_BIT_KHR exactly while no micromap is attached, and drops
    // the bit for a legal attach because the micromap then decides opacity
    // ("the micromap is used to determine geometry opacity instead of the
    // VK_GEOMETRY_OPAQUE_BIT_KHR flag", raytraversal.adoc).
    acceleration_structure_geometry->flags = detail::plan_geometry_flags({
        .triangle_geometry = true,
        .opaque_requested = true,
        .micromap_chained = opacity_micromap != nullptr})
                                                .flags;

    if (_option.motion.is_enabled() && keyframe_count == 2u) {
        // VkAccelerationStructureGeometryMotionTrianglesDataNV is NV-only;
        // guard it against devices without VK_NV_ray_tracing_motion_blur.
        if (!device()->enable_motion_blur()) [[unlikely]] {
            LUISA_ERROR("BLAS vertex motion blur requires VK_NV_ray_tracing_motion_blur, "
                        "which is not enabled on this device.");
        }
        // BLAS vertex motion blur: provide two keyframes via
        // VkAccelerationStructureGeometryMotionTrianglesDataNV.
        // Keyframe 0 goes into triangles.vertexData (the base geometry).
        // Keyframe 1 goes into the motion triangles pNext extension.
        auto keyframe_size_bytes = vertex_count_per_keyframe * cmd->vertex_stride();

        auto motion_triangles = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryMotionTrianglesDataNV>();
        motion_triangles->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV;
        motion_triangles->pNext = nullptr;
        motion_triangles->vertexData.deviceAddress = vertex_buffer_base + keyframe_size_bytes;
        triangles.pNext = motion_triangles;

        // Base geometry uses keyframe 0 vertices only
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData = vertex_data_device_address;
        triangles.maxVertex = vertex_count_per_keyframe - 1;
        triangles.vertexStride = cmd->vertex_stride();
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData = index_data_device_address;
    } else {
        // Non-motion path or unsupported keyframe count (>2 not supported by NV extension)
        if (_option.motion.is_enabled() && keyframe_count != 2u) {
            LUISA_WARNING("BLAS vertex motion blur only supports exactly 2 keyframes, but {} were provided. "
                          "Falling back to static BLAS build.", keyframe_count);
        }
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData = vertex_data_device_address;
        triangles.maxVertex = total_vertex_count - 1;
        triangles.vertexStride = cmd->vertex_stride();
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData = index_data_device_address;
        // Null unless a backend-internal caller attached a built micromap.
        triangles.pNext = opacity_micromap;
    }

    _pre_build(cmdbuffer, acceleration_structure_geometry, cmd->triangle_buffer_size() / 12, cmd->request());
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->vertex_buffer()),
        ResourceBarrier::Usage::kAccelInstanceBuffer);
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->triangle_buffer()),
        ResourceBarrier::Usage::kAccelInstanceBuffer);
}
void Blas::build(
    CommandBuffer &cmdbuffer,
    ProceduralPrimitiveBuildCommand const *cmd) {
    _acceleration_build_geometry_info->dstAccelerationStructure = _accel;
    _acceleration_build_geometry_info->srcAccelerationStructure =
        _acceleration_build_geometry_info->mode ==
                VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR ?
            _accel :
            VK_NULL_HANDLE;
    _acceleration_build_geometry_info->scratchData.deviceAddress = _scratch_buffer->get_device_address() + _scratch_buffer_offset;
    auto acceleration_structure_build_range_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildRangeInfoKHR>();
    acceleration_structure_build_range_info->primitiveCount = cmd->aabb_buffer_size() / sizeof(luisa::compute::AABB);
    acceleration_structure_build_range_info->primitiveOffset = 0;
    acceleration_structure_build_range_info->firstVertex = 0;
    acceleration_structure_build_range_info->transformOffset = 0;
    vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        _acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
}
void Blas::build(
    CommandBuffer &cmdbuffer,
    MeshBuildCommand const *cmd) {
    if (_micromap != nullptr) {
        // BUILD: (re)build the micromap from the bits this build declares, in this
        // very command buffer, right before the acceleration structure that uses it.
        // UPDATE: the micromap content must not change, so an unbuilt one is a hard
        // error - re-baking needs Mesh::BuildRequest::FORCE_BUILD.
        if (_acceleration_build_geometry_info->mode ==
            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
            _micromap->build_device(cmdbuffer, _micromap_data_buffer,
                                    _micromap_data_offset);
            // The micromap build has to complete before the acceleration structure
            // that references it is built; without this dependency the driver may
            // read a half-built micromap (which looks like an all-opaque cut-out).
            LUISA_ASSERT(device()->sync2_capable(),
                         "attaching an opacity micromap needs the micromap build "
                         "stage of synchronization2");
            VkMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            barrier.pNext = nullptr;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
            barrier.srcAccessMask = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
            barrier.dstStageMask =
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            barrier.dstAccessMask =
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.pNext = nullptr;
            dependency.dependencyFlags = 0u;
            dependency.memoryBarrierCount = 1u;
            dependency.pMemoryBarriers = &barrier;
            detail::cmd_pipeline_barrier(cmdbuffer.cmdbuffer(), device(), &dependency);
        } else if (!_micromap->built()) [[unlikely]] {
            LUISA_ERROR(
                "Re-baking an opacity micromap requires an explicit mesh rebuild: a "
                "PREFER_UPDATE build cannot rebuild the micromap it is attached to. "
                "Pass Mesh::BuildRequest::FORCE_BUILD.");
        }
    }
    _acceleration_build_geometry_info->dstAccelerationStructure = _accel;
    _acceleration_build_geometry_info->srcAccelerationStructure =
        _acceleration_build_geometry_info->mode ==
                VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR ?
            _accel :
            VK_NULL_HANDLE;
    _acceleration_build_geometry_info->scratchData.deviceAddress = _scratch_buffer->get_device_address() + _scratch_buffer_offset;
    auto acceleration_structure_build_range_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildRangeInfoKHR>();
    acceleration_structure_build_range_info->primitiveCount = cmd->triangle_buffer_size() / 12;
    acceleration_structure_build_range_info->primitiveOffset = 0;
    acceleration_structure_build_range_info->firstVertex = 0;
    acceleration_structure_build_range_info->transformOffset = 0;
    vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        _acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
}
Blas::~Blas() {
    for (auto &&i : _handles) {
        auto accel = i->accel;
        accel->_all_instance[i->accel_index].handle = nullptr;
        // A mesh-refresh entry queued by the last BLAS recreate (_sync_tlas)
        // may still reference this BLAS; drop it so a later TLAS build never
        // bakes the address of a destroyed BLAS into an instance descriptor.
        accel->_drop_refresh(i->accel_index, this);
        MeshHandle::destroy_handle(i);
    }
    vkDestroyAccelerationStructureKHR(device()->logic_device(), _accel, Device::alloc_callbacks());
}
uint64_t Blas::get_accel_device_address() const {
    if (!_accel) [[unlikely]] {
        LUISA_ERROR("BLAS not initialized.");
    }
    VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
    acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    acceleration_device_address_info.accelerationStructure = _accel;
    return vkGetAccelerationStructureDeviceAddressKHR(device()->logic_device(), &acceleration_device_address_info);
}
void Blas::_remove_accel_ref(MeshHandle *handle) {
    LUISA_ASSUME(handle->mesh == this);
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
MeshHandle *Blas::_add_accel_ref(Tlas *accel, uint index) {
    auto meshHandle = MeshHandle::allocate_handle();
    meshHandle->mesh = this;
    meshHandle->accel = accel;
    meshHandle->accel_index = index;
    {
        std::lock_guard lck(_handle_mtx);
        meshHandle->mesh_index = _handles.size();
        _handles.emplace_back(meshHandle);
    }
    return meshHandle;
}
void Blas::_sync_tlas() {
    std::lock_guard lck(_handle_mtx);
    for (auto &&i : _handles) {
        LUISA_ASSUME(i->mesh == this);
        i->accel->_update_mesh(i);
    }
}

namespace detail {
static vstd::Pool<MeshHandle> meshHandlePool(256, false);
static vstd::spin_mutex meshHandleMtx;
}// namespace detail
MeshHandle *MeshHandle::allocate_handle() {
    using namespace detail;
    return meshHandlePool.create_lock(meshHandleMtx);
}
void MeshHandle::destroy_handle(MeshHandle *handle) {
    using namespace detail;
    meshHandlePool.destroy_lock(meshHandleMtx, handle);
}
}// namespace lc::vk
