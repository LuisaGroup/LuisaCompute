#include "blas.h"
#include "device.h"
#include "log.h"
#include "tlas.h"
#include "stream.h"
#include <luisa/runtime/rtx/aabb.h>
namespace lc::vk {
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
    // Enable BLAS-level vertex motion blur when the mesh has motion keyframes
    if (_option.motion.is_enabled()) {
        _acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
    }
    _acceleration_build_geometry_info->geometryCount = 1;
    _acceleration_build_geometry_info->pGeometries = acceleration_structure_geometry;
    bool update = _option.allow_update && request == AccelBuildRequest::PREFER_UPDATE;

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
    bool sync = _accel;
    if (_accel) {
        cmdbuffer.states()->callbacks.emplace_back([a = _accel, device = device()]() {
            vkDestroyAccelerationStructureKHR(device->logic_device(), a, Device::alloc_callbacks());
        });
    }
    VK_CHECK_RESULT(vkCreateAccelerationStructureKHR(device()->logic_device(), &acceleration_structure_create_info, Device::alloc_callbacks(), &_accel));
    scratch_buffer_size = (scratch_buffer_size + 255) & (~(255u));
    auto scratch_chunk = cmdbuffer.scratch_buffer_alloc->allocate(scratch_buffer_size);

    _scratch_buffer = reinterpret_cast<Buffer const *>(scratch_chunk.handle);
    _scratch_buffer_offset = scratch_chunk.offset;
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
    acceleration_structure_geometry->flags = 0;
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
    acceleration_structure_geometry->flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto &triangles = acceleration_structure_geometry->geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.pNext = nullptr;

    auto keyframe_count = _option.motion.is_enabled() ? _option.motion.keyframe_count : 1u;
    auto total_vertex_count = cmd->vertex_buffer_size() / cmd->vertex_stride();
    auto vertex_count_per_keyframe = total_vertex_count / keyframe_count;

    if (_option.motion.is_enabled() && keyframe_count == 2u) {
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
    _acceleration_build_geometry_info->dstAccelerationStructure = _accel;
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
        i->accel->_all_instance[i->accel_index].handle = nullptr;
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