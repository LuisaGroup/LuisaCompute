#include "tlas.h"
#include "stream.h"
#include "buffer.h"
#include "compute_shader.h"
#include "device.h"
#include "log.h"
#include "blas.h"
#include "motion_instance.h"
namespace lc::vk {
namespace tlas_detail {
struct TlasInputInst {
    float affine[12];
    std::array<uint, 2> mesh;
    uint index : 24;
    uint vis_mask : 8;
    uint user_id : 24;
    uint flags : 8;
};
// Resolve a primitive handle to a Blas pointer.
// If the handle is a MotionInstance, returns its child Blas.
static Blas *resolve_to_blas(uint64_t primitive_handle) {
    auto prim = reinterpret_cast<PrimitiveBase *>(primitive_handle);
    if (!prim) return nullptr;
    if (prim->is_motion_instance()) {
        auto mi = static_cast<MotionInstance *>(prim);
        return mi->child();
    }
    return static_cast<Blas *>(prim);
}
}// namespace tlas_detail
Tlas::Tlas(Device *device, AccelOption const &option)
    : Resource(device), _option(option), _acceleration_build_geometry_info(nullptr) {
    if (!device->enable_raytracing()) [[unlikely]] {
        LUISA_ERROR("Raytracing not enabled, TLAS can not be loaded.");
    }
}
void Tlas::pre_build(
    CommandBuffer &cmdbuffer,
    uint instance_count,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<AccelBuildCommand::Modification const> modifications,
    AccelBuildRequest request) {
    using namespace tlas_detail;
    _resize_instance(instance_count);

    // Pre-scan modifications to detect motion before calculating buffer sizes
    if (!_has_motion && !(modifications.empty() && _set_map.empty())) {
        for (auto &&i : modifications) {
            if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                if (prim) {
                    if (prim->is_motion_instance()) {
                        _has_motion = true;
                        break;
                    }
                    auto blas = tlas_detail::resolve_to_blas(i.primitive);
                    if (blas && blas->has_motion()) {
                        _has_motion = true;
                        break;
                    }
                }
            }
        }
    }

    // _instance_buffer always uses 64-byte stride (VkAccelerationStructureInstanceKHR)
    // for shader reads via StructuredBuffer<_MeshInst>.
    // When motion is enabled, _motion_instance_buffer uses 160-byte stride
    // (VkAccelerationStructureMotionInstanceNV padded to 16-byte alignment) for TLAS build.
    auto std_inst_size = static_cast<size_t>(instance_count) * sizeof(VkAccelerationStructureInstanceKHR);
    std_inst_size = (std_inst_size + 65535u) & (~65535u);
    // resize
    auto resource_barrier = cmdbuffer.resource_barrier;
    bool update = _option.allow_update && request == AccelBuildRequest::PREFER_UPDATE && (!_require_rebuild);
    _require_rebuild = false;
    if (_last_instance_count != instance_count) {
        update = false;
        _last_instance_count = instance_count;
    }
    // Resize the standard 64-byte instance buffer (for shader reads)
    if (_instance_buffer && _instance_buffer->byte_size() < std_inst_size) {
        update = false;
        auto new_inst_buffer = vstd::make_unique<DefaultBuffer>(device(), std_inst_size, true, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        resource_barrier->record(
            BufferView{new_inst_buffer.get()},
            ResourceBarrier::Usage::kCopyDest);
        resource_barrier->record(
            BufferView{
                _instance_buffer.get()},
            ResourceBarrier::Usage::kCopySource);
        resource_barrier->update_states(cmdbuffer.cmdbuffer());
        VkBufferCopy2 buffer_copy{
            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            nullptr,
            0, 0,
            _instance_buffer->byte_size()};
        VkCopyBufferInfo2 copy_info2{
            VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            nullptr,
            _instance_buffer->vk_buffer(),
            new_inst_buffer->vk_buffer(),
            1,
            &buffer_copy};
        vkCmdCopyBuffer2(
            cmdbuffer.cmdbuffer(),
            &copy_info2);
        cmdbuffer.states()->dispose_after_flush(std::move(_instance_buffer));
        _instance_buffer = std::move(new_inst_buffer);
    } else if (!_instance_buffer) {
        update = false;
        _instance_buffer = vstd::make_unique<DefaultBuffer>(device(), std_inst_size, false, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    }
    // When motion is enabled, also manage the motion instance buffer for TLAS build
    if (_has_motion) {
        auto motion_buf_size = static_cast<size_t>(instance_count) * kMotionInstanceStride;
        motion_buf_size = (motion_buf_size + 65535u) & (~65535u);
        if (_motion_instance_buffer && _motion_instance_buffer->byte_size() < motion_buf_size) {
            update = false;
            cmdbuffer.states()->dispose_after_flush(std::move(_motion_instance_buffer));
        }
        if (!_motion_instance_buffer) {
            update = false;
            _motion_instance_buffer = vstd::make_unique<DefaultBuffer>(
                device(),
                motion_buf_size,
                false,
                static_cast<VkBufferUsageFlagBits>(
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                )
            );
        }
    }
    if (!(modifications.empty() && _set_map.empty())) {
        // First pass: resolve primitives and update mesh references
        // Store resolved Blas pointers for each modification that has a primitive
        luisa::vector<Blas *> resolved_meshes(modifications.size(), nullptr);
        for (size_t idx = 0; idx < modifications.size(); idx++) {
            auto &&i = modifications[idx];
            auto ite = _set_map.find(i.index);
            bool updateMesh = (i.flags & AccelBuildCommand::Modification::flag_primitive);
            if (ite != _set_map.end()) {
                resolved_meshes[idx] = ite->second->mesh;
                if (!updateMesh) {
                    const_cast<uint &>(i.flags) = i.flags | AccelBuildCommand::Modification::flag_primitive;
                    updateMesh = true;
                }
                _set_map.erase(ite);
            } else if (updateMesh) {
                resolved_meshes[idx] = tlas_detail::resolve_to_blas(i.primitive);
            }
            if (updateMesh) {
                _set_mesh(resolved_meshes[idx], i.index);
                update = false;
                // Check if any child BLAS has motion enabled
                if (resolved_meshes[idx] && resolved_meshes[idx]->has_motion()) {
                    _has_motion = true;
                }
                // Check if the primitive is a MotionInstance
                if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                    auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                    if (prim && prim->is_motion_instance()) {
                        _has_motion = true;
                    }
                }
            }
        }
        // When motion is enabled, fill both instance buffers from CPU:
        // 1. _motion_instance_buffer (160-byte stride, 16-byte aligned) for TLAS build
        // 2. _instance_buffer (64-byte stride) for shader reads via StructuredBuffer<_MeshInst>
        if (_has_motion) {
            // Allocate upload buffer for motion instances (160-byte stride)
            auto motion_upload_size = static_cast<size_t>(instance_count) * kMotionInstanceStride;
            // Allocate upload buffer for standard instances (64-byte stride)
            auto std_inst_size_bytes = static_cast<size_t>(instance_count) * sizeof(VkAccelerationStructureInstanceKHR);
            auto total_upload_size = motion_upload_size + std_inst_size_bytes;
            auto upload_buf = cmdbuffer.states()->upload_alloc.allocate(total_upload_size, 16);
            auto upload_base = reinterpret_cast<uint8_t *>(
                static_cast<UploadBuffer const *>(upload_buf.buffer)->mapped_ptr()) + upload_buf.offset;
            auto motion_data = upload_base;
            auto std_data = upload_base + motion_upload_size;
            memset(motion_data, 0, motion_upload_size);
            memset(std_data, 0, std_inst_size_bytes);

            // Fill each instance from modifications
            for (size_t idx = 0; idx < modifications.size(); idx++) {
                auto &&i = modifications[idx];
                if (i.index >= instance_count) continue;

                // Determine if this modification refers to a MotionInstance with SRT/Matrix keyframes
                MotionInstance *mi = nullptr;
                if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                    auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                    if (prim && prim->is_motion_instance()) {
                        mi = static_cast<MotionInstance *>(prim);
                    }
                }

                // --- Fill motion instance at 160-byte stride ---
                auto inst_base = motion_data + static_cast<size_t>(i.index) * kMotionInstanceStride;

                // --- Fill 64-byte standard instance ---
                auto std_inst = reinterpret_cast<VkAccelerationStructureInstanceKHR *>(
                    std_data + static_cast<size_t>(i.index) * sizeof(VkAccelerationStructureInstanceKHR));

                // Common instance fields
                uint custom_index;
                if (i.flags & AccelBuildCommand::Modification::flag_user_id) {
                    custom_index = i.user_id;
                } else {
                    custom_index = static_cast<uint>(i.index);
                }
                uint mask;
                if (i.flags & AccelBuildCommand::Modification::flag_visibility) {
                    mask = i.vis_mask;
                } else {
                    mask = 0xFF;
                }
                VkGeometryInstanceFlagsKHR geom_flags = 0;
                if (i.flags & AccelBuildCommand::Modification::flag_opaque_on) {
                    geom_flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
                } else if (i.flags & AccelBuildCommand::Modification::flag_opaque_off) {
                    geom_flags = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
                }
                uint64_t accel_ref = 0;
                if ((i.flags & AccelBuildCommand::Modification::flag_primitive) && resolved_meshes[idx]) {
                    accel_ref = resolved_meshes[idx]->get_accel_device_address();
                    resource_barrier->record(BufferView{resolved_meshes[idx]->_accel_buffer.get()},
                                             ResourceBarrier::Usage::kAccelInstanceBuffer);
                }

                // Fill standard instance buffer (for shader reads)
                if (i.flags & AccelBuildCommand::Modification::flag_transform) {
                    memcpy(&std_inst->transform, i.affine, sizeof(float) * 12);
                }
                std_inst->instanceCustomIndex = custom_index;
                std_inst->mask = mask;
                std_inst->instanceShaderBindingTableRecordOffset = 0;
                std_inst->flags = geom_flags;
                std_inst->accelerationStructureReference = accel_ref;

                // Fill motion instance buffer (for TLAS build)
                if (mi && mi->mode() == AccelMotionMode::MATRIX && mi->keyframe_count() >= 2) {
                    // Matrix Motion Instance: type = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV (1)
                    *reinterpret_cast<uint32_t *>(inst_base + 0) = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV;
                    *reinterpret_cast<uint32_t *>(inst_base + 4) = 0u; // flags = 0

                    // VkAccelerationStructureMatrixMotionInstanceNV layout:
                    //   transformT0 (VkTransformMatrixKHR, 48 bytes) at offset 8
                    //   transformT1 (VkTransformMatrixKHR, 48 bytes) at offset 56
                    //   instanceCustomIndex:24 | mask:8 at offset 104
                    //   instanceShaderBindingTableRecordOffset:24 | flags:8 at offset 108
                    //   accelerationStructureReference (uint64) at offset 112
                    auto &keyframes = mi->keyframes();
                    auto &mat0 = keyframes[0].as_matrix();
                    auto &mat1 = keyframes[mi->keyframe_count() - 1].as_matrix();

                    // Write VkTransformMatrixKHR (row-major float[3][4]) from float4x4 (column-major)
                    auto write_vk_matrix = [](uint8_t *dst, const float4x4 &m) {
                        auto *f = reinterpret_cast<float *>(dst);
                        // Row 0
                        f[0] = m[0][0]; f[1] = m[1][0]; f[2] = m[2][0]; f[3] = m[3][0];
                        // Row 1
                        f[4] = m[0][1]; f[5] = m[1][1]; f[6] = m[2][1]; f[7] = m[3][1];
                        // Row 2
                        f[8] = m[0][2]; f[9] = m[1][2]; f[10] = m[2][2]; f[11] = m[3][2];
                    };

                    write_vk_matrix(inst_base + 8, mat0);       // transformT0
                    write_vk_matrix(inst_base + 8 + 48, mat1);  // transformT1

                    // Instance fields after the two matrix transforms
                    auto *mat_inst_fields = inst_base + 8 + 48 + 48; // offset 104
                    *reinterpret_cast<uint32_t *>(mat_inst_fields + 0) =
                        (custom_index & 0x00FFFFFFu) | (static_cast<uint32_t>(mask) << 24u);
                    *reinterpret_cast<uint32_t *>(mat_inst_fields + 4) =
                        (0u & 0x00FFFFFFu) | (static_cast<uint32_t>(geom_flags) << 24u);
                    *reinterpret_cast<uint64_t *>(mat_inst_fields + 8) = accel_ref;

                } else if (mi && mi->mode() == AccelMotionMode::SRT && mi->keyframe_count() >= 2) {
                    // SRT Motion Instance: type = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV (2)
                    *reinterpret_cast<uint32_t *>(inst_base + 0) = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV;
                    *reinterpret_cast<uint32_t *>(inst_base + 4) = 0u; // flags = 0

                    // VkAccelerationStructureSRTMotionInstanceNV layout at offset 8:
                    //   transformT0 (VkSRTDataNV, 64 bytes) at offset 8
                    //   transformT1 (VkSRTDataNV, 64 bytes) at offset 72
                    //   instanceCustomIndex:24 | mask:8 at offset 136
                    //   instanceShaderBindingTableRecordOffset:24 | flags:8 at offset 140
                    //   accelerationStructureReference (uint64) at offset 144

                    // Convert MotionInstanceTransformSRT -> VkSRTDataNV (field remapping)
                    auto &keyframes = mi->keyframes();
                    auto &srt0 = keyframes[0].as_srt();
                    auto &srt1 = keyframes[mi->keyframe_count() - 1].as_srt();

                    auto write_vk_srt = [](uint8_t *dst, const MotionInstanceTransformSRT &srt) {
                        auto *f = reinterpret_cast<float *>(dst);
                        f[0]  = srt.scale[0];       // sx
                        f[1]  = srt.shear[0];       // a
                        f[2]  = srt.shear[1];       // b
                        f[3]  = srt.pivot[0];       // pvx
                        f[4]  = srt.scale[1];       // sy
                        f[5]  = srt.shear[2];       // c
                        f[6]  = srt.pivot[1];       // pvy
                        f[7]  = srt.scale[2];       // sz
                        f[8]  = srt.pivot[2];       // pvz
                        f[9]  = srt.quaternion[0];  // qx
                        f[10] = srt.quaternion[1];  // qy
                        f[11] = srt.quaternion[2];  // qz
                        f[12] = srt.quaternion[3];  // qw
                        f[13] = srt.translation[0]; // tx
                        f[14] = srt.translation[1]; // ty
                        f[15] = srt.translation[2]; // tz
                    };

                    write_vk_srt(inst_base + 8, srt0);       // transformT0
                    write_vk_srt(inst_base + 8 + 64, srt1);  // transformT1

                    // Instance fields after the two SRT transforms
                    auto *srt_inst_fields = inst_base + 8 + 64 + 64; // offset 136
                    *reinterpret_cast<uint32_t *>(srt_inst_fields + 0) =
                        (custom_index & 0x00FFFFFFu) | (static_cast<uint32_t>(mask) << 24u);
                    *reinterpret_cast<uint32_t *>(srt_inst_fields + 4) =
                        (0u & 0x00FFFFFFu) | (static_cast<uint32_t>(geom_flags) << 24u);
                    *reinterpret_cast<uint64_t *>(srt_inst_fields + 8) = accel_ref;

                    if (mi->keyframe_count() > 2) {
                        LUISA_WARNING("VK_NV_ray_tracing_motion_blur only supports 2 keyframes for SRT motion. "
                                      "Using first and last keyframes (ignoring {} intermediate keyframes).",
                                      mi->keyframe_count() - 2);
                    }
                } else {
                    // Static instance: type = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV (0)
                    *reinterpret_cast<uint32_t *>(inst_base + 0) = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV;
                    *reinterpret_cast<uint32_t *>(inst_base + 4) = 0u; // flags = 0
                    // data.staticInstance = VkAccelerationStructureInstanceKHR at offset 8
                    auto motion_inst = reinterpret_cast<VkAccelerationStructureInstanceKHR *>(inst_base + 8);
                    if (i.flags & AccelBuildCommand::Modification::flag_transform) {
                        memcpy(&motion_inst->transform, i.affine, sizeof(float) * 12);
                    }
                    motion_inst->instanceCustomIndex = custom_index;
                    motion_inst->mask = mask;
                    motion_inst->instanceShaderBindingTableRecordOffset = 0;
                    motion_inst->flags = geom_flags;
                    motion_inst->accelerationStructureReference = accel_ref;
                }
            }

            // Also process remaining _set_map entries (BLAS updates not covered by modifications).
            // Without this, when a BLAS is rebuilt (address changes) and notifies the TLAS via
            // _update_mesh/_set_map, the motion instance buffer would retain stale BLAS addresses,
            // causing GPU hangs on the next TraceRay.
            for (auto &[set_idx, handle] : _set_map) {
                if (set_idx >= instance_count) continue;
                auto *mesh = handle->mesh;
                if (!mesh) continue;
                uint64_t accel_ref = mesh->get_accel_device_address();
                resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                         ResourceBarrier::Usage::kAccelInstanceBuffer);

                // Update accelerationStructureReference in motion instance buffer
                auto inst_base = motion_data + static_cast<size_t>(set_idx) * kMotionInstanceStride;
                uint32_t motion_type = *reinterpret_cast<uint32_t *>(inst_base + 0);
                if (motion_type == VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV) {
                    // SRT motion instance: accel ref at offset 8 + 64 + 64 + 8 = 144
                    *reinterpret_cast<uint64_t *>(inst_base + 8 + 64 + 64 + 8) = accel_ref;
                } else if (motion_type == VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV) {
                    // Matrix motion instance: accel ref at offset 8 + 48 + 48 + 8 = 112
                    *reinterpret_cast<uint64_t *>(inst_base + 8 + 48 + 48 + 8) = accel_ref;
                } else {
                    // Static motion instance: accel ref at offset 8 + offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference)
                    auto motion_inst = reinterpret_cast<VkAccelerationStructureInstanceKHR *>(inst_base + 8);
                    motion_inst->accelerationStructureReference = accel_ref;
                }

                // Update accelerationStructureReference in standard instance buffer
                auto std_inst = reinterpret_cast<VkAccelerationStructureInstanceKHR *>(
                    std_data + static_cast<size_t>(set_idx) * sizeof(VkAccelerationStructureInstanceKHR));
                std_inst->accelerationStructureReference = accel_ref;

                _set_mesh(mesh, set_idx);
                update = false;
            }

            // Copy motion instances to _motion_instance_buffer (for TLAS build)
            resource_barrier->record(
                BufferView{_motion_instance_buffer.get()},
                ResourceBarrier::Usage::kCopyDest);
            // Copy standard instances to _instance_buffer (for shader reads)
            resource_barrier->record(
                BufferView{_instance_buffer.get()},
                ResourceBarrier::Usage::kCopyDest);
            resource_barrier->update_states(cmdbuffer.cmdbuffer());
            // Copy motion buffer
            {
                VkBufferCopy2 buffer_copy{
                    VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    nullptr,
                    upload_buf.offset, 0,
                    motion_upload_size};
                VkCopyBufferInfo2 copy_info2{
                    VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    nullptr,
                    upload_buf.buffer->vk_buffer(),
                    _motion_instance_buffer->vk_buffer(),
                    1,
                    &buffer_copy};
                vkCmdCopyBuffer2(cmdbuffer.cmdbuffer(), &copy_info2);
            }
            // Copy standard buffer
            {
                VkBufferCopy2 buffer_copy{
                    VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    nullptr,
                    upload_buf.offset + motion_upload_size, 0,
                    std_inst_size_bytes};
                VkCopyBufferInfo2 copy_info2{
                    VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    nullptr,
                    upload_buf.buffer->vk_buffer(),
                    _instance_buffer->vk_buffer(),
                    1,
                    &buffer_copy};
                vkCmdCopyBuffer2(cmdbuffer.cmdbuffer(), &copy_info2);
            }
            _set_map.clear();
        } else {
        // Non-motion path: use compute shader to fill instance buffer
        resource_barrier->record(
            BufferView{
                _instance_buffer.get()},
            ResourceBarrier::Usage::kComputeUAV);
        auto shader = device()->set_accel_kernel.get(device());
        resource_barrier->update_states(cmdbuffer.cmdbuffer());
        VkDescriptorSet desc_set;
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = cmdbuffer.states()->desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = shader->desc_set_layout().data()};
        VK_CHECK_RESULT(
            vkAllocateDescriptorSets(
                device()->logic_device(),
                &alloc_info,
                &desc_set));
        const uint modification_size = modifications.size() + _set_map.size();
        uint2 value = {
            modification_size,
            instance_count};
        vkCmdPushConstants(
            cmdbuffer.cmdbuffer(),
            shader->pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(value),
            &value);

        auto dsc_buffer = cmdbuffer.states()->upload_alloc.allocate((modification_size) * sizeof(TlasInputInst), 16);
        cache.clear();
        luisa::enlarge_by(cache, (dsc_buffer.size_bytes + sizeof(uint4) - 1) / sizeof(uint4));
        std::memset(cache.data(), 0, luisa::size_bytes(cache));
        auto inst_ptr = reinterpret_cast<TlasInputInst *>(cache.data());

        for (size_t idx = 0; idx < modifications.size(); idx++) {
            auto &&i = modifications[idx];
            std::memcpy(inst_ptr->affine, i.affine, sizeof(float) * 12);
            inst_ptr->index = i.index;
            inst_ptr->vis_mask = i.vis_mask;
            inst_ptr->user_id = i.user_id;
            inst_ptr->flags = i.flags;
            if ((i.flags & AccelBuildCommand::Modification::flag_primitive) && resolved_meshes[idx]) {
                auto mesh = resolved_meshes[idx];
                auto addr = mesh->get_accel_device_address();
                inst_ptr->mesh = reinterpret_cast<std::array<uint, 2> const &>(addr);
                resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                         ResourceBarrier::Usage::kAccelInstanceBuffer);
            }
            inst_ptr++;
        }
        for (auto &i : _set_map) {
            if (i.first >= _all_instance.size()) continue;
            inst_ptr->index = i.first;
            inst_ptr->flags = AccelBuildCommand::Modification::flag_primitive;
            resource_barrier->record(BufferView{i.second->mesh->_accel_buffer.get()},
                                     ResourceBarrier::Usage::kAccelInstanceBuffer);
            auto addr = i.second->mesh->get_accel_device_address();
            inst_ptr->mesh = reinterpret_cast<std::array<uint, 2> &>(addr);
            ++inst_ptr;
        }
        static_cast<UploadBuffer const *>(dsc_buffer.buffer)->copy_from(cache.data(), dsc_buffer.offset, dsc_buffer.size_bytes);
        _set_map.clear();
        VkDescriptorBufferInfo arg_buffer_info{
            dsc_buffer.buffer->vk_buffer(),
            dsc_buffer.offset,
            dsc_buffer.size_bytes};
        VkDescriptorBufferInfo buffer_info{
            _instance_buffer->vk_buffer(),
            0,
            _instance_buffer->byte_size()};
        write_desc_sets.emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &arg_buffer_info,
            nullptr});
        write_desc_sets.emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            1,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &buffer_info,
            nullptr});
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
        vkCmdBindDescriptorSets(
            cmdbuffer.cmdbuffer(),
            VK_PIPELINE_BIND_POINT_COMPUTE,
            shader->pipeline_layout(),
            0,
            1,
            &desc_set,
            0,
            nullptr);
        vkCmdBindPipeline(cmdbuffer.cmdbuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, shader->pipeline());
        vkCmdDispatch(cmdbuffer.cmdbuffer(), (modification_size + 255) / 256, 1, 1);
        } // end non-motion path
    }
    VkDeviceOrHostAddressConstKHR instance_data_device_address{};
    // When motion is enabled, TLAS build reads from the 160-byte stride motion instance buffer.
    // The 64-byte _instance_buffer is only for shader reads (StructuredBuffer<_MeshInst>).
    auto *build_instance_buffer = _has_motion ? _motion_instance_buffer.get() : _instance_buffer.get();
    instance_data_device_address.deviceAddress = build_instance_buffer->get_device_address();
    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    acceleration_structure_geometry->flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    acceleration_structure_geometry->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    acceleration_structure_geometry->geometry.instances.arrayOfPointers = VK_FALSE;
    acceleration_structure_geometry->geometry.instances.data = instance_data_device_address;

    _acceleration_build_geometry_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildGeometryInfoKHR>();
    _acceleration_build_geometry_info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    _acceleration_build_geometry_info->type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    _acceleration_build_geometry_info->flags = _option.hint == AccelOption::UsageHint::FAST_BUILD ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    _acceleration_build_geometry_info->geometryCount = 1;
    _acceleration_build_geometry_info->pGeometries = acceleration_structure_geometry;
    if (_option.allow_update) {
        _acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    // Add motion bit if any child BLAS has motion or any MotionInstance is present
    if (_has_motion) {
        _acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
    }
    _acceleration_build_geometry_info->mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
    acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        device()->logic_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        _acceleration_build_geometry_info,
        &instance_count,
        &acceleration_structure_build_sizes_info);
    uint scratch_buffer_size = update ? acceleration_structure_build_sizes_info.updateScratchSize : acceleration_structure_build_sizes_info.buildScratchSize;
    if (_accel_buffer && _accel_buffer->byte_size() < acceleration_structure_build_sizes_info.accelerationStructureSize) {
        cmdbuffer.states()->dispose_after_flush(std::move(_accel_buffer));
    }
    if (!_accel_buffer) {
        update = false;
        _accel_buffer = vstd::make_unique<DefaultBuffer>(
            device(),
            acceleration_structure_build_sizes_info.accelerationStructureSize,
            false, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    }
    resource_barrier->record(
        BufferView{build_instance_buffer},
        ResourceBarrier::Usage::kAccelInstanceBuffer);
    resource_barrier->record(
        _accel_buffer.get(),
        ResourceBarrier::Usage::kBuildAccel);
    VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
    acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    acceleration_structure_create_info.buffer = _accel_buffer->vk_buffer();
    acceleration_structure_create_info.size = acceleration_structure_build_sizes_info.accelerationStructureSize;
    acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // Motion info for TLAS with motion instances
    VkAccelerationStructureMotionInfoNV motion_info{};
    motion_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV;
    motion_info.maxInstances = instance_count;
    motion_info.flags = 0;
    if (_has_motion) {
        acceleration_structure_create_info.createFlags = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
        acceleration_structure_create_info.pNext = &motion_info;
    }
    if (!update) {
        if (_accel) {
            cmdbuffer.states()->callbacks.emplace_back([a = _accel, device = device()]() {
                vkDestroyAccelerationStructureKHR(device->logic_device(), a, Device::alloc_callbacks());
            });
            _accel = VK_NULL_HANDLE;
        }
        VK_CHECK_RESULT(vkCreateAccelerationStructureKHR(device()->logic_device(), &acceleration_structure_create_info, Device::alloc_callbacks(), &_accel));
    }
    scratch_buffer_size = (scratch_buffer_size + 255) & (~(255u));
    auto scratch_chunk = cmdbuffer.scratch_buffer_alloc->allocate(scratch_buffer_size);
    _scratch_buffer = reinterpret_cast<Buffer const *>(scratch_chunk.handle);
    _scratch_buffer_offset = scratch_chunk.offset;
    LUISA_DEBUG_ASSERT(((_scratch_buffer->get_device_address() + scratch_chunk.offset) & 255) == 0, "tlas scratch buffer alignment failed.");
    cmdbuffer.resource_barrier->record(
        _scratch_buffer,
        ResourceBarrier::Usage::kComputeUAV);
}
void Tlas::_update_mesh(
    MeshHandle *handle) {
    auto instIndex = handle->accel_index;
    LUISA_ASSUME(_all_instance[instIndex].handle == handle);
    _set_map[instIndex] = handle;
    _require_rebuild = true;
}
void Tlas::build(
    CommandBuffer &cmdbuffer,
    uint instance_count) {
    _acceleration_build_geometry_info->dstAccelerationStructure = _accel;
    if (_acceleration_build_geometry_info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
        _acceleration_build_geometry_info->srcAccelerationStructure = _accel;
    }
    _acceleration_build_geometry_info->scratchData.deviceAddress = _scratch_buffer->get_device_address() + _scratch_buffer_offset;
    auto acceleration_structure_build_range_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildRangeInfoKHR>();
    acceleration_structure_build_range_info->primitiveCount = instance_count;
    acceleration_structure_build_range_info->primitiveOffset = 0;
    acceleration_structure_build_range_info->firstVertex = 0;
    acceleration_structure_build_range_info->transformOffset = 0;
    vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        _acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
    // possible?
    cmdbuffer.resource_barrier->record(
        BufferView{
            _instance_buffer.get()},
        ResourceBarrier::Usage::kComputeRead);
    cmdbuffer.resource_barrier->record(
        _accel_buffer.get(),
        ResourceBarrier::Usage::kComputeRead);
}
void Tlas::_resize_instance(size_t size) {
    if (size < _all_instance.size()) {
        for (auto &i : vstd::ptr_range(_all_instance.data() + size, _all_instance.data() + _all_instance.size())) {
            if (!i.handle) continue;
            i.handle->mesh->_remove_accel_ref(i.handle);
        }
    }
    _all_instance.resize(size);
}

Tlas::~Tlas() {
    for (auto &&i : _all_instance) {
        auto mesh = i.handle;
        if (mesh)
            mesh->mesh->_remove_accel_ref(mesh);
    }
    vkDestroyAccelerationStructureKHR(device()->logic_device(), _accel, Device::alloc_callbacks());
}
void Tlas::_set_mesh(Blas *mesh, uint64 index) {
    auto &&inst = _all_instance[index].handle;
    if (inst != nullptr) {
        if (inst->mesh == mesh) return;
        inst->mesh->_remove_accel_ref(inst);
    }
    inst = mesh->_add_accel_ref(this, index);
    inst->accel_index = index;
}
}// namespace lc::vk