#include "tlas.h"
#include "stream.h"
#include "buffer.h"
#include "compute_shader.h"
#include "device.h"
#include "command_buffer_sync.h"
#include "log.h"
#include "blas.h"
#include "motion_instance.h"
#include "vulkan_builtin_contract.h"
namespace lc::vk {
namespace tlas_detail {
using TlasInputInst = detail::VulkanAccelUpdateInput;
static_assert(
    detail::VulkanAccelUpdateLayout::flag_mesh ==
    AccelBuildCommand::Modification::flag_primitive);
static_assert(
    detail::VulkanAccelUpdateLayout::flag_transform ==
    AccelBuildCommand::Modification::flag_transform);
static_assert(
    detail::VulkanAccelUpdateLayout::flag_opaque_on ==
    AccelBuildCommand::Modification::flag_opaque_on);
static_assert(
    detail::VulkanAccelUpdateLayout::flag_opaque_off ==
    AccelBuildCommand::Modification::flag_opaque_off);
static_assert(
    detail::VulkanAccelUpdateLayout::flag_visibility ==
    AccelBuildCommand::Modification::flag_visibility);
static_assert(
    detail::VulkanAccelUpdateLayout::flag_user_id ==
    AccelBuildCommand::Modification::flag_user_id);
static_assert(
    detail::VulkanAccelUpdateLayout::instance_force_opaque ==
    VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR);
static_assert(
    detail::VulkanAccelUpdateLayout::instance_force_no_opaque ==
    VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR);
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

    // Recompute motion state from ground truth: the live instance table plus
    // any incoming primitive modifications. A TLAS must stay motion-capable
    // for as long as any referenced child requires it, no matter which
    // instances the current modification list touches — recomputing from the
    // modification list alone would silently flip a motion TLAS to the
    // non-motion path on e.g. a transform-only update, dropping motion blur
    // and mixing instance formats between builds.
    _has_motion = false;
    for (auto &inst : _all_instance) {
        if (inst.handle == nullptr) continue;
        if (inst.is_motion_instance || inst.handle->mesh->has_motion()) {
            _has_motion = true;
            break;
        }
    }
    if (!_has_motion) {
        for (auto &&i : modifications) {
            if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                if (prim && prim->is_motion_instance()) {
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
    if (_has_motion != _built_with_motion) {
        // The TLAS create flags and instance format changed between builds; an
        // in-place update would mix layouts, so force a full rebuild.
        update = false;
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
        detail::cmd_copy_buffer(cmdbuffer.cmdbuffer(), device(), &copy_info2);
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
            _motion_instance_buffer = vstd::make_unique<DefaultBuffer>(device(), motion_buf_size, false, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        }
    }
    if (!(modifications.empty() && _pending_refresh_count == 0u)) {
        // First pass: resolve primitives and update mesh references
        // Store resolved Blas pointers for each modification that has a primitive
        luisa::vector<Blas *> resolved_meshes(modifications.size(), nullptr);
        for (size_t idx = 0; idx < modifications.size(); idx++) {
            auto &&i = modifications[idx];
            bool updateMesh = (i.flags & AccelBuildCommand::Modification::flag_primitive);
            // Whether the primitive comes from the modification itself (as
            // opposed to a pending refresh entry force-enabling the rewrite).
            bool explicit_primitive = updateMesh;
            // Pending refresh state lives per-slot (dense instance index), so
            // the merge below is a plain array access — no hash lookup.
            auto index = static_cast<size_t>(i.index);
            auto &pending = _all_instance[index].refresh_blas;
            if (updateMesh) {
                // An explicit primitive modification is authoritative: it wins
                // over any pending BLAS-recreate refresh entry for the same
                // slot (mirrors the dx backend's ProcessSetDesc merge order).
                // Otherwise the replacement would be silently dropped and the
                // TLAS would keep tracing the old mesh.
                resolved_meshes[idx] = tlas_detail::resolve_to_blas(i.primitive);
            } else if (pending != nullptr) {
                resolved_meshes[idx] = pending;
                const_cast<uint &>(i.flags) = i.flags | AccelBuildCommand::Modification::flag_primitive;
                updateMesh = true;
            }
            // A modification touching this slot (explicit or folded) consumes
            // or overrides any pending refresh for it.
            if (pending != nullptr) {
                pending = nullptr;
                --_pending_refresh_count;
            }
            if (updateMesh) {
                auto motion_instance = false;
                if (explicit_primitive) {
                    auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                    motion_instance = prim && prim->is_motion_instance();
                }
                _set_mesh(resolved_meshes[idx], i.index, motion_instance, explicit_primitive);
                update = false;
                // Check if any child BLAS has motion enabled
                if (resolved_meshes[idx] && resolved_meshes[idx]->has_motion()) {
                    if (!device()->enable_motion_blur()) [[unlikely]] {
                        LUISA_ERROR("TLAS motion requires VK_NV_ray_tracing_motion_blur, "
                                    "which is not enabled on this device.");
                    }
                    _has_motion = true;
                }
                // Check if the primitive is a MotionInstance
                if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                    auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                    if (prim && prim->is_motion_instance()) {
                        if (!device()->enable_motion_blur()) [[unlikely]] {
                            LUISA_ERROR("TLAS motion requires VK_NV_ray_tracing_motion_blur, "
                                        "which is not enabled on this device.");
                        }
                        _has_motion = true;
                    }
                }
            }
        }
        // When motion is enabled, fill both instance buffers from CPU:
        // 1. _motion_instance_buffer (160-byte stride, 16-byte aligned) for TLAS build
        // 2. _instance_buffer (64-byte stride) for shader reads via StructuredBuffer<_MeshInst>
        if (_has_motion) {
            auto motion_upload_size = static_cast<size_t>(instance_count) * kMotionInstanceStride;
            auto std_inst_size_bytes = static_cast<size_t>(instance_count) * sizeof(VkAccelerationStructureInstanceKHR);
            // Both instance buffers are consumed whole by the TLAS build, so
            // keep persistent host-side shadow copies, update them
            // incrementally and upload them whole: refilling from the current
            // modification list alone would zero every untouched instance and
            // drop pending slot refreshes (which carry the new device
            // addresses of recreated BLAS).
            _motion_instance_cache.resize(motion_upload_size);
            _std_instance_cache.resize(std_inst_size_bytes);

            static_assert(sizeof(VkAccelerationStructureInstanceKHR) == 64u);
            static_assert(offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference) == 56u);

            // Convert MotionInstanceTransformSRT -> VkSRTDataNV (field remapping)
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

            // Fold pending BLAS-recreate refreshes into the shadow copies,
            // touching only the acceleration-structure reference fields.
            // _all_instance is already sized to instance_count (resized at the
            // top of pre_build), so a single contiguous slot scan replaces the
            // hash-map iteration.
            for (auto index = 0u; index < _all_instance.size(); index++) {
                auto mesh = _all_instance[index].refresh_blas;
                if (mesh == nullptr) continue;
                if (index >= instance_count) continue;// defensive
                auto addr = mesh->get_accel_device_address();
                resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                         ResourceBarrier::Usage::kAccelInstanceBuffer);
                reinterpret_cast<VkAccelerationStructureInstanceKHR *>(
                    _std_instance_cache.data() + index * sizeof(VkAccelerationStructureInstanceKHR))
                    ->accelerationStructureReference = addr;
                auto refresh_base = _motion_instance_cache.data() + index * kMotionInstanceStride;
                switch (*reinterpret_cast<uint32_t const *>(refresh_base)) {
                    case VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV:
                        // VkAccelerationStructureSRTMotionInstanceNV
                        *reinterpret_cast<uint64_t *>(refresh_base + 144u) = addr;
                        break;
                    case VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV:
                        // VkAccelerationStructureMatrixMotionInstanceNV
                        *reinterpret_cast<uint64_t *>(refresh_base + 112u) = addr;
                        break;
                    default:// static instance: VkAccelerationStructureInstanceKHR at offset 8
                        *reinterpret_cast<uint64_t *>(refresh_base + 8u + 56u) = addr;
                        break;
                }
                _all_instance[index].refresh_blas = nullptr;
                --_pending_refresh_count;
            }

            // Apply the modification list on top of the preserved shadow
            // contents; only fields whose flag is set are overwritten (the
            // same semantics as the non-motion update shader).
            for (size_t idx = 0; idx < modifications.size(); idx++) {
                auto &&i = modifications[idx];
                if (i.index >= instance_count) continue;

                // Determine if this modification assigns a MotionInstance with
                // SRT keyframes (matrix-mode motion instances are encoded as
                // static instances, a pre-existing limitation of this path).
                MotionInstance *mi = nullptr;
                if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                    auto prim = reinterpret_cast<PrimitiveBase *>(i.primitive);
                    if (prim && prim->is_motion_instance()) {
                        mi = static_cast<MotionInstance *>(prim);
                    }
                }

                // --- 64-byte standard instance (shader reads) ---
                auto std_inst = reinterpret_cast<VkAccelerationStructureInstanceKHR *>(
                    _std_instance_cache.data() + static_cast<size_t>(i.index) * sizeof(VkAccelerationStructureInstanceKHR));
                if (i.flags & AccelBuildCommand::Modification::flag_transform) {
                    memcpy(&std_inst->transform, i.affine, sizeof(float) * 12);
                }
                if (i.flags & AccelBuildCommand::Modification::flag_user_id) {
                    std_inst->instanceCustomIndex = i.user_id;
                }
                if (i.flags & AccelBuildCommand::Modification::flag_visibility) {
                    std_inst->mask = i.vis_mask;
                }
                if (i.flags & AccelBuildCommand::Modification::flag_opaque_on) {
                    std_inst->flags = (std_inst->flags & ~VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR) |
                                      VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
                } else if (i.flags & AccelBuildCommand::Modification::flag_opaque_off) {
                    std_inst->flags = (std_inst->flags & ~VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR) |
                                      VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
                }
                if ((i.flags & AccelBuildCommand::Modification::flag_primitive) && resolved_meshes[idx]) {
                    auto mesh = resolved_meshes[idx];
                    std_inst->accelerationStructureReference = mesh->get_accel_device_address();
                    resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                             ResourceBarrier::Usage::kAccelInstanceBuffer);
                }

                // --- 160-byte motion instance (TLAS build) ---
                auto inst_base = _motion_instance_cache.data() + static_cast<size_t>(i.index) * kMotionInstanceStride;
                if (mi != nullptr && mi->mode() == AccelMotionMode::SRT && mi->keyframe_count() >= 2u) {
                    // An SRT motion instance (re-)assignment: rewrite the
                    // whole entry from the current keyframes and the
                    // just-updated standard instance fields.
                    *reinterpret_cast<uint32_t *>(inst_base + 0) = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV;
                    *reinterpret_cast<uint32_t *>(inst_base + 4) = 0u; // flags = 0

                    // VkAccelerationStructureSRTMotionInstanceNV layout at offset 8:
                    //   transformT0 (VkSRTDataNV, 64 bytes) at offset 8
                    //   transformT1 (VkSRTDataNV, 64 bytes) at offset 72
                    //   instanceCustomIndex:24 | mask:8 at offset 136
                    //   instanceShaderBindingTableRecordOffset:24 | flags:8 at offset 140
                    //   accelerationStructureReference (uint64) at offset 144
                    auto &keyframes = mi->keyframes();
                    auto &srt0 = keyframes[0].as_srt();
                    auto &srt1 = keyframes[mi->keyframe_count() - 1].as_srt();
                    write_vk_srt(inst_base + 8, srt0);      // transformT0
                    write_vk_srt(inst_base + 8 + 64, srt1); // transformT1

                    // Instance fields after the two SRT transforms
                    auto *srt_inst_fields = inst_base + 8 + 64 + 64; // offset 136
                    *reinterpret_cast<uint32_t *>(srt_inst_fields + 0) =
                        detail::VulkanAccelUpdateInput::pack_index_visibility(
                            std_inst->instanceCustomIndex, std_inst->mask);
                    *reinterpret_cast<uint32_t *>(srt_inst_fields + 4) =
                        detail::VulkanAccelUpdateInput::pack_user_id_flags(
                            0u, static_cast<uint32_t>(std_inst->flags));
                    *reinterpret_cast<uint64_t *>(srt_inst_fields + 8) =
                        std_inst->accelerationStructureReference;

                    if (mi->keyframe_count() > 2u) {
                        LUISA_WARNING("VK_NV_ray_tracing_motion_blur only supports 2 keyframes for SRT motion. "
                                      "Using first and last keyframes (ignoring {} intermediate keyframes).",
                                      mi->keyframe_count() - 2u);
                    }
                } else if (mi != nullptr || !_all_instance[i.index].is_motion_instance) {
                    // Static-encoded instance: (re-)write the entry type and
                    // mirror the just-updated standard instance fields. A slot
                    // whose motion instance was replaced by a plain primitive
                    // flips back to the static encoding here (the slot's
                    // is_motion_instance was already updated by _set_mesh).
                    *reinterpret_cast<uint32_t *>(inst_base + 0) = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV;
                    *reinterpret_cast<uint32_t *>(inst_base + 4) = 0u; // flags = 0
                    memcpy(inst_base + 8, std_inst, sizeof(VkAccelerationStructureInstanceKHR));
                }
                // else: a motion-instance slot touched by a non-primitive
                // modification keeps its cached motion entry (including the
                // keyframes) unchanged.
            }

            // Upload both shadow copies whole.
            auto total_upload_size = motion_upload_size + std_inst_size_bytes;
            auto upload_buf = cmdbuffer.states()->upload_alloc.allocate(total_upload_size, 16);
            auto upload_base = reinterpret_cast<uint8_t *>(
                static_cast<UploadBuffer const *>(upload_buf.buffer)->mapped_ptr()) + upload_buf.offset;
            memcpy(upload_base, _motion_instance_cache.data(), motion_upload_size);
            memcpy(upload_base + motion_upload_size, _std_instance_cache.data(), std_inst_size_bytes);

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
                detail::cmd_copy_buffer(cmdbuffer.cmdbuffer(), device(), &copy_info2);
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
                detail::cmd_copy_buffer(cmdbuffer.cmdbuffer(), device(), &copy_info2);
            }
            _pending_refresh_count = 0;// all pending refreshes were folded into the shadows above
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
        const uint modification_size = static_cast<uint>(modifications.size() + _pending_refresh_count);
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
            std::memcpy(
                inst_ptr->affine.data(), i.affine,
                sizeof(float) * inst_ptr->affine.size());
            inst_ptr->index_visibility =
                TlasInputInst::pack_index_visibility(
                    i.index, i.vis_mask);
            inst_ptr->user_id_flags =
                TlasInputInst::pack_user_id_flags(
                    i.user_id, i.flags);
            if ((i.flags & AccelBuildCommand::Modification::flag_primitive) && resolved_meshes[idx]) {
                auto mesh = resolved_meshes[idx];
                auto addr = mesh->get_accel_device_address();
                inst_ptr->mesh =
                    TlasInputInst::device_address_words(addr);
                resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                         ResourceBarrier::Usage::kAccelInstanceBuffer);
            }
            inst_ptr++;
        }
        // Append the pending refreshes not consumed by a modification: a single
        // contiguous pass over the instance slots (dense index space) replaces
        // the hash-map iteration.
        for (auto index = 0u; index < _all_instance.size(); index++) {
            auto mesh = _all_instance[index].refresh_blas;
            if (mesh == nullptr) continue;
            inst_ptr->index_visibility =
                TlasInputInst::pack_index_visibility(
                    static_cast<uint32_t>(index), 0u);
            inst_ptr->user_id_flags =
                TlasInputInst::pack_user_id_flags(
                    0u,
                    AccelBuildCommand::Modification::flag_primitive);
            resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                     ResourceBarrier::Usage::kAccelInstanceBuffer);
            auto addr = mesh->get_accel_device_address();
            inst_ptr->mesh =
                TlasInputInst::device_address_words(addr);
            ++inst_ptr;
            _all_instance[index].refresh_blas = nullptr;
            --_pending_refresh_count;
        }
        static_cast<UploadBuffer const *>(dsc_buffer.buffer)->copy_from(cache.data(), dsc_buffer.offset, dsc_buffer.size_bytes);
        LUISA_ASSERT(_pending_refresh_count == 0u,
                     "pending refresh accounting desynced ({} left)",
                     _pending_refresh_count);
        _pending_refresh_count = 0;// defensive clear, mirrors the old map clear()
        VkDescriptorBufferInfo arg_buffer_info{
            dsc_buffer.buffer->vk_buffer(),
            dsc_buffer.offset,
            dsc_buffer.size_bytes};
        VkDescriptorBufferInfo buffer_info{
            _instance_buffer->vk_buffer(),
            0,
            _instance_buffer->byte_size()};
        auto local_write_begin = write_desc_sets.size();
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
        LUISA_ASSERT(
            write_desc_sets.size() - local_write_begin ==
                shader->local_descriptor_binding_count(),
            "Vulkan acceleration-update kernel consumed {} local descriptor "
            "bindings but its validated interface requires {}.",
            write_desc_sets.size() - local_write_begin,
            shader->local_descriptor_binding_count());
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
    // The TLAS build dereferences every referenced BLAS through the raw device
    // addresses in the instance buffer (instance AABBs derive from child BLAS
    // contents), so each child BLAS buffer must be synchronized with this build
    // even when the instance list itself is untouched — an in-place BLAS update
    // leaves both `modifications` and the pending-refresh slots empty and would otherwise race
    // with the BLAS build, letting the TLAS pick up stale geometry.
    for (auto &inst : _all_instance) {
        if (inst.handle != nullptr) {
            resource_barrier->record(
                BufferView{inst.handle->mesh->_accel_buffer.get()},
                ResourceBarrier::Usage::kAccelInstanceBuffer);
        }
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
        if (!device()->enable_motion_blur()) [[unlikely]] {
            LUISA_ERROR("TLAS motion requires VK_NV_ray_tracing_motion_blur, "
                        "which is not enabled on this device.");
        }
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
        if (!device()->enable_motion_blur()) [[unlikely]] {
            LUISA_ERROR("TLAS motion requires VK_NV_ray_tracing_motion_blur, "
                        "which is not enabled on this device.");
        }
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
        _built_with_motion = _has_motion;
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
}
void Tlas::_update_mesh(
    MeshHandle *handle) {
    auto instIndex = handle->accel_index;
    LUISA_ASSUME(_all_instance[instIndex].handle == handle);
    // Queue a refresh of the instance's BLAS address. Storing the stable Blas
    // instead of the pooled MeshHandle keeps the entry valid even if the
    // handle is later destroyed/recycled before the next TLAS build.
    _queue_refresh(instIndex, handle->mesh);
    _require_rebuild = true;
}
void Tlas::_queue_refresh(size_t instance_index, Blas *blas) noexcept {
    auto &pending = _all_instance[instance_index].refresh_blas;
    if (pending == nullptr) {
        ++_pending_refresh_count;
    }
    pending = blas;
}
void Tlas::_drop_refresh(size_t instance_index, Blas *blas) noexcept {
    auto &pending = _all_instance[instance_index].refresh_blas;
    if (pending == blas) {
        pending = nullptr;
        --_pending_refresh_count;
    }
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
            // Pending refreshes of removed slots vanish with the Instance
            // structs themselves (resize below); just keep the counter in sync.
            if (i.refresh_blas != nullptr) {
                --_pending_refresh_count;
            }
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
void Tlas::_set_mesh(Blas *mesh, uint64 index, bool is_motion_instance, bool explicit_primitive) {
    auto &&inst = _all_instance[index];
    if (inst.handle != nullptr) {
        if (inst.handle->mesh == mesh) {
            // Same child BLAS: only an explicit primitive assignment may change
            // whether the slot counts as a motion instance — a pending refresh
            // folded into the build must preserve the existing flag.
            if (explicit_primitive) { inst.is_motion_instance = is_motion_instance; }
            return;
        }
        inst.handle->mesh->_remove_accel_ref(inst.handle);
    }
    inst.handle = mesh->_add_accel_ref(this, index);
    inst.handle->accel_index = index;
    inst.is_motion_instance = is_motion_instance;
}
}// namespace lc::vk
