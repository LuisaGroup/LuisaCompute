//
// Created by mike on 3/19/26.
//

#include <algorithm>
#include <cstring>
#include <limits>

#include "hip_check.h"
#include "hip_geometry.h"
#include "hip_command_encoder.h"
#include "hip_stream.h"
#include "hip_accel.h"

namespace luisa::compute::hip {

namespace {

[[nodiscard]] uint64_t hiprt_instance_handle(const hiprtInstance &instance) noexcept {
    switch (instance.type) {
        case hiprtInstanceTypeGeometry:
            return reinterpret_cast<uint64_t>(instance.geometry);
        case hiprtInstanceTypeScene:
            return reinterpret_cast<uint64_t>(instance.scene);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid HIPRT instance type.");
}

[[nodiscard]] bool same_hiprt_instance(
    const hiprtInstance &lhs, const hiprtInstance &rhs) noexcept {
    return lhs.type == rhs.type &&
           hiprt_instance_handle(lhs) == hiprt_instance_handle(rhs);
}

}// namespace

HIPAccel::HIPAccel(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {}

HIPAccel::~HIPAccel() noexcept {
    if (_scene || _instance_buffer || _scene_build_buffer) {
        // Scene builds and traces are asynchronous, while HIPRT destruction and
        // hipFree below have no stream on which to order their deallocation.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
    }
    if (_scene) {
        LUISA_CHECK_HIPRT(hiprtDestroyScene(_hiprt_ctx, _scene));
    }
    if (_instance_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_instance_buffer)));
    }
    if (_scene_build_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_scene_build_buffer)));
    }
}

hiprtSceneBuildInput HIPAccel::_make_scene_build_input(HIPCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_instance_count > 0u && _instance_buffer,
                 "Cannot build a HIPRT scene without instance data.");

    auto hip_stream = encoder.stream()->handle();
    auto n = static_cast<uint32_t>(_instance_count);
    auto instance_bytes = _instance_count * sizeof(hiprtInstance);

    if (_scene_build_capacity < _instance_count) {
        auto aligned_size = [](size_t size, size_t alignment) noexcept {
            return (size + alignment - 1u) & ~(alignment - 1u);
        };
        auto new_capacity = _instance_count;
        auto new_instance_bytes = new_capacity * sizeof(hiprtInstance);
        auto new_frame_bytes = new_capacity * sizeof(hiprtFrameMatrix);
        auto new_mask_bytes = new_capacity * sizeof(uint32_t);
        auto frame_offset = aligned_size(new_instance_bytes, alignof(hiprtFrameMatrix));
        auto mask_offset = aligned_size(frame_offset + new_frame_bytes, alignof(uint32_t));
        auto allocation_size = mask_offset + new_mask_bytes;

        hipDeviceptr_t new_buffer{};
        LUISA_CHECK_HIP(hipMallocAsync(
            reinterpret_cast<void **>(&new_buffer), allocation_size, hip_stream));
        // hiprtFrameMatrix includes a time field and tail padding outside the
        // affine matrix copied below. Initialize the allocation once so every
        // persistent frame remains a zero-time static transform.
        LUISA_CHECK_HIP(hipMemsetAsync(
            reinterpret_cast<void *>(new_buffer), 0, allocation_size, hip_stream));
        if (_scene_build_buffer) {
            LUISA_CHECK_HIP(hipFreeAsync(
                reinterpret_cast<void *>(_scene_build_buffer), hip_stream));
        }
        auto new_buffer_bytes = reinterpret_cast<std::byte *>(new_buffer);
        _scene_build_buffer = new_buffer;
        _scene_build_capacity = new_capacity;
        _scene_instances = new_buffer;
        _scene_frames = reinterpret_cast<hipDeviceptr_t>(new_buffer_bytes + frame_offset);
        _scene_masks = reinterpret_cast<hipDeviceptr_t>(new_buffer_bytes + mask_offset);
        _hiprt_instances_dirty = true;
    }

    if (_hiprt_instances_dirty) {
        encoder.with_upload_buffer(instance_bytes, [&](auto upload_buffer) noexcept {
            std::memcpy(upload_buffer->address(), _hiprt_instances.data(), instance_bytes);
            LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                _scene_instances, upload_buffer->address(), instance_bytes, hip_stream));
        });
        _hiprt_instances_dirty = false;
    }

    // GPU-side instance mutations update CodegenInstance directly. Gather the
    // transform and visibility fields from that authoritative device buffer so
    // both HIPRT refits and rebuilds observe all preceding shader writes.
    auto instance_data = reinterpret_cast<const std::byte *>(_instance_buffer);
    LUISA_CHECK_HIP(hipMemcpy2DAsync(
        reinterpret_cast<void *>(_scene_frames), sizeof(hiprtFrameMatrix),
        instance_data + offsetof(CodegenInstance, affine), sizeof(CodegenInstance),
        sizeof(CodegenInstance::affine), n, hipMemcpyDeviceToDevice, hip_stream));
    LUISA_CHECK_HIP(hipMemcpy2DAsync(
        reinterpret_cast<void *>(_scene_masks), sizeof(uint32_t),
        instance_data + offsetof(CodegenInstance, visibility_mask), sizeof(CodegenInstance),
        sizeof(uint32_t), n, hipMemcpyDeviceToDevice, hip_stream));

    hiprtSceneBuildInput input{};
    input.instanceCount = n;
    input.frameCount = n;
    input.frameType = hiprtFrameTypeMatrix;
    input.instances = reinterpret_cast<hiprtDevicePtr>(_scene_instances);
    input.instanceFrames = reinterpret_cast<hiprtDevicePtr>(_scene_frames);
    input.instanceTransformHeaders = nullptr;
    input.instanceMasks = reinterpret_cast<hiprtDevicePtr>(_scene_masks);
    return input;
}

void HIPAccel::_build(HIPCommandEncoder &encoder) noexcept {
    auto hip_stream = encoder.stream()->handle();

    // HIPRT scene storage is sized by instance count. A compacted allocation
    // has no spare capacity for a fresh full build and must be recreated.
    auto recreate_scene =
        _scene == nullptr || _scene_instance_count != _instance_count ||
        _option.allow_compaction;
    if (_scene && recreate_scene) {
        // Finish earlier same-stream scene use before HIPRT frees the old scene.
        LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
        LUISA_CHECK_HIPRT(hiprtDestroyScene(_hiprt_ctx, _scene));
        _scene = nullptr;
    }

    auto build_input = _make_scene_build_input(encoder);

    hiprtBuildOptions build_options{};
    build_options.buildFlags = make_hiprt_build_flags(_option);

    if (!_scene) {
        LUISA_CHECK_HIPRT(hiprtCreateScene(_hiprt_ctx, build_input,
                                           build_options, _scene));
        _scene_instance_count = _instance_count;
    }

    size_t temp_size = 0;
    LUISA_CHECK_HIPRT(hiprtGetSceneBuildTemporaryBufferSize(_hiprt_ctx, build_input, build_options, temp_size));
    auto temp_buffer =
        encoder.stream()->rt_scratch_buffer(temp_size);

    LUISA_CHECK_HIPRT(hiprtBuildScene(_hiprt_ctx, hiprtBuildOperationBuild,
                                      build_input, build_options,
                                      temp_buffer, hip_stream, _scene));

    if (_option.allow_compaction) {
        LUISA_CHECK_HIPRT(hiprtCompactScene(
            _hiprt_ctx, hip_stream, _scene, _scene));
    }
}

void HIPAccel::_update(HIPCommandEncoder &encoder) noexcept {
    auto hip_stream = encoder.stream()->handle();
    auto build_input = _make_scene_build_input(encoder);

    hiprtBuildOptions build_options{};
    build_options.buildFlags = make_hiprt_build_flags(_option);

    LUISA_CHECK_HIPRT(hiprtBuildScene(_hiprt_ctx, hiprtBuildOperationUpdate,
                                      build_input, build_options,
                                      nullptr, hip_stream, _scene));
}

void HIPAccel::build(HIPCommandEncoder &encoder, AccelBuildCommand *command) noexcept {

    std::scoped_lock lock{_mutex};

    auto hip_stream = encoder.stream()->handle();
    auto instance_count = command->instance_count();
    LUISA_ASSERT(instance_count > 0u &&
                     instance_count <= std::numeric_limits<uint32_t>::max(),
                 "HIP acceleration structures require a nonzero 32-bit instance count.");

    auto old_instance_count = _instance_count;
    auto instance_count_changed = _instance_count != instance_count;
    if (instance_count_changed) {
        _instance_count = instance_count;
        _host_instances.resize(instance_count);
        _hiprt_instances.resize(instance_count);
        _primitives.resize(instance_count);
        _requires_rebuild = true;
    }

    auto required_size = instance_count * sizeof(CodegenInstance);
    if (_instance_buffer_size < required_size) {
        hipDeviceptr_t new_instance_buffer{};
        LUISA_CHECK_HIP(hipMallocAsync(
            reinterpret_cast<void **>(&new_instance_buffer), required_size, hip_stream));
        LUISA_CHECK_HIP(hipMemsetAsync(
            reinterpret_cast<void *>(new_instance_buffer), 0, required_size, hip_stream));
        if (_instance_buffer) {
            auto copy_count = std::min(old_instance_count, static_cast<size_t>(instance_count));
            if (copy_count > 0u) {
                LUISA_CHECK_HIP(hipMemcpyDtoDAsync(
                    new_instance_buffer, _instance_buffer,
                    copy_count * sizeof(CodegenInstance), hip_stream));
            }
            LUISA_CHECK_HIP(hipFreeAsync(
                reinterpret_cast<void *>(_instance_buffer), hip_stream));
        }
        _instance_buffer = new_instance_buffer;
        _instance_buffer_size = required_size;
    } else if (instance_count > old_instance_count) {
        auto tail = reinterpret_cast<std::byte *>(_instance_buffer) +
                    old_instance_count * sizeof(CodegenInstance);
        LUISA_CHECK_HIP(hipMemsetAsync(
            tail, 0,
            (instance_count - old_instance_count) * sizeof(CodegenInstance),
            hip_stream));
    }

    auto mods = command->modifications();
    for (auto &m : mods) {
        auto idx = m.index;
        LUISA_ASSERT(idx < instance_count, "Modification index out of range.");
        auto &inst = _host_instances[idx];
        auto &hiprt_inst = _hiprt_instances[idx];

        if (m.flags & AccelBuildCommand::Modification::flag_primitive) {
            _requires_rebuild = true;
            _hiprt_instances_dirty = true;
            auto primitive = reinterpret_cast<const HIPPrimitive *>(m.primitive);
            _primitives[idx] = primitive;
            auto binding = primitive->binding();
            LUISA_ASSERT(hiprt_instance_handle(binding.instance) != 0u,
                         "Cannot bind an unbuilt HIP primitive to an acceleration structure.");
            hiprt_inst = binding.instance;
            inst.sbt_offset = static_cast<uint32_t>(binding.kind);
            inst.mesh_handle = binding.codegen_handle;
            inst.motion_data = binding.motion_data;
        }

        if (m.flags & AccelBuildCommand::Modification::flag_transform) {
            std::memcpy(inst.affine, m.affine, 12 * sizeof(float));
        }

        if (m.flags & AccelBuildCommand::Modification::flag_visibility) {
            inst.visibility_mask = m.vis_mask;
        }

        if (m.flags & AccelBuildCommand::Modification::flag_user_id) {
            inst.user_id = m.user_id;
        }

        if (m.flags & AccelBuildCommand::Modification::flag_opaque_on) {
            inst.flags |= CodegenInstance::flag_opaque;
        } else if (m.flags & AccelBuildCommand::Modification::flag_opaque_off) {
            inst.flags &= ~CodegenInstance::flag_opaque;
        }
    }

    // A geometry or nested scene can recreate its HIPRT handle while retaining
    // the same runtime resource. Refresh cached bindings so a TLAS update never
    // retains a destroyed child handle or stale delegated metadata.
    luisa::vector<uint32_t> primitive_metadata_updates;
    for (auto i = 0u; i < instance_count; i++) {
        auto primitive = _primitives[i];
        LUISA_ASSERT(primitive != nullptr,
                     "HIP acceleration structure instance {} has no primitive.", i);
        auto binding = primitive->binding();
        LUISA_ASSERT(hiprt_instance_handle(binding.instance) != 0u,
                     "HIP acceleration structure instance {} refers to an unbuilt primitive.", i);
        if (!same_hiprt_instance(_hiprt_instances[i], binding.instance)) {
            _hiprt_instances[i] = binding.instance;
            _requires_rebuild = true;
            _hiprt_instances_dirty = true;
        }
        auto primitive_kind = static_cast<uint32_t>(binding.kind);
        if (_host_instances[i].sbt_offset != primitive_kind ||
            _host_instances[i].mesh_handle != binding.codegen_handle ||
            _host_instances[i].motion_data != binding.motion_data) {
            _host_instances[i].sbt_offset = primitive_kind;
            _host_instances[i].mesh_handle = binding.codegen_handle;
            _host_instances[i].motion_data = binding.motion_data;
            primitive_metadata_updates.emplace_back(i);
        }
    }

    // Primitive kind/data are not shader-mutable, so refreshing just these two
    // fields cannot overwrite device-authoritative transform or mask state.
    if (!primitive_metadata_updates.empty()) {
        auto kind_bytes = primitive_metadata_updates.size() * sizeof(uint32_t);
        auto data_offset = (kind_bytes + alignof(uint64_t) - 1u) &
                           ~(alignof(uint64_t) - 1u);
        auto motion_offset = data_offset +
                             primitive_metadata_updates.size() * sizeof(uint64_t);
        auto upload_size = motion_offset +
                           primitive_metadata_updates.size() * sizeof(uint64_t);
        encoder.with_upload_buffer(upload_size, [&](auto upload_buffer) noexcept {
            auto staging = static_cast<std::byte *>(upload_buffer->address());
            for (auto i = 0u; i < primitive_metadata_updates.size(); i++) {
                auto instance_index = primitive_metadata_updates[i];
                auto &instance = _host_instances[instance_index];
                std::memcpy(staging + i * sizeof(uint32_t),
                            &instance.sbt_offset, sizeof(instance.sbt_offset));
                std::memcpy(staging + data_offset + i * sizeof(uint64_t),
                            &instance.mesh_handle, sizeof(instance.mesh_handle));
                std::memcpy(staging + motion_offset + i * sizeof(uint64_t),
                            &instance.motion_data, sizeof(instance.motion_data));
                auto device_instance = reinterpret_cast<std::byte *>(_instance_buffer) +
                                       instance_index * sizeof(CodegenInstance);
                LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                    reinterpret_cast<hipDeviceptr_t>(
                        device_instance + offsetof(CodegenInstance, sbt_offset)),
                    staging + i * sizeof(uint32_t), sizeof(uint32_t), hip_stream));
                LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                    reinterpret_cast<hipDeviceptr_t>(
                        device_instance + offsetof(CodegenInstance, mesh_handle)),
                    staging + data_offset + i * sizeof(uint64_t),
                    sizeof(uint64_t), hip_stream));
                LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                    reinterpret_cast<hipDeviceptr_t>(
                        device_instance + offsetof(CodegenInstance, motion_data)),
                    staging + motion_offset + i * sizeof(uint64_t),
                    sizeof(uint64_t), hip_stream));
            }
        });
    }

    // Upload only fields explicitly changed by the host. Copying whole stale
    // host descriptors would overwrite transform/mask/opacity/user-id values
    // written by an earlier GPU dispatch.
    if (!mods.empty()) {
        _sorted_modifications.clear();
        _sorted_modifications.reserve(mods.size());
        for (auto &m : mods) { _sorted_modifications.emplace_back(&m); }
        std::sort(_sorted_modifications.begin(), _sorted_modifications.end(),
                  [](auto lhs, auto rhs) noexcept { return lhs->index < rhs->index; });
        encoder.with_upload_buffer(
            mods.size() * sizeof(CodegenInstance),
            [&](auto upload_buffer) noexcept {
                auto staging = static_cast<std::byte *>(upload_buffer->address());
                auto device_instances = reinterpret_cast<std::byte *>(_instance_buffer);
                for (auto i = 0u; i < _sorted_modifications.size(); i++) {
                    auto m = _sorted_modifications[i];
                    auto &host_instance = _host_instances[m->index];
                    auto staging_instance = staging + i * sizeof(CodegenInstance);
                    std::memcpy(staging_instance, &host_instance, sizeof(CodegenInstance));
                }
                auto upload_field = [&](uint32_t required_flags,
                                        size_t offset, size_t size) noexcept {
                    for (auto begin = 0u; begin < _sorted_modifications.size();) {
                        while (begin < _sorted_modifications.size() &&
                               !(_sorted_modifications[begin]->flags & required_flags)) {
                            begin++;
                        }
                        if (begin == _sorted_modifications.size()) { break; }
                        auto end = begin + 1u;
                        while (end < _sorted_modifications.size() &&
                               (_sorted_modifications[end]->flags & required_flags) &&
                               _sorted_modifications[end]->index == _sorted_modifications[end - 1u]->index + 1u) {
                            end++;
                        }
                        auto device_field = device_instances +
                                            _sorted_modifications[begin]->index * sizeof(CodegenInstance) + offset;
                        auto staging_field = staging + begin * sizeof(CodegenInstance) + offset;
                        if (end == begin + 1u) {
                            LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                                reinterpret_cast<hipDeviceptr_t>(device_field),
                                staging_field, size, hip_stream));
                        } else {
                            LUISA_CHECK_HIP(hipMemcpy2DAsync(
                                device_field, sizeof(CodegenInstance),
                                staging_field, sizeof(CodegenInstance),
                                size, end - begin, hipMemcpyHostToDevice, hip_stream));
                        }
                        begin = end;
                    }
                };
                using Mod = AccelBuildCommand::Modification;
                upload_field(Mod::flag_primitive,
                             offsetof(CodegenInstance, sbt_offset),
                             sizeof(CodegenInstance::sbt_offset));
                upload_field(Mod::flag_primitive,
                             offsetof(CodegenInstance, mesh_handle),
                             sizeof(CodegenInstance::mesh_handle));
                upload_field(Mod::flag_primitive,
                             offsetof(CodegenInstance, motion_data),
                             sizeof(CodegenInstance::motion_data));
                upload_field(Mod::flag_transform,
                             offsetof(CodegenInstance, affine),
                             sizeof(CodegenInstance::affine));
                upload_field(Mod::flag_visibility,
                             offsetof(CodegenInstance, visibility_mask),
                             sizeof(CodegenInstance::visibility_mask));
                upload_field(Mod::flag_user_id,
                             offsetof(CodegenInstance, user_id),
                             sizeof(CodegenInstance::user_id));
                upload_field(Mod::flag_opaque,
                             offsetof(CodegenInstance, flags),
                             sizeof(CodegenInstance::flags));
            });
    }

    _requires_rebuild = _requires_rebuild ||
                        command->request() == AccelBuildRequest::FORCE_BUILD ||
                        !_option.allow_update ||
                        _scene == nullptr;

    if (!command->update_instance_buffer_only()) {
        // Device-side motion-key writes target the public frame input, while
        // HIPRT traversal consumes a private copy owned by each nested scene.
        // Refit every distinct motion primitive on the same stream before the
        // outer scene observes its bounds. Static primitives implement this as
        // a no-op.
        luisa::vector<const HIPPrimitive *> prepared_primitives;
        prepared_primitives.reserve(_primitives.size());
        for (auto primitive : _primitives) {
            if (std::find(prepared_primitives.cbegin(),
                          prepared_primitives.cend(), primitive) ==
                prepared_primitives.cend()) {
                primitive->prepare_for_tlas_build(encoder);
                prepared_primitives.emplace_back(primitive);
            }
        }
        if (_requires_rebuild) {
            _build(encoder);
        } else {
            _update(encoder);
        }
        _requires_rebuild = false;
    }
}

HIPAccel::Binding HIPAccel::binding() const noexcept {
    std::scoped_lock lock{_mutex};
    return Binding{
        reinterpret_cast<uint64_t>(_scene),
        reinterpret_cast<uint64_t>(_instance_buffer)};
}

}// namespace luisa::compute::hip
