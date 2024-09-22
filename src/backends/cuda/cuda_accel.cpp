#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include "optix_api.h"
#include "cuda_error.h"
#include "cuda_stream.h"
#include "cuda_command_encoder.h"
#include "cuda_device.h"
#include "cuda_curve.h"
#include "cuda_motion_instance.h"
#include "cuda_accel.h"

namespace luisa::compute::cuda {

[[nodiscard]] inline optix::BuildInput cuda_accel_build_inputs(uint64_t instance_buffer,
                                                               uint instance_count) noexcept {
    optix::BuildInput input{};
    input.type = optix::BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = instance_buffer;
    input.instanceArray.numInstances = instance_count;
    return input;
}

void CUDAAccel::_build(CUDACommandEncoder &encoder) noexcept {

    if (!_name.empty()) { nvtxRangePushA(luisa::format("{}::build", _name).c_str()); }

    // build IAS
    auto instance_count = static_cast<uint>(_primitives.size());
    auto build_input = cuda_accel_build_inputs(_instance_buffer, instance_count);
    auto build_options = make_optix_build_options(_option, optix::BUILD_OPERATION_BUILD);
    auto optix_ctx = encoder.stream()->device()->handle().optix_context();
    auto cuda_stream = encoder.stream()->handle();

    Clock clock;
    optix::AccelBufferSizes sizes{};
    LUISA_CHECK_OPTIX(optix::api().accelComputeMemoryUsage(
        optix_ctx, &build_options, &build_input, 1u, &sizes));
    _update_buffer_size = sizes.tempUpdateSizeInBytes;
    // LUISA_VERBOSE("Computed accel memory usage in {} ms: "
    //               "temp = {}, temp_update = {}, output = {}.",
    //               clock.toc(), sizes.tempSizeInBytes,
    //               sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);

    if (_option.allow_compaction) {// with compaction

        static constexpr auto align = [](size_t x) noexcept {
            constexpr auto a = optix::ACCEL_BUFFER_BYTE_ALIGNMENT;
            return (x + a - 1u) / a * a;
        };

        // compute the required buffer sizes
        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);

        // allocate buffers
        auto build_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&build_buffer, build_buffer_size, cuda_stream));
        auto compacted_size_buffer = build_buffer + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer + temp_buffer_offset;
        auto output_buffer = build_buffer + output_buffer_offset;

        // build the acceleration structure and query the compacted size
        if (!_name.empty()) { nvtxRangePushA("build"); }
        optix::AccelEmitDesc emit_desc{};
        emit_desc.type = optix::PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream, &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes, output_buffer,
            sizes.outputSizeInBytes, &_handle, &emit_desc, 1u));
        if (!_name.empty()) { nvtxRangePop(); }

        // read back the compacted size
        size_t compacted_size;
        encoder.with_download_pool_no_fallback(sizeof(size_t), [&](auto temp) noexcept {
            if (temp) {
                LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(temp->address(), compacted_size_buffer, sizeof(size_t), cuda_stream));
                LUISA_CHECK_CUDA(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(&compacted_size),
                                               reinterpret_cast<CUdeviceptr>(temp->address()),
                                               sizeof(size_t), cuda_stream));
            } else {
                LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
            }
        });
        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
        // LUISA_VERBOSE("CUDAAccel compaction: before = {}B, after = {}B, ratio = {}.",
        //               sizes.outputSizeInBytes, compacted_size,
        //               compacted_size / static_cast<double>(sizes.outputSizeInBytes));

        // do compaction
        if (_bvh_buffer_size < compacted_size) {
            _bvh_buffer_size = compacted_size;
            if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer, _bvh_buffer_size, cuda_stream));
        }
        if (!_name.empty()) { nvtxRangePushA("compact"); }
        LUISA_CHECK_OPTIX(optix::api().accelCompact(
            optix_ctx, cuda_stream, _handle, _bvh_buffer, _bvh_buffer_size, &_handle));
        if (!_name.empty()) { nvtxRangePop(); }
        LUISA_CHECK_CUDA(cuMemFreeAsync(build_buffer, cuda_stream));
    } else {// without compaction
        // re-allocate buffers if necessary
        if (_bvh_buffer_size < sizes.outputSizeInBytes) {
            _bvh_buffer_size = sizes.outputSizeInBytes;
            if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer, _bvh_buffer_size, cuda_stream));
        }
        // allocate the temporary update buffer
        auto temp_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp_buffer, sizes.tempSizeInBytes, cuda_stream));
        // perform the build
        if (!_name.empty()) { nvtxRangePushA("build"); }
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream, &build_options, &build_input, 1, temp_buffer,
            sizes.tempSizeInBytes, _bvh_buffer, _bvh_buffer_size, &_handle, nullptr, 0u));
        if (!_name.empty()) { nvtxRangePop(); }
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, cuda_stream));
    }

    if (!_name.empty()) { nvtxRangePop(); }
}

void CUDAAccel::_update(CUDACommandEncoder &encoder) noexcept {
    if (!_name.empty()) { nvtxRangePushA(luisa::format("{}::update", _name).c_str()); }
    auto instance_count = static_cast<uint>(_primitives.size());
    auto build_input = cuda_accel_build_inputs(_instance_buffer, instance_count);
    auto build_options = make_optix_build_options(_option, optix::BUILD_OPERATION_UPDATE);
    auto cuda_stream = encoder.stream()->handle();
    auto update_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, _update_buffer_size, cuda_stream));
    LUISA_CHECK_OPTIX(optix::api().accelBuild(
        encoder.stream()->device()->handle().optix_context(), cuda_stream,
        &build_options, &build_input, 1u, update_buffer, _update_buffer_size,
        _bvh_buffer, _bvh_buffer_size, &_handle, nullptr, 0u));
    LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
    if (!_name.empty()) { nvtxRangePop(); }
}

CUDAAccel::CUDAAccel(const AccelOption &option) noexcept
    : _option{option} {}

CUDAAccel::~CUDAAccel() noexcept {
    if (_instance_buffer) { LUISA_CHECK_CUDA(cuMemFree(_instance_buffer)); }
    if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer)); }
}

void CUDAAccel::build(CUDACommandEncoder &encoder, AccelBuildCommand *command) noexcept {

    std::scoped_lock lock{_mutex};

    // prepare instance buffer
    auto cuda_stream = encoder.stream()->handle();// the worker stream has to be pinned for dependencies
    auto instance_count = command->instance_count();
    LUISA_ASSERT(instance_count > 0u, "Instance count must be greater than 0.");
    if (auto size = instance_count * sizeof(optix::Instance); _instance_buffer_size < size) {
        auto old_instance_buffer = _instance_buffer;
        auto new_instance_buffer_size = next_pow2(size);
        LUISA_CHECK_CUDA(cuMemAllocAsync(&_instance_buffer, new_instance_buffer_size, cuda_stream));
        if (old_instance_buffer) {
            LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(
                _instance_buffer, old_instance_buffer, _instance_buffer_size, cuda_stream));
            LUISA_CHECK_CUDA(cuMemFreeAsync(old_instance_buffer, cuda_stream));
        }
        _instance_buffer_size = new_instance_buffer_size;
    }
    auto instance_count_changed = _primitives.size() != instance_count;
    _primitives.resize(instance_count);
    _prim_handles.resize(instance_count);
    // update the instance buffer
    auto mods = command->modifications();
    if (auto n = static_cast<uint32_t>(mods.size())) {
        using Mod = AccelBuildCommand::Modification;
        auto update_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, mods.size_bytes(), cuda_stream));
        encoder.with_upload_buffer(mods.size_bytes(), [&](auto host_update_buffer) noexcept {
            auto host_updates = reinterpret_cast<Mod *>(host_update_buffer->address());
            for (auto i = 0u; i < n; i++) {
                auto m = mods[i];
                if (m.flags & Mod::flag_primitive) {
                    _requires_rebuild = true;
                    static constexpr auto mod_flag_procedural = 1u << 8u;
                    static constexpr auto mod_flag_curve_piecewise_linear = 1u << 9u;
                    static constexpr auto mod_flag_curve_cubic_bspline = 1u << 10u;
                    static constexpr auto mod_flag_curve_catmull_rom = 1u << 11u;
                    static constexpr auto mod_flag_curve_bezier = 1u << 12u;
                    auto prim = reinterpret_cast<const CUDAPrimitiveBase *>(m.primitive);
                    _primitives[m.index] = prim;
                    auto handle = prim->handle();
                    m.primitive = handle;
                    _prim_handles[m.index] = handle;
                    auto mark_prim_mod_flag = [&m](auto p) noexcept {
                        if (p->tag() == CUDAPrimitive::Tag::PROCEDURAL) {
                            m.flags |= mod_flag_procedural;
                        } else if (p->tag() == CUDAPrimitive::Tag::CURVE) {
                            auto curve = static_cast<const CUDACurve *>(p);
                            switch (auto basis = curve->basis()) {
                                case optix::PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
                                    m.flags |= mod_flag_curve_cubic_bspline;
                                    break;
                                case optix::PRIMITIVE_TYPE_ROUND_LINEAR:
                                    m.flags |= mod_flag_curve_piecewise_linear;
                                    break;
                                case optix::PRIMITIVE_TYPE_ROUND_CATMULLROM:
                                    m.flags |= mod_flag_curve_catmull_rom;
                                    break;
                                case optix::PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER:
                                    m.flags |= mod_flag_curve_bezier;
                                    break;
                                default: LUISA_ERROR_WITH_LOCATION(
                                    "Invalid curve type (0x{:x}).", static_cast<uint64_t>(basis));
                            }
                        }
                    };
                    if (prim->tag() == CUDAPrimitiveBase::Tag::MOTION_INSTANCE) {
                        auto instance = static_cast<const CUDAMotionInstance *>(prim);
                        auto child = instance->child();
                        mark_prim_mod_flag(child);
                    } else {
                        mark_prim_mod_flag(prim);
                    }
                }
                host_updates[i] = m;
            }
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                update_buffer, host_updates, mods.size_bytes(), cuda_stream));
        });
        auto update_kernel = encoder.stream()->device()->accel_update_function();
        std::array<void *, 3u> args{&_instance_buffer, &update_buffer, &n};
        constexpr auto block_size = 256u;
        auto block_count = (n + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            update_kernel, block_count, 1u, 1u, block_size, 1u, 1u,
            0u, cuda_stream, args.data(), nullptr));
        LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
    }

    // check if any primitive handle changed due to
    // rebuild but not presented in modifications
    auto changed_handle_count = 0u;
    auto any_motion_instance_child_changed = false;
    for (auto i = 0u; i < instance_count; i++) {
        if (_primitives[i]->handle() != _prim_handles[i]) {
            changed_handle_count++;
        }
        if (_primitives[i]->tag() == CUDAPrimitiveBase::Tag::MOTION_INSTANCE) {
            auto instance = static_cast<const CUDAMotionInstance *>(_primitives[i]);
            auto child = instance->child();
            if (auto iter = _motion_instance_to_primitive.find(instance);
                iter == _motion_instance_to_primitive.end() || iter->second != child) {
                any_motion_instance_child_changed = true;
            }
        }
    }

    // find out whether we really need to build (or rebuild) the BVH
    _requires_rebuild = _requires_rebuild /* pending rebuild */ ||
                        command->request() == AccelBuildRequest::FORCE_BUILD /* user requires rebuilding */ ||
                        !_option.allow_update /* update is not allowed in the accel */ ||
                        _handle == 0u /* the accel is not yet built */ ||
                        instance_count_changed /* instance count changed */ ||
                        changed_handle_count > 0u /* additional handle changes due to rebuild */ ||
                        any_motion_instance_child_changed /* motion instance child changed */;

    // rebuild the motion instance to primitive map if necessary
    if (any_motion_instance_child_changed) {
        _motion_instance_to_primitive.clear();
        for (auto i = 0u; i < instance_count; i++) {
            if (_primitives[i]->tag() == CUDAPrimitiveBase::Tag::MOTION_INSTANCE) {
                auto instance = static_cast<const CUDAMotionInstance *>(_primitives[i]);
                auto child = instance->child();
                _motion_instance_to_primitive.emplace(instance, child);
            }
        }
    }

    // gather changed handles if any
    if (changed_handle_count > 0u) {
        struct alignas(16u) ChangedHandle {
            size_t index;
            optix::TraversableHandle handle;
        };
        auto size_bytes = changed_handle_count * sizeof(ChangedHandle);
        auto device_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&device_buffer, size_bytes, cuda_stream));
        encoder.with_upload_buffer(size_bytes, [&](auto host_buffer) noexcept {
            auto host_handles = reinterpret_cast<ChangedHandle *>(host_buffer->address());
            for (auto i = 0u, j = 0u; i < instance_count; i++) {
                if (auto handle = _primitives[i]->handle(); handle != _prim_handles[i]) {
                    host_handles[j++] = {i, handle};
                    _prim_handles[i] = handle;
                }
            }
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(device_buffer, host_handles, size_bytes, cuda_stream));
        });
        auto kernel = encoder.stream()->device()->instance_handle_update_function();
        std::array<void *, 3u> args{&_instance_buffer, &device_buffer, &changed_handle_count};
        constexpr auto block_size = 256u;
        auto block_count = (changed_handle_count + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            kernel, block_count, 1u, 1u, block_size, 1u, 1u,
            0u, cuda_stream, args.data(), nullptr));
        LUISA_CHECK_CUDA(cuMemFreeAsync(device_buffer, cuda_stream));
    }
    if (!command->update_instance_buffer_only()) {// build or update the BVH
        if (_requires_rebuild) {
            _build(encoder);
        } else {
            _update(encoder);
        }
        // reset the flag
        _requires_rebuild = false;
    }
}

void CUDAAccel::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_mutex};
    _name = std::move(name);
}

optix::TraversableHandle CUDAAccel::handle() const noexcept {
    std::scoped_lock lock{_mutex};
    return _handle;
}

CUdeviceptr CUDAAccel::instance_buffer() const noexcept {
    std::scoped_lock lock{_mutex};
    return _instance_buffer;
}

CUDAAccel::Binding CUDAAccel::binding() const noexcept {
    std::scoped_lock lock{_mutex};
    return Binding{_handle, _instance_buffer};
}

}// namespace luisa::compute::cuda
