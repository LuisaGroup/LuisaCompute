//
// Created by Mike on 2021/12/2.
//

#include <core/clock.h>

#include <backends/cuda/optix_api.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_accel.h>

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
    LUISA_INFO("Computed accel memory usage in {} ms: "
               "temp = {}, temp_update = {}, output = {}.",
               clock.toc(), sizes.tempSizeInBytes,
               sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);

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
        optix::AccelEmitDesc emit_desc{};
        emit_desc.type = optix::PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream, &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes, output_buffer,
            sizes.outputSizeInBytes, &_handle, &emit_desc, 1u));
        // read back the compacted size
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
        LUISA_INFO("CUDAAccel compaction: before = {}B, after = {}B, ratio = {}.",
                   sizes.outputSizeInBytes, compacted_size,
                   compacted_size / static_cast<double>(sizes.outputSizeInBytes));
        // do compaction
        if (_bvh_buffer_size < compacted_size) {
            _bvh_buffer_size = compacted_size;
            if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer, _bvh_buffer_size, cuda_stream));
        }
        LUISA_CHECK_OPTIX(optix::api().accelCompact(
            optix_ctx, cuda_stream, _handle, _bvh_buffer, _bvh_buffer_size, &_handle));
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
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream, &build_options, &build_input, 1, temp_buffer,
            sizes.tempSizeInBytes, _bvh_buffer, _bvh_buffer_size, &_handle, nullptr, 0u));
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, cuda_stream));
    }
}

void CUDAAccel::_update(CUDACommandEncoder &encoder) noexcept {
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
}

CUDAAccel::CUDAAccel(const AccelOption &option) noexcept
    : _option{option} {}

CUDAAccel::~CUDAAccel() noexcept {
    if (_instance_buffer) { LUISA_CHECK_CUDA(cuMemFree(_instance_buffer)); }
    if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer)); }
}

void CUDAAccel::build(CUDACommandEncoder &encoder, AccelBuildCommand *command) noexcept {
    // prepare instance buffer
    auto cuda_stream = encoder.stream()->handle();// the worker stream has to be pinned for dependencies
    auto instance_count = command->instance_count();
    if (auto size = instance_count * sizeof(optix::Instance); _instance_buffer_size < size) {
        _instance_buffer_size = next_pow2(size);
        if (_instance_buffer) { LUISA_CHECK_CUDA(cuMemFreeAsync(_instance_buffer, cuda_stream)); }
        LUISA_CHECK_CUDA(cuMemAllocAsync(&_instance_buffer, _instance_buffer_size, cuda_stream));
    }
    _primitives.resize(instance_count);
    // update the instance buffer
    auto mods = command->modifications();
    if (auto n = mods.size()) {
        using Mod = AccelBuildCommand::Modification;
        auto update_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, mods.size_bytes(), cuda_stream));
        encoder.with_upload_buffer(mods.size_bytes(), [&](auto host_update_buffer) noexcept {
            auto host_updates = reinterpret_cast<Mod *>(host_update_buffer->address());
            for (auto i = 0u; i < n; i++) {
                auto m = mods[i];
                if (m.flags & Mod::flag_primitive) {
                    auto prim = reinterpret_cast<const CUDAPrimitive *>(m.primitive);
                    _primitives[i] = prim;
                    m.primitive = prim->handle();
                }
                host_updates[i] = m;
            }
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(update_buffer, host_updates, mods.size_bytes(), cuda_stream));
        });
        auto update_kernel = encoder.stream()->device()->accel_update_function();
        std::array<void *, 3u> args{&_instance_buffer, &update_buffer, &n};
        constexpr auto block_size = 1024u;
        auto block_count = (n + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            update_kernel, block_count, 1u, 1u, block_size, 1u, 1u,
            0u, cuda_stream, args.data(), nullptr));
        LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
    }
    // find out whether we really need to build (or rebuild) the BVH
    auto requires_build = command->request() == AccelBuildRequest::FORCE_BUILD /* user requires rebuilding */ ||
                          !_option.allow_update /* update is not allowed in the accel */ ||
                          _handle == 0u /* the accel is not yet built */ ||
                          _prim_handles.size() != instance_count;// number of instances changed
    // check if any primitive handle changed
    _prim_handles.resize(instance_count);
    for (auto i = 0u; i < instance_count; i++) {
        auto prim = _primitives[i];
        requires_build |= prim->handle() != _prim_handles[i];// primitive handle changed
        _prim_handles[i] = prim->handle();
    }
    // perform the build or update
    if (requires_build) {
        _build(encoder);
    } else {
        _update(encoder);
    }
}

}// namespace luisa::compute::cuda
