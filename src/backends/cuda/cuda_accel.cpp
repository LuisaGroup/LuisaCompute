//
// Created by Mike on 2021/12/2.
//

#include <optix_stubs.h>

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_accel.h>

namespace luisa::compute::cuda {

CUDAAccel::CUDAAccel(AccelUsageHint hint) noexcept : _build_hint{hint} {}

CUDAAccel::~CUDAAccel() noexcept {
    if (_instance_buffer) { LUISA_CHECK_CUDA(cuMemFree(_instance_buffer)); }
    if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer)); }
}

[[nodiscard]] inline auto cuda_accel_build_options(AccelUsageHint hint, OptixBuildOperation op) noexcept {
    OptixAccelBuildOptions build_options{};
    build_options.operation = op;
    switch (hint) {
        case AccelUsageHint::FAST_TRACE:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                       OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            break;
        case AccelUsageHint::FAST_UPDATE:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                       OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            break;
        case AccelUsageHint::FAST_BUILD:
            build_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            break;
    }
    return build_options;
}

[[nodiscard]] inline OptixBuildInput cuda_accel_build_inputs(
    uint64_t instance_buffer, uint instance_count) noexcept {
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = instance_buffer;
    input.instanceArray.numInstances = instance_count;
    return input;
}

void CUDAAccel::build(CUDADevice *device, CUDAStream *stream, const AccelBuildCommand *command) noexcept {
    // prepare instance buffer
    auto cuda_stream = stream->handle();// the worker stream has to be pinned for dependencies
    auto instance_count = command->instance_count();
    if (auto size = instance_count * sizeof(OptixInstance); _instance_buffer_size < size) {
        _instance_buffer_size = next_pow2(size);
        if (_instance_buffer) { LUISA_CHECK_CUDA(cuMemFreeAsync(_instance_buffer, cuda_stream)); }
        LUISA_CHECK_CUDA(cuMemAllocAsync(&_instance_buffer, _instance_buffer_size, cuda_stream));
    }
    // update the instance buffer
    auto mods = command->modifications();
    _meshes.resize(instance_count);
    if (auto n = mods.size()) {
        using Mod = AccelBuildCommand::Modification;
        auto host_update_buffer = stream->upload_pool()->allocate(mods.size_bytes());
        auto host_updates = reinterpret_cast<Mod *>(host_update_buffer->address());
        for (auto i = 0u; i < n; i++) {
            auto m = mods[i];
            if (m.flags & Mod::flag_mesh) {
                auto mesh = reinterpret_cast<const CUDAMesh *>(m.mesh);
                _meshes[m.index] = mesh;
                m.mesh = mesh->handle();
            }
            host_updates[i] = m;
        }
        auto update_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, mods.size_bytes(), cuda_stream));
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(update_buffer, host_updates, mods.size_bytes(), cuda_stream));
        auto update_kernel = device->accel_update_function();
        std::array<void *, 3u> args{&_instance_buffer, &update_buffer, &n};
        constexpr auto block_size = 1024u;
        auto block_count = (n + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            update_kernel, block_count, 1u, 1u, block_size, 1u, 1u,
            0u, cuda_stream, args.data(), nullptr));
        LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
        // recycle the update buffers after the kernel is finished
        stream->emplace_callback(host_update_buffer);
    }
    // find out whether we really need to build (or rebuild) the BVH
    auto requires_build = command->request() == AccelBuildRequest::FORCE_BUILD ||
                          _build_hint != AccelUsageHint::FAST_UPDATE ||
                          _handle == 0u || _mesh_handles.size() != instance_count;
    _mesh_handles.resize(instance_count);
    for (auto i = 0u; i < instance_count; i++) {
        if (auto handle = _meshes[i]->handle(); _mesh_handles[i] != handle) {
            requires_build = true;
            _mesh_handles[i] = handle;
        }
    }
    // perform the build or update
    if (requires_build) {
        _build(device, stream, cuda_stream);
    } else {
        _update(device, stream, cuda_stream);
    }
}

void CUDAAccel::_build(CUDADevice *device, CUDAStream *stream, CUstream cuda_stream) noexcept {

    // build IAS
    auto instance_count = static_cast<uint>(_meshes.size());
    auto build_input = cuda_accel_build_inputs(_instance_buffer, instance_count);
    auto build_options = cuda_accel_build_options(_build_hint, OPTIX_BUILD_OPERATION_BUILD);

    Clock clock;
    OptixAccelBufferSizes sizes;
    LUISA_CHECK_OPTIX(optixAccelComputeMemoryUsage(
        device->handle().optix_context(), &build_options,
        &build_input, 1u, &sizes));
    _update_buffer_size = sizes.tempUpdateSizeInBytes;
    LUISA_INFO("Computed accel memory usage in {} ms: "
               "temp = {}, temp_update = {}, output = {}.",
               clock.toc(), sizes.tempSizeInBytes,
               sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);

    if (_build_hint == AccelUsageHint::FAST_BUILD) {// no compaction
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
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream, &build_options, &build_input, 1, temp_buffer,
            sizes.tempSizeInBytes, _bvh_buffer, _bvh_buffer_size, &_handle, nullptr, 0u));
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, cuda_stream));
    } else {// with compaction
        static constexpr auto align = [](size_t x) noexcept {
            constexpr auto a = OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
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
        OptixAccelEmitDesc emit_desc{};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream, &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes, output_buffer, sizes.outputSizeInBytes,
            &_handle, &emit_desc, 1u));
        // read back the compacted size
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
        LUISA_INFO("CUDAAccel compaction: before = {}B, after = {}B, ratio = {}.",
                   sizes.outputSizeInBytes, compacted_size,
                   compacted_size / static_cast<double>(sizes.outputSizeInBytes));
        // do compaction...
        if (_bvh_buffer_size < compacted_size) {
            _bvh_buffer_size = compacted_size;
            if (_bvh_buffer) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer, _bvh_buffer_size, cuda_stream));
        }
        LUISA_CHECK_OPTIX(optixAccelCompact(
            device->handle().optix_context(), cuda_stream, _handle,
            _bvh_buffer, _bvh_buffer_size, &_handle));
        LUISA_CHECK_CUDA(cuMemFreeAsync(build_buffer, cuda_stream));
    }
}

void CUDAAccel::_update(CUDADevice *device, CUDAStream *stream, CUstream cuda_stream) noexcept {
    auto instance_count = static_cast<uint>(_meshes.size());
    auto build_input = cuda_accel_build_inputs(_instance_buffer, instance_count);
    auto build_options = cuda_accel_build_options(_build_hint, OPTIX_BUILD_OPERATION_UPDATE);
    auto update_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, _update_buffer_size, cuda_stream));
    LUISA_CHECK_OPTIX(optixAccelBuild(
        device->handle().optix_context(), cuda_stream,
        &build_options, &build_input, 1u, update_buffer, _update_buffer_size,
        _bvh_buffer, _bvh_buffer_size, &_handle, nullptr, 0u));
    LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
}

}// namespace luisa::compute::cuda
