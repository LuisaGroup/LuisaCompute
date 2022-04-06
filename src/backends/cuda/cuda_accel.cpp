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

CUDAAccel::CUDAAccel(AccelBuildHint hint) noexcept
    : _build_hint{hint} {}

CUDAAccel::~CUDAAccel() noexcept {
    if (_heap != nullptr) {
        _heap->free(_instance_buffer);
        _heap->free(_bvh_buffer);
    }
}

[[nodiscard]] inline auto make_build_options(AccelBuildHint hint, OptixBuildOperation op) noexcept {
    OptixAccelBuildOptions build_options{};
    build_options.operation = op;
    switch (hint) {
        case AccelBuildHint::FAST_TRACE:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                       OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                       OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            break;
        case AccelBuildHint::FAST_UPDATE:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                       OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            break;
        case AccelBuildHint::FAST_REBUILD:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                       OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            break;
    }
    return build_options;
}

[[nodiscard]] inline OptixBuildInput make_build_input(
    uint64_t instance_buffer, uint instance_count) noexcept {
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = CUDAHeap::buffer_address(instance_buffer);
    input.instanceArray.numInstances = instance_count;
    return input;
}

void CUDAAccel::build(CUDADevice *device, CUDAStream *stream,
                      luisa::span<const uint64_t> mesh_handles,
                      luisa::span<const AccelUpdateRequest> requests) noexcept {

    static constexpr auto round_up = [](size_t size, size_t alignment) noexcept {
        return (size + alignment - 1u) / alignment * alignment;
    };

    // create instance buffer
    _heap = device->heap();
    auto cuda_stream = stream->handle(true);
    if (auto instance_buffer_size = mesh_handles.size() * sizeof(OptixInstance);
        _instance_buffer_size < instance_buffer_size) {
        stream->emplace_callback(CUDAHeap::BufferFreeContext::create(
            _heap, _instance_buffer));// release old instance buffer
        _instance_buffer_size = next_pow2(instance_buffer_size);
        _instance_buffer = _heap->allocate(_instance_buffer_size);
    }

    // materialize instance buffer if required
    if (auto n = static_cast<uint>(requests.size())) [[likely]] {
        auto gas_buffer_size = round_up(mesh_handles.size() * sizeof(OptixTraversableHandle), 16u);
        auto buffer_size = gas_buffer_size + requests.size_bytes();
        auto host_buffer = stream->upload_pool()->allocate(buffer_size);
        for (auto i = 0u; i < mesh_handles.size(); i++) {
            auto handle = reinterpret_cast<const CUDAMesh *>(mesh_handles[i])->handle();
            reinterpret_cast<OptixTraversableHandle *>(host_buffer->address())[i] = handle;
        }
        std::memcpy(host_buffer->address() + gas_buffer_size,
                    requests.data(), requests.size_bytes());
        auto device_buffer = _heap->allocate(buffer_size);
        auto update_buffer_address = CUDAHeap::buffer_address(device_buffer);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            update_buffer_address, host_buffer->address(),
            buffer_size, cuda_stream));
        auto init_kernel = device->accel_initialize_function();
        auto gas_buffer = update_buffer_address;
        auto request_buffer = update_buffer_address + gas_buffer_size;
        auto instance_buffer = CUDAHeap::buffer_address(_instance_buffer);
        std::array<void *, 4u> args{
            &instance_buffer, &gas_buffer,
            &request_buffer, &n};
        constexpr auto block_size = 256u;
        auto block_count = (n + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            init_kernel, block_count, 1u, 1u, block_size, 1u, 1u,
            0u, cuda_stream, args.data(), nullptr));
        stream->emplace_callback(host_buffer);
        stream->emplace_callback(CUDAHeap::BufferFreeContext::create(_heap, device_buffer));
    }

    // build IAS
    _instance_count = static_cast<uint>(mesh_handles.size());
    auto build_input = make_build_input(_instance_buffer, _instance_count);
    auto build_options = make_build_options(_build_hint, OPTIX_BUILD_OPERATION_BUILD);

    Clock clock;
    OptixAccelBufferSizes sizes;
    LUISA_CHECK_OPTIX(optixAccelComputeMemoryUsage(
        device->handle().optix_context(), &build_options,
        &build_input, 1u, &sizes));
    _update_buffer_size = sizes.tempUpdateSizeInBytes;
    LUISA_INFO(
        "Computed accel memory usage in {} ms: "
        "temp = {}, temp_update = {}, output = {}.",
        clock.toc(),
        sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);

    static constexpr auto align = [](size_t x) noexcept {
        return round_up(x, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
    };
    if (_build_hint == AccelBuildHint::FAST_REBUILD) {// no compaction
        if (_bvh_buffer_size < sizes.outputSizeInBytes) {
            stream->emplace_callback(
                CUDAHeap::BufferFreeContext::create(
                    _heap, _bvh_buffer));
            _bvh_buffer_size = sizes.outputSizeInBytes;
            _bvh_buffer = _heap->allocate(_bvh_buffer_size);
        }
        auto temp_buffer = _heap->allocate(sizes.tempSizeInBytes);
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream,
            &build_options, &build_input, 1,
            CUDAHeap::buffer_address(temp_buffer),
            sizes.tempSizeInBytes,
            CUDAHeap::buffer_address(_bvh_buffer),
            _bvh_buffer_size,
            &_handle, nullptr, 0u));
        stream->emplace_callback(
            CUDAHeap::BufferFreeContext::create(
                _heap, temp_buffer));
    } else {// with compaction

        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);

        auto build_buffer = _heap->allocate(build_buffer_size);
        auto build_buffer_address = CUDAHeap::buffer_address(build_buffer);
        auto compacted_size_buffer = build_buffer_address + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer_address + temp_buffer_offset;
        auto output_buffer = build_buffer_address + output_buffer_offset;

        OptixAccelEmitDesc emit_desc{};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream,
            &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes,
            output_buffer, sizes.outputSizeInBytes,
            &_handle, &emit_desc, 1u));
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
        LUISA_INFO("Compacted size: {}.", compacted_size);

        // do compaction...
        if (_bvh_buffer_size < compacted_size) {
            stream->emplace_callback(
                CUDAHeap::BufferFreeContext::create(
                    _heap, _bvh_buffer));
            _bvh_buffer_size = compacted_size;
            _bvh_buffer = _heap->allocate(_bvh_buffer_size);
        }
        LUISA_CHECK_OPTIX(optixAccelCompact(
            device->handle().optix_context(),
            cuda_stream, _handle,
            CUDAHeap::buffer_address(_bvh_buffer),
            _bvh_buffer_size, &_handle));
        stream->emplace_callback(
            CUDAHeap::BufferFreeContext::create(_heap, build_buffer));
    }
}

void CUDAAccel::update(CUDADevice *device, CUDAStream *stream,
                       luisa::span<const AccelUpdateRequest> requests) noexcept {

    // update instance buffer if dirty
    auto cuda_stream = stream->handle(true);
    if (auto n = static_cast<uint>(requests.size())) {
        auto host_request_buffer = stream->upload_pool()->allocate(requests.size_bytes());
        std::memcpy(host_request_buffer->address(), requests.data(), requests.size_bytes());
        auto request_buffer = device->heap()->allocate(requests.size_bytes());
        auto request_buffer_address = CUDAHeap::buffer_address(request_buffer);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            request_buffer_address, host_request_buffer->address(),
            requests.size_bytes(), cuda_stream));
        auto instance_buffer_address = CUDAHeap::buffer_address(_instance_buffer);
        auto update_kernel = device->accel_update_function();
        std::array<void *, 3u> args{
            &instance_buffer_address,
            &request_buffer_address, &n};
        constexpr auto block_size = 256u;
        auto block_count = (n + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            update_kernel, block_count, 1u, 1u, block_size, 1u, 1u,
            0u, cuda_stream, args.data(), nullptr));
        stream->emplace_callback(host_request_buffer);
        stream->emplace_callback(CUDAHeap::BufferFreeContext::create(
            device->heap(), request_buffer));
    }

    // update IAS
    auto build_input = make_build_input(_instance_buffer, _instance_count);
    auto build_options = make_build_options(_build_hint, OPTIX_BUILD_OPERATION_UPDATE);
    auto update_buffer = _heap->allocate(_update_buffer_size);
    LUISA_CHECK_OPTIX(optixAccelBuild(
        device->handle().optix_context(), cuda_stream,
        &build_options, &build_input, 1u,
        CUDAHeap::buffer_address(update_buffer),
        _update_buffer_size,
        CUDAHeap::buffer_address(_bvh_buffer),
        _bvh_buffer_size,
        &_handle, nullptr, 0u));
    stream->emplace_callback(
        CUDAHeap::BufferFreeContext::create(
            _heap, update_buffer));
}

}// namespace luisa::compute::cuda
