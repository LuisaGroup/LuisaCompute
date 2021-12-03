//
// Created by Mike on 2021/12/2.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_accel.h>

namespace luisa::compute::cuda {

CUDAAccel::CUDAAccel(AccelBuildHint hint) noexcept
    : _build_hint{hint} {}

CUDAAccel::~CUDAAccel() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_instance_buffer));
    LUISA_CHECK_CUDA(cuMemFree(_update_buffer));
    LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer));
}

void CUDAAccel::add_instance(CUDAMesh *mesh, float4x4 transform) noexcept {
    _instance_meshes.emplace_back(mesh);
    _instance_transforms.emplace_back(transform);
    _resources.emplace(mesh->vertex_buffer_handle());
    _resources.emplace(mesh->triangle_buffer_handle());
    _resources.emplace(reinterpret_cast<uint64_t>(mesh));
}

void CUDAAccel::set_transform(size_t index, float4x4 transform) noexcept {
    _instance_transforms[index] = transform;
    _dirty_range.mark(index);
}

bool CUDAAccel::uses_resource(uint64_t handle) const noexcept {
    return _resources.contains(handle);
}

[[nodiscard]] inline auto make_optix_instance(
    size_t index, OptixTraversableHandle handle,
    float4x4 transform) noexcept {
    OptixInstance instance{};
    instance.transform[0] = transform[0].x;
    instance.transform[1] = transform[1].x;
    instance.transform[2] = transform[2].x;
    instance.transform[3] = transform[3].x;
    instance.transform[4] = transform[0].y;
    instance.transform[5] = transform[1].y;
    instance.transform[6] = transform[2].y;
    instance.transform[7] = transform[3].y;
    instance.transform[8] = transform[0].z;
    instance.transform[9] = transform[1].z;
    instance.transform[10] = transform[2].z;
    instance.transform[11] = transform[3].z;
    instance.instanceId = static_cast<uint32_t>(index);
    instance.sbtOffset = 0u;
    instance.visibilityMask = 0xffu;
    instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    instance.traversableHandle = handle;
    return instance;
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

void CUDAAccel::build(CUDADevice *device, CUDAStream *stream) noexcept {

    static constexpr auto round_up = [](size_t size, size_t alignment) noexcept {
        return (size + alignment - 1u) / alignment * alignment;
    };

    // upload instance buffer
    _dirty_range.clear();
    auto instance_buffer_size = _instance_meshes.size() * sizeof(OptixInstance);
    if (_instance_buffer == 0u || _instance_buffer_size < instance_buffer_size) {
        LUISA_CHECK_CUDA(cuMemFree(_instance_buffer));
        _instance_buffer_size = round_up(instance_buffer_size, 16u * sizeof(OptixInstance));
        LUISA_CHECK_CUDA(cuMemAlloc(&_instance_buffer, _instance_buffer_size));
    }
    auto instance_buffer = stream->upload_pool().allocate(instance_buffer_size);
    auto instances = reinterpret_cast<OptixInstance *>(instance_buffer.address());
    for (auto i = 0u; i < _instance_meshes.size(); i++) {
        instances[i] = make_optix_instance(
            i, _instance_meshes[i]->handle(),
            _instance_transforms[i]);
    }
    LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
        _instance_buffer, instance_buffer.address(),
        instance_buffer_size, stream->handle()));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        stream->handle(), [](void *user_data) noexcept {
            auto ctx = static_cast<CUDARingBuffer::RecycleContext *>(user_data);
            ctx->recycle();
        },
        CUDARingBuffer::RecycleContext::create(instance_buffer, &stream->upload_pool())));

    // build IAS
    auto build_input = _make_build_input();
    auto build_options = make_build_options(_build_hint, OPTIX_BUILD_OPERATION_BUILD);

    Clock clock;
    OptixAccelBufferSizes sizes;
    LUISA_CHECK_OPTIX(optixAccelComputeMemoryUsage(
        device->handle().optix_context(), &build_options,
        &build_input, 1u, &sizes));
    LUISA_INFO(
        "Computed accel memory usage in {} ms: "
        "temp = {}, temp_update = {}, output = {}.",
        clock.toc(),
        sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);
    if (_update_buffer_size < sizes.tempUpdateSizeInBytes) {
        LUISA_CHECK_CUDA(cuMemFree(_update_buffer));
        _update_buffer = 0u;
        _update_buffer_size = sizes.tempUpdateSizeInBytes;
    }

    static constexpr auto align = [](size_t x) noexcept {
        return round_up(x, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
    };
    if (_build_hint == AccelBuildHint::FAST_REBUILD) {// no compaction
        auto temp_buffer_offset = align(sizes.outputSizeInBytes);
        auto build_buffer_size = align(temp_buffer_offset + sizes.tempSizeInBytes);
        if (_bvh_buffer == 0u || _bvh_buffer_size < build_buffer_size) {
            LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer));
            LUISA_CHECK_CUDA(cuMemAlloc(&_bvh_buffer, build_buffer_size));
            _bvh_buffer_size = build_buffer_size;
        }
        auto temp_buffer = _bvh_buffer + temp_buffer_offset;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), stream->handle(),
            &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes,
            _bvh_buffer, sizes.outputSizeInBytes,
            &_handle, nullptr, 0u));
    } else {// with compaction

        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);

        CUdeviceptr build_buffer;
        LUISA_CHECK_CUDA(cuMemAlloc(&build_buffer, build_buffer_size));
        auto compacted_size_buffer = build_buffer + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer + temp_buffer_offset;
        auto output_buffer = build_buffer + output_buffer_offset;

        OptixAccelEmitDesc emit_desc{};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), stream->handle(),
            &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes,
            output_buffer, sizes.outputSizeInBytes,
            &_handle, &emit_desc, 1u));
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), stream->handle()));
        LUISA_CHECK_CUDA(cuStreamSynchronize(stream->handle()));
        LUISA_INFO("Compacted size: {}.", compacted_size);

        // do compaction...
        if (_bvh_buffer == 0u || _bvh_buffer_size < compacted_size) {
            LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer));
            _bvh_buffer_size = compacted_size;
            LUISA_CHECK_CUDA(cuMemAlloc(&_bvh_buffer, _bvh_buffer_size));
        }
        LUISA_CHECK_OPTIX(optixAccelCompact(
            device->handle().optix_context(), stream->handle(),
            _handle, _bvh_buffer, _bvh_buffer_size, &_handle));
        LUISA_CHECK_CUDA(cuMemFree(build_buffer));
    }
}

void CUDAAccel::update(CUDADevice *device, CUDAStream *stream) noexcept {

    // update instance buffer if dirty
    if (!_dirty_range.empty()) {
        auto dirty_update_buffer_size = _dirty_range.size() * sizeof(OptixInstance);
        auto dirty_update_buffer = stream->upload_pool().allocate(dirty_update_buffer_size);
        auto instances = reinterpret_cast<OptixInstance *>(dirty_update_buffer.address());
        for (auto i = 0u; i < _dirty_range.size(); i++) {
            auto index = i + _dirty_range.offset();
            instances[i] = make_optix_instance(
                index, _instance_meshes[index]->handle(),
                _instance_transforms[index]);
        }
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            _instance_buffer + _dirty_range.offset() * sizeof(OptixInstance),
            dirty_update_buffer.address(), dirty_update_buffer_size,
            stream->handle()));
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            stream->handle(), [](void *user_data) noexcept {
                auto recycle_context = static_cast<CUDARingBuffer::RecycleContext *>(user_data);
                recycle_context->recycle();
            },
            CUDARingBuffer::RecycleContext::create(dirty_update_buffer, &stream->upload_pool())));
        _dirty_range.clear();
    }

    // update IAS
    if (_update_buffer == 0u) {
        LUISA_CHECK_CUDA(cuMemAlloc(
            &_update_buffer, _update_buffer_size));
    }
    auto build_input = _make_build_input();
    auto build_options = make_build_options(_build_hint, OPTIX_BUILD_OPERATION_UPDATE);
    LUISA_CHECK_OPTIX(optixAccelBuild(
        device->handle().optix_context(), stream->handle(),
        &build_options, &build_input, 1u,
        _update_buffer, _update_buffer_size,
        _bvh_buffer, _bvh_buffer_size,
        &_handle, nullptr, 0u));
}

inline OptixBuildInput CUDAAccel::_make_build_input() const noexcept {
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = _instance_buffer;
    input.instanceArray.numInstances = static_cast<uint32_t>(_instance_meshes.size());
    return input;
}

}// namespace luisa::compute::cuda
