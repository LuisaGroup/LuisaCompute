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

void CUDAAccel::add_instance(CUDAMesh *mesh, float4x4 transform, bool visible) noexcept {
    _instance_meshes.emplace_back(mesh);
    _instance_transforms.emplace_back(transform);
    _instance_visibilities.push_back(visible);
    _resources.emplace(mesh->vertex_buffer_handle());
    _resources.emplace(mesh->triangle_buffer_handle());
    _resources.emplace(reinterpret_cast<uint64_t>(mesh));
}

void CUDAAccel::set_instance(size_t index, CUDAMesh *mesh, float4x4 transform, bool visible) noexcept {
    _instance_meshes[index] = mesh;
    _instance_transforms[index] = transform;
    _instance_visibilities[index] = visible;
    _resources.emplace(mesh->vertex_buffer_handle());
    _resources.emplace(mesh->triangle_buffer_handle());
    _resources.emplace(reinterpret_cast<uint64_t>(mesh));
}

void CUDAAccel::set_visibility(size_t index, bool visible) noexcept {
    _instance_visibilities[index] = visible;
    _dirty_range.mark(index);
}

void CUDAAccel::pop_instance() noexcept {
    _instance_meshes.pop_back();
    _instance_transforms.pop_back();
    _instance_visibilities.pop_back();
}

void CUDAAccel::set_transform(size_t index, float4x4 transform) noexcept {
    _instance_transforms[index] = transform;
    _dirty_range.mark(index);
}

bool CUDAAccel::uses_resource(uint64_t handle) const noexcept {
    return _resources.count(handle) != 0u;
}

[[nodiscard]] inline auto make_optix_instance(
    size_t index, OptixTraversableHandle handle,
    float4x4 transform, bool visible) noexcept {
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
    instance.visibilityMask = visible ? 0xffu : 0x00u;
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
    _heap = device->heap();

    // upload instance buffer
    _dirty_range.clear();
    if (auto instance_buffer_size = _instance_meshes.size() * sizeof(OptixInstance);
        _instance_buffer_size < instance_buffer_size) {
        stream->emplace_callback(CUDAHeap::BufferFreeContext::create(
            _heap, _instance_buffer));
        _instance_buffer_size = instance_buffer_size;
        _instance_buffer = _heap->allocate(_instance_buffer_size);
    }
    auto instance_buffer = stream->upload_pool()->allocate(_instance_buffer_size);
    auto instances = reinterpret_cast<OptixInstance *>(instance_buffer.address());
    for (auto i = 0u; i < _instance_meshes.size(); i++) {
        instances[i] = make_optix_instance(
            i, _instance_meshes[i]->handle(),
            _instance_transforms[i],
            _instance_visibilities[i]);
    }
    LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
        CUDAHeap::buffer_address(_instance_buffer),
        instance_buffer.address(),
        _instance_buffer_size, stream->handle()));
    stream->emplace_callback(
        CUDARingBuffer::RecycleContext::create(
            instance_buffer, stream->upload_pool()));

    // build IAS
    auto build_input = _make_build_input();
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
            device->handle().optix_context(), stream->handle(),
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
        if (_bvh_buffer_size < compacted_size) {
            stream->emplace_callback(
                CUDAHeap::BufferFreeContext::create(
                    _heap, _bvh_buffer));
            _bvh_buffer_size = compacted_size;
            _bvh_buffer = _heap->allocate(_bvh_buffer_size);
        }
        LUISA_CHECK_OPTIX(optixAccelCompact(
            device->handle().optix_context(),
            stream->handle(), _handle,
            CUDAHeap::buffer_address(_bvh_buffer),
            _bvh_buffer_size, &_handle));
        stream->emplace_callback(
            CUDAHeap::BufferFreeContext::create(
                _heap, build_buffer));
    }
}

void CUDAAccel::update(CUDADevice *device, CUDAStream *stream) noexcept {

    // update instance buffer if dirty
    if (!_dirty_range.empty()) {
        auto dirty_update_buffer_size = _dirty_range.size() * sizeof(OptixInstance);
        auto dirty_update_buffer = stream->upload_pool()->allocate(dirty_update_buffer_size);
        auto instances = reinterpret_cast<OptixInstance *>(dirty_update_buffer.address());
        for (auto i = 0u; i < _dirty_range.size(); i++) {
            auto index = i + _dirty_range.offset();
            instances[i] = make_optix_instance(
                index, _instance_meshes[index]->handle(),
                _instance_transforms[index],
                _instance_visibilities[index]);
        }
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            CUDAHeap::buffer_address(_instance_buffer) +
                _dirty_range.offset() * sizeof(OptixInstance),
            dirty_update_buffer.address(), dirty_update_buffer_size,
            stream->handle()));
        stream->emplace_callback(
            CUDARingBuffer::RecycleContext::create(
                dirty_update_buffer, stream->upload_pool()));
        _dirty_range.clear();
    }

    // update IAS
    auto build_input = _make_build_input();
    auto build_options = make_build_options(_build_hint, OPTIX_BUILD_OPERATION_UPDATE);
    auto update_buffer = _heap->allocate(_update_buffer_size);
    LUISA_CHECK_OPTIX(optixAccelBuild(
        device->handle().optix_context(), stream->handle(),
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

inline OptixBuildInput CUDAAccel::_make_build_input() const noexcept {
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = CUDAHeap::buffer_address(_instance_buffer);
    input.instanceArray.numInstances = static_cast<uint32_t>(_instance_meshes.size());
    return input;
}

}// namespace luisa::compute::cuda
