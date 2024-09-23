#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include "cuda_device.h"
#include "cuda_command_encoder.h"
#include "cuda_primitive.h"

namespace luisa::compute::cuda {

CUDAPrimitive::CUDAPrimitive(CUDAPrimitive::Tag tag,
                             const AccelOption &option) noexcept
    : CUDAPrimitiveBase{tag}, _option{option} {}

CUDAPrimitive::~CUDAPrimitive() noexcept {
    if (_bvh_buffer_handle) {
        LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer_handle));
    }
}

void CUDAPrimitive::_build(CUDACommandEncoder &encoder) noexcept {

    if (!_name.empty()) { nvtxRangePushA(luisa::format("{}::build", _name).c_str()); }

    auto build_input = _make_build_input();
    auto cuda_stream = encoder.stream()->handle();
    auto optix_ctx = encoder.stream()->device()->handle().optix_context();
    auto build_options = make_optix_build_options(option(), optix::BUILD_OPERATION_BUILD);

    optix::AccelBufferSizes sizes{};
    LUISA_CHECK_OPTIX(optix::api().accelComputeMemoryUsage(
        optix_ctx, &build_options,
        &build_input, 1u, &sizes));
    // LUISA_VERBOSE(
    //     "Computed mesh memory usage: temp = {}, temp_update = {}, output = {}.",
    //     sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);
    _update_buffer_size = sizes.tempUpdateSizeInBytes;
    static constexpr auto align = [](size_t x) noexcept -> size_t {
        static constexpr auto alignment = optix::ACCEL_BUFFER_BYTE_ALIGNMENT;
        return (x + alignment - 1u) / alignment * alignment;
    };

    if (option().allow_compaction) {// with compaction
        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);
        auto build_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&build_buffer, build_buffer_size, cuda_stream));
        auto compacted_size_buffer = build_buffer + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer + temp_buffer_offset;
        auto output_buffer = build_buffer + output_buffer_offset;

        if (!_name.empty()) { nvtxRangePushA("build"); }
        optix::AccelEmitDesc emit_desc{};
        emit_desc.type = optix::PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream,
            &build_options, &build_input, 1, temp_buffer, sizes.tempSizeInBytes,
            output_buffer, sizes.outputSizeInBytes, &_handle, &emit_desc, 1u));
        if (!_name.empty()) { nvtxRangePop(); }

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
        // LUISA_VERBOSE("CUDAMesh compaction sizes: before = {}B, after = {}B, ratio = {}.",
        //               sizes.outputSizeInBytes, compacted_size,
        //               compacted_size / static_cast<double>(sizes.outputSizeInBytes));

        if (_bvh_buffer_size < compacted_size) {
            _bvh_buffer_size = compacted_size;
            if (_bvh_buffer_handle) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer_handle, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer_handle, _bvh_buffer_size, cuda_stream));
        }
        if (!_name.empty()) { nvtxRangePushA("compact"); }
        LUISA_CHECK_OPTIX(optix::api().accelCompact(
            optix_ctx, cuda_stream, _handle,
            _bvh_buffer_handle, _bvh_buffer_size, &_handle));
        if (!_name.empty()) { nvtxRangePop(); }
        LUISA_CHECK_CUDA(cuMemFreeAsync(build_buffer, cuda_stream));
    } else {// without compaction
        if (_bvh_buffer_size < sizes.outputSizeInBytes) {
            _bvh_buffer_size = sizes.outputSizeInBytes;
            if (_bvh_buffer_handle) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer_handle, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer_handle, _bvh_buffer_size, cuda_stream));
        }
        auto temp_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp_buffer, sizes.tempSizeInBytes, cuda_stream));
        if (!_name.empty()) { nvtxRangePushA("build"); }
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream, &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes, _bvh_buffer_handle,
            _bvh_buffer_size, &_handle, nullptr, 0u));
        if (!_name.empty()) { nvtxRangePop(); }
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, cuda_stream));
    }
    // update handle
    LUISA_ASSERT(_handle != 0ull, "OptiX BVH build failed.");
    if (!_name.empty()) { nvtxRangePop(); }
    LUISA_VERBOSE("Built CUDA primitive: handle = {}, bvh_buffer = {}, size = {}.",
                  reinterpret_cast<void *>(_handle),
                  reinterpret_cast<void *>(_bvh_buffer_handle),
                  _bvh_buffer_size);
}

void CUDAPrimitive::_update(CUDACommandEncoder &encoder) noexcept {
    if (!_name.empty()) { nvtxRangePushA(luisa::format("{}::update", _name).c_str()); }
    auto build_input = _make_build_input();
    auto build_options = make_optix_build_options(option(), optix::BUILD_OPERATION_UPDATE);
    auto cuda_stream = encoder.stream()->handle();
    auto update_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, _update_buffer_size, cuda_stream));
    LUISA_CHECK_OPTIX(optix::api().accelBuild(
        encoder.stream()->device()->handle().optix_context(), cuda_stream,
        &build_options, &build_input, 1u, update_buffer, _update_buffer_size,
        _bvh_buffer_handle, _bvh_buffer_size, &_handle, nullptr, 0u));
    LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
    if (!_name.empty()) { nvtxRangePop(); }
    LUISA_VERBOSE("Updated CUDA primitive: handle = {}, bvh_buffer = {}, size = {}.",
                  reinterpret_cast<void *>(_handle),
                  reinterpret_cast<void *>(_bvh_buffer_handle),
                  _bvh_buffer_size);
}

const CUdeviceptr *CUDAPrimitive::_motion_buffer_pointers(CUdeviceptr base, size_t total_size) const noexcept {
    static thread_local CUdeviceptr pointers[max_motion_keyframe_count] = {};
    auto n = motion_keyframe_count();
    LUISA_ASSERT(n <= max_motion_keyframe_count, "Too many motion keyframes.");
    LUISA_ASSERT(total_size % n == 0u, "Motion buffer size is not divisible by keyframe count.");
    auto pitch = total_size / n;
    for (auto i = 0u; i < n; i++) {
        pointers[i] = base + i * pitch;
    }
    return pointers;
}

void CUDAPrimitiveBase::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_mutex};
    _name = std::move(name);
}

optix::TraversableHandle CUDAPrimitiveBase::handle() const noexcept {
    std::scoped_lock lock{_mutex};
    return _handle;
}

}// namespace luisa::compute::cuda
