#include <Resource/ClusterAccel.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/UploadBuffer.h>
#include <Shader/ComputeShader.h>
#include <luisa/runtime/rtx/mesh.h>
#include <cstddef>
#include <limits>

namespace lc::dx {
using namespace luisa::compute;

namespace cluster_accel_detail {

static_assert(sizeof(ClusterBuildInfo) == 64u);
static_assert(sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS) == 80u);
static_assert(alignof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS) == 8u);
static_assert(offsetof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS, TriangleCount) == 8u);
static_assert(offsetof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS, VertexBuffer) == 32u);
static_assert(offsetof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS, OpacityMicromapIndexBuffer) == 72u);
static_assert(sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) == 16u);

constexpr D3D12_INDEX_FORMAT kIndexFormats[] = {
    D3D12_INDEX_FORMAT_UINT8,
    D3D12_INDEX_FORMAT_UINT16,
    D3D12_INDEX_FORMAT_UINT32,
};

[[nodiscard]] uint32_t saturated_product(uint32_t lhs, uint32_t rhs) noexcept {
    auto value = static_cast<uint64_t>(lhs) * rhs;
    return value > std::numeric_limits<uint32_t>::max() ?
               std::numeric_limits<uint32_t>::max() :
               static_cast<uint32_t>(value);
}

[[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS address(BufferView const &view) noexcept {
    return view.buffer == nullptr ? 0u : view.buffer->GetAddress() + view.offset;
}

[[nodiscard]] size_t checked_array_size(uint32_t count, size_t element_size) {
    LUISA_ASSERT(element_size == 0u ||
                     count <= std::numeric_limits<size_t>::max() / element_size,
                 "Cluster build array size overflows the host address space.");
    return static_cast<size_t>(count) * element_size;
}

void record_read_only(EnhancedBarrierTracker &tracker, BufferView const &view) {
    tracker.Record(view, EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
}

void record_as_read(
    EnhancedBarrierTracker &tracker,
    Buffer const *buffer,
    D3D12_BARRIER_SYNC sync = D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE) {
    tracker.Record(
        BufferView{buffer},
        sync,
        D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
        D3D12_BARRIER_LAYOUT_UNDEFINED);
}

}// namespace cluster_accel_detail

ClusterAccel::ClusterAccel(Device *device, AccelOption const &option)
    : BottomAccel(device, option), _usage_hint(option.hint) {
    // Classic compaction/update flags must never leak into a cluster path.
    hint = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
    if (!device->feature_check.cluster_acceleration_structure_supported() ||
        !device->raytracing2) [[unlikely]] {
        LUISA_ERROR(
            "Cluster (mega-geometry) acceleration structures require DXR "
            "ClustersAndPTLASSupported and ID3D12DeviceRaytracing2.");
    }
    if (option.motion.is_enabled()) [[unlikely]] {
        LUISA_WARNING(
            "Cluster (mega-geometry) meshes do not support motion blur; "
            "the motion option is ignored and every build is a full rebuild.");
    }
    if (option.allow_compaction || option.allow_update) [[unlikely]] {
        LUISA_WARNING(
            "Cluster (mega-geometry) meshes have no classic compaction or "
            "update path; those options are ignored.");
    }
}

void ClusterAccel::_query_build_sizes(uint32_t count) {
    using namespace cluster_accel_detail;
    if (_sizes_cluster_count == count) return;

    _operation_flags = _usage_hint == AccelOption::UsageHint::FAST_BUILD ?
                           D3D12_RTAS_OPERATION_FLAG_FAST_OPERATION :
                           D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;

    uint64_t clas_stride = 0u;
    for (auto format = 0u; format < kIndexFormatCount; ++format) {
        auto &input = _clas_inputs[format];
        input = {};
        auto &limits = input.ClusterLimits;
        limits.MaxArgCount = count;
        limits.MaxGeometryIndexValue = D3D12_RAYTRACING_MAXIMUM_GEOMETRY_INDEX;
        // There can be at most one geometry-index/flags value per triangle.
        limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 256u;
        limits.MaxTriangleCountPerCluster = 256u;
        limits.MaxUniqueVertexIndexCountPerCluster = 256u;
        limits.MaxTotalTriangleCount = saturated_product(count, 256u);
        limits.MaxSumOfUniqueVertexIndexCounts = saturated_product(count, 256u);
        // The public record layout has no D3D12 OMM base location and the
        // initial DX path deliberately disables OMM consumption.
        limits.MaxOpacityMicromapIndicesPerCluster = 0u;
        input.Flags = _operation_flags;
        input.Mode = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
        input.VertexFormat = D3D12_VERTEX_FORMAT_FLOAT32_3;
        input.IndexFormat = kIndexFormats[format];
        input.GeometryIndexAndFlagsIndexFormat = D3D12_INDEX_FORMAT_NONE;
        input.OpacityMicromapIndexFormat = D3D12_INDEX_FORMAT_NONE;
        input.MinPositionTruncateBitCount = 0u;

        D3D12_RTAS_OPERATION_INPUTS inputs{};
        inputs.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
        inputs.pClusterTrianglesDesc = &input;
        device->raytracing2->GetRTASOperationPrebuildInfo(
            &inputs, &_clas_batch_info[format]);

        // A per-object query supplies a conservative fixed stride for explicit
        // destinations. The batch query above independently supplies scratch.
        auto one = input;
        one.ClusterLimits.MaxArgCount = 1u;
        one.ClusterLimits.MaxTotalTriangleCount = 256u;
        one.ClusterLimits.MaxSumOfUniqueVertexIndexCounts = 256u;
        inputs.pClusterTrianglesDesc = &one;
        D3D12_RTAS_OPERATION_PREBUILD_INFO one_info{};
        device->raytracing2->GetRTASOperationPrebuildInfo(&inputs, &one_info);
        LUISA_ASSERT(one_info.ResultDataMaxSizeInBytes > 0u,
                     "DXR returned an empty CLAS prebuild size.");
        clas_stride = std::max(
            clas_stride,
            CalcAlign(one_info.ResultDataMaxSizeInBytes,
                      D3D12_RAYTRACING_CLAS_BYTE_ALIGNMENT));
    }
    _clas_stride = clas_stride;

    _blas_inputs = {};
    _blas_inputs.Flags = _operation_flags;
    _blas_inputs.MaxArgCount = 1u;
    _blas_inputs.Mode = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
    _blas_inputs.MaxTotalClasCount = count;
    _blas_inputs.MaxClasCountPerArg = count;
    D3D12_RTAS_OPERATION_INPUTS blas_inputs{};
    blas_inputs.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
    blas_inputs.pClasDesc = &_blas_inputs;
    device->raytracing2->GetRTASOperationPrebuildInfo(&blas_inputs, &_blas_info);
    LUISA_ASSERT(_blas_info.ResultDataMaxSizeInBytes > 0u,
                 "DXR returned an empty Cluster BLAS prebuild size.");
    _sizes_cluster_count = count;
}

void ClusterAccel::_ensure_result_buffers(
    CommandBufferBuilder &builder, uint32_t count) {
    auto ensure = [&](vstd::unique_ptr<DefaultBuffer> &buffer,
                      uint64_t required_size,
                      char const *name) {
        if (buffer && buffer->GetByteSize() >= required_size) return false;
        if (buffer) {
            builder.get_cb()->get_alloc()->dispose_after_complete(std::move(buffer));
        }
        buffer = vstd::create_unique(new DefaultBuffer(
            device,
            CalcAlign(required_size, 65536u),
            device->default_allocator.get(),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            false,
            name));
        return true;
    };

    LUISA_ASSERT(_clas_stride != 0u &&
                     count <= std::numeric_limits<uint64_t>::max() / _clas_stride,
                 "CLAS pool size overflows a 64-bit allocation size.");
    ensure(_clas_buffer, _clas_stride * count, "dx-cluster-clas-pool");
    LUISA_ASSERT(_clas_buffer->GetAddress() % D3D12_RAYTRACING_CLAS_BYTE_ALIGNMENT == 0u,
                 "CLAS pool base is not 128-byte aligned.");
    if (ensure(accelBuffer, _blas_info.ResultDataMaxSizeInBytes,
               "dx-cluster-blas-pool")) {
        // Explicit destinations make the Cluster BLAS address host-known during
        // preprocessing, before a later TLAS layer bakes its instance buffer.
        SyncTopAccel();
    }
    LUISA_ASSERT(accelBuffer->GetAddress() % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT == 0u,
                 "Cluster BLAS pool base is not 256-byte aligned.");
}

void ClusterAccel::_record_input_states(
    EnhancedBarrierTracker &tracker,
    MeshBuildCommand const *cmd) const {
    auto source = reinterpret_cast<Buffer const *>(cmd->cluster_buffer());
    tracker.Record(
        BufferView{source, cmd->cluster_buffer_offset(), cmd->cluster_buffer_size()},
        EnhancedBarrierTracker::Usage::ComputeRead);
    tracker.Record(_counts, EnhancedBarrierTracker::Usage::CopyDest);
    for (auto format = 0u; format < kIndexFormatCount; ++format) {
        tracker.Record(_clas_args[format], EnhancedBarrierTracker::Usage::ComputeUAV);
        tracker.Record(_clas_destinations[format], EnhancedBarrierTracker::Usage::ComputeUAV);
    }
    tracker.Record(_clas_addresses, EnhancedBarrierTracker::Usage::ComputeUAV);
    tracker.Record(_blas_args, EnhancedBarrierTracker::Usage::CopyDest);
    tracker.Record(_blas_destination, EnhancedBarrierTracker::Usage::CopyDest);
    tracker.Record(_clas_buffer.get(), EnhancedBarrierTracker::Range{},
                   EnhancedBarrierTracker::Usage::BuildAccel);
    tracker.Record(GetAccelBuffer(), EnhancedBarrierTracker::Range{},
                   EnhancedBarrierTracker::Usage::BuildAccel);
}

size_t ClusterAccel::PreProcessStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    MeshBuildCommand const *cmd) {
    if (_build_pending) [[unlikely]] {
        LUISA_ERROR(
            "The same Cluster BLAS cannot be preprocessed twice before its "
            "first build executes. Split the builds into separate submissions.");
    }
    auto count = cmd->cluster_count();
    LUISA_ASSERT(count > 0u,
                 "A cluster (mega-geometry) build needs at least one record.");
    LUISA_ASSERT(cmd->cluster_buffer_size() % sizeof(ClusterBuildInfo) == 0u,
                 "Cluster build input is not a whole array of 64-byte records.");
    _cluster_count = count;
    _query_build_sizes(count);
    _ensure_result_buffers(builder, count);

    auto alloc = builder.get_cb()->get_alloc();
    _counts = alloc->get_temp_default_buffer(
        kIndexFormatCount * sizeof(uint), alignof(uint));
    for (auto format = 0u; format < kIndexFormatCount; ++format) {
        _clas_args[format] = alloc->get_temp_default_buffer(
            cluster_accel_detail::checked_array_size(
                count, sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS)),
            alignof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS));
        _clas_destinations[format] = alloc->get_temp_default_buffer(
            cluster_accel_detail::checked_array_size(count, sizeof(uint64_t)),
            alignof(uint64_t));
    }
    _clas_addresses = alloc->get_temp_default_buffer(
        cluster_accel_detail::checked_array_size(count, sizeof(uint64_t)),
        alignof(uint64_t));
    _blas_args = alloc->get_temp_default_buffer(
        sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
        alignof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS));
    _blas_destination = alloc->get_temp_default_buffer(
        sizeof(D3D12_GPU_VIRTUAL_ADDRESS), alignof(D3D12_GPU_VIRTUAL_ADDRESS));
    _record_input_states(tracker, cmd);

    size_t scratch_size = 0u;
    for (auto format = 0u; format < kIndexFormatCount; ++format) {
        _scratch_offsets[format] = scratch_size;
        scratch_size += CalcAlign(
            _clas_batch_info[format].ScratchDataSizeInBytes,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    }
    _scratch_offsets[kIndexFormatCount] = scratch_size;
    scratch_size += CalcAlign(
        _blas_info.ScratchDataSizeInBytes,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    _build_pending = true;
    return scratch_size;
}

void ClusterAccel::BuildClusters(
    EnhancedBarrierTracker &tracker,
    CommandBufferBuilder &builder,
    BufferView const &scratch_buffer,
    MeshBuildCommand const *cmd) {
    using namespace cluster_accel_detail;
    if (!_build_pending) [[unlikely]] {
        LUISA_ERROR("Cluster BLAS encoding has no matching preprocessing pass.");
    }
    auto count = _cluster_count;
    LUISA_ASSERT(count > 0u && cmd->cluster_count() == count &&
                     _clas_buffer && accelBuffer,
                 "Cluster BLAS build without matching preprocessing.");

    auto alloc = builder.get_cb()->get_alloc();
    auto upload = [&](void const *data, size_t size, size_t alignment) {
        auto view = alloc->get_temp_upload_buffer(size, alignment);
        static_cast<UploadBuffer const *>(view.buffer)->CopyData(
            view.offset, {static_cast<uint8_t const *>(data), size});
        return view;
    };
    auto copy_to = [&](BufferView const &dst, void const *data,
                       size_t size, size_t alignment) {
        auto src = upload(data, size, alignment);
        builder.copy_buffer(src.buffer, dst.buffer,
                            src.offset, dst.offset, size);
    };

    std::array<uint, kIndexFormatCount> zeros{};
    copy_to(_counts, zeros.data(), sizeof(zeros), alignof(uint));

    D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS blas_args{};
    blas_args.ClasAddressCount = count;
    blas_args.ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    blas_args.ClasAddressArray = address(_clas_addresses);
    copy_to(_blas_args, &blas_args, sizeof(blas_args), alignof(decltype(blas_args)));
    auto blas_address = accelBuffer->GetAddress();
    copy_to(_blas_destination, &blas_address, sizeof(blas_address),
            alignof(D3D12_GPU_VIRTUAL_ADDRESS));

    GraphicsCmdlistBarrierCallback callback(builder);
    tracker.Record(_counts, EnhancedBarrierTracker::Usage::ComputeUAV);
    tracker.UpdateState(&callback);

    struct alignas(8) ConvertConstants {
        uint count;
        uint clas_stride;
        uint pool_low;
        uint pool_high;
    } constants{};
    LUISA_ASSERT(_clas_stride <= std::numeric_limits<uint>::max(),
                 "CLAS stride does not fit the conversion shader constant.");
    constants.count = count;
    constants.clas_stride = static_cast<uint>(_clas_stride);
    constants.pool_low = static_cast<uint>(_clas_buffer->GetAddress());
    constants.pool_high = static_cast<uint>(_clas_buffer->GetAddress() >> 32u);
    auto constant_buffer = upload(&constants, sizeof(constants),
                                  D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    auto source = reinterpret_cast<Buffer const *>(cmd->cluster_buffer());
    BufferView source_view{source, cmd->cluster_buffer_offset(),
                           cmd->cluster_buffer_size()};
    BindProperty properties[] = {
        constant_buffer,
        source_view,
        _clas_args[0],
        _clas_args[1],
        _clas_args[2],
        _counts,
        _clas_destinations[0],
        _clas_destinations[1],
        _clas_destinations[2],
        _clas_addresses,
    };
    builder.dispatch_compute(
        device->cluster_build_info_kernel.get(device),
        uint3(count, 1u, 1u), properties);

    // Conversion UAV writes become read-only operation arguments. In
    // particular, IndirectArgumentArraySize points directly at each count.
    tracker.Record(_counts, EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
    for (auto format = 0u; format < kIndexFormatCount; ++format) {
        record_read_only(tracker, _clas_args[format]);
        record_read_only(tracker, _clas_destinations[format]);
    }
    record_read_only(tracker, _clas_addresses);
    tracker.UpdateState(&callback);

    ComPtr<ID3D12CommandListRaytracing2> rt_commands;
    if (FAILED(builder.get_cb()->cmd_list()->QueryInterface(
            IID_PPV_ARGS(rt_commands.GetAddressOf())))) [[unlikely]] {
        LUISA_ERROR(
            "ID3D12CommandListRaytracing2 is unavailable on a device that "
            "reported cluster acceleration-structure support.");
    }

    auto scratch_address = address(scratch_buffer);
    for (auto format = 0u; format < kIndexFormatCount; ++format) {
        D3D12_RTAS_BATCHED_OPERATION_DATA data{};
        data.BatchScratchData = scratch_address + _scratch_offsets[format];
        LUISA_ASSERT(data.BatchScratchData % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT == 0u,
                     "CLAS scratch address is not 256-byte aligned.");
        data.ResultAddressArray.StartAddress = address(_clas_destinations[format]);
        data.ResultAddressArray.StrideInBytes = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        data.IndirectArgumentArray.StartAddress = address(_clas_args[format]);
        data.IndirectArgumentArray.StrideInBytes =
            sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
        data.IndirectArgumentArraySize = address(_counts) + format * sizeof(uint);

        D3D12_RTAS_OPERATION_DESC desc{};
        desc.Inputs.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
        desc.Inputs.pClusterTrianglesDesc = &_clas_inputs[format];
        desc.pBatchedOperationData = &data;
        rt_commands->ExecuteIndirectRTASOperations(
            1u, &desc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    }

    // All three CLAS batches feed the single Cluster BLAS operation. Their
    // scratch regions are disjoint; this barrier covers the actual CLAS data
    // and the host-known address arrays produced by the converter.
    record_as_read(tracker, _clas_buffer.get());
    record_read_only(tracker, _blas_args);
    record_read_only(tracker, _blas_destination);
    tracker.UpdateState(&callback);

    D3D12_RTAS_BATCHED_OPERATION_DATA blas_data{};
    blas_data.BatchScratchData = scratch_address + _scratch_offsets[kIndexFormatCount];
    LUISA_ASSERT(blas_data.BatchScratchData % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT == 0u,
                 "Cluster BLAS scratch address is not 256-byte aligned.");
    blas_data.ResultAddressArray.StartAddress = address(_blas_destination);
    blas_data.ResultAddressArray.StrideInBytes = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    blas_data.IndirectArgumentArray.StartAddress = address(_blas_args);
    blas_data.IndirectArgumentArray.StrideInBytes = sizeof(blas_args);
    // A null count pointer selects MaxArgCount == 1 from _blas_inputs.
    blas_data.IndirectArgumentArraySize = 0u;

    D3D12_RTAS_OPERATION_DESC blas_desc{};
    blas_desc.Inputs.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
    blas_desc.Inputs.pClasDesc = &_blas_inputs;
    blas_desc.pBatchedOperationData = &blas_data;
    rt_commands->ExecuteIndirectRTASOperations(
        1u, &blas_desc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    // Publish the built Cluster BLAS to subsequent TLAS builds and ray queries.
    tracker.Record(GetAccelBuffer(), EnhancedBarrierTracker::Range{},
                   EnhancedBarrierTracker::Usage::ComputeAccelRead);
    tracker.UpdateState(&callback);
    _build_pending = false;
}

}// namespace lc::dx
