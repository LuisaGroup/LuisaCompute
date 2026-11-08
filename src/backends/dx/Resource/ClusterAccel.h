#pragma once

#include <Resource/BottomAccel.h>

namespace lc::dx {

// A DXR 2.0 Cluster BLAS and the CLAS objects it references. The class derives
// from BottomAccel only to reuse its TLAS back-reference bookkeeping and the
// accelBuffer address interface; classic BuildRaytracingAccelerationStructure
// is never used for this resource.
class ClusterAccel final : public BottomAccel {
    static constexpr size_t kIndexFormatCount = 3u;

    luisa::compute::AccelOption::UsageHint _usage_hint;
    vstd::unique_ptr<DefaultBuffer> _clas_buffer;
    uint64_t _clas_stride{0u};
    uint32_t _cluster_count{0u};
    uint32_t _sizes_cluster_count{~0u};
    bool _build_pending{false};

    D3D12_RTAS_OPERATION_FLAGS _operation_flags{D3D12_RTAS_OPERATION_FLAG_NONE};
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC _clas_inputs[kIndexFormatCount]{};
    D3D12_RTAS_CLAS_INPUTS_DESC _blas_inputs{};
    D3D12_RTAS_OPERATION_PREBUILD_INFO _clas_batch_info[kIndexFormatCount]{};
    D3D12_RTAS_OPERATION_PREBUILD_INFO _blas_info{};

    // Per-command temporary storage. Explicit CLAS destinations let the
    // conversion kernel publish the final addresses before any build starts.
    BufferView _counts;
    BufferView _clas_args[kIndexFormatCount];
    BufferView _clas_destinations[kIndexFormatCount];
    BufferView _clas_addresses;
    BufferView _blas_args;
    BufferView _blas_destination;
    size_t _scratch_offsets[kIndexFormatCount + 1u]{};

    void _query_build_sizes(uint32_t count);
    void _ensure_result_buffers(
        CommandBufferBuilder &builder,
        uint32_t count);
    void _record_input_states(
        EnhancedBarrierTracker &tracker,
        luisa::compute::MeshBuildCommand const *cmd) const;

public:
    ClusterAccel(Device *device, luisa::compute::AccelOption const &option);
    ~ClusterAccel() override = default;

    size_t PreProcessStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker,
        luisa::compute::MeshBuildCommand const *cmd);
    void BuildClusters(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const &scratch_buffer,
        luisa::compute::MeshBuildCommand const *cmd);

    [[nodiscard]] uint64_t cluster_blas_address() const noexcept {
        return accelBuffer ? accelBuffer->GetAddress() : 0u;
    }
};

}// namespace lc::dx
