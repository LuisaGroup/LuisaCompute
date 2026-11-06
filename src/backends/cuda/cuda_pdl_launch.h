// Internal, dependency-free seam for the stream-form programmatic dependent
// launch (PDL) decision of the CUDA backend.
//
// The predicate is a pure function of shader state and device/driver versions,
// so it lives in a header instead of `cuda_shader_native.cpp` to make the
// decision matrix host-testable (src/tests/unit/core/
// test_cuda_pdl_launch_predicate.cpp). It must stay free of CUDA API
// dependencies (<atomic>, <cstdint> only) so a host-only test can include it
// without the CUDA driver, exactly like the capability struct in
// include/luisa/backends/ext/cuda/cuda_graph_ext.h.

#pragma once

#include <atomic>
#include <cstdint>

namespace luisa::compute::cuda::cuda_pdl_detail {

// Stream-form programmatic dependent launch. `cuLaunchKernel` keeps the grids of
// one stream strictly serialized; a kernel that waits for its programmatic
// dependency (`grid_dependency_synchronize()`) can instead be launched through
// `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`,
// which allows the driver to start this grid before its predecessor completed.
// Everything else takes the very same `cuLaunchKernel` call as before.
[[nodiscard]] inline bool should_relax_stream_serialization(
    bool uses_grid_dependency_synchronize,
    const std::atomic_bool &latch, uint32_t compute_capability,
    uint32_t driver_version) noexcept {
    // The shader opts in by waiting for its programmatic dependency: the wait is
    // what makes an early launch correct, so a kernel that carries no such call
    // falls back to the classic serialized `cuLaunchKernel` launch of
    // cuda_shader_native_detail::launch_kernel. Note that the whole
    // predicate is about this shader, not about the grid's index inside a
    // multi-dispatch command: the canonical producer -> consumer chain is two
    // commands on one stream, i.e. two *first* grids, so gating on the kernel id
    // would make the feature inert for exactly the case it exists for. When the
    // predecessor is not a kernel the attribute yields no overlap, but it also
    // changes nothing, because the wait is still correct.
    // PDL is a sm_90+ device feature, and the launch attribute belongs to the
    // driver that is running this binary.
    return uses_grid_dependency_synchronize &&
           latch.load(std::memory_order_relaxed) &&
           compute_capability >= 90u &&
           driver_version >= 12000u;
}

}// namespace luisa::compute::cuda::cuda_pdl_detail
