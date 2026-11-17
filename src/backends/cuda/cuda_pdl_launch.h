// Internal, dependency-free seam for the stream-form programmatic dependent
// launch (PDL) decision of the CUDA backend.
//
// The predicate is a pure function of shader, stream and device/driver state,
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

// What the stream slot the next launch enters currently holds. A relaxed launch
// is defined by CUDA as one that may start before "the previous kernel in the
// stream" completed, so the state that matters is exactly "is the immediate
// predecessor of this launch a kernel launch?".
enum class PdlPredecessor : uint32_t {
    // Nothing recorded yet: a fresh stream, a synchronize(), or the state was
    // invalidated after an error. Conservative: never relax.
    unknown = 0u,
    // The immediately preceding stream operation is a kernel launch. This is the
    // only state in which the launch attribute can pay for itself.
    kernel_launch = 1u,
    // The immediately preceding stream operation is anything else: a copy, a
    // build, a bindless-array update, a custom command, an event record/wait or a
    // host callback. There is no preceding kernel to overlap with (so the
    // attribute gains nothing) and for the ordering operations the effect of the
    // attribute is not documented (so it stays off).
    non_kernel = 2u,
};

// Every term of the stream-form PDL decision, named so a call site reads as a
// sentence and a unit test can build the full cube.
struct PdlLaunchConditions {
    // The shader itself waits for its programmatic dependency
    // (`grid_dependency_synchronize()`, recorded in CUDAShaderMetadata). Without
    // that wait an early launch would read stale data.
    bool shader_waits_for_dependency{false};
    // PdlPredecessor::kernel_launch (see above).
    bool predecessor_is_kernel{false};
    // Backend policy: CUDADeviceConfigExt + CUDAPdlStreamExt, minus the
    // LUISA_DISABLE_PDL_STREAM kill switch.
    bool stream_policy_enabled{false};
    // The per-shader latch: false once the driver refused the attribute.
    bool driver_accepts_attribute{false};
    // PDL is an sm_90+ device feature...
    uint32_t compute_capability{0u};
    // ...and the launch attribute belongs to the driver running this binary.
    uint32_t driver_version{0u};
};

[[nodiscard]] inline bool should_relax_stream_serialization(
    const PdlLaunchConditions &conditions) noexcept {
    return conditions.shader_waits_for_dependency &&
           conditions.predecessor_is_kernel &&
           conditions.stream_policy_enabled &&
           conditions.driver_accepts_attribute &&
           conditions.compute_capability >= 90u &&
           conditions.driver_version >= 12000u;
}

}// namespace luisa::compute::cuda::cuda_pdl_detail
