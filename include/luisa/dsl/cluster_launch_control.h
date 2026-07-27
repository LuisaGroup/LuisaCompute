#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/builtin.h>

namespace luisa::compute {

/// Cluster Launch Control work-stealing loop for 1D kernels.
///
/// Implements the standard work-stealing pattern using NVIDIA Cluster Launch Control
/// (CUDA Blackwell, SM 10.0+). The user provides a computation body that processes
/// one thread block worth of work.
///
/// Usage:
/// \code
/// Kernel1D k = [&](BufferFloat data, UInt n) noexcept {
///     cluster_launch_control_work_stealing_1d([&](Int bx) noexcept {
///         auto i = bx * blockDim.x + threadIdx.x;
///         $if (i < n) {
///             data.write(i, data.read(i) * 2.0f);
///         };
///     });
/// };
/// \endcode
///
/// \tparam ComputeBody Callable with signature void(Int bx) — bx is the current block x-index.
template<typename ComputeBody>
    requires std::is_invocable_r_v<void, ComputeBody, Int>
void cluster_launch_control_work_stealing_1d(ComputeBody &&body) noexcept {
    Shared<uint4> result{1};
    Shared<ulong> bar{1};
    Var<int> phase = def(0);

    // Initialize barrier from thread 0
    if_(thread_x() == 0u, [&] {
        mbarrier_init(bar, 1u);
    });
    sync_block();

    // Initial block index
    Var<int> bx = def(block_x());

    loop([&] {
        sync_block();

        // Cancellation request from thread 0
        if_(thread_x() == 0u, [&] {
            fence_proxy_async_acquire();
            clc_try_cancel(result, bar);
            mbarrier_arrive_expect_tx(bar, 16u); // sizeof(uint4)
        });

        // Computation
        body(bx);

        // Wait for cancellation (while loop)
        loop([&] {
            if_(mbarrier_try_wait_parity(bar, phase), break_);
        });
        phase = phase ^ 1;

        // Check result
        if_(clc_query_is_canceled(result) != true, break_);

        // Get next block index
        bx = clc_query_get_ctaid_x(result);

        fence_proxy_async_release();
    });
}

} // namespace luisa::compute
