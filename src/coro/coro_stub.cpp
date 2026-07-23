#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/coro_frame.h>

namespace luisa::compute::detail {

void coroutine_chained_await_impl(
    CoroFrame &frame, size_t node_count,
    luisa::move_only_function<void(size_t, CoroFrame &)> node) noexcept {
    // Stub: the chained-await logic is compiled as a device-side helper.
    // In practice, coroutine execution is dispatched via a scheduler
    // (StateMachineCoroScheduler, PersistentThreadsCoroScheduler, etc.)
    // which handles token-chained execution on the GPU.
    // This host-side stub is provided so the API compiles.
    LUISA_WARNING("coroutine_chained_await_impl called on host — no-op stub");
}

} // namespace luisa::compute::detail
