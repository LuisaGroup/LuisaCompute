/// StateMachineCoroScheduler implementation.
/// The template methods are defined in the header (state_machine.h) since
/// they depend on the coroutine's Args... parameter pack.
/// This translation unit serves as a compilation guard and contains
/// explicit template instantiations for common argument combinations.

#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute::coro {

// Explicit instantiations for common argument combinations.
template class StateMachineCoroScheduler<Buffer<int>>;
template class StateMachineCoroScheduler<int>;
template class StateMachineCoroScheduler<Buffer<float4>, uint2>;

}// namespace luisa::compute::coro
