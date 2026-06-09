/// PersistentThreadsCoroScheduler implementation.
/// The template methods are defined in the header (persistent.h) since
/// they depend on the coroutine's Args... parameter pack.
/// This translation unit serves as a compilation guard and contains
/// explicit template instantiations for common argument combinations.

#include <luisa/coro/schedulers/persistent.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute::coro {

// Explicit instantiations for common argument combinations.
template class PersistentThreadsCoroScheduler<>;
template class PersistentThreadsCoroScheduler<int>;
template class PersistentThreadsCoroScheduler<Buffer<int>>;
template class PersistentThreadsCoroScheduler<Buffer<uint>>;

}// namespace luisa::compute::coro
