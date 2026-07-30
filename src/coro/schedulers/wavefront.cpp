/// WavefrontCoroScheduler implementation.
/// The template methods are defined in the header (wavefront.h) since
/// they depend on the coroutine's Args... parameter pack.
/// This translation unit serves as a compilation guard and contains
/// explicit template instantiations for common argument combinations.

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute::coro {

// Explicit instantiations for common argument combinations.
template class WavefrontCoroScheduler<Buffer<int>>;
template class WavefrontCoroScheduler<Buffer<uint>>;
template class WavefrontCoroScheduler<int>;
template class WavefrontCoroScheduler<>;

}// namespace luisa::compute::coro
