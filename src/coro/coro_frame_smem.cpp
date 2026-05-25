#include <luisa/core/logging.h>

namespace luisa::compute::detail {
void error_coro_frame_smem_subscript_on_soa() noexcept {
    LUISA_ERROR_WITH_LOCATION("Subscripting on SOA shared frame is not allowed.");
}
}// namespace luisa::compute::detail
