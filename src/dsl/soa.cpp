#include <luisa/core/logging.h>
#include <luisa/dsl/soa.h>

namespace luisa::compute::detail {

void error_soa_subview_out_of_range() noexcept {
    LUISA_ERROR_WITH_LOCATION("SOAView::subview out of range.");
}

void error_soa_view_exceeds_uint_max() noexcept {
    LUISA_ERROR_WITH_LOCATION("SOAView exceeds the maximum indexable size of 'uint'.");
}

void error_soa_index_out_of_range() noexcept {
    LUISA_ERROR_WITH_LOCATION("SOAView::operator[] out of range.");
}

}// namespace luisa::compute::detail
