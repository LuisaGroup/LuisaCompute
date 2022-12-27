//
// Created by Mike Smith on 2022/12/27.
//

#include <core/logging.h>
#include <dsl/ref.h>

namespace luisa::compute::detail {

void ref_dynamic_struct_error_member_out_of_range(size_t member_count, size_t index) noexcept {
    LUISA_ERROR_WITH_LOCATION("Member index {} out of range [0, {}).", index, member_count);
}

void ref_dynamic_struct_error_member_type_mismatched(const Type *requested, const Type *actual) noexcept {
    LUISA_ERROR_WITH_LOCATION("Member type mismatched: requested {}, actual {}.",
                              requested->description(), actual->description());
}

void ref_dynamic_struct_error_member_not_found(luisa::string_view name) noexcept {
    LUISA_ERROR_WITH_LOCATION("Member {} not found.", name);
}

}// namespace luisa::compute::detail
