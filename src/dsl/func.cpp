//
// Created by Mike Smith on 2022/12/25.
//

#include <core/logging.h>
#include <dsl/func.h>

namespace luisa::compute::detail {

void CallableInvoke::_error_too_many_arguments() noexcept {
    LUISA_ERROR_WITH_LOCATION("Too many arguments for callable.");
}

}// namespace luisa::compute::detail
