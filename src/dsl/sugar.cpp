//
// Created by Mike on 9/11/2023.
//

#include <luisa/core/stl/format.h>
#include <luisa/dsl/sugar.h>

namespace luisa::compute::dsl_detail {
luisa::string format_source_location(const char *file, int line) noexcept {
    return luisa::format("{}:{}", file, line);
}
}// namespace luisa::compute::dsl_detail
