#pragma once

#include <EASTL/variant.h>

namespace luisa {

using eastl::get;
using eastl::get_if;
using eastl::holds_alternative;
using eastl::monostate;
using eastl::variant;
using eastl::variant_alternative_t;
using eastl::variant_size_v;
using eastl::visit;

}// namespace luisa
