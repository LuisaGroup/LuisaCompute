#include <luisa/xir/value.h>

namespace luisa::compute::xir {

Value::Value(const Type *type, const Name *name) noexcept
    : _type{type}, _name{name} {}

}// namespace luisa::compute::xir
