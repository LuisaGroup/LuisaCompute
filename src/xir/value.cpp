#include <luisa/xir/value.h>

namespace luisa::compute::xir {

Value::Value(Pool *pool, const Type *type, const Name *name) noexcept
    : PooledObject{pool}, _type{type}, _name{name} {}

}// namespace luisa::compute::xir
