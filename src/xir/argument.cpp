#include <luisa/xir/argument.h>

namespace luisa::compute::xir {

Argument::Argument(Pool *pool, const Type *type,
                   bool by_ref, const Name *name) noexcept
    : Value{pool, type, name}, _by_ref{by_ref} {}

}// namespace luisa::compute::xir
