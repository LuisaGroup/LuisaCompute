#include <luisa/xir/argument.h>

namespace luisa::compute::xir {

Argument::Argument(const Type *type, bool by_ref, const Name *name) noexcept
    : Value{type, name}, _by_ref{by_ref} {}

}// namespace luisa::compute::xir
