#include <luisa/xir/function.h>

namespace luisa::compute::xir {

Function::Function(Pool *pool, Tag tag, const Type *type, const Name *name) noexcept
    : Value{pool, type, name}, _tag{tag}, _body{pool->create<BasicBlock>()} {}

void Function::add_argument(Argument *argument) noexcept {
    _arguments.emplace_back(argument);
}

}// namespace luisa::compute::xir
