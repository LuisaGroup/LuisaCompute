#include <luisa/xir/function.h>

namespace luisa::compute::xir {

Function::Function(Pool *pool, FunctionTag tag, const Type *type, const Name *name) noexcept
    : Value{pool, type, name},
      _function_tag{tag}, _body{pool->create<FunctionBodyBlock>(this)},
      _arguments{pool}, _shared_variables{pool}, _local_variables{pool} {
    _arguments.head_sentinel()->set_parent_function(this);
    _arguments.tail_sentinel()->set_parent_function(this);
    _shared_variables.head_sentinel()->set_parent_function(this);
    _shared_variables.tail_sentinel()->set_parent_function(this);
    _local_variables.head_sentinel()->set_parent_function(this);
    _local_variables.tail_sentinel()->set_parent_function(this);
}

void Function::add_argument(Argument *argument) noexcept {
    _arguments.insert_back(argument);
}

void Function::add_shared_variable(SharedVariable *shared) noexcept {
    _shared_variables.insert_back(shared);
}

void Function::add_local_variable(LocalVariable *local) noexcept {
    _local_variables.insert_back(local);
}

}// namespace luisa::compute::xir
