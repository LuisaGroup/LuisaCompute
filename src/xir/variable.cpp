#include <luisa/xir/variable.h>

namespace luisa::compute::xir {

Variable::Variable(Pool *pool,
                   Function *parent_function,
                   const Type *type,
                   const Name *name) noexcept
    : Value{pool, type, name},
      _parent_function{parent_function} {}

void Variable::set_parent_function(Function *func) noexcept {
    _parent_function = func;
}

Argument::Argument(Pool *pool, bool by_ref,
                   Function *parent_function,
                   const Type *type,
                   const Name *name) noexcept
    : Super{pool, parent_function, type, name},
      _by_ref{by_ref} {}

void Argument::remove_self() noexcept {
    Super::remove_self();
    set_parent_function(nullptr);
}

void Argument::insert_before_self(Argument *node) noexcept {
    Super::insert_before_self(node);
    node->set_parent_function(this->parent_function());
}

void Argument::insert_after_self(Argument *node) noexcept {
    Super::insert_after_self(node);
    node->set_parent_function(this->parent_function());
}

void LocalVariable::remove_self() noexcept {
    Super::remove_self();
    set_parent_function(nullptr);
}

void LocalVariable::insert_before_self(LocalVariable *node) noexcept {
    Super::insert_before_self(node);
    node->set_parent_function(this->parent_function());
}

void LocalVariable::insert_after_self(LocalVariable *node) noexcept {
    Super::insert_after_self(node);
    node->set_parent_function(this->parent_function());
}

void SharedVariable::remove_self() noexcept {
    Super::remove_self();
    set_parent_function(nullptr);
}

void SharedVariable::insert_before_self(SharedVariable *node) noexcept {
    Super::insert_before_self(node);
    node->set_parent_function(this->parent_function());
}

void SharedVariable::insert_after_self(SharedVariable *node) noexcept {
    Super::insert_after_self(node);
    node->set_parent_function(this->parent_function());
}

}// namespace luisa::compute::xir
