#include "luisa/xir/value.h"

#include <luisa/xir/use.h>

namespace luisa::compute::xir {

Use::Use(Pool *pool, Value *value, User *user) noexcept : Super{pool} {
    set_value(value);
    set_user(user);
}

void Use::set_value(Value *value) noexcept {
    if (_value == value) { return; }
    remove_self();
    _value = value;
    if (_value != nullptr) {
        add_to_list(_value->use_list());
    }
}

void Use::set_user(User *user) noexcept {
    _user = user;
    if (!is_linked() && _value != nullptr) {
        add_to_list(_value->use_list());
    }
}

}// namespace luisa::compute::xir
