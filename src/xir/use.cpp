#include "luisa/xir/value.h"

#include <luisa/xir/use.h>

namespace luisa::compute::xir {

Use::Use(Value *value, User *user) noexcept {
    set_value(value);
    set_user(user);
}

void Use::set_value(Value *value) noexcept {
    if (_value == value) { return; }
    remove_self();
    _value = value;
    _value->use_list().insert_front(this);
}

void Use::set_user(User *user) noexcept {
    _user = user;
}

}// namespace luisa::compute::xir
