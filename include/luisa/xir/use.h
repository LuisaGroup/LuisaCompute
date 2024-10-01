#pragma once

#include <luisa/xir/ilist.h>

namespace luisa::compute::xir {

class Value;
class User;

class LC_XIR_API Use : public IntrusiveForwardNode<Use> {

private:
    Value *_value = nullptr;
    User *_user = nullptr;

public:
    explicit Use(Pool *pool, Value *value = nullptr, User *user = nullptr) noexcept;

    // set value, also update the use list of the old and new values
    void set_value(Value *value) noexcept;
    // set user, also update the use list of the old and new users
    void set_user(User *user) noexcept;

    [[nodiscard]] auto value() noexcept { return _value; }
    [[nodiscard]] auto value() const noexcept { return const_cast<const Value *>(_value); }

    [[nodiscard]] auto user() noexcept { return _user; }
    [[nodiscard]] auto user() const noexcept { return const_cast<const User *>(_user); }
};

using UseList = IntrusiveForwardList<Use>;

}// namespace luisa::compute::xir
