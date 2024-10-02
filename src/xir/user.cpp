#include <luisa/core/logging.h>
#include <luisa/xir/user.h>

namespace luisa::compute::xir {

void User::_remove_operand_uses() noexcept {
    for (auto o : _operands) {
        if (o != nullptr) {
            o->remove_self();
        }
    }
}

void User::_add_operand_uses() noexcept {
    for (auto o : _operands) {
        if (o != nullptr) {
            LUISA_DEBUG_ASSERT(o->user() == this, "Use::user() should be the same as this.");
            if (!o->is_linked() && o->value() != nullptr) {
                o->add_to_list(o->value()->use_list());
            }
        }
    }
}

void User::set_operand(size_t index, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    if (auto old = _operands[index]) {
        old->set_value(value);
    } else {
        auto use = pool()->create<Use>(value, this);
        _operands[index] = use;
    }
}

Use *User::operand_use(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    return _operands[index];
}

const Use *User::operand_use(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    return _operands[index];
}

Value *User::operand(size_t index) noexcept {
    return operand_use(index)->value();
}

const Value *User::operand(size_t index) const noexcept {
    return operand_use(index)->value();
}

void User::set_operand_count(size_t n) noexcept {
    for (auto i = n; i < _operands.size(); i++) {
        _operands[i]->remove_self();
    }
    _operands.resize(n);
}

void User::set_operands(luisa::span<Value *const> operands) noexcept {
    _remove_operand_uses();
    _operands.clear();
    for (auto o : operands) {
        auto use = pool()->create<Use>(o, this);
        _operands.emplace_back(use);
    }
}

void User::add_operand(Value *value) noexcept {
    auto use = pool()->create<Use>(value, this);
    _operands.emplace_back(use);
}

void User::insert_operand(size_t index, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(index <= _operands.size(), "Index out of range.");
    auto use = pool()->create<Use>(value, this);
    _operands.insert(_operands.cbegin() + index, use);
}

void User::remove_operand(size_t index) noexcept {
    if (index < _operands.size()) {
        _operands[index]->remove_self();
        _operands.erase(_operands.cbegin() + index);
    }
}

luisa::span<Use *> User::operand_uses() noexcept {
    return _operands;
}

luisa::span<const Use *const> User::operand_uses() const noexcept {
    return _operands;
}

size_t User::operand_count() const noexcept {
    return _operands.size();
}

}// namespace luisa::compute::xir
