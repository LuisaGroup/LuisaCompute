#pragma once

#include <luisa/core/managed_ilist.h>
#include <luisa/xir/traits.h>

namespace luisa::compute::xir {

class Value;
class User;
class UseList;

class LUISA_XIR_API Use final : public ManagedIntrusiveForwardNode<Use> {

private:
    friend class UseList;
    User *_user;
    Value *_value;
    UseList *_list_owner{nullptr};

public:
    explicit Use(User *user, Value *value = nullptr) noexcept;
    ManagedPtr<Use> remove_self() noexcept override;
    void set_value(Value *value) noexcept;

    [[nodiscard]] auto value() noexcept {
        validate_canary();
        return _value;
    }
    [[nodiscard]] auto value() const noexcept {
        return const_cast<const Value *>(_value);
    }
    [[nodiscard]] auto user() noexcept {
        validate_canary();
        return _user;
    }
    [[nodiscard]] auto user() const noexcept {
        return const_cast<const User *>(_user);
    }
};

// A Use's logical operand value and its physical intrusive-list owner are two
// independently verifiable relations. The wrapper keeps the physical owner on
// every linked node, making exact membership an O(1) identity predicate while
// retaining O(1) insertion and removal.
class LUISA_XIR_API UseList final {

private:
    ManagedIntrusiveForwardList<Use> _nodes;

public:
    UseList() noexcept = default;

    [[nodiscard]] bool empty() const noexcept { return _nodes.empty(); }
    [[nodiscard]] Use *front() noexcept { return _nodes.front(); }
    [[nodiscard]] const Use *front() const noexcept { return _nodes.front(); }
    [[nodiscard]] size_t count_size() const noexcept { return _nodes.count_size(); }
    [[nodiscard]] bool contains(const Use *use) const noexcept {
        return use != nullptr &&
               use->_list_owner == this &&
               use->is_linked();
    }

    [[nodiscard]] auto begin() noexcept { return _nodes.begin(); }
    [[nodiscard]] auto begin() const noexcept { return _nodes.begin(); }
    [[nodiscard]] auto end() const noexcept { return _nodes.end(); }
    [[nodiscard]] auto cbegin() const noexcept { return _nodes.cbegin(); }
    [[nodiscard]] auto cend() const noexcept { return _nodes.cend(); }

    Use *push_front(ManagedPtr<Use> use) noexcept;
    [[nodiscard]] ManagedPtr<Use> pop_front() noexcept {
        return _nodes.pop_front();
    }
};

}// namespace luisa::compute::xir
