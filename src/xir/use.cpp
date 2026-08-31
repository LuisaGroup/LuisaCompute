#include <luisa/core/logging.h>
#include <luisa/xir/value.h>
#include <luisa/xir/user.h>
#include <luisa/xir/use.h>

namespace luisa::compute::xir {

Use::Use(User *user, Value *value) noexcept : _user{user}, _value{value} {
    LUISA_DEBUG_ASSERT(user != nullptr, "User must not be null.");
}

Use *UseList::push_front(ManagedPtr<Use> use) noexcept {
    LUISA_DEBUG_ASSERT(
        use != nullptr && use->_list_owner == nullptr &&
            !use->is_linked(),
        "Use is already linked to an owner list.");
    auto *node = _nodes.push_front(std::move(use));
    node->_list_owner = this;
    return node;
}

ManagedPtr<Use> Use::remove_self() noexcept {
    auto was_linked = is_linked();
    LUISA_DEBUG_ASSERT(
        was_linked == (_list_owner != nullptr),
        "Use intrusive linkage and owner-list identity disagree.");
    auto self = Super::remove_self();
    if (self != nullptr) {
        LUISA_DEBUG_ASSERT(
            was_linked && self.get() == this,
            "Removed Use ownership is inconsistent.");
        _list_owner = nullptr;
    }
    return self;
}

void Use::set_value(Value *value) noexcept {
    validate_canary();
    _value = value;
}

}// namespace luisa::compute::xir
