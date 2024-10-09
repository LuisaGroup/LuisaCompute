#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/xir/constant.h>

namespace luisa::compute::xir {

Constant::Constant(Pool *pool, const Type *type, const void *data, const Name *name) noexcept
    : Value{pool, type, name} {
    LUISA_ASSERT(type != nullptr, "Constant type must be specified.");
    if (!_is_small()) {
        _large = luisa::allocate_with_allocator<std::byte>(type->size());
    }
    set_data(data);
}

Constant::~Constant() noexcept {
    if (!_is_small()) {
        luisa::deallocate_with_allocator(static_cast<std::byte *>(_large));
    }
}

bool Constant::_is_small() const noexcept {
    return type()->size() <= sizeof(void *);
}

void Constant::set_data(const void *data) noexcept {
    if (data == nullptr) {
        memset(this->data(), 0, type()->size());
    } else {
        memmove(this->data(), data, type()->size());
    }
}

void Constant::set_type(const Type *type) noexcept {
    LUISA_ERROR_WITH_LOCATION("Constant type cannot be changed.");
}

void *Constant::data() noexcept {
    return _is_small() ? _small : _large;
}

const void *Constant::data() const noexcept {
    return const_cast<Constant *>(this)->data();
}

}// namespace luisa::compute::xir
