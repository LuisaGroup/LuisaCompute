//
// Created by Mike Smith on 2021/3/6.
//

#include <core/hash.h>
#include <core/spin_mutex.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] auto &constant_registry() noexcept {
    static ArenaVector<ConstantData> r{Arena::global()};
    return r;
}

[[nodiscard]] auto &constant_registry_mutex() noexcept {
    static spin_mutex m;
    return m;
}

}// namespace detail

uint64_t ConstantData::create(ConstantData::View data) noexcept {
    return std::visit(
        [](auto view) noexcept {
            using T = std::remove_const_t<typename decltype(view)::value_type>;
            auto type = Type::of<T>();
            auto hash = xxh3_hash64(view.data(), view.size_bytes(), type->hash());
            std::scoped_lock lock{detail::constant_registry_mutex()};
            if (std::none_of(
                    detail::constant_registry().cbegin(),
                    detail::constant_registry().cend(),
                    [hash](auto &&item) noexcept { return item._hash == hash; })) {
                auto ptr = Arena::global().allocate<T>(view.size());
                std::memmove(ptr, view.data(), view.size_bytes());
                std::span<const T> new_view{ptr, view.size()};
                detail::constant_registry().emplace_back(ConstantData{new_view, hash});
            }
            return hash;
        },
        data);
}

ConstantData::View ConstantData::view(uint64_t hash) noexcept {

    auto iter = std::find_if(
        detail::constant_registry().cbegin(),
        detail::constant_registry().cend(),
        [hash](auto &&item) noexcept { return item._hash == hash; });

    if (iter == detail::constant_registry().cend()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid constant data with hash {}.", hash);
    }
    return iter->_view;
}

}// namespace luisa::compute
