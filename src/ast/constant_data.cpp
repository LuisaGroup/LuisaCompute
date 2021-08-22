//
// Created by Mike Smith on 2021/3/6.
//

#include <core/hash.h>
#include <util/spin_mutex.h>
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

ConstantData ConstantData::create(ConstantData::View data) noexcept {
    return std::visit(
        [](auto view) noexcept -> ConstantData {
            using T = std::remove_const_t<typename decltype(view)::value_type>;
            auto type = Type::of<T>();
            auto hash = hash64(view, type->hash());
            std::scoped_lock lock{detail::constant_registry_mutex()};
            if (auto iter = std::find_if(detail::constant_registry().cbegin(),
                                         detail::constant_registry().cend(),
                                         [hash](auto &&item) noexcept { return item._hash == hash; });
                iter != detail::constant_registry().cend()) { return *iter; }
            auto &&arena = Arena::global();
            auto ptr = arena.allocate<T>(view.size());
            std::memmove(ptr, view.data(), view.size_bytes());
            std::span<const T> new_view{ptr, view.size()};
            return detail::constant_registry().emplace_back(ConstantData{new_view, hash});
        },
        data);
}

}// namespace luisa::compute
