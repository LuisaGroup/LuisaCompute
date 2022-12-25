//
// Created by Mike Smith on 2021/3/6.
//

#include <core/stl/hash.h>
#include <core/spin_mutex.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] auto &constant_registry() noexcept {
    static luisa::vector<std::pair<ConstantData, luisa::vector<std::byte>>> r;
    return r;
}

[[nodiscard]] auto &constant_registry_mutex() noexcept {
    static spin_mutex m;
    return m;
}

}// namespace detail

ConstantData ConstantData::create(ConstantData::View data) noexcept {
    return luisa::visit(
        [](auto view) noexcept -> ConstantData {
            using T = std::remove_const_t<typename decltype(view)::value_type>;
            auto type = Type::of<T>();
            using namespace std::string_view_literals;
            static thread_local auto seed = hash_value("__hash_constant_data"sv);
            auto hash = hash_value(type->hash(), seed);
            hash = luisa::hash64(view.data(), view.size_bytes(), hash);
            std::scoped_lock lock{detail::constant_registry_mutex()};
            if (auto iter = std::find_if(
                    detail::constant_registry().cbegin(),
                    detail::constant_registry().cend(),
                    [hash](auto &&item) noexcept {
                        return item.first._hash == hash;
                    });
                iter != detail::constant_registry().cend()) { return iter->first; }
            luisa::vector<std::byte> storage(view.size_bytes());
            std::memcpy(storage.data(), view.data(), view.size_bytes());
            luisa::span<const T> new_view{reinterpret_cast<const T *>(storage.data()), view.size()};
            return detail::constant_registry()
                .emplace_back(std::make_pair(
                    ConstantData{new_view, hash}, std::move(storage)))
                .first;
        },
        data);
}

}// namespace luisa::compute
