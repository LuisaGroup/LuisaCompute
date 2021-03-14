//
// Created by Mike Smith on 2021/3/6.
//

#include <core/hash.h>
#include <core/spin_mutex.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] std::vector<ConstantData> &constant_registry() noexcept {
    static std::vector<ConstantData> r;
    return r;
}

[[nodiscard]] auto &constant_registry_mutex() noexcept {
    static spin_mutex m;
    return m;
}

}

uint64_t ConstantData::create(ConstantData::View data) noexcept {
    return std::visit(
        [](auto view) noexcept {
            using T = std::remove_const_t<typename decltype(view)::value_type>;
            auto type = Type::of<T>();
            auto hash = xxh3_hash64(view.data(), view.size_bytes(), type->hash());
            std::scoped_lock lock{detail::constant_registry_mutex()};
            if (auto iter = std::find_if(
                    detail::constant_registry().cbegin(),
                    detail::constant_registry().cend(),
                    [hash](auto &&item) noexcept { return item._hash == hash; });
                iter == detail::constant_registry().cend()) {
                auto ptr = std::make_unique<std::byte[]>(view.size_bytes());
                std::memmove(ptr.get(), view.data(), view.size_bytes());
                std::span<const T> new_view{reinterpret_cast<T *>(ptr.get()), view.size()};
                detail::constant_registry().emplace_back(ConstantData{std::move(ptr), new_view, hash});
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
        LUISA_ERROR_WITH_LOCATION("Invalid constant data with hash {}.", hash);
    }
    return iter->_view;
}

}// namespace luisa::compute
