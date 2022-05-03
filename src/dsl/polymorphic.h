//
// Created by Mike Smith on 2022/5/3.
//

#pragma once

#include <core/stl.h>
#include <dsl/expr_traits.h>
#include <dsl/builtin.h>
#include <dsl/stmt.h>

namespace luisa::compute {

template<typename T>
class Polymorphic {

private:
    luisa::vector<luisa::unique_ptr<T>> _impl;

public:
    [[nodiscard]] auto size() const noexcept { return _impl.size(); }
    [[nodiscard]] auto impl(size_t i) noexcept { return _impl[i].get(); }
    [[nodiscard]] auto impl(size_t i) const noexcept { return _impl[i].get(); }

    // clang-format off
    template<typename Impl, typename... Args>
        requires std::derived_from<Impl, T>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        auto impl = luisa::make_unique<Impl>(std::forward<Args>(args)...);
        auto impl_ptr = impl.get();
        auto tag = static_cast<uint>(_impl.size());
        _impl.emplace_back(std::move(impl));
        return std::make_pair(tag, impl_ptr);
    }
    // clang-format on

    template<typename Tag, typename F>
        requires is_integral_expr_v<Tag> && std::invocable<F, const T *>
    void dispatch(Tag &&tag, F &&f) const noexcept {
        auto s = switch_(std::forward<Tag>(tag));
        for (auto i = 0u; i < _impl.size(); i++) {
            s = std::move(s).case_(i, [&f, this, i] { f(impl(i)); });
        }
        std::move(s).default_(unreachable);
    }
};

}// namespace luisa::compute
