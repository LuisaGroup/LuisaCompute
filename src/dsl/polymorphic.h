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
    [[nodiscard]] auto empty() const noexcept { return _impl.empty(); }
    [[nodiscard]] auto size() const noexcept { return _impl.size(); }
    [[nodiscard]] auto impl(size_t i) noexcept { return _impl.at(i).get(); }
    [[nodiscard]] auto impl(size_t i) const noexcept { return _impl.at(i).get(); }

    [[nodiscard]] auto emplace(luisa::unique_ptr<T> impl) noexcept {
        auto tag = static_cast<uint>(_impl.size());
        _impl.emplace_back(std::move(impl));
        return tag;
    }

    // clang-format off
    template<typename Impl, typename... Args>
        requires std::derived_from<Impl, T>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return emplace(luisa::make_unique<Impl>(std::forward<Args>(args)...));
    }
    // clang-format on

    template<typename Tag>
        requires is_integral_expr_v<Tag>
    void dispatch(Tag &&tag, const luisa::function<void(const T *)> &f) const noexcept {
        if (empty()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("No implementations registered.");
        }
        if (_impl.size() == 1u) {
            f(_impl.front().get());
        } else {
            detail::SwitchStmtBuilder{std::forward<Tag>(tag)} % [&] {
                for (auto i = 0u; i < _impl.size(); i++) {
                    detail::SwitchCaseStmtBuilder{i} % [&f, this, i] { f(impl(i)); };
                }
                detail::SwitchDefaultStmtBuilder{} % unreachable;
            };
        }
    }
};

}// namespace luisa::compute
