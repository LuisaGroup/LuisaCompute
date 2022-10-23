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

    template<typename Impl, typename... Args>
        requires std::derived_from<Impl, T>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return emplace(luisa::make_unique<Impl>(std::forward<Args>(args)...));
    }

    template<typename Tag>
        requires is_integral_expr_v<Tag>
    void dispatch(Tag &&tag, const luisa::function<void(const T *)> &f) const noexcept {
        if (empty()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("No implementations registered.");
        }
        if (_impl.size() == 1u) {
            f(impl(0u));
        } else {
            detail::SwitchStmtBuilder{std::forward<Tag>(tag)} % [&] {
                for (auto i = 0u; i < _impl.size(); i++) {
                    detail::SwitchCaseStmtBuilder{i} % [&f, this, i] { f(impl(i)); };
                }
                detail::SwitchDefaultStmtBuilder{} % unreachable;
            };
        }
    }

    template<typename Tag>
        requires is_integral_expr_v<Tag>
    void dispatch_range(Tag &&tag, uint lo, uint hi,
                        const luisa::function<void(const T *)> &f) const noexcept {
        LUISA_ASSERT(lo < hi, "Invalid polymorphic tag range [{}, {}).", lo, hi);
        if (hi > _impl.size()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Out-of-bound polymorphic tag range [{}, {}). "
                "Registered tag count: {}.",
                lo, hi, _impl.size());
            hi = _impl.size();
        }
        if (hi == lo) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("No implementations registered.");
        }
        if (hi == lo + 1u) {// only one implementation
            f(impl(lo));
        } else {
            detail::SwitchStmtBuilder{std::forward<Tag>(tag)} % [&] {
                for (auto i = lo; i < hi; i++) {
                    detail::SwitchCaseStmtBuilder{i} % [&f, this, i] { f(impl(i)); };
                }
                detail::SwitchDefaultStmtBuilder{} % unreachable;
            };
        }
    }

    template<typename Tag, typename Group>
        requires is_integral_expr_v<Tag>
    void dispatch_group(Tag &&tag, const Group &group,
                        const luisa::function<void(const T *)> &f) const noexcept {
        luisa::vector<uint> tags;
        tags.reserve(std::size(group));
        for (auto &&t : group) {
            if (t < _impl.size()) {
                tags.emplace_back(t);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Out-of-bound polymorphic tag {}. "
                    "Registered tag count: {}.",
                    t, _impl.size());
            }
        }
        std::sort(tags.begin(), tags.end());
        tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
        if (tags.empty()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("No implementations registered.");
        }
        LUISA_ASSERT(group.size() > 0, "Empty polymorphic tag group.");
        if (tags.size() == 1u) {
            f(impl(tags.front()));
        } else {
            detail::SwitchStmtBuilder{std::forward<Tag>(tag)} % [&] {
                for (auto t : tags) {
                    detail::SwitchCaseStmtBuilder{t} % [&f, this, t] { f(impl(t)); };
                }
                detail::SwitchDefaultStmtBuilder{} % unreachable;
            };
        }
    }
};

}// namespace luisa::compute
