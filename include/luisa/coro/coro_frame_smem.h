//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/dsl/shared.h>
#include <luisa/coro/coro_frame.h>

namespace luisa::compute {

namespace detail {
[[noreturn]] LUISA_CORO_API void error_coro_frame_smem_subscript_on_soa() noexcept;
}// namespace detail

template<>
class Shared<coroutine::CoroFrame> {

private:
    luisa::shared_ptr<const coroutine::CoroFrameDesc> _desc;
    luisa::vector<const RefExpr *> _expressions;
    // we need to return a `CoroFrame &` for operator[] with proper lifetime
    luisa::vector<luisa::unique_ptr<coroutine::CoroFrame>> _temp_frames;
    size_t _size;

private:
    void _create(bool soa, luisa::span<const uint> soa_excluded) noexcept {
        auto fb = detail::FunctionBuilder::current();
        if (!soa) {
            auto s = fb->shared(Type::array(_desc->type(), _size));
            _expressions.emplace_back(s);
        } else {
            auto fields = _desc->type()->members();
            _expressions.reserve(fields.size());
            for (auto i = 0u; i < fields.size(); i++) {
                if (std::find(soa_excluded.begin(), soa_excluded.end(), i) != soa_excluded.end()) {
                    _expressions.emplace_back(nullptr);
                } else {
                    auto type = i == 0u ? Type::array(Type::of<uint>(), 3u) : fields[i];
                    auto s = fb->shared(Type::array(type, _size));
                    _expressions.emplace_back(s);
                }
            }
        }
    }

public:
    Shared(luisa::shared_ptr<const coroutine::CoroFrameDesc> desc,
           size_t n, bool soa = false,
           luisa::span<const uint> soa_excluded_fields = {}) noexcept
        : _desc{std::move(desc)}, _size{n} { _create(soa, soa_excluded_fields); }

    Shared(Shared &&) noexcept = default;
    Shared(const Shared &) noexcept = delete;
    Shared &operator=(Shared &&) noexcept = delete;
    Shared &operator=(const Shared &) noexcept = delete;

    [[nodiscard]] auto desc() const noexcept { return _desc.get(); }
    [[nodiscard]] auto is_soa() const noexcept { return _expressions.size() > 1; }
    [[nodiscard]] auto size() const noexcept { return _size; }

private:
    /// Read index with active fields
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto _read(I &&index, luisa::optional<luisa::span<const uint>> active_fields) const noexcept {
        auto i = def(std::forward<I>(index));
        auto fb = detail::FunctionBuilder::current();
        auto frame = fb->local(_desc->type());
        if (!is_soa()) {
            auto expr = fb->access(_desc->type(), _expressions[0], i.expression());
            fb->assign(frame, expr);
        } else {
            auto fields = _desc->type()->members();
            for (auto m = 0u; m < fields.size(); m++) {
                if (_expressions[m] == nullptr) { continue; }
                if (active_fields && std::find(active_fields->begin(), active_fields->end(), m) == active_fields->end()) { continue; }
                auto f = fb->member(fields[m], frame, m);
                if (m == 0u) {
                    auto t = Type::of<uint>();
                    auto s = fb->access(Type::array(t, 3u), _expressions[m], i.expression());
                    std::array<const Expression *, 3u> elems{};
                    elems[0] = fb->access(t, s, fb->literal(t, 0u));
                    elems[1] = fb->access(t, s, fb->literal(t, 1u));
                    elems[2] = fb->access(t, s, fb->literal(t, 2u));
                    auto v = fb->call(Type::of<uint3>(), CallOp::MAKE_UINT3, elems);
                    fb->assign(f, v);
                } else {
                    auto s = fb->access(fields[m], _expressions[m], i.expression());
                    fb->assign(f, s);
                }
            }
        }
        return coroutine::CoroFrame{_desc, frame};
    }

    /// Write index with active fields
    template<typename I>
        requires is_integral_expr_v<I>
    void _write(I &&index, const coroutine::CoroFrame &frame, luisa::optional<luisa::span<const uint>> active_fields) const noexcept {
        auto i = def(std::forward<I>(index));
        auto fb = detail::FunctionBuilder::current();
        if (!is_soa()) {
            auto ref = fb->access(_desc->type(), _expressions[0], i.expression());
            fb->assign(ref, frame.expression());
        } else {
            auto fields = _desc->type()->members();
            for (auto m = 0u; m < fields.size(); m++) {
                if (_expressions[m] == nullptr) { continue; }
                if (active_fields && std::find(active_fields->begin(), active_fields->end(), m) == active_fields->end()) { continue; }
                auto f = fb->member(fields[m], frame.expression(), m);
                if (m == 0u) {
                    auto t = Type::of<uint>();
                    auto s = fb->access(Type::array(t, 3u), _expressions[m], i.expression());
                    for (auto j = 0u; j < 3u; j++) {
                        auto fj = fb->swizzle(t, f, 1u, j);
                        auto sj = fb->access(t, s, fb->literal(t, j));
                        fb->assign(sj, fj);
                    }
                } else {
                    auto s = fb->access(fields[m], _expressions[m], i.expression());
                    fb->assign(s, f);
                }
            }
        }
    }

public:
    /// Reference to the i-th element
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] coroutine::CoroFrame &operator[](I &&index) noexcept {
        if (is_soa()) { detail::error_coro_frame_smem_subscript_on_soa(); }
        auto fb = detail::FunctionBuilder::current();
        auto ref = fb->access(
            _desc->type(), _expressions[0],
            detail::extract_expression(std::forward<I>(index)));
        auto temp_frame = luisa::make_unique<coroutine::CoroFrame>(_desc, ref);
        return *_temp_frames.emplace_back(std::move(temp_frame));
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] coroutine::CoroFrame read(I &&index) const noexcept {
        return _read(std::forward<I>(index), luisa::nullopt);
    }
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] coroutine::CoroFrame read(I &&index, luisa::span<const uint> active_fields) const noexcept {
        return _read(std::forward<I>(index), luisa::make_optional(active_fields));
    }
    template<typename I>
        requires is_integral_expr_v<I>
    void write(I &&index, const coroutine::CoroFrame &frame) const noexcept {
        _write(std::forward<I>(index), frame, luisa::nullopt);
    }
    template<typename I>
        requires is_integral_expr_v<I>
    void write(I &&index, const coroutine::CoroFrame &frame, luisa::span<const uint> active_fields) const noexcept {
        _write(std::forward<I>(index), frame, luisa::make_optional(active_fields));
    }
};

}// namespace luisa::compute
