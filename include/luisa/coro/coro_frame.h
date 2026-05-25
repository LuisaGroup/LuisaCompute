//
// Created by mike on 5/7/24.
//

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/coro/coro_frame_desc.h>
#include <luisa/dsl/expr.h>

namespace luisa::compute::coroutine {

class LUISA_CORO_API CoroFrame {

private:
    luisa::shared_ptr<const CoroFrameDesc> _desc;
    const Expression *_expression;

public:
    Var<uint3> &coro_id;
    Var<uint> &target_token;

public:
    CoroFrame(luisa::shared_ptr<const CoroFrameDesc> desc, const Expression *expr) noexcept;
    CoroFrame(CoroFrame &&another) noexcept;
    CoroFrame(const CoroFrame &another) noexcept;
    CoroFrame &operator=(const CoroFrame &rhs) noexcept;
    CoroFrame &operator=(CoroFrame &&rhs) noexcept;

public:
    [[nodiscard]] static CoroFrame create(luisa::shared_ptr<const CoroFrameDesc> desc) noexcept;
    [[nodiscard]] static CoroFrame create(luisa::shared_ptr<const CoroFrameDesc> desc, Expr<uint3> coro_id) noexcept;

private:
    void _check_member_index(uint index) const noexcept;

public:
    [[nodiscard]] auto description() const noexcept { return _desc.get(); }
    [[nodiscard]] auto expression() const noexcept { return _expression; }

public:
    template<typename T>
    [[nodiscard]] Var<T> &get(uint index) noexcept {
        _check_member_index(index);
        auto fb = luisa::compute::detail::FunctionBuilder::current();
        auto member = fb->member(_desc->type()->members()[index], _expression, index);
        return *fb->create_temporary<Var<T>>(member);
    }
    template<typename T>
    [[nodiscard]] const Var<T> &get(uint index) const noexcept {
        return const_cast<CoroFrame *>(this)->get<T>(index);
    }
    template<typename T>
    [[nodiscard]] Var<T> &get(luisa::string_view name) noexcept {
        return get<T>(_desc->designated_field(name));
    }
    template<typename T>
    [[nodiscard]] const Var<T> &get(luisa::string_view name) const noexcept {
        return const_cast<CoroFrame *>(this)->get<T>(name);
    }
    [[nodiscard]] Var<bool> is_terminated() const noexcept;
};

}// namespace luisa::compute::coroutine
