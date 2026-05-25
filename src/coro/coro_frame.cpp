//
// Created by mike on 5/7/24.
//

#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include <luisa/coro/coro_token.h>
#include <luisa/coro/coro_frame.h>

namespace luisa::compute::coroutine {

CoroFrame::CoroFrame(luisa::shared_ptr<const CoroFrameDesc> desc, const Expression *expr) noexcept
    : _desc{std::move(desc)},
      _expression{expr},
      coro_id{this->get<uint3>(0u)},
      target_token{this->get<uint>(1u)} {
    LUISA_ASSERT(expr != nullptr, "CoroFrame expression must not be null.");
    LUISA_ASSERT(expr->type() == _desc->type(), "CoroFrame expression type mismatch.");
}

CoroFrame::CoroFrame(CoroFrame &&another) noexcept
    : CoroFrame{std::move(another._desc), another._expression} {
    another._expression = nullptr;
}

CoroFrame::CoroFrame(const CoroFrame &another) noexcept
    : CoroFrame{another._desc, [e = another.expression()]() noexcept {
                    auto fb = detail::FunctionBuilder::current();
                    auto copy = fb->local(e->type());
                    fb->assign(copy, e);
                    return copy;
                }()} {}

CoroFrame &CoroFrame::operator=(const CoroFrame &rhs) noexcept {
    if (this == std::addressof(rhs)) { return *this; }
    LUISA_ASSERT(this->description() == rhs.description(), "CoroFrame description mismatch.");
    auto fb = detail::FunctionBuilder::current();
    fb->assign(_expression, rhs.expression());
    return *this;
}

CoroFrame &CoroFrame::operator=(CoroFrame &&rhs) noexcept {
    return *this = static_cast<const CoroFrame &>(rhs);
}

CoroFrame CoroFrame::create(luisa::shared_ptr<const CoroFrameDesc> desc) noexcept {
    auto fb = detail::FunctionBuilder::current();
    // create an variable for the coro frame
    auto expr = fb->local(desc->type());
    // initialize the coro frame members
    auto zero_init = fb->call(desc->type(), CallOp::ZERO, {});
    fb->assign(expr, zero_init);
    return CoroFrame{std::move(desc), expr};
}

CoroFrame CoroFrame::create(luisa::shared_ptr<const CoroFrameDesc> desc, Expr<uint3> coro_id) noexcept {
    auto frame = create(std::move(desc));
    frame.coro_id = coro_id;
    return frame;
}

void CoroFrame::_check_member_index(uint index) const noexcept {
    LUISA_ASSERT(index < _desc->type()->members().size(),
                 "CoroFrame member index out of range.");
}

Var<bool> CoroFrame::is_terminated() const noexcept {
    return (target_token & coro_token_terminal) != 0u;
}

}// namespace luisa::compute::coroutine
