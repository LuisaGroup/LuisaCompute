#include <luisa/ir_v2/ir_v2.h>

namespace luisa::compute::ir_v2 {
BasicBlock::BasicBlock(Pool &pool) noexcept {
    _first = pool.alloc<Node>();
    _last = pool.alloc<Node>();
    _first->inst = pool.alloc<BasicBlockSential>();
    _last->inst = pool.alloc<BasicBlockSential>();
    _first->next = _last;
    _last->prev = _first;
    _first->scope = this;
    _last->scope = this;
}
Node *IrBuilder::call(const Func *f, luisa::span<Node *> args_span, const Type *ty) noexcept {
    auto args = luisa::vector<Node *>{args_span.begin(), args_span.end()};
    auto call = _pool->alloc<Call>(f, args);
    return append(_pool->alloc<Node>(call, ty));
}

Node *IrBuilder::if_(Node *cond, BasicBlock *true_branch, BasicBlock *false_branch) noexcept {
    auto if_ = _pool->alloc<If>(cond, true_branch, false_branch);
    return append(_pool->alloc<Node>(if_, Type::of<void>()));
}
Node *IrBuilder::generic_loop(BasicBlock *perpare, Node *cond, BasicBlock *body, BasicBlock *after) noexcept {
    auto loop = _pool->alloc<GenericLoop>(perpare, cond, body, after);
    return append(_pool->alloc<Node>(loop, Type::of<void>()));
}
Node *IrBuilder::switch_(Node *value, luisa::span<SwitchCase> cases, BasicBlock *default_branch) noexcept {
    luisa::vector<SwitchCase> cases_{cases.begin(), cases.end()};
    auto switch_ = _pool->alloc<Switch>(value, cases_, default_branch);
    return append(_pool->alloc<Node>(switch_, Type::of<void>()));
}
}// namespace luisa::compute::ir_v2