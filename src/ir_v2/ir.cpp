#include <luisa/ir_v2/ir_v2.h>

namespace luisa::compute::ir_v2 {
bool Node::is_lvalue() const noexcept {
    if (this->is_local()) return true;
    if (this->is_gep()) return true;
    if (this->is_argument()) {
        auto arg = this->inst->as<Argument>();
        return !arg->by_value;
    }
    return false;
}
BasicBlock::BasicBlock(Pool &pool) noexcept {
    _first = pool.alloc<Node>();
    _last = pool.alloc<Node>();
    _first->inst = pool.alloc<BasicBlockSentinel>();
    _last->inst = pool.alloc<BasicBlockSentinel>();
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

void validate(Module &module) noexcept {
    luisa::unordered_set<Node *> defined;
    for (auto arg : module.args) {
        defined.insert(arg);
    }
    if (module.kind() == Module::Kind::KERNEL) {
        auto kernel = static_cast<KernelModule *>(&module);
        for (auto &c : kernel->captures) {
            defined.insert(c.node);
        }
    }
    struct Visitor {
        std::function<void(Node *)> visit_node;
        std::function<void(BasicBlock *)> visit_block;
    };
    Visitor vis;
    auto check = [&](Node *n) {
        if (!n) { return; }
        if (defined.contains(n)) { return; }
        LUISA_ERROR_WITH_LOCATION("use of undefined node: {}", (void *)n);
    };
    vis.visit_node = [&](Node *node) {
        auto inst = node->inst;
        auto tag = inst->tag();
        switch (tag) {
            case InstructionTag::IF: {
                auto if_ = inst->as<If>();
                check(if_->cond);
                vis.visit_block(if_->true_branch);
                vis.visit_block(if_->false_branch);
            } break;
            case InstructionTag::LOCAL: {
                auto local = inst->as<Local>();
                check(local->init);
            } break;
            case InstructionTag::GENERIC_LOOP: {
                auto loop = inst->as<GenericLoop>();
                vis.visit_block(loop->prepare);
                check(loop->cond);
                vis.visit_block(loop->body);
                vis.visit_block(loop->update);
            } break;
            case InstructionTag::CALL: {
                auto call = inst->as<Call>();
                for (auto arg : call->args) {
                    check(arg);
                }
            } break;
            case InstructionTag::SWITCH: {
                auto switch_ = inst->as<Switch>();
                check(switch_->value);
                for (auto &c : switch_->cases) {
                    vis.visit_block(c.block);
                }
                vis.visit_block(switch_->default_);
            } break;
            case InstructionTag::RETURN: {
                auto ret = inst->as<Return>();
                check(ret->value);
            } break;
            default:
                break;
        }
        defined.insert(node);
    };
    vis.visit_block = [&](BasicBlock *bb) {
        bb->for_each([&](Node *n) {
            vis.visit_node(n);
        });
    };
}

}// namespace luisa::compute::ir_v2