#include <luisa/ir_v2/ir_v2.h>
#include <luisa/core/string_scratch.h>
namespace luisa::compute::ir_v2 {
bool Node::is_lvalue() const noexcept {
    if (is_local()) return true;
    if (is_gep()) return true;
    if (is_argument()) {
        auto arg = inst.as<ArgumentInst>();
        return !arg->by_value;
    }
    return false;
}
BasicBlock::BasicBlock(Pool &pool) noexcept {
    _first = pool.alloc<Node>(Instruction(InstructionTag::BASIC_BLOCK_SENTINEL), Type::of<void>());
    _last = pool.alloc<Node>(Instruction(InstructionTag::BASIC_BLOCK_SENTINEL), Type::of<void>());
    _first->next = _last;
    _last->prev = _first;
    _first->scope = this;
    _last->scope = this;
}
Node *IrBuilder::call(Func f, luisa::span<const Node *const> args_span, const Type *ty) noexcept {
    auto args = luisa::vector<Node *>{(Node *)args_span.begin(), (Node *)args_span.end()};
    auto call = Instruction(CallInst(std::move(f), args));
    return append(_pool->alloc<Node>(std::move(call), ty));
}

Node *IrBuilder::if_(const Node *cond, const BasicBlock *true_branch, const BasicBlock *false_branch) noexcept {
    auto if_ = Instruction(IfInst(const_cast<Node *>(cond), true_branch, false_branch));
    return append(_pool->alloc<Node>(std::move(if_), Type::of<void>()));
}
Node *IrBuilder::generic_loop(const BasicBlock *perpare, const Node *cond, const BasicBlock *body, const BasicBlock *after) noexcept {
    auto loop = Instruction(GenericLoopInst(perpare, const_cast<Node *>(cond), body, after));
    return append(_pool->alloc<Node>(std::move(loop), Type::of<void>()));
}
Node *IrBuilder::switch_(const Node *value, luisa::span<const SwitchCase> cases, const BasicBlock *default_branch) noexcept {
    luisa::vector<SwitchCase> cases_{cases.begin(), cases.end()};
    auto switch_ = Instruction(SwitchInst(const_cast<Node *>(value), cases_, default_branch));
    return append(_pool->alloc<Node>(std::move(switch_), Type::of<void>()));
}

void validate(Module &module) noexcept {
    luisa::unordered_set<const Node *> defined;
    for (auto arg : module.args) {
        defined.insert(arg);
    }
    if (module.kind() == Module::Kind::KERNEL) {
        auto kernel = static_cast<KernelModule *>(&module);
        // for (auto &c : kernel->captures) {
        //     defined.insert(c.node);
        // }
    }
    struct Visitor {
        std::function<void(const Node *)> visit_node;
        std::function<void(const BasicBlock *)> visit_block;
    };
    Visitor vis;
    auto check = [&](const Node *n) {
        if (!n) { return; }
        if (defined.contains(n)) { return; }
        LUISA_ERROR_WITH_LOCATION("use of undefined node: {}", (void *)n);
    };
    vis.visit_node = [&](const Node *node) {
        auto &inst = node->inst;
        auto tag = inst.tag();
        switch (tag) {
            case InstructionTag::IF: {
                auto if_ = inst.as<IfInst>();
                check(if_->cond);
                vis.visit_block(if_->true_branch);
                vis.visit_block(if_->false_branch);
            } break;
            case InstructionTag::LOCAL: {
                auto local = inst.as<LocalInst>();
                check(local->init);
            } break;
            case InstructionTag::GENERIC_LOOP: {
                auto loop = inst.as<GenericLoopInst>();
                vis.visit_block(loop->prepare);
                check(loop->cond);
                vis.visit_block(loop->body);
                vis.visit_block(loop->update);
            } break;
            case InstructionTag::CALL: {
                auto call = inst.as<CallInst>();
                for (auto arg : call->args) {
                    check(arg);
                }
            } break;
            case InstructionTag::SWITCH: {
                auto switch_ = inst.as<SwitchInst>();
                check(switch_->value);
                for (auto &c : switch_->cases) {
                    vis.visit_block(c.block);
                }
                vis.visit_block(switch_->default_);
            } break;
            case InstructionTag::RETURN: {
                auto ret = inst.as<ReturnInst>();
                check(ret->value);
            } break;
            default:
                break;
        }
        defined.insert(node);
    };
    vis.visit_block = [&](const BasicBlock *bb) {
        bb->for_each([&](Node *n) {
            vis.visit_node(n);
        });
    };
}
class IrDebugDump {
    StringScratch _scratch;
    luisa::unordered_map<const Node *, luisa::string> _node_id;
    luisa::unordered_map<const BasicBlock *, luisa::string> _block_id;
    size_t _indent = 0;
    void def(const Node *node, luisa::string name = "") {
        LUISA_ASSERT(!_node_id.contains(node), "node already defined");
        if (name.empty()) {
            name = luisa::format("${}", _node_id.size());
        } else {
            name = luisa::format("{}", name);
        }
        _node_id[node] = name;
    }

    luisa::string gen(const Node *node) {
        auto it = _node_id.find(node);
        if (it != _node_id.end()) { return it->second; }
        LUISA_ERROR_WITH_LOCATION("node {} not defined", (void *)node);
    }
    luisa::string gen(const BasicBlock *bb) {
        auto it = _block_id.find(bb);
        if (it != _block_id.end()) { return it->second; }
        LUISA_ERROR_WITH_LOCATION("block {} not defined", (void *)bb);
    }
    luisa::string gen_ty(const Type *ty) {
        return nullptr == ty ? luisa::string("void") : luisa::string(ty->description());
    }
    void gen_block_def(const BasicBlock *bb) {
        LUISA_ASSERT(!_block_id.contains(bb), "block already defined");
        auto label = luisa::format("$BB_{}:", _block_id.size());
        _block_id[bb] = label;
        writeln("{}: begin", label);
        with_indent([&] {
            bb->for_each([&](Node *n) {
                def(n);
                gen_def(n);
            });
        });
        writeln("{}: end", label);
    }
    void gen_def(const Node *node) {
        auto &inst = node->inst;
        auto ty = node->ty;
        auto tag = inst.tag();
        switch (tag) {
            case InstructionTag::ARGUMENT: {
                auto arg = inst.as<ArgumentInst>();
                writeln("{}, by_value:{}, ty: {}, ", gen(node), gen_ty(ty), arg->by_value);
                break;
            }
            case InstructionTag::BUFFER: [[fallthrough]];
            case InstructionTag::TEXTURE2D: [[fallthrough]];
            case InstructionTag::TEXTURE3D: [[fallthrough]];
            case InstructionTag::BINDLESS_ARRAY: [[fallthrough]];
            case InstructionTag::ACCEL: [[fallthrough]];
            case InstructionTag::SHARED: [[fallthrough]];
            case InstructionTag::UNIFORM: {
                writeln("{}, ty: {}, ", gen(node), gen_ty(ty));
                break;
            }
            case InstructionTag::BREAK: {
                (void)gen(node);
                writeln("break");
                break;
            }
            case InstructionTag::CONTINUE: {
                (void)gen(node);
                writeln("continue");
                break;
            }
            case InstructionTag::RETURN: {
                (void)gen(node);
                auto ret = inst.as<ReturnInst>();
                writeln("return {}", gen(ret->value));
                break;
            }
            case InstructionTag::IF: {
                auto if_ = inst.as<IfInst>();
                auto cond = gen(if_->cond);
                writeln("if {}", cond);
                gen_block_def(if_->true_branch);
                if (if_->false_branch) {
                    writeln("else");
                    gen_block_def(if_->false_branch);
                }
                break;
            }
            case InstructionTag::SWITCH: {
                auto sw = inst.as<SwitchInst>();
                auto value = gen(sw->value);
                writeln("switch {} {{", value);
                with_indent([&] {
                    for (auto &c : sw->cases) {
                        writeln("case {}=>", c.value);
                        gen_block_def(c.block);
                    }
                    writeln("default=>");
                    gen_block_def(sw->default_);
                });
                writeln("}");
                break;
            }
            case InstructionTag::GENERIC_LOOP: {
                auto loop = inst.as<GenericLoopInst>();
                writeln("loop {{");
                with_indent([&] {
                    writeln("prepare=>");
                    gen_block_def(loop->prepare);
                    writeln("break if !{}", gen(loop->cond));
                    writeln("body=>");
                    gen_block_def(loop->body);
                    writeln("update=>");
                    gen_block_def(loop->update);
                });
                writeln("}}");
                break;
            }
            case InstructionTag::RAY_QUERY: {
                auto rq = inst.as<RayQueryInst>();
                writeln("ray_query {}, ty: {} {{", gen(node), gen_ty(ty));
                with_indent([&] {
                    writeln("on_triangle_hit=>");
                    gen_block_def(rq->on_triangle_hit);
                    writeln("on_procedural_hit=>");
                    gen_block_def(rq->on_procedural_hit);
                });
                writeln("}}");
                break;
            }
            case InstructionTag::CALL: {
                auto call = inst.as<CallInst>();
                auto &func = call->func;
                auto args_ = StringScratch();
                for (auto arg : call->args) {
                    if (!args_.empty()) {
                        args_ << ", ";
                    }
                    args_ << gen(arg);
                }
                writeln("{} = {}({}), ty: {}", gen(node), tag_name(func.tag()), args_.string_view(), gen_ty(ty));
                break;
            }
            case InstructionTag::CONSTANT: {
                auto constant = inst.as<ConstantInst>();
                auto value = StringScratch();
                for (auto i = 0u; i < constant->value.size(); i++) {
                    if (i != 0) {
                        value << ", ";
                    }
                    value << constant->value[i];
                }
                writeln("{} = Constant({}), ty: {}", gen(node), value.string_view(), gen_ty(ty));
                break;
            }
            case InstructionTag::LOCAL: {
                auto local = inst.as<LocalInst>();
                writeln("{} = Local({}), ty: {}", gen(node), gen(local->init), gen_ty(ty));
                break;
            }
            case InstructionTag::UPDATE: {
                auto update = inst.as<UpdateInst>();
                writeln("store {}, {}", gen(node), gen(update->value), gen_ty(ty));
                break;
            }
            default:
                LUISA_ERROR_WITH_LOCATION("unsupported instruction tag: {}", tag_name(tag));
        }
    }
    void _dump(const Module &module) noexcept {
        _scratch << "module at " << (size_t)(void *)&module << " {\n";
        if (module.kind() == Module::Kind::KERNEL) {
            auto &km = static_cast<const KernelModule &>(module);
            writeln("block size: {}", km.block_size);
        }
        with_indent([&] {
            for (auto i = 0; i < module.args.size(); i++) {
                auto arg = module.args[i];
                def(arg, luisa::format("$arg_{}", i));
                gen_def(arg);
            }
            gen_block_def(module.entry);
        });
        _scratch << "}";
    }
    template<class F>
        requires std::invocable<F>
    void with_indent(F &&f) {
        _indent++;
        f();
        _indent--;
    }
    template<typename Fmt, class... Args>
    void writeln(Fmt &&fmt, Args &&...args) noexcept {
        for (size_t i = 0; i < _indent; i++) { _scratch << " "; }
        _scratch << luisa::format(std::forward<Fmt>(fmt), std::forward<Args>(args)...) << "\n";
    }
public:
    static luisa::string dump(Module &module) noexcept {
        IrDebugDump d;
        d._dump(module);
        return d._scratch.string();
    }
};
luisa::string dump_human_readable(Module &module) noexcept {
    return IrDebugDump::dump(module);
}
}// namespace luisa::compute::ir_v2