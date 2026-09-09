#include <luisa/xir/passes/coro_call.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/constant.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <algorithm>
#include "coro_call_alias.h"
#include "helpers.h"

namespace luisa::compute::xir {
namespace {

struct Region final : InstructionCloneValueResolver {
    Function *function{nullptr};
    Function *root{nullptr};
    size_t id{0u};
    luisa::unordered_map<const Value *, Value *> values;
    luisa::unordered_map<uint32_t, uint32_t> tokens;
    luisa::vector<AllocaInst *> parameters;
    AllocaInst *return_site{nullptr};
    AllocaInst *result{nullptr};
    BasicBlock *entry{nullptr};
    IndexedBranchInst *return_dispatch{nullptr};

    Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (auto it = values.find(value); it != values.end()) { return it->second; }
        switch (value->derived_value_tag()) {
            case DerivedValueTag::CONSTANT:
            case DerivedValueTag::UNDEFINED:
            case DerivedValueTag::FUNCTION:
            case DerivedValueTag::SPECIAL_REGISTER: return const_cast<Value *>(value);
            default: break;
        }
        LUISA_ERROR("Shared coroutine callable has an unresolved value.");
    }
};

class Lowering {
    Function *_root;
    luisa::unordered_map<Function *, uint32_t> _state;
    luisa::unordered_map<Function *, bool> _suspending;
    luisa::vector<Function *> _order;
    luisa::unordered_map<Function *, luisa::unique_ptr<Region>> _regions;
    luisa::unordered_set<uint32_t> _used_tokens;
    uint32_t _next_token{1u};
    CoroCallInfo _info;
    luisa::unordered_map<CallInst *, uint32_t> _call_owners;

    uint32_t _allocate_token() {
        // Root IDs belong to the public coroutine ABI and can be sparse,
        // including TERMINAL_TOKEN - 1. Allocate from the unused IDs instead
        // of incrementing the largest source ID into the reserved terminal.
        while (_next_token != TERMINAL_TOKEN && _used_tokens.contains(_next_token)) {
            ++_next_token;
        }
        LUISA_ASSERT(_next_token != TERMINAL_TOKEN,
                     "Coroutine suspend token space is exhausted.");
        auto token = _next_token++;
        _used_tokens.emplace(token);
        return token;
    }

    bool _visit(Function *f) {
        auto &state = _state[f];
        LUISA_ASSERT(state != 1u, "Recursive coroutine calls are unsupported.");
        if (state == 2u) { return _suspending.at(f); }
        state = 1u;
        bool suspends = false;
        if (auto *def = f->definition()) {
            for (auto *bb : def->basic_blocks()) {
                for (auto *inst : bb->instructions()) {
                    suspends |= inst->isa<CoroSuspendInst>();
                    if (inst->isa<CallInst>()) {
                        suspends |= _visit(static_cast<CallInst *>(inst)->callee());
                    } else {
                        for (size_t i = 0u; i < inst->operand_count(); ++i) {
                            auto *operand = inst->operand(i);
                            if (operand != nullptr && operand->isa<Function>()) {
                                LUISA_ASSERT(!_visit(static_cast<Function *>(operand)),
                                             "Suspension in a synchronous callback is unsupported.");
                            }
                        }
                    }
                }
            }
        }
        _state[f] = 2u;
        _suspending[f] = suspends;
        if (suspends && f != _root) { _order.emplace_back(f); }
        return suspends;
    }

public:
    explicit Lowering(Function *root) noexcept : _root{root} {}

    CoroCallInfo run() {
        _visit(_root);
        if (_order.empty()) { return _info; }
        auto normalize = [](Function *f) {
            auto info = destructure_cfg_pass_run_on_function(f);
            LUISA_ASSERT(info.succeeded(), "Shared coroutine call CFG normalization failed.");
            for (auto *bb : f->definition()->basic_blocks()) {
                for (auto *inst : bb->instructions()) {
                    LUISA_ASSERT(!inst->isa<PhiInst>(), "Shared coroutine call lowering requires PHI-free XIR.");
                }
            }
        };
        _info.graph.functions.emplace_back();
        normalize(_root);
        for (auto *f : _order) { normalize(f); }
        for (auto *bb : _root->definition()->basic_blocks()) {
            for (auto *inst : bb->instructions()) {
                if (inst->isa<CoroSuspendInst>()) {
                    auto token = static_cast<CoroSuspendInst *>(inst)->token();
                    _info.graph.functions.front().resume_tokens.emplace_back(token);
                    _used_tokens.emplace(token);
                }
            }
        }
        // Build all activation storage in the root entry. Callee allocas are
        // hoisted here too: their lifetime is the complete suspended task.
        detail::CoroCallAliasLowering aliases{_root};
        XIRBuilder prologue;
        prologue.set_insertion_point(_root->definition()->body_block()->instructions().head_sentinel());
        luisa::vector<AllocaInst *> root_allocas;
        for (auto *bb : _root->definition()->basic_blocks()) {
            for (auto *inst : bb->instructions()) {
                if (inst->isa<AllocaInst>()) { root_allocas.emplace_back(static_cast<AllocaInst *>(inst)); }
            }
        }
        for (auto *alloca : root_allocas) { prologue.append(alloca->remove_self()); }
        for (auto *f : _order) {
            auto region = luisa::make_unique<Region>();
            auto &r = *region;
            r.function = f;
            r.root = _root;
            r.id = ++_info.callable_count;
            r.return_site = prologue.alloca_local(Type::of<uint32_t>());
            r.return_site->set_coro_return_selector(static_cast<uint32_t>(r.id));
            prologue.store(r.return_site, _root->parent_module()->create_constant_zero(Type::of<uint32_t>()));
            if (f->type() != nullptr) { r.result = prologue.alloca_local(f->type()); }
            for (auto *arg : f->arguments()) {
                if (arg->is_resource() || arg->is_reference()) {
                    r.parameters.emplace_back(nullptr);
                    r.values[arg] = arg;
                    aliases.add_parameter(arg, prologue);
                } else {
                    r.parameters.emplace_back(prologue.alloca_local(arg->type()));
                }
            }
            for (auto *bb : f->definition()->basic_blocks()) {
                r.values[bb] = _root->create_basic_block();
                for (auto *inst : bb->instructions()) {
                    if (inst->isa<AllocaInst>()) {
                        r.values[inst] = inst->clone_with_metadata(prologue, r);
                    }
                    if (inst->isa<CoroSuspendInst>()) {
                        auto local_token = static_cast<CoroSuspendInst *>(inst)->token();
                        LUISA_ASSERT(!r.tokens.contains(local_token),
                                     "Duplicate suspend token in a coroutine callable.");
                        r.tokens.emplace(local_token, _allocate_token());
                    }
                }
            }
            r.entry = static_cast<BasicBlock *>(r.resolve(f->definition()->body_block()));
            XIRBuilder b;
            auto *exit = _root->create_basic_block();
            b.set_insertion_point(exit);
            auto *return_tag = b.load(Type::of<uint32_t>(), r.return_site);
            b.store(r.return_site, _root->parent_module()->create_constant_zero(Type::of<uint32_t>()));
            r.return_dispatch = b.indexed_branch(return_tag);
            auto *invalid = _root->create_basic_block();
            r.return_dispatch->set_default_block(invalid);
            b.set_insertion_point(invalid);
            b.unreachable_("Invalid coroutine callable return site.");
            CoroCallGraph::Function function;
            function.id = static_cast<uint32_t>(r.id);
            for (auto [local, global] : r.tokens) { function.resume_tokens.emplace_back(global); }
            std::sort(function.resume_tokens.begin(), function.resume_tokens.end());
            _info.graph.functions.emplace_back(std::move(function));
            _regions.emplace(f, std::move(region));
        }
        // Clone each definition once. RPO includes the synthetic resume edge
        // emitted by AST-to-XIR; all allocas and blocks are already mapped.
        for (auto *f : _order) {
            auto &r = *_regions.at(f);
            XIRBuilder b;
            b.set_insertion_point(r.entry);
            size_t index = 0u;
            for (auto *arg : f->arguments()) {
                if (auto *slot = r.parameters[index++]) { r.values[arg] = b.load(arg->type(), slot); }
            }
            f->definition()->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *bb) noexcept {
                b.set_insertion_point(static_cast<BasicBlock *>(r.resolve(bb)));
                for (auto *inst : bb->instructions()) {
                    if (inst->isa<AllocaInst>()) { continue; }
                    if (inst->isa<ReturnInst>()) {
                        auto *ret = static_cast<ReturnInst *>(inst);
                        if (r.result != nullptr) { b.store(r.result, r.resolve(ret->return_value())); }
                        b.br(r.return_dispatch->parent_block());
                    } else if (inst->isa<CoroSuspendInst>()) {
                        auto *s = static_cast<CoroSuspendInst *>(inst);
                        luisa::vector<luisa::string> names;
                        luisa::vector<Value *> exports, bindings;
                        luisa::vector<CoroSuspendExtensionPtr> extensions;
                        for (size_t i = 0u; i < s->frame_export_count(); ++i) {
                            names.emplace_back(luisa::format("callable{}.{}", r.id, s->frame_export_name(i)));
                            exports.emplace_back(r.resolve(s->frame_export_value(i)));
                        }
                        for (auto &&extension : s->extensions()) { extensions.emplace_back(extension->clone()); }
                        for (size_t i = 0u; i < s->extension_binding_value_count(); ++i) { bindings.emplace_back(r.resolve(s->extension_binding_value(i))); }
                        b.coro_suspend(r.tokens.at(s->token()), s->name(), nullptr, names, exports, std::move(extensions), bindings);
                    } else if (inst->isa<CoroResumeInst>()) {
                        b.coro_resume(r.tokens.at(static_cast<CoroResumeInst *>(inst)->token()), nullptr);
                    } else {
                        auto *clone = inst->clone_with_metadata(b, r);
                        r.values[inst] = clone;
                        if (clone->isa<CallInst>()) {
                            _call_owners.emplace(static_cast<CallInst *>(clone), static_cast<uint32_t>(r.id));
                        }
                    }
                }
            });
        }
        for (auto *f : _order) {
            auto &r = *_regions.at(f);
            for (auto *bb : f->definition()->basic_blocks()) {
                auto *clone = static_cast<BasicBlock *>(r.resolve(bb));
                if (!clone->is_terminated()) {
                    XIRBuilder b;
                    b.set_insertion_point(clone);
                    b.unreachable_();
                }
            }
        }
        luisa::vector<CallInst *> calls;
        for (auto *bb : _root->definition()->basic_blocks()) {
            for (auto *inst : bb->instructions()) {
                if (inst->isa<CallInst>() && _regions.contains(static_cast<CallInst *>(inst)->callee())) {
                    calls.emplace_back(static_cast<CallInst *>(inst));
                }
            }
        }
        // Bind alias families in caller-before-callee order. Region IDs were
        // assigned bottom up; root is the only special (zero) ID.
        std::stable_sort(calls.begin(), calls.end(), [&](CallInst *a, CallInst *b) {
            auto owner = [&](CallInst *call) { auto i = _call_owners.find(call); return i == _call_owners.end() ? UINT32_MAX : i->second; };
            return owner(a) > owner(b);
        });
        for (auto *call : calls) {
            auto &r = *_regions.at(call->callee());
            auto *before = call->parent_block();
            auto *after = _root->create_basic_block();
            luisa::vector<Instruction *> tail;
            for (auto *inst = call->next(); inst != before->instructions().tail_sentinel(); inst = inst->next()) { tail.emplace_back(inst); }
            XIRBuilder b;
            b.set_insertion_point(call);
            for (size_t i = 0u; i < r.parameters.size(); ++i) {
                if (auto *slot = r.parameters[i]) { b.store(slot, call->argument(i)); }
            }
            aliases.bind_call(call, b, prologue);
            auto site = static_cast<uint32_t>(++_info.call_site_count);
            b.store(r.return_site, _root->parent_module()->create_constant(Type::of<uint32_t>(), &site));
            b.br(r.entry);
            b.set_insertion_point(after);
            if (r.result != nullptr) { call->replace_all_uses_with(b.load(call->type(), r.result)); }
            for (auto *inst : tail) { b.append(inst->remove_self()); }
            call->remove_self();
            r.return_dispatch->add_case(site, after);
            auto owner = _call_owners.find(call);
            _info.graph.edges.push_back({owner == _call_owners.end() ? 0u : owner->second,
                                         static_cast<uint32_t>(r.id), site});
        }
        aliases.lower_uses(prologue);
        // The original definitions are private to this compilation module;
        // erase only after every shared region and call operand was resolved.
        for (auto *f : _order) {
            for (auto *bb : f->definition()->basic_blocks()) {
                for (auto *inst : bb->instructions()) { inst->set_operands({}); }
            }
        }
        for (auto *f : _order) { f->remove_self(); }
        return _info;
    }
};

void rematerialize_cross_block_addresses(Function *function) {
    luisa::vector<GEPInst *> addresses;
    for (auto *block : function->definition()->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<GEPInst>()) { addresses.emplace_back(static_cast<GEPInst *>(inst)); }
        }
    }
    XIRBuilder b;
    for (auto *address : addresses) {
        luisa::vector<Use *> uses;
        for (auto *use : address->use_list()) {
            auto *user = use->user();
            if (user != nullptr && user->isa<Instruction>() &&
                static_cast<Instruction *>(user)->parent_block() != address->parent_block()) {
                uses.emplace_back(use);
            }
        }
        for (auto *use : uses) {
            auto *user = static_cast<Instruction *>(use->user());
            b.set_insertion_point(user->prev());
            auto reconstruct = [&](auto &&self, GEPInst *gep) -> GEPInst * {
                auto *base = gep->base();
                if (base->isa<GEPInst>() &&
                    static_cast<GEPInst *>(base)->parent_block() != user->parent_block()) {
                    base = self(self, static_cast<GEPInst *>(base));
                }
                luisa::vector<Value *> indices;
                for (auto *index : gep->index_uses()) { indices.emplace_back(index->value()); }
                auto *copy = b.gep(gep->type(), base, indices);
                for (auto *metadata : gep->metadata_list()) {
                    copy->metadata_list().push_front(metadata->clone());
                }
                return copy;
            };
            User::set_operand_use_value(use, reconstruct(reconstruct, address));
        }
    }
}
}// namespace

void coro_call_demote_cross_block_values(Function *root) {
    luisa::vector<Instruction *> values;
    for (auto *block : root->definition()->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->type() != nullptr && !inst->is_lvalue() && !inst->type()->is_resource()) {
                values.emplace_back(inst);
            }
        }
    }
    XIRBuilder b;
    for (auto *value : values) {
        luisa::vector<Use *> uses;
        for (auto *use : value->use_list()) {
            auto *user = use->user();
            if (user != nullptr && user->isa<Instruction>() &&
                static_cast<Instruction *>(user)->parent_block() != value->parent_block()) {
                uses.emplace_back(use);
            }
        }
        if (uses.empty()) { continue; }
        b.set_insertion_point(root->definition()->body_block()->instructions().head_sentinel());
        auto *slot = b.alloca_local(value->type());
        b.set_insertion_point(value);
        b.store(slot, value);
        for (auto *use : uses) {
            auto *user = static_cast<Instruction *>(use->user());
            b.set_insertion_point(user->prev());
            User::set_operand_use_value(use, b.load(value->type(), slot));
        }
    }
}

void coro_call_structure_continuation(Function *function) {
    auto *definition = function->definition();
    auto *module = function->parent_module();
    luisa::vector<BasicBlock *> blocks;
    definition->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER,
                                      [&](BasicBlock *block) { blocks.emplace_back(block); });
    // Each dispatcher arm is a distinct lexical region. References cannot be
    // spilled as values: rebuild each cross-block address at its use first,
    // then transport its captured scalar indices with the other live values.
    // This happens after frame materialization and creates no frame state.
    rematerialize_cross_block_addresses(function);
    coro_call_demote_cross_block_values(function);
    luisa::unordered_map<BasicBlock *, uint32_t> ids;
    for (auto *block : blocks) { ids.emplace(block, static_cast<uint32_t>(ids.size())); }
    auto *old_entry = definition->body_block();
    auto *entry = definition->create_basic_block();
    definition->set_body_block(entry);
    XIRBuilder b;
    b.set_insertion_point(entry);
    luisa::vector<AllocaInst *> allocas;
    for (auto *block : blocks) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<AllocaInst>()) { allocas.emplace_back(static_cast<AllocaInst *>(inst)); }
        }
    }
    for (auto *alloca : allocas) { b.append(alloca->remove_self()); }
    auto *pc = b.alloca_local(Type::of<uint32_t>());
    pc->add_comment("continuation-local block selector; not a suspend token");
    auto constant = [&](BasicBlock *target) {
        auto id = ids.at(target);
        return module->create_constant(Type::of<uint32_t>(), &id);
    };
    b.store(pc, constant(old_entry));
    auto *loop = b.simple_loop();
    auto *dispatch = definition->create_basic_block();
    auto *loop_exit = definition->create_basic_block();
    auto *next = definition->create_basic_block();
    auto *invalid = definition->create_basic_block();
    loop->set_body_block(dispatch);
    loop->set_merge_block(loop_exit);
    b.set_insertion_point(dispatch);
    auto *choice = b.switch_(b.load(Type::of<uint32_t>(), pc));
    choice->set_merge_block(next);
    choice->set_default_block(invalid);
    for (auto *block : blocks) { choice->add_case(ids.at(block), block); }
    b.set_insertion_point(invalid);
    b.unreachable_("Invalid continuation block selector.");
    b.set_insertion_point(loop_exit);
    b.unreachable_();
    b.set_insertion_point(next);
    b.continue_(dispatch);
    for (auto *block : blocks) {
        auto *term = block->terminator();
        if (term->isa<BranchInst>()) {
            auto *target = static_cast<BranchInst *>(term)->target_block();
            auto removed = term->remove_self();
            b.set_insertion_point(block);
            b.store(pc, constant(target));
            b.break_(next);
        } else if (term->isa<ConditionalBranchInst>()) {
            auto *branch = static_cast<ConditionalBranchInst *>(term);
            auto *condition = branch->condition();
            auto *true_target = branch->true_block();
            auto *false_target = branch->false_block();
            auto removed = term->remove_self();
            b.set_insertion_point(block);
            auto *selection = b.if_(condition);
            auto *yes = definition->create_basic_block();
            auto *no = definition->create_basic_block();
            auto *merge = definition->create_basic_block();
            selection->set_true_target(yes);
            selection->set_false_target(no);
            selection->set_merge_block(merge);
            b.set_insertion_point(yes);
            b.store(pc, constant(true_target));
            b.br(merge);
            b.set_insertion_point(no);
            b.store(pc, constant(false_target));
            b.br(merge);
            b.set_insertion_point(merge);
            b.break_(next);
        } else if (term->isa<IndexedBranchInst>()) {
            auto *branch = static_cast<IndexedBranchInst *>(term);
            auto removed = term->remove_self();
            b.set_insertion_point(block);
            auto *selection = b.switch_(branch->value());
            auto *merge = definition->create_basic_block();
            selection->set_merge_block(merge);
            auto target_arm = [&](BasicBlock *target) {
                auto *arm = definition->create_basic_block();
                b.set_insertion_point(arm);
                b.store(pc, constant(target));
                b.break_(merge);
                return arm;
            };
            selection->set_default_block(target_arm(branch->default_block()));
            for (size_t i = 0; i < branch->case_count(); ++i) {
                selection->add_case(branch->case_value(i), target_arm(branch->case_block(i)));
            }
            b.set_insertion_point(merge);
            b.break_(next);
        } else {
            LUISA_ASSERT(term->isa<ReturnInst>() || term->isa<UnreachableInst>(),
                         "Expected a raw materialized coroutine continuation.");
        }
    }
}

CoroCallInfo coro_call_pass_run_on_function(Function *root) {
    return Lowering{root}.run();
}
}// namespace luisa::compute::xir
