#include <luisa/xir/passes/coro_call.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/post_dom_tree.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
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
            luisa::sort(function.resume_tokens.begin(), function.resume_tokens.end());
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
        // Single-call-site suspending callables always return to one fixed
        // continuation block, so collapse their return_dispatch directly to a
        // branch and remove the unused return_site selector and unreachable block.
        luisa::vector<ManagedPtr<Instruction>> removed_insts;
        luisa::vector<ManagedPtr<BasicBlock>> removed_blocks;
        for (auto &[f, region_ptr] : _regions) {
            auto &r = *region_ptr;
            if (r.return_dispatch == nullptr || r.return_dispatch->case_count() != 1u) {
                continue;
            }
            auto *single_after = r.return_dispatch->case_block(0u);
            auto *invalid = r.return_dispatch->default_block();
            auto *exit_bb = r.return_dispatch->parent_block();
            auto *return_tag = r.return_dispatch->value();
            removed_insts.emplace_back(r.return_dispatch->remove_self());
            r.return_dispatch = nullptr;
            {
                XIRBuilder exit_builder;
                exit_builder.set_insertion_point(exit_bb);
                exit_builder.br(single_after);
            }
            if (invalid != nullptr) {
                while (!invalid->instructions().empty()) {
                    removed_insts.emplace_back(invalid->instructions().front()->remove_self());
                }
                removed_blocks.emplace_back(invalid->remove_self());
            }
            if (return_tag != nullptr && return_tag->isa<LoadInst>() &&
                return_tag->use_list().empty()) {
                removed_insts.emplace_back(static_cast<LoadInst *>(return_tag)->remove_self());
            }
            if (r.return_site != nullptr) {
                luisa::vector<StoreInst *> stores;
                auto only_stores = true;
                for (auto *use : r.return_site->use_list()) {
                    auto *user = use == nullptr ? nullptr : use->user();
                    if (user == nullptr || !user->isa<StoreInst>() ||
                        static_cast<StoreInst *>(user)->variable() != r.return_site) {
                        only_stores = false;
                        break;
                    }
                    stores.emplace_back(static_cast<StoreInst *>(user));
                }
                if (only_stores) {
                    for (auto *store : stores) {
                        removed_insts.emplace_back(store->remove_self());
                    }
                    removed_insts.emplace_back(r.return_site->remove_self());
                    r.return_site = nullptr;
                }
            }
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
    auto *def = root->definition();
    luisa::vector<Instruction *> values;
    luisa::unordered_map<Instruction *, size_t> ordinals;
    for (auto *block : def->basic_blocks()) {
        auto ordinal = size_t{0u};
        for (auto *inst : block->instructions()) {
            ordinals.emplace(inst, ordinal++);
            if (inst->type() != nullptr && !inst->is_lvalue() && !inst->type()->is_resource()) {
                values.emplace_back(inst);
            }
        }
    }
    XIRBuilder b;
    for (auto *value : values) {
        auto *def_block = value->parent_block();
        luisa::vector<BasicBlock *> user_block_order;
        luisa::unordered_map<BasicBlock *, luisa::vector<Use *>> uses_by_block;
        luisa::unordered_map<BasicBlock *, std::pair<size_t, Instruction *>> earliest_user;
        for (auto *use : value->use_list()) {
            auto *user = use->user();
            if (user == nullptr || !user->isa<Instruction>()) { continue; }
            auto *user_inst = static_cast<Instruction *>(user);
            auto *user_block = user_inst->parent_block();
            if (user_block == def_block) { continue; }
            auto ord = ordinals.at(user_inst);
            auto [it, inserted] = uses_by_block.try_emplace(user_block);
            if (inserted) {
                user_block_order.emplace_back(user_block);
                earliest_user.emplace(user_block, std::pair{ord, user_inst});
            } else {
                auto &best = earliest_user.at(user_block);
                if (ord < best.first) {
                    best = {ord, user_inst};
                }
            }
            it->second.emplace_back(use);
        }
        if (user_block_order.empty()) { continue; }
        b.set_insertion_point(def->body_block()->instructions().head_sentinel());
        auto *slot = b.alloca_local(value->type());
        b.set_insertion_point(value);
        b.store(slot, value);
        for (auto *user_block : user_block_order) {
            auto *anchor = earliest_user.at(user_block).second;
            b.set_insertion_point(anchor->prev());
            auto *loaded = b.load(value->type(), slot);
            for (auto *use : uses_by_block.at(user_block)) {
                User::set_operand_use_value(use, loaded);
            }
        }
    }
}

void coro_call_structure_continuation(Function *function) {
    if (function == nullptr) { return; }
    auto *definition = function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) { return; }
    auto *module = function->parent_module();

    // Each dispatcher arm is a distinct lexical region. References cannot be
    // spilled as values: rebuild each cross-block address at its use first,
    // then transport its captured scalar indices with the other live values.
    // This happens after frame materialization and creates no frame state.
    rematerialize_cross_block_addresses(function);
    coro_call_demote_cross_block_values(function);

    luisa::vector<BasicBlock *> blocks;
    luisa::unordered_map<BasicBlock *, uint32_t> raw_ids;
    definition->traverse_basic_blocks(
        BasicBlockTraversalOrder::REVERSE_POST_ORDER,
        [&](BasicBlock *block) noexcept {
            raw_ids.emplace(block, static_cast<uint32_t>(blocks.size()));
            blocks.emplace_back(block);
        });
    if (blocks.empty()) { return; }

    const size_t nblocks = blocks.size();
    constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    luisa::vector<luisa::vector<uint32_t>> succs(nblocks);
    luisa::vector<luisa::vector<uint32_t>> preds(nblocks);
    for (size_t i = 0u; i < nblocks; ++i) {
        blocks[i]->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (auto it = raw_ids.find(succ); it != raw_ids.end()) {
                succs[i].emplace_back(it->second);
            }
        });
        blocks[i]->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
            if (auto it = raw_ids.find(pred); it != raw_ids.end()) {
                preds[i].emplace_back(it->second);
            }
        });
    }

    auto dom = compute_dom_tree(definition, {.compute_dominance_frontiers = false});
    auto pdom = compute_post_dom_tree(
        definition, {.account_for_infinite_paths = true, .ignore_unreachable = true});

    luisa::vector<uint8_t> must_dispatch(nblocks, 0u);
    auto *old_entry = definition->body_block();
    const uint32_t old_entry_id = raw_ids.at(old_entry);
    if (!preds[old_entry_id].empty()) {
        must_dispatch[old_entry_id] = 1u;
    }

    for (size_t u = 0u; u < nblocks; ++u) {
        for (uint32_t p : preds[u]) {
            if (p == u || dom.dominates(blocks[u], blocks[p])) {
                must_dispatch[u] = 1u;
                break;
            }
        }
        auto *term = blocks[u]->terminator();
        if (term != nullptr && term->isa<IndexedBranchInst>()) {
            auto *branch = static_cast<IndexedBranchInst *>(term);
            auto *def_bb = branch->default_block();
            for (size_t ci = 0u; ci < branch->case_count(); ++ci) {
                if (branch->case_block(ci) == def_bb) {
                    if (auto it = raw_ids.find(def_bb); it != raw_ids.end()) {
                        must_dispatch[it->second] = 1u;
                    }
                    break;
                }
            }
        }
    }

    // Detect pure single-entry single-exit (SESE) selections bottom-up in
    // dominator post-order so linear chains and structured diamonds stay inline
    // rather than flattening onto the top-level dispatcher switch.
    luisa::vector<uint32_t> dom_postorder;
    dom_postorder.reserve(nblocks);
    if (auto *root_node = dom.root()) {
        struct DomFrame {
            const DomTreeNode *node;
            size_t next_child;
        };
        luisa::vector<DomFrame> stack;
        stack.emplace_back(DomFrame{root_node, 0u});
        while (!stack.empty()) {
            auto &frame = stack.back();
            if (frame.next_child < frame.node->children().size()) {
                auto *child = frame.node->children()[frame.next_child++];
                stack.emplace_back(DomFrame{child, 0u});
            } else {
                if (auto it = raw_ids.find(frame.node->block()); it != raw_ids.end()) {
                    dom_postorder.emplace_back(it->second);
                }
                stack.pop_back();
            }
        }
    }

    luisa::vector<uint32_t> sese_merge(nblocks, invalid_id);
    luisa::vector<uint32_t> claimed_owner(nblocks, invalid_id);
    luisa::vector<uint32_t> dfs_active_epoch(nblocks, 0u);
    luisa::vector<uint32_t> dfs_done_epoch(nblocks, 0u);
    luisa::vector<uint32_t> m_edge_epoch(nblocks, 0u);
    uint32_t check_epoch = 0u;
    luisa::vector<uint32_t> cand_interior;
    struct SeseDfsFrame {
        uint32_t u;
        size_t succ_idx;
        bool pushed_active;
    };
    luisa::vector<SeseDfsFrame> sese_stack;

    for (uint32_t h : dom_postorder) {
        auto *h_bb = blocks[h];
        auto *term = h_bb->terminator();
        if (term == nullptr) { continue; }
        if (term->isa<ConditionalBranchInst>()) {
            auto *cbr = static_cast<ConditionalBranchInst *>(term);
            if (cbr->true_block() == cbr->false_block()) { continue; }
        } else if (!term->isa<IndexedBranchInst>()) {
            continue;
        }
        auto *m_bb = pdom.immediate_post_dominator(h_bb);
        if (m_bb == nullptr) { continue; }
        auto m_it = raw_ids.find(m_bb);
        if (m_it == raw_ids.end()) { continue; }
        const uint32_t m = m_it->second;
        if (m == h || must_dispatch[m] != 0u || claimed_owner[m] != invalid_id ||
            !dom.strictly_dominates(h_bb, m_bb)) {
            continue;
        }

        if (++check_epoch == 0u) {
            std::fill(dfs_active_epoch.begin(), dfs_active_epoch.end(), 0u);
            std::fill(dfs_done_epoch.begin(), dfs_done_epoch.end(), 0u);
            std::fill(m_edge_epoch.begin(), m_edge_epoch.end(), 0u);
            check_epoch = 1u;
        }
        cand_interior.clear();
        sese_stack.clear();
        size_t reached_preds_of_m = 0u;
        auto record_edge_to_m = [&](uint32_t src) noexcept {
            if (m_edge_epoch[src] != check_epoch) {
                m_edge_epoch[src] = check_epoch;
                ++reached_preds_of_m;
            }
        };

        auto push_step = [&](uint32_t target, uint32_t src) noexcept -> bool {
            if (target == m) {
                record_edge_to_m(src);
                return true;
            }
            if (target == h || !dom.strictly_dominates(h_bb, blocks[target]) ||
                must_dispatch[target] != 0u) {
                return false;
            }
            if (dfs_active_epoch[target] == check_epoch) {
                return false;
            }
            if (dfs_done_epoch[target] == check_epoch) {
                return true;
            }
            if (claimed_owner[target] != invalid_id || preds[target].size() != 1u) {
                return false;
            }
            cand_interior.emplace_back(target);
            // If `target` heads an already-verified sub-SESE region, teleport
            // directly to its merge node(s) instead of re-walking its interior.
            uint32_t cur = target;
            dfs_done_epoch[cur] = check_epoch;
            while (sese_merge[cur] != invalid_id) {
                uint32_t next_m = sese_merge[cur];
                if (next_m == h || next_m == m || must_dispatch[next_m] != 0u ||
                    !dom.strictly_dominates(h_bb, blocks[next_m]) ||
                    dfs_active_epoch[next_m] == check_epoch) {
                    return false;
                }
                if (dfs_done_epoch[next_m] == check_epoch) {
                    return true;
                }
                dfs_done_epoch[next_m] = check_epoch;
                cur = next_m;
            }
            dfs_active_epoch[cur] = check_epoch;
            sese_stack.emplace_back(SeseDfsFrame{cur, 0u, true});
            return true;
        };

        bool ok = true;
        for (uint32_t s : succs[h]) {
            if (!push_step(s, h)) {
                ok = false;
                break;
            }
        }
        while (ok && !sese_stack.empty()) {
            auto &frame = sese_stack.back();
            const uint32_t u = frame.u;
            if (frame.succ_idx < succs[u].size()) {
                uint32_t s = succs[u][frame.succ_idx++];
                if (!push_step(s, u)) {
                    ok = false;
                    break;
                }
            } else {
                if (frame.pushed_active) {
                    dfs_active_epoch[u] = 0u;
                }
                sese_stack.pop_back();
            }
        }

        if (!ok || reached_preds_of_m != preds[m].size()) { continue; }
        sese_merge[h] = m;
        claimed_owner[m] = h;
        for (uint32_t interior : cand_interior) {
            claimed_owner[interior] = h;
        }
    }

    luisa::vector<uint8_t> is_dispatch(nblocks, 0u);
    for (size_t u = 0u; u < nblocks; ++u) {
        if (must_dispatch[u] != 0u ||
            (claimed_owner[u] == invalid_id && preds[u].size() >= 2u)) {
            is_dispatch[u] = 1u;
        }
    }

    bool needs_dispatcher = false;
    for (size_t u = 0u; u < nblocks && !needs_dispatcher; ++u) {
        for (uint32_t v : succs[u]) {
            if (is_dispatch[v] != 0u) {
                needs_dispatcher = true;
                break;
            }
        }
    }

    if (needs_dispatcher && is_dispatch[old_entry_id] == 0u) {
        // Hoist only an unconditional prefix. Keep selection and return
        // nesting inside the same dispatcher arm as before.
        auto prefix = old_entry_id;
        while (is_dispatch[prefix] == 0u) {
            auto *terminator = blocks[prefix]->terminator();
            if (terminator == nullptr || !terminator->isa<BranchInst>() ||
                succs[prefix].size() != 1u) {
                is_dispatch[old_entry_id] = 1u;
                break;
            }
            prefix = succs[prefix].front();
        }
    }

    bool has_return = false;
    for (auto *block : blocks) {
        if (auto *term = block->terminator(); term != nullptr && term->isa<ReturnInst>()) {
            has_return = true;
            break;
        }
    }

    luisa::vector<AllocaInst *> allocas;
    for (auto *block : blocks) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<AllocaInst>()) { allocas.emplace_back(static_cast<AllocaInst *>(inst)); }
        }
    }

    XIRBuilder b;
    AllocaInst *pc = nullptr;
    AllocaInst *return_slot = nullptr;
    uint32_t exit_id = invalid_id;
    BasicBlock *next = nullptr;
    BasicBlock *loop_preheader = nullptr;
    luisa::vector<BasicBlock *> dispatch_blocks;
    luisa::unordered_map<BasicBlock *, uint32_t> dispatch_ids;
    auto constant_id = [&](uint32_t id) noexcept -> Value * {
        return module->create_constant(Type::of<uint32_t>(), &id);
    };
    const bool hoist_entry = needs_dispatcher && is_dispatch[old_entry_id] == 0u;

    if (!needs_dispatcher) {
        b.set_insertion_point(old_entry->instructions().head_sentinel());
        for (auto *alloca : allocas) { b.append(alloca->remove_self()); }
    } else {
        for (size_t u = 0u; u < nblocks; ++u) {
            if (is_dispatch[u] != 0u) {
                dispatch_ids.emplace(blocks[u], static_cast<uint32_t>(dispatch_blocks.size()));
                dispatch_blocks.emplace_back(blocks[u]);
            }
        }
        exit_id = static_cast<uint32_t>(dispatch_blocks.size());
        if (hoist_entry) {
            b.set_insertion_point(old_entry->instructions().head_sentinel());
        } else {
            auto *entry = definition->create_basic_block();
            definition->set_body_block(entry);
            b.set_insertion_point(entry);
        }
        for (auto *alloca : allocas) { b.append(alloca->remove_self()); }
        pc = b.alloca_local(Type::of<uint32_t>());
        pc->add_comment("continuation-local block selector; not a suspend token");
        if (has_return && function->type() != nullptr) {
            return_slot = b.alloca_local(function->type());
            return_slot->add_comment("continuation-local return value");
        }
        if (return_slot != nullptr && !function->type()->is_resource() && !function->type()->is_custom()) {
            b.store(return_slot, module->create_constant_zero(function->type()));
        }
        if (hoist_entry) {
            loop_preheader = definition->create_basic_block();
            b.set_insertion_point(loop_preheader);
        } else {
            b.store(pc, constant_id(dispatch_ids.at(old_entry)));
        }
        auto *loop = b.simple_loop();
        auto *dispatch = definition->create_basic_block();
        auto *loop_exit = definition->create_basic_block();
        next = definition->create_basic_block();
        auto *invalid = definition->create_basic_block();
        loop->set_body_block(dispatch);
        loop->set_merge_block(loop_exit);
        b.set_insertion_point(dispatch);
        auto *choice = b.switch_(b.load(Type::of<uint32_t>(), pc));
        choice->set_merge_block(next);
        choice->set_default_block(invalid);
        for (auto *block : dispatch_blocks) { choice->add_case(dispatch_ids.at(block), block); }
        if (has_return) {
            b.set_insertion_point(invalid);
            b.store(pc, constant_id(exit_id));
            b.break_(next);
            b.set_insertion_point(loop_exit);
            if (return_slot != nullptr) {
                b.return_(b.load(function->type(), return_slot));
            } else {
                b.return_void();
            }
            auto *exit_arm = definition->create_basic_block();
            auto *cont_arm = definition->create_basic_block();
            auto *after_if = definition->create_basic_block();
            b.set_insertion_point(next);
            auto *is_exit = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                {b.load(Type::of<uint32_t>(), pc), constant_id(exit_id)});
            auto *done_if = b.if_(is_exit);
            done_if->set_true_target(exit_arm);
            done_if->set_false_target(cont_arm);
            done_if->set_merge_block(after_if);
            b.set_insertion_point(exit_arm);
            b.break_(loop_exit);
            b.set_insertion_point(cont_arm);
            b.br(after_if);
            b.set_insertion_point(after_if);
            b.continue_(dispatch);
        } else {
            b.set_insertion_point(invalid);
            b.unreachable_("Invalid continuation block selector.");
            b.set_insertion_point(loop_exit);
            b.unreachable_();
            b.set_insertion_point(next);
            b.continue_(dispatch);
        }
    }

    auto constant_bb = [&](BasicBlock *target) noexcept -> Value * {
        auto id = dispatch_ids.at(target);
        return module->create_constant(Type::of<uint32_t>(), &id);
    };
    auto is_dispatch_block = [&](BasicBlock *bb) noexcept -> bool {
        if (!needs_dispatcher) { return false; }
        auto it = raw_ids.find(bb);
        return it != raw_ids.end() && is_dispatch[it->second] != 0u;
    };
    auto get_sese_merge = [&](BasicBlock *bb) noexcept -> BasicBlock * {
        auto it = raw_ids.find(bb);
        if (it == raw_ids.end()) { return nullptr; }
        uint32_t mid = sese_merge[it->second];
        return mid == invalid_id ? nullptr : blocks[mid];
    };

    enum class ExitKind : uint8_t {
        BranchToMerge,
        BreakToMerge,
    };
    struct EmitTask {
        enum class Kind : uint8_t {
            RunLinear,
            FinalizeSynthMerge,
        };
        Kind kind{Kind::RunLinear};
        BasicBlock *cur{nullptr};
        BasicBlock *stop{nullptr};
        ExitKind exit_kind{ExitKind::BranchToMerge};
        BasicBlock *exit_target{nullptr};
        uint32_t reached_id{invalid_id};
        uint32_t flag_id{invalid_id};
    };

    auto emit_exit_at = [&](BasicBlock *block, ExitKind ek, BasicBlock *target) noexcept {
        b.set_insertion_point(block);
        if (ek == ExitKind::BranchToMerge) {
            b.br(target);
        } else {
            b.break_(target);
        }
    };

    luisa::vector<EmitTask> emit_stack;
    luisa::vector<uint8_t> reached_flags;
    auto run_emitter = [&](BasicBlock *root_block, ExitKind top_kind, BasicBlock *top_target) noexcept {
        emit_stack.clear();
        reached_flags.clear();
        emit_stack.emplace_back(EmitTask{
            EmitTask::Kind::RunLinear, root_block, nullptr, top_kind, top_target, invalid_id, invalid_id});
        while (!emit_stack.empty()) {
            auto task = emit_stack.back();
            emit_stack.pop_back();
            if (task.kind == EmitTask::Kind::FinalizeSynthMerge) {
                if (reached_flags[task.flag_id] != 0u) {
                    emit_exit_at(task.cur, task.exit_kind, task.exit_target);
                    if (task.reached_id != invalid_id) { reached_flags[task.reached_id] = 1u; }
                } else {
                    b.set_insertion_point(task.cur);
                    b.unreachable_();
                }
                continue;
            }

            BasicBlock *cur = task.cur;
            while (cur != nullptr) {
                auto *term = cur->terminator();
                if (term->isa<ReturnInst>()) {
                    if (needs_dispatcher) {
                        LUISA_ASSERT(task.stop == nullptr,
                                     "Return cannot occur inside an SESE region.");
                        auto *ret = static_cast<ReturnInst *>(term);
                        auto *ret_val = ret->return_value();
                        auto removed = term->remove_self();
                        b.set_insertion_point(cur);
                        if (return_slot != nullptr && ret_val != nullptr) {
                            b.store(return_slot, ret_val);
                        }
                        b.store(pc, constant_id(exit_id));
                        emit_exit_at(cur, task.exit_kind, task.exit_target);
                        if (task.reached_id != invalid_id) { reached_flags[task.reached_id] = 1u; }
                    }
                    break;
                }
                if (term->isa<UnreachableInst>()) {
                    break;
                }
                if (term->isa<BranchInst>() ||
                    (term->isa<ConditionalBranchInst>() &&
                     static_cast<ConditionalBranchInst *>(term)->true_block() ==
                         static_cast<ConditionalBranchInst *>(term)->false_block())) {
                    BasicBlock *target = term->isa<BranchInst>() ?
                                             static_cast<BranchInst *>(term)->target_block() :
                                             static_cast<ConditionalBranchInst *>(term)->true_block();
                    if (task.stop != nullptr && target == task.stop) {
                        if (!term->isa<BranchInst>() ||
                            task.exit_kind != ExitKind::BranchToMerge ||
                            target != task.exit_target) {
                            auto removed = term->remove_self();
                            emit_exit_at(cur, task.exit_kind, task.exit_target);
                        }
                        if (task.reached_id != invalid_id) { reached_flags[task.reached_id] = 1u; }
                        break;
                    }
                    if (!is_dispatch_block(target)) {
                        if (!term->isa<BranchInst>()) {
                            auto removed = term->remove_self();
                            b.set_insertion_point(cur);
                            b.br(target);
                        }
                        cur = target;
                        continue;
                    }
                    auto removed = term->remove_self();
                    b.set_insertion_point(cur);
                    b.store(pc, constant_bb(target));
                    emit_exit_at(cur, task.exit_kind, task.exit_target);
                    if (task.reached_id != invalid_id) { reached_flags[task.reached_id] = 1u; }
                    break;
                }
                if (term->isa<ConditionalBranchInst>()) {
                    auto *cbr = static_cast<ConditionalBranchInst *>(term);
                    auto *cond = cbr->condition();
                    auto *t_bb = cbr->true_block();
                    auto *f_bb = cbr->false_block();
                    auto *merge_bb = get_sese_merge(cur);
                    if (merge_bb != nullptr) {
                        auto removed = term->remove_self();
                        b.set_insertion_point(cur);
                        auto *if_inst = b.if_(cond);
                        if_inst->set_true_target(t_bb);
                        if_inst->set_false_target(f_bb);
                        if_inst->set_merge_block(merge_bb);
                        emit_stack.emplace_back(EmitTask{
                            EmitTask::Kind::RunLinear,
                            merge_bb,
                            task.stop,
                            task.exit_kind,
                            task.exit_target,
                            task.reached_id,
                            invalid_id});
                        if (f_bb != merge_bb) {
                            emit_stack.emplace_back(EmitTask{
                                EmitTask::Kind::RunLinear,
                                f_bb,
                                merge_bb,
                                ExitKind::BranchToMerge,
                                merge_bb,
                                invalid_id,
                                invalid_id});
                        }
                        if (t_bb != merge_bb) {
                            emit_stack.emplace_back(EmitTask{
                                EmitTask::Kind::RunLinear,
                                t_bb,
                                merge_bb,
                                ExitKind::BranchToMerge,
                                merge_bb,
                                invalid_id,
                                invalid_id});
                        }
                        break;
                    }
                    auto *synth_merge = definition->create_basic_block();
                    auto removed = term->remove_self();
                    b.set_insertion_point(cur);
                    auto *if_inst = b.if_(cond);
                    if_inst->set_merge_block(synth_merge);

                    const uint32_t my_flag_id = static_cast<uint32_t>(reached_flags.size());
                    reached_flags.emplace_back(0u);
                    emit_stack.emplace_back(EmitTask{
                        EmitTask::Kind::FinalizeSynthMerge,
                        synth_merge,
                        nullptr,
                        task.exit_kind,
                        task.exit_target,
                        task.reached_id,
                        my_flag_id});

                    auto prepare_arm = [&](BasicBlock *arm_target, bool is_true) noexcept {
                        if (task.stop != nullptr && arm_target == task.stop) {
                            auto *tramp = definition->create_basic_block();
                            b.set_insertion_point(tramp);
                            b.br(synth_merge);
                            if (is_true) {
                                if_inst->set_true_target(tramp);
                            } else {
                                if_inst->set_false_target(tramp);
                            }
                            reached_flags[my_flag_id] = 1u;
                        } else if (is_dispatch_block(arm_target)) {
                            auto *tramp = definition->create_basic_block();
                            b.set_insertion_point(tramp);
                            b.store(pc, constant_bb(arm_target));
                            b.br(synth_merge);
                            if (is_true) {
                                if_inst->set_true_target(tramp);
                            } else {
                                if_inst->set_false_target(tramp);
                            }
                            reached_flags[my_flag_id] = 1u;
                        } else {
                            if (is_true) {
                                if_inst->set_true_target(arm_target);
                            } else {
                                if_inst->set_false_target(arm_target);
                            }
                            emit_stack.emplace_back(EmitTask{
                                EmitTask::Kind::RunLinear,
                                arm_target,
                                task.stop,
                                ExitKind::BranchToMerge,
                                synth_merge,
                                my_flag_id,
                                invalid_id});
                        }
                    };
                    prepare_arm(f_bb, false);
                    prepare_arm(t_bb, true);
                    break;
                }
                if (term->isa<IndexedBranchInst>()) {
                    auto *ibr = static_cast<IndexedBranchInst *>(term);
                    auto *sel_val = ibr->value();
                    auto *def_bb = ibr->default_block();
                    luisa::vector<std::pair<IndexedBranchInst::case_value_type, BasicBlock *>> cases;
                    cases.reserve(ibr->case_count());
                    for (size_t ci = 0u; ci < ibr->case_count(); ++ci) {
                        cases.emplace_back(ibr->case_value(ci), ibr->case_block(ci));
                    }
                    auto *merge_bb = get_sese_merge(cur);
                    auto removed = term->remove_self();
                    b.set_insertion_point(cur);
                    auto *sw = b.switch_(sel_val);

                    if (merge_bb != nullptr) {
                        sw->set_merge_block(merge_bb);
                        emit_stack.emplace_back(EmitTask{
                            EmitTask::Kind::RunLinear,
                            merge_bb,
                            task.stop,
                            task.exit_kind,
                            task.exit_target,
                            task.reached_id,
                            invalid_id});
                        luisa::unordered_set<BasicBlock *> emitted_arms;
                        if (def_bb == merge_bb) {
                            auto *tramp = definition->create_basic_block();
                            b.set_insertion_point(tramp);
                            b.break_(merge_bb);
                            sw->set_default_block(tramp);
                        } else {
                            sw->set_default_block(def_bb);
                            if (emitted_arms.emplace(def_bb).second) {
                                emit_stack.emplace_back(EmitTask{
                                    EmitTask::Kind::RunLinear,
                                    def_bb,
                                    merge_bb,
                                    ExitKind::BreakToMerge,
                                    merge_bb,
                                    invalid_id,
                                    invalid_id});
                            }
                        }
                        BasicBlock *case_merge_tramp = nullptr;
                        for (auto &[cval, cbb] : cases) {
                            if (cbb == merge_bb) {
                                if (case_merge_tramp == nullptr) {
                                    case_merge_tramp = definition->create_basic_block();
                                    b.set_insertion_point(case_merge_tramp);
                                    b.break_(merge_bb);
                                }
                                sw->add_case(cval, case_merge_tramp);
                            } else {
                                sw->add_case(cval, cbb);
                                if (emitted_arms.emplace(cbb).second) {
                                    emit_stack.emplace_back(EmitTask{
                                        EmitTask::Kind::RunLinear,
                                        cbb,
                                        merge_bb,
                                        ExitKind::BreakToMerge,
                                        merge_bb,
                                        invalid_id,
                                        invalid_id});
                                }
                            }
                        }
                        break;
                    }

                    auto *synth_merge = definition->create_basic_block();
                    sw->set_merge_block(synth_merge);
                    const uint32_t my_flag_id = static_cast<uint32_t>(reached_flags.size());
                    reached_flags.emplace_back(0u);
                    emit_stack.emplace_back(EmitTask{
                        EmitTask::Kind::FinalizeSynthMerge,
                        synth_merge,
                        nullptr,
                        task.exit_kind,
                        task.exit_target,
                        task.reached_id,
                        my_flag_id});

                    auto make_dispatch_tramp = [&](BasicBlock *target) noexcept {
                        auto *tramp = definition->create_basic_block();
                        b.set_insertion_point(tramp);
                        if (is_dispatch_block(target)) {
                            b.store(pc, constant_bb(target));
                        }
                        b.break_(synth_merge);
                        reached_flags[my_flag_id] = 1u;
                        return tramp;
                    };

                    luisa::unordered_set<BasicBlock *> emitted_arms;
                    if ((task.stop != nullptr && def_bb == task.stop) ||
                        is_dispatch_block(def_bb)) {
                        // Keep default block distinct from case arms.
                        sw->set_default_block(make_dispatch_tramp(def_bb));
                    } else {
                        sw->set_default_block(def_bb);
                        if (emitted_arms.emplace(def_bb).second) {
                            emit_stack.emplace_back(EmitTask{
                                EmitTask::Kind::RunLinear,
                                def_bb,
                                task.stop,
                                ExitKind::BreakToMerge,
                                synth_merge,
                                my_flag_id,
                                invalid_id});
                        }
                    }

                    luisa::unordered_map<BasicBlock *, BasicBlock *> case_tramps;
                    for (auto &[cval, cbb] : cases) {
                        if ((task.stop != nullptr && cbb == task.stop) ||
                            is_dispatch_block(cbb)) {
                            auto [it, ins] = case_tramps.try_emplace(cbb, nullptr);
                            if (ins) { it->second = make_dispatch_tramp(cbb); }
                            sw->add_case(cval, it->second);
                        } else {
                            sw->add_case(cval, cbb);
                            if (emitted_arms.emplace(cbb).second) {
                                emit_stack.emplace_back(EmitTask{
                                    EmitTask::Kind::RunLinear,
                                    cbb,
                                    task.stop,
                                    ExitKind::BreakToMerge,
                                    synth_merge,
                                    my_flag_id,
                                    invalid_id});
                            }
                        }
                    }
                    break;
                }
                LUISA_ASSERT(term->isa<ReturnInst>() || term->isa<UnreachableInst>(),
                             "Expected a raw materialized coroutine continuation.");
                break;
            }
        }
    };

    if (!needs_dispatcher) {
        run_emitter(old_entry, ExitKind::BranchToMerge, nullptr);
    } else {
        if (hoist_entry) {
            run_emitter(old_entry, ExitKind::BranchToMerge, loop_preheader);
        }
        for (auto *arm : dispatch_blocks) {
            run_emitter(arm, ExitKind::BreakToMerge, next);
        }
    }
}

CoroCallInfo coro_call_pass_run_on_function(Function *root) {
    return Lowering{root}.run();
}
}// namespace luisa::compute::xir
