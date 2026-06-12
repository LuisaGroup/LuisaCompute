#include <algorithm>

#include <luisa/core/stl/deque.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static luisa::optional<uint32_t> resume_token_of_block(BasicBlock *bb) noexcept {
    if (bb == nullptr) { return luisa::nullopt; }
    for (auto *inst : bb->instructions()) {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
            return static_cast<CoroResumeInst *>(inst)->token();
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] static AllocaInst *trace_local_alloca(Value *value) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *inst = static_cast<Instruction *>(value);
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::ALLOCA: {
                auto *alloca = static_cast<AllocaInst *>(inst);
                return alloca->is_local() ? alloca : nullptr;
            }
            case DerivedInstructionTag::GEP: {
                value = static_cast<GEPInst *>(inst)->base();
                break;
            }
            default:
                return nullptr;
        }
    }
    return nullptr;
}

[[nodiscard]] static luisa::optional<luisa::string> local_alloca_name(Value *value) noexcept {
    if (auto *alloca = trace_local_alloca(value)) {
        if (auto name = alloca->name()) {
            return luisa::string{name.value()};
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] static bool is_always_available(Value *value) noexcept {
    if (value == nullptr) { return true; }
    switch (value->derived_value_tag()) {
        case DerivedValueTag::UNDEFINED:
        case DerivedValueTag::FUNCTION:
        case DerivedValueTag::BASIC_BLOCK:
        case DerivedValueTag::CONSTANT:
        case DerivedValueTag::ARGUMENT:
        case DerivedValueTag::SPECIAL_REGISTER:
            return true;
        case DerivedValueTag::INSTRUCTION:
            return false;
    }
    return true;
}

[[nodiscard]] static bool is_frameable_ssa_value(Value *value) noexcept {
    if (value == nullptr || value->type() == nullptr || value->is_lvalue()) { return false; }
    if (value->derived_value_tag() != DerivedValueTag::INSTRUCTION) { return false; }
    auto *inst = static_cast<Instruction *>(value);
    return !inst->is_terminator();
}

[[nodiscard]] static Value *frame_value_for_operand(Value *value) noexcept {
    if (auto *alloca = trace_local_alloca(value)) { return alloca; }
    return is_frameable_ssa_value(value) ? value : nullptr;
}

[[nodiscard]] static luisa::string frame_value_name(Value *value, size_t index) noexcept {
    if (auto *alloca = trace_local_alloca(value)) {
        if (auto name = alloca->name()) {
            return luisa::string{name.value()};
        }
    }
    return luisa::format("_coro_frame_{}", index);
}

static void append_sorted_names(luisa::vector<luisa::string> &dst,
                                const luisa::unordered_set<Value *> &src,
                                const luisa::unordered_map<Value *, luisa::string> &names) noexcept {
    dst.clear();
    dst.reserve(src.size());
    for (auto *value : src) {
        if (auto it = names.find(value); it != names.end()) {
            dst.emplace_back(it->second);
        }
    }
    std::sort(dst.begin(), dst.end());
}

static void append_ordered_values(luisa::vector<Value *> &dst,
                                  const luisa::unordered_set<Value *> &src,
                                  const luisa::unordered_map<Value *, size_t> &order) noexcept {
    dst.clear();
    dst.reserve(src.size());
    for (auto *value : src) { dst.emplace_back(value); }
    std::sort(dst.begin(), dst.end(), [&](auto *lhs, auto *rhs) noexcept {
        auto li = order.find(lhs);
        auto ri = order.find(rhs);
        auto lo = li == order.end() ? static_cast<size_t>(-1) : li->second;
        auto ro = ri == order.end() ? static_cast<size_t>(-1) : ri->second;
        return lo < ro;
    });
}

[[nodiscard]] static bool same_set(const luisa::unordered_set<Value *> &a,
                                   const luisa::unordered_set<Value *> &b) noexcept {
    if (a.size() != b.size()) { return false; }
    for (auto &v : a) {
        if (!b.contains(v)) { return false; }
    }
    return true;
}

struct ScopeDataflowState {
    luisa::unordered_set<Value *> killed;
    luisa::unordered_set<Value *> external;
    luisa::unordered_set<Value *> touched;
};

[[nodiscard]] static bool same_state(const ScopeDataflowState &a,
                                     const ScopeDataflowState &b) noexcept {
    return same_set(a.killed, b.killed) &&
           same_set(a.external, b.external) &&
           same_set(a.touched, b.touched);
}

static void merge_state_union(ScopeDataflowState &dst,
                              const ScopeDataflowState &src) noexcept {
    for (auto *value : src.external) { dst.external.emplace(value); }
    for (auto *value : src.touched) { dst.touched.emplace(value); }
}

static void merge_state_into_entry(ScopeDataflowState &dst,
                                   const ScopeDataflowState &src,
                                   bool first_pred) noexcept {
    merge_state_union(dst, src);
    if (first_pred) {
        dst.killed = src.killed;
    } else {
        luisa::unordered_set<Value *> killed;
        for (auto *value : dst.killed) {
            if (src.killed.contains(value)) { killed.emplace(value); }
        }
        dst.killed = std::move(killed);
    }
}

static void touch_value(Value *value, ScopeDataflowState &state) noexcept {
    if (value == nullptr) { return; }
    state.killed.emplace(value);
    state.touched.emplace(value);
}

static void may_touch_value(Value *value, ScopeDataflowState &state) noexcept {
    if (value == nullptr) { return; }
    state.touched.emplace(value);
}

static void use_value(Value *value, ScopeDataflowState &state) noexcept {
    if (is_always_available(value)) { return; }
    if (auto *frame_value = frame_value_for_operand(value)) {
        if (!state.killed.contains(frame_value)) {
            state.external.emplace(frame_value);
        }
    }
}

static void use_pointer_indices(Value *value, ScopeDataflowState &state) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *inst = static_cast<Instruction *>(value);
        if (inst->derived_instruction_tag() != DerivedInstructionTag::GEP) { break; }
        auto *gep = static_cast<GEPInst *>(inst);
        for (size_t i = 0u; i < gep->index_count(); ++i) {
            use_value(gep->index(i), state);
        }
        value = gep->base();
    }
}

static void transfer_call_instruction(CallInst *call, ScopeDataflowState &state) noexcept {
    auto arg_iter = call->callee()->arguments().begin();
    for (auto *arg_use : call->argument_uses()) {
        auto *argument = arg_use->value();
        if (arg_iter != call->callee()->arguments().end() &&
            (*arg_iter)->is_reference()) {
            use_pointer_indices(argument, state);
            if (auto *alloca = trace_local_alloca(argument)) {
                use_value(alloca, state);
                may_touch_value(alloca, state);
            } else {
                use_value(argument, state);
            }
        } else {
            use_value(argument, state);
        }
        if (arg_iter != call->callee()->arguments().end()) { ++arg_iter; }
    }
    if (call->type() != nullptr && !call->is_lvalue()) {
        touch_value(call, state);
    }
}

static void transfer_instruction(Instruction *inst, ScopeDataflowState &state) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA:
            break;
        case DerivedInstructionTag::CALL: {
            transfer_call_instruction(static_cast<CallInst *>(inst), state);
            break;
        }
        case DerivedInstructionTag::LOAD: {
            auto *load = static_cast<LoadInst *>(inst);
            use_pointer_indices(load->variable(), state);
            if (auto *alloca = trace_local_alloca(load->variable())) {
                use_value(alloca, state);
                touch_value(inst, state);
            } else {
                use_value(load->variable(), state);
                touch_value(inst, state);
            }
            break;
        }
        case DerivedInstructionTag::STORE: {
            auto *store = static_cast<StoreInst *>(inst);
            use_value(store->value(), state);
            use_pointer_indices(store->variable(), state);
            if (auto *alloca = trace_local_alloca(store->variable())) {
                touch_value(alloca, state);
            } else {
                use_value(store->variable(), state);
            }
            break;
        }
        case DerivedInstructionTag::CORO_SUSPEND:
        case DerivedInstructionTag::CORO_TERMINATE:
            break;
        default: {
            for (auto *op_use : inst->operand_uses()) {
                use_value(op_use->value(), state);
            }
            if (inst->type() != nullptr && !inst->is_lvalue() && !inst->is_terminator()) {
                touch_value(inst, state);
            }
            break;
        }
    }
}

struct ScopeDataflowResult {
    luisa::unordered_set<Value *> external;
    luisa::unordered_set<Value *> touched;
    luisa::unordered_map<uint32_t, luisa::unordered_set<Value *>> killed_at_suspend;
    luisa::unordered_map<uint32_t, luisa::unordered_set<Value *>> touched_at_suspend;
};

[[nodiscard]] static ScopeDataflowResult analyze_scope_use_def(
    const CoroCfgDistillResult::Scope &scope) noexcept {
    ScopeDataflowResult result;
    if (scope.blocks.empty()) { return result; }

    luisa::unordered_set<BasicBlock *> scope_blocks;
    for (auto *bb : scope.blocks) { scope_blocks.emplace(bb); }

    luisa::unordered_map<BasicBlock *, ScopeDataflowState> in_states;
    luisa::unordered_map<BasicBlock *, ScopeDataflowState> out_states;
    for (;;) {
        auto changed = false;
        for (auto *bb : scope.blocks) {
            ScopeDataflowState next_in;
            auto first_pred = true;
            if (bb == scope.blocks.front()) {
                first_pred = false;
            }
            bb->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                if (!scope_blocks.contains(pred)) { return; }
                auto it = out_states.find(pred);
                if (it == out_states.end()) { return; }
                merge_state_into_entry(next_in, it->second, first_pred);
                first_pred = false;
            });
            auto old_in = in_states.find(bb);
            if (old_in == in_states.end() || !same_state(old_in->second, next_in)) {
                in_states[bb] = next_in;
                changed = true;
            }

            auto state = next_in;
            for (auto *inst : bb->instructions()) {
                if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
                    auto *suspend = static_cast<CoroSuspendInst *>(inst);
                    result.killed_at_suspend[suspend->token()] = state.killed;
                    result.touched_at_suspend[suspend->token()] = state.touched;
                }
                transfer_instruction(inst, state);
            }
            auto old_out = out_states.find(bb);
            if (old_out == out_states.end() || !same_state(old_out->second, state)) {
                out_states[bb] = std::move(state);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    for (auto &[bb, state] : out_states) {
        static_cast<void>(bb);
        for (auto *value : state.external) { result.external.emplace(value); }
        for (auto *value : state.touched) { result.touched.emplace(value); }
    }
    return result;
}

[[nodiscard]] static luisa::unordered_set<Value *> set_difference(
    const luisa::unordered_set<Value *> &a,
    const luisa::unordered_set<Value *> &b) noexcept {
    luisa::unordered_set<Value *> r;
    for (auto *value : a) {
        if (!b.contains(value)) { r.emplace(value); }
    }
    return r;
}

[[nodiscard]] static luisa::unordered_set<Value *> set_intersection(
    const luisa::unordered_set<Value *> &a,
    const luisa::unordered_set<Value *> &b) noexcept {
    luisa::unordered_set<Value *> r;
    for (auto *value : a) {
        if (b.contains(value)) { r.emplace(value); }
    }
    return r;
}

static void append_set(luisa::unordered_set<Value *> &dst,
                       const luisa::unordered_set<Value *> &src) noexcept {
    for (auto *value : src) { dst.emplace(value); }
}

static void append_names_from_values(luisa::vector<luisa::string> &dst,
                                     const luisa::vector<Value *> &values,
                                     const luisa::unordered_map<Value *, luisa::string> &names) noexcept {
    dst.clear();
    dst.reserve(values.size());
    for (auto *value : values) {
        if (auto it = names.find(value); it != names.end()) {
            dst.emplace_back(it->second);
        }
    }
}

static void analyze_live_variables(CoroCfgDistillResult &result, FunctionDefinition *def) noexcept {
    auto n = result.scopes.size();
    luisa::vector<ScopeDataflowResult> scope_data;
    scope_data.reserve(n);
    for (auto &scope : result.scopes) {
        scope_data.emplace_back(analyze_scope_use_def(scope));
    }

    luisa::unordered_map<uint32_t, size_t> trigger_to_scope;
    for (size_t i = 0u; i < n; ++i) {
        trigger_to_scope.emplace(result.scopes[i].trigger_token, i);
    }

    result.transition_edges.clear();
    for (size_t from = 0u; from < n; ++from) {
        for (auto &sp : result.scopes[from].suspend_points) {
            auto iter = trigger_to_scope.find(sp.token);
            if (iter == trigger_to_scope.end()) { continue; }
            auto to = iter->second;
            CoroCfgDistillResult::Edge edge;
            edge.from_scope = from;
            edge.to_scope = to;
            edge.token = sp.token;
            if (auto killed = scope_data[from].killed_at_suspend.find(sp.token);
                killed != scope_data[from].killed_at_suspend.end()) {
                for (auto *value : killed->second) { edge.killed_values.emplace_back(value); }
            }
            if (auto touched = scope_data[from].touched_at_suspend.find(sp.token);
                touched != scope_data[from].touched_at_suspend.end()) {
                for (auto *value : touched->second) { edge.touched_values.emplace_back(value); }
            }
            result.transition_edges.emplace_back(std::move(edge));
        }
    }

    luisa::vector<luisa::unordered_set<Value *>> live_begin(n);
    for (;;) {
        auto changed = false;
        for (size_t ri = 0u; ri < n; ++ri) {
            auto s = n - 1u - ri;
            auto next = scope_data[s].external;
            for (auto &edge : result.transition_edges) {
                if (edge.from_scope != s || edge.to_scope >= n) { continue; }
                luisa::unordered_set<Value *> killed;
                for (auto *value : edge.killed_values) { killed.emplace(value); }
                auto propagated = set_difference(live_begin[edge.to_scope], killed);
                append_set(next, propagated);
            }
            if (!same_set(live_begin[s], next)) {
                live_begin[s] = std::move(next);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    luisa::vector<luisa::unordered_set<Value *>> live_in(n);
    luisa::vector<luisa::unordered_set<Value *>> live_out(n);
    for (size_t s = 0u; s < n; ++s) {
        live_in[s] = scope_data[s].external;
        for (auto &edge : result.transition_edges) {
            if (edge.from_scope != s || edge.to_scope >= n) { continue; }
            luisa::unordered_set<Value *> killed;
            for (auto *value : edge.killed_values) { killed.emplace(value); }
            auto propagated = set_difference(live_begin[edge.to_scope], killed);
            auto reload = set_intersection(propagated, scope_data[s].touched);
            append_set(live_in[s], reload);
            luisa::unordered_set<Value *> touched;
            for (auto *value : edge.touched_values) { touched.emplace(value); }
            auto store = set_intersection(live_begin[edge.to_scope], touched);
            edge.store_values.clear();
            for (auto *value : store) {
                edge.store_values.emplace_back(value);
                live_out[s].emplace(value);
            }
        }
    }

    luisa::unordered_set<Value *> frame_value_set;
    for (size_t i = 0u; i < n; ++i) {
        append_set(frame_value_set, live_begin[i]);
        append_set(frame_value_set, live_in[i]);
        append_set(frame_value_set, live_out[i]);
    }

    luisa::unordered_map<Value *, size_t> order;
    if (def != nullptr) {
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (frame_value_set.contains(inst) && !order.contains(inst)) {
                order.emplace(inst, order.size());
            }
        });
    }

    result.frame_values.clear();
    result.frame_values.reserve(frame_value_set.size());
    luisa::vector<Value *> ordered_frame_values;
    append_ordered_values(ordered_frame_values, frame_value_set, order);
    luisa::unordered_map<Value *, luisa::string> names;
    for (auto *value : ordered_frame_values) {
        auto name = frame_value_name(value, result.frame_values.size());
        names.emplace(value, name);
        result.frame_values.emplace_back(CoroCfgDistillResult::FrameValue{
            .value = value,
            .name = std::move(name),
            .type = value->type(),
        });
    }

    for (size_t i = 0u; i < n; ++i) {
        append_ordered_values(result.scopes[i].external_values, scope_data[i].external, order);
        append_ordered_values(result.scopes[i].touched_values, scope_data[i].touched, order);
        append_ordered_values(result.scopes[i].live_in_values, live_in[i], order);
        append_ordered_values(result.scopes[i].live_out_values, live_out[i], order);
        append_names_from_values(result.scopes[i].external_variables, result.scopes[i].external_values, names);
        append_names_from_values(result.scopes[i].touched_variables, result.scopes[i].touched_values, names);
        append_names_from_values(result.scopes[i].live_in_variables, result.scopes[i].live_in_values, names);
        append_names_from_values(result.scopes[i].live_out_variables, result.scopes[i].live_out_values, names);
    }

    for (auto &edge : result.transition_edges) {
        luisa::unordered_set<Value *> killed_set;
        luisa::unordered_set<Value *> touched_set;
        luisa::unordered_set<Value *> store_set;
        for (auto *value : edge.killed_values) { killed_set.emplace(value); }
        for (auto *value : edge.touched_values) { touched_set.emplace(value); }
        for (auto *value : edge.store_values) { store_set.emplace(value); }
        append_ordered_values(edge.killed_values, killed_set, order);
        append_ordered_values(edge.touched_values, touched_set, order);
        append_ordered_values(edge.store_values, store_set, order);
        append_names_from_values(edge.killed_variables, edge.killed_values, names);
        append_names_from_values(edge.touched_variables, edge.touched_values, names);
        append_names_from_values(edge.store_variables, edge.store_values, names);
    }
}

[[nodiscard]] static CoroCfgDistillResult distill_function(FunctionDefinition *def) noexcept {

    CoroCfgDistillResult result;
    if (def == nullptr || def->body_block() == nullptr) { return result; }

    luisa::unordered_set<BasicBlock *> reachable;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        reachable.insert(bb);
    });

    luisa::unordered_map<uint32_t, BasicBlock *> token_to_resume;
    luisa::unordered_map<uint32_t, luisa::string> token_to_name;
    for (auto *bb : def->basic_blocks()) {
        if (!reachable.contains(bb)) { continue; }
        for (auto *inst : bb->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::CORO_RESUME: {
                    auto *r = static_cast<CoroResumeInst *>(inst);
                    token_to_resume.emplace(r->token(), bb);
                    break;
                }
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto *s = static_cast<CoroSuspendInst *>(inst);
                    token_to_name.emplace(s->token(), s->name());
                    break;
                }
                default:
                    break;
            }
        }
    }

    struct Root {
        BasicBlock *block;
        uint32_t token;
        luisa::optional<luisa::string> name;
    };
    luisa::vector<Root> roots;
    roots.emplace_back(Root{def->body_block(), 0u, luisa::nullopt});
    luisa::vector<uint32_t> resume_tokens;
    resume_tokens.reserve(token_to_resume.size());
    for (auto &[token, bb] : token_to_resume) {
        static_cast<void>(bb);
        resume_tokens.emplace_back(token);
    }
    std::sort(resume_tokens.begin(), resume_tokens.end());
    for (auto token : resume_tokens) {
        auto name = luisa::optional<luisa::string>{};
        if (auto it = token_to_name.find(token); it != token_to_name.end()) {
            name = it->second;
        }
        roots.emplace_back(Root{token_to_resume.at(token), token, std::move(name)});
    }

    result.scopes.reserve(roots.size());
    for (size_t i = 0u; i < roots.size(); ++i) {
        auto &scope = result.scopes.emplace_back();
        scope.scope_id = static_cast<int>(i);
        scope.trigger_token = roots[i].token;
        scope.trigger_name = roots[i].name;

        luisa::unordered_set<BasicBlock *> visited;
        luisa::deque<BasicBlock *> worklist;
        worklist.emplace_back(roots[i].block);

        while (!worklist.empty()) {
            auto *bb = worklist.front();
            worklist.pop_front();
            if (bb == nullptr || !reachable.contains(bb)) { continue; }
            if (bb != roots[i].block) {
                if (auto resume_token = resume_token_of_block(bb);
                    resume_token.has_value() && *resume_token != roots[i].token) {
                    continue;
                }
            }
            if (!visited.emplace(bb).second) { continue; }
            scope.blocks.emplace_back(bb);
            if (!bb->is_terminated()) { continue; }
            auto *term = bb->terminator();
            switch (term->derived_instruction_tag()) {
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto *s = static_cast<CoroSuspendInst *>(term);
                    scope.suspend_points.emplace_back(
                        CoroCfgDistillResult::Scope::SuspendPoint{bb, s->token(), s->name()});
                    if (!scope.suspend_token.has_value()) {
                        scope.suspend_token = s->token();
                        scope.suspend_name = s->name();
                    }
                    break;
                }
                case DerivedInstructionTag::CORO_TERMINATE:
                    scope.is_terminal = true;
                    break;
                default:
                    bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                        worklist.emplace_back(succ);
                    });
                    break;
            }
        }
    }

    luisa::vector<luisa::unordered_set<BasicBlock *>> scope_blocks;
    scope_blocks.reserve(result.scopes.size());
    for (auto &scope : result.scopes) {
        auto &set = scope_blocks.emplace_back();
        for (auto *bb : scope.blocks) { set.emplace(bb); }
    }

    luisa::unordered_map<uint32_t, size_t> trigger_to_scope;
    for (size_t i = 0u; i < result.scopes.size(); ++i) {
        trigger_to_scope.emplace(result.scopes[i].trigger_token, i);
    }

    luisa::vector<luisa::unordered_set<size_t>> edge_sets(result.scopes.size());
    for (size_t i = 0u; i < result.scopes.size(); ++i) {
        for (auto &sp : result.scopes[i].suspend_points) {
            if (auto it = trigger_to_scope.find(sp.token); it != trigger_to_scope.end()) {
                edge_sets[i].emplace(it->second);
            }
        }
        for (auto *bb : result.scopes[i].blocks) {
            if (!bb->is_terminated()) { continue; }
            bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                if (scope_blocks[i].contains(succ)) { return; }
                for (size_t j = 0u; j < scope_blocks.size(); ++j) {
                    if (j != i && scope_blocks[j].contains(succ)) {
                        edge_sets[i].emplace(j);
                    }
                }
            });
        }
    }

    result.edges.resize(edge_sets.size());
    for (size_t i = 0u; i < edge_sets.size(); ++i) {
        result.edges[i].assign(edge_sets[i].begin(), edge_sets[i].end());
        std::sort(result.edges[i].begin(), result.edges[i].end());
    }

    analyze_live_variables(result, def);
    return result;
}

}// namespace detail

CoroCfgDistillResult coro_cfg_distill_pass_run_on_function(Function *f) noexcept {
    auto *def = f->definition();
    if (def == nullptr) { return {}; }
    return detail::distill_function(def);
}

size_t coro_cfg_distill_pass_run_on_module(Module *m) noexcept {
    size_t count = 0u;
    for (auto *f : m->function_list()) {
        if (f->is_definition()) {
            static_cast<void>(detail::distill_function(static_cast<FunctionDefinition *>(f)));
            ++count;
        }
    }
    return count;
}

}// namespace luisa::compute::xir
