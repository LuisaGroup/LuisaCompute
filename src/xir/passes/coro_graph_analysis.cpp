#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/passes/coro_graph_analysis.h>

namespace luisa::compute::xir::coro {

namespace {

// Stage 1: translate the structured XIR CFG into a flat CoroInstruction vector.

struct PreliminaryBuilder {
    luisa::vector<CoroInstruction> instructions;
    luisa::unordered_map<Instruction *, CoroInstrRef> source_to_instr;

    CoroInstrRef add(CoroInstruction node) noexcept {
        CoroInstrRef ref{instructions.size()};
        instructions.emplace_back(std::move(node));
        return ref;
    }

    CoroInstrRef translate_inst(Instruction *inst) noexcept {
        if (auto it = source_to_instr.find(inst); it != source_to_instr.end()) {
            return it->second;
        }
        CoroInstruction node{};
        node.source_inst = inst;
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::RETURN:
            case DerivedInstructionTag::UNREACHABLE:
            case DerivedInstructionTag::RASTER_DISCARD: {
                node.tag = CoroInstruction::Tag::TERMINATE;
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            case DerivedInstructionTag::CORO_SUSPEND: {
                auto suspend = static_cast<CoroSuspendInst *>(inst);
                node.tag = CoroInstruction::Tag::SUSPEND;
                node.suspend_token = suspend->token();
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            case DerivedInstructionTag::CORO_REGISTER: {
                node.tag = CoroInstruction::Tag::SIMPLE;
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            case DerivedInstructionTag::LOOP: {
                auto loop = static_cast<LoopInst *>(inst);
                node.tag = CoroInstruction::Tag::LOOP;
                // XIR LoopInst: prepare → cond_br(body, merge); body; update → br(prepare)
                // Translate as: body = [prepare_instrs..., body_instrs..., update_instrs...]
                // The loop condition is the last instruction of prepare (a cond_br).
                // For the coro graph, we model it as a do-while: body runs, then cond checked.
                // Actually XIR loops are for(;;) style: prepare checks cond, body runs, update runs.
                // We'll flatten: body = translate_block(prepare) ++ translate_block(body_block) ++ translate_block(update)
                // cond = the conditional branch in prepare (if any)
                // But the Rust IR uses do-while. XIR uses for-style. We need to adapt.
                // Simplification: treat the whole loop as a single node with sub-blocks.
                // The materializer will clone the loop structure directly.
                if (auto prepare = loop->prepare_block()) {
                    node.body = translate_block(prepare);
                }
                luisa::vector<CoroInstrRef> body_instrs;
                if (auto body_bb = loop->body_block()) {
                    body_instrs = translate_block(body_bb);
                }
                luisa::vector<CoroInstrRef> update_instrs;
                if (auto update = loop->update_block()) {
                    update_instrs = translate_block(update);
                }
                // Store as: body = body_block instructions, cond references prepare,
                // true_branch = update instructions (for the materializer to know structure)
                // Actually let's keep it simple: store all sub-blocks in the node.
                // body = prepare instructions
                // true_branch = body_block instructions
                // false_branch = update instructions
                // The materializer knows LoopInst structure: prepare→body→update→prepare cycle.
                node.true_branch = std::move(body_instrs);
                node.false_branch = std::move(update_instrs);
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            case DerivedInstructionTag::SIMPLE_LOOP: {
                auto loop = static_cast<SimpleLoopInst *>(inst);
                node.tag = CoroInstruction::Tag::LOOP;
                if (auto body_bb = loop->body_block()) {
                    node.body = translate_block(body_bb);
                }
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            case DerivedInstructionTag::IF: {
                auto if_inst = static_cast<IfInst *>(inst);
                node.tag = CoroInstruction::Tag::IF;
                // cond is the condition operand — we'll reference it via source_inst
                if (auto tb = if_inst->true_block()) {
                    node.true_branch = translate_block(tb);
                }
                if (auto fb = if_inst->false_block()) {
                    node.false_branch = translate_block(fb);
                }
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            case DerivedInstructionTag::SWITCH: {
                auto sw = static_cast<SwitchInst *>(inst);
                node.tag = CoroInstruction::Tag::SWITCH;
                auto case_vals = sw->case_values();
                for (size_t i = 0; i < sw->case_count(); ++i) {
                    CoroSwitchCase c;
                    c.value = case_vals[i];
                    if (auto cb = sw->case_block(i)) {
                        c.body = translate_block(cb);
                    }
                    node.cases.emplace_back(std::move(c));
                }
                if (auto db = sw->default_block()) {
                    node.default_branch = translate_block(db);
                }
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
            default: {
                node.tag = CoroInstruction::Tag::SIMPLE;
                auto ref = add(std::move(node));
                source_to_instr.emplace(inst, ref);
                return ref;
            }
        }
    }

    luisa::vector<CoroInstrRef> translate_block(BasicBlock *block) noexcept {
        luisa::vector<CoroInstrRef> refs;
        if (block == nullptr) { return refs; }
        for (auto inst : block->instructions()) {
            refs.emplace_back(translate_inst(inst));
            // For structured CF nodes, also translate the merge block contents
            // (instructions after the construct live there in XIR).
            if (auto *cfm = inst->control_flow_merge()) {
                if (auto *merge = cfm->merge_block()) {
                    auto merge_refs = translate_block(merge);
                    refs.insert(refs.end(), merge_refs.begin(), merge_refs.end());
                }
            }
        }
        return refs;
    }
};

// Find terminators recursively: a node is a terminator if it's a Suspend,
// Terminate, or a control-flow node where ALL branches terminate.
void find_terminators(const luisa::vector<CoroInstruction> &instructions,
                      CoroInstrRef ref,
                      luisa::unordered_map<size_t, bool> &known) noexcept {
    if (!ref.valid()) { return; }
    if (known.contains(ref.index)) { return; }
    auto &instr = instructions[ref.index];
    auto block_terminates = [&](const luisa::vector<CoroInstrRef> &block) -> bool {
        for (auto r : block) {
            find_terminators(instructions, r, known);
        }
        for (auto r : block) {
            if (known.contains(r.index) && known[r.index]) { return true; }
        }
        return false;
    };
    bool result = false;
    switch (instr.tag) {
        case CoroInstruction::Tag::SUSPEND:
        case CoroInstruction::Tag::TERMINATE:
            result = true;
            break;
        case CoroInstruction::Tag::ENTRY_SCOPE:
            result = block_terminates(instr.body);
            break;
        case CoroInstruction::Tag::LOOP: {
            // A loop terminates if its body terminates (contains suspend/return).
            // Check all sub-blocks: body (prepare), true_branch (body_block), false_branch (update)
            bool any = block_terminates(instr.body) ||
                       block_terminates(instr.true_branch) ||
                       block_terminates(instr.false_branch);
            result = any;
            break;
        }
        case CoroInstruction::Tag::IF: {
            bool t = block_terminates(instr.true_branch);
            bool f = block_terminates(instr.false_branch);
            result = t && f;
            break;
        }
        case CoroInstruction::Tag::SWITCH: {
            bool all = true;
            for (auto &c : instr.cases) {
                if (!block_terminates(c.body)) { all = false; }
            }
            if (!block_terminates(instr.default_branch)) { all = false; }
            result = all;
            break;
        }
        default:
            result = false;
            break;
    }
    known.emplace(ref.index, result);
}

// Stage 2: split into scopes. Walk from entry, when a suspend is hit, start a
// new scope for the continuation. Insert condition-replay and first-flag nodes.

struct ScopeSplitter {
    CoroPreliminaryGraph &pg;
    luisa::vector<CoroScope> scopes;
    luisa::unordered_map<uint32_t, CoroScopeRef> token_to_scope;
    luisa::unordered_map<luisa::string, Value *> designated_values;

    // Instruction stack: parent control-flow nodes on the path to current position.
    struct StackFrame {
        CoroInstrRef instr;
        int32_t branch_value{};// which branch we took (for condition replay)
        bool is_loop{};
    };

    CoroScopeRef create_scope() noexcept {
        CoroScopeRef ref{scopes.size()};
        scopes.emplace_back();
        return ref;
    }

    // Collect reachable nodes from a position in a block, stopping at the next
    // suspend. Returns the suspend token if one was hit, or -1 if the block
    // ended without a suspend.
    int64_t collect_reachable(
        const luisa::vector<CoroInstrRef> &block,
        size_t start_index,
        CoroScopeRef scope,
        const luisa::vector<StackFrame> &stack) noexcept {

        for (size_t i = start_index; i < block.size(); ++i) {
            auto ref = block[i];
            auto &instr = pg.instructions[ref.index];
            switch (instr.tag) {
                case CoroInstruction::Tag::SUSPEND: {
                    // Hit a suspend. Create a new scope for the continuation.
                    auto cont_scope = create_scope();
                    token_to_scope.emplace(instr.suspend_token, cont_scope);
                    // Build condition-replay items from the stack.
                    if (!stack.empty()) {
                        CoroInstruction replay_node{};
                        replay_node.tag = CoroInstruction::Tag::CONDITION_STACK_REPLAY;
                        for (auto &frame : stack) {
                            if (!frame.is_loop) {
                                ConditionStackItem item{};
                                item.control_flow_inst = pg.instructions[frame.instr.index].source_inst;
                                item.value = frame.branch_value;
                                replay_node.replay_items.emplace_back(item);
                            }
                        }
                        if (!replay_node.replay_items.empty()) {
                            auto replay_ref = CoroInstrRef{pg.instructions.size()};
                            pg.instructions.emplace_back(std::move(replay_node));
                            scopes[cont_scope.index].instructions.emplace_back(replay_ref);
                        }
                    }
                    // Collect the rest of this block into the continuation scope.
                    // But first: if there are enclosing loops in the stack, we need
                    // to emit the first-flag + loop reconstruction.
                    bool has_enclosing_loop = false;
                    for (auto &frame : stack) {
                        if (frame.is_loop) { has_enclosing_loop = true; break; }
                    }
                    if (!has_enclosing_loop) {
                        collect_reachable(block, i + 1, cont_scope, {});
                    } else {
                        // Collect dominated instructions (rest of this block after suspend).
                        luisa::vector<CoroInstrRef> cont_instrs;
                        for (size_t j = i + 1; j < block.size(); ++j) {
                            cont_instrs.emplace_back(block[j]);
                        }
                        // Reconstruct enclosing loops with first-flag guards.
                        reconstruct_loops(cont_scope, stack, block, i);
                        // Recursively process the continuation to find further suspends.
                        auto &cont_scope_instrs = scopes[cont_scope.index].instructions;
                        auto copy = cont_scope_instrs;
                        cont_scope_instrs.clear();
                        collect_reachable(copy, 0, cont_scope, {});
                    }
                    return static_cast<int64_t>(instr.suspend_token);
                }
                case CoroInstruction::Tag::TERMINATE: {
                    scopes[scope.index].instructions.emplace_back(ref);
                    return -1;
                }
                case CoroInstruction::Tag::LOOP: {
                    // Recurse into loop sub-blocks looking for suspends.
                    auto new_stack = stack;
                    new_stack.push_back(StackFrame{ref, 0, true});
                    // Check prepare (body field), body_block (true_branch), update (false_branch)
                    bool found_suspend = false;
                    auto check_block = [&](const luisa::vector<CoroInstrRef> &sub_block) {
                        if (found_suspend) return;
                        auto token = collect_reachable(sub_block, 0, scope, new_stack);
                        if (token >= 0) { found_suspend = true; }
                    };
                    // If the loop contains a suspend, we've already handled it via
                    // collect_reachable recursion. If not, add the loop as-is.
                    if (contains_suspend(instr)) {
                        check_block(instr.body);       // prepare
                        check_block(instr.true_branch);// body_block
                        check_block(instr.false_branch);// update
                        if (!found_suspend) {
                            scopes[scope.index].instructions.emplace_back(ref);
                        }
                    } else {
                        scopes[scope.index].instructions.emplace_back(ref);
                    }
                    if (found_suspend) { return 1; }// signal that we handled it
                    break;
                }
                case CoroInstruction::Tag::IF: {
                    if (contains_suspend(instr)) {
                        // Recurse into branches.
                        auto new_stack_t = stack;
                        new_stack_t.push_back(StackFrame{ref, 1, false});
                        auto new_stack_f = stack;
                        new_stack_f.push_back(StackFrame{ref, 0, false});
                        auto t = collect_reachable(instr.true_branch, 0, scope, new_stack_t);
                        auto f = collect_reachable(instr.false_branch, 0, scope, new_stack_f);
                        if (t >= 0 || f >= 0) { return std::max(t, f); }
                    } else {
                        scopes[scope.index].instructions.emplace_back(ref);
                    }
                    break;
                }
                case CoroInstruction::Tag::SWITCH: {
                    if (contains_suspend(instr)) {
                        for (size_t ci = 0; ci < instr.cases.size(); ++ci) {
                            auto new_stack_c = stack;
                            new_stack_c.push_back(StackFrame{ref, instr.cases[ci].value, false});
                            collect_reachable(instr.cases[ci].body, 0, scope, new_stack_c);
                        }
                        auto new_stack_d = stack;
                        new_stack_d.push_back(StackFrame{ref, -1, false});
                        collect_reachable(instr.default_branch, 0, scope, new_stack_d);
                    } else {
                        scopes[scope.index].instructions.emplace_back(ref);
                    }
                    break;
                }
                default: {
                    scopes[scope.index].instructions.emplace_back(ref);
                    break;
                }
            }
        }
        return -1;
    }

    bool contains_suspend(const CoroInstruction &instr) const noexcept {
        auto check_block = [&](const luisa::vector<CoroInstrRef> &block) -> bool {
            for (auto r : block) {
                if (r.valid() && pg.instructions[r.index].tag == CoroInstruction::Tag::SUSPEND) return true;
                if (r.valid() && contains_suspend(pg.instructions[r.index])) return true;
            }
            return false;
        };
        switch (instr.tag) {
            case CoroInstruction::Tag::SUSPEND: return true;
            case CoroInstruction::Tag::LOOP:
                return check_block(instr.body) || check_block(instr.true_branch) || check_block(instr.false_branch);
            case CoroInstruction::Tag::IF:
                return check_block(instr.true_branch) || check_block(instr.false_branch);
            case CoroInstruction::Tag::SWITCH: {
                for (auto &c : instr.cases) { if (check_block(c.body)) return true; }
                return check_block(instr.default_branch);
            }
            case CoroInstruction::Tag::ENTRY_SCOPE:
                return check_block(instr.body);
            default: return false;
        }
    }

    void reconstruct_loops(CoroScopeRef scope, const luisa::vector<StackFrame> &stack,
                           const luisa::vector<CoroInstrRef> &suspend_block, size_t suspend_idx) noexcept {
        // Find enclosing loops from innermost to outermost.
        // For each loop: emit a first_flag, then clone the loop with:
        //   - pre-suspend nodes guarded by if(!first_flag)
        //   - post-suspend nodes run normally
        //   - first_flag cleared at the suspend point
        // Then emit post-loop code (merge block instructions after the loop in the parent).

        // Create first_flag node
        CoroInstruction flag_node{};
        flag_node.tag = CoroInstruction::Tag::MAKE_FIRST_FLAG;
        auto flag_ref = CoroInstrRef{pg.instructions.size()};
        pg.instructions.emplace_back(std::move(flag_node));
        scopes[scope.index].instructions.emplace_back(flag_ref);

        // Find the innermost enclosing loop in the stack.
        for (int si = static_cast<int>(stack.size()) - 1; si >= 0; --si) {
            if (!stack[si].is_loop) continue;
            auto loop_ref = stack[si].instr;
            auto &loop_instr = pg.instructions[loop_ref.index];

            // Build the reconstructed loop body:
            // For each sub-block of the loop, wrap pre-suspend instructions in
            // SkipIfFirstFlag, leave post-suspend instructions as-is, and insert
            // ClearFirstFlag at the suspend point.
            CoroInstruction reconstructed_loop{};
            reconstructed_loop.tag = CoroInstruction::Tag::LOOP;
            reconstructed_loop.source_inst = loop_instr.source_inst;

            // We need to rebuild the loop's sub-blocks with first-flag guards.
            // For simplicity in this initial impl: clone the entire loop body
            // with a SkipIfFirstFlag wrapping all instructions before the suspend,
            // and a ClearFirstFlag at the suspend position.
            auto wrap_block_with_first_flag = [&](const luisa::vector<CoroInstrRef> &block) -> luisa::vector<CoroInstrRef> {
                luisa::vector<CoroInstrRef> result;
                luisa::vector<CoroInstrRef> pre_suspend;
                bool past_suspend = false;
                for (auto r : block) {
                    if (r.index == suspend_block[suspend_idx].index) {
                        // Insert SkipIfFirstFlag for pre-suspend
                        if (!pre_suspend.empty()) {
                            CoroInstruction skip{};
                            skip.tag = CoroInstruction::Tag::SKIP_IF_FIRST_FLAG;
                            skip.first_flag = flag_ref;
                            skip.body = std::move(pre_suspend);
                            auto skip_ref = CoroInstrRef{pg.instructions.size()};
                            pg.instructions.emplace_back(std::move(skip));
                            result.emplace_back(skip_ref);
                            pre_suspend.clear();
                        }
                        // Insert ClearFirstFlag
                        CoroInstruction clear{};
                        clear.tag = CoroInstruction::Tag::CLEAR_FIRST_FLAG;
                        clear.first_flag = flag_ref;
                        auto clear_ref = CoroInstrRef{pg.instructions.size()};
                        pg.instructions.emplace_back(std::move(clear));
                        result.emplace_back(clear_ref);
                        past_suspend = true;
                    } else if (!past_suspend) {
                        // Check if this node contains the suspend (nested)
                        if (pg.instructions[r.index].tag == CoroInstruction::Tag::IF ||
                            pg.instructions[r.index].tag == CoroInstruction::Tag::SWITCH ||
                            pg.instructions[r.index].tag == CoroInstruction::Tag::LOOP) {
                            if (contains_suspend(pg.instructions[r.index])) {
                                // Wrap everything before this in SkipIfFirstFlag
                                if (!pre_suspend.empty()) {
                                    CoroInstruction skip{};
                                    skip.tag = CoroInstruction::Tag::SKIP_IF_FIRST_FLAG;
                                    skip.first_flag = flag_ref;
                                    skip.body = std::move(pre_suspend);
                                    auto skip_ref = CoroInstrRef{pg.instructions.size()};
                                    pg.instructions.emplace_back(std::move(skip));
                                    result.emplace_back(skip_ref);
                                    pre_suspend.clear();
                                }
                                // Recursively handle this nested CF node
                                // For now, add it as-is (the materializer will handle nested replay)
                                result.emplace_back(r);
                                past_suspend = true;
                            } else {
                                pre_suspend.emplace_back(r);
                            }
                        } else {
                            pre_suspend.emplace_back(r);
                        }
                    } else {
                        result.emplace_back(r);
                    }
                }
                // If we never found the suspend in this block, just return as-is with guard
                if (!past_suspend && !pre_suspend.empty()) {
                    CoroInstruction skip{};
                    skip.tag = CoroInstruction::Tag::SKIP_IF_FIRST_FLAG;
                    skip.first_flag = flag_ref;
                    skip.body = std::move(pre_suspend);
                    auto skip_ref = CoroInstrRef{pg.instructions.size()};
                    pg.instructions.emplace_back(std::move(skip));
                    result.emplace_back(skip_ref);
                }
                return result;
            };

            // Reconstruct each sub-block of the loop
            reconstructed_loop.body = wrap_block_with_first_flag(loop_instr.body);          // prepare
            reconstructed_loop.true_branch = wrap_block_with_first_flag(loop_instr.true_branch); // body_block
            reconstructed_loop.false_branch = wrap_block_with_first_flag(loop_instr.false_branch); // update

            auto recon_ref = CoroInstrRef{pg.instructions.size()};
            pg.instructions.emplace_back(std::move(reconstructed_loop));
            scopes[scope.index].instructions.emplace_back(recon_ref);

            // After the loop, collect the merge block (post-loop) instructions.
            // These are the instructions in the parent block after the loop instruction.
            // We need to find them from the parent scope. For now, we rely on the
            // caller (collect_reachable) having already handled post-loop collection
            // by continuing iteration after the loop in the parent block.
            break;// only handle innermost loop for now; nested loops need recursion
        }
    }
};

}// namespace

CoroPreliminaryGraph coro_preliminary_graph_build(Function *function) noexcept {
    CoroPreliminaryGraph result;
    if (function == nullptr || !function->is_definition()) {
        result.diagnostics.emplace_back("coro_preliminary_graph_build: function is null or not a definition");
        return result;
    }
    auto def = static_cast<FunctionDefinition *>(function);
    auto body = def->body_block();
    if (body == nullptr) {
        result.diagnostics.emplace_back("coro_preliminary_graph_build: function has no body block");
        return result;
    }
    PreliminaryBuilder builder;
    auto body_refs = builder.translate_block(body);

    // Wrap in an EntryScope node.
    CoroInstruction entry_node{};
    entry_node.tag = CoroInstruction::Tag::ENTRY;
    auto entry_ref = CoroInstrRef{builder.instructions.size()};
    builder.instructions.emplace_back(std::move(entry_node));
    body_refs.insert(body_refs.begin(), entry_ref);

    CoroInstruction entry_scope{};
    entry_scope.tag = CoroInstruction::Tag::ENTRY_SCOPE;
    entry_scope.body = std::move(body_refs);
    auto entry_scope_ref = CoroInstrRef{builder.instructions.size()};
    builder.instructions.emplace_back(std::move(entry_scope));

    // Find terminators.
    luisa::unordered_map<size_t, bool> known;
    find_terminators(builder.instructions, entry_scope_ref, known);
    luisa::unordered_set<size_t> terminators;
    for (auto &[idx, is_term] : known) {
        if (is_term) { terminators.emplace(idx); }
    }

    result.instructions = std::move(builder.instructions);
    result.source_to_instr = std::move(builder.source_to_instr);
    result.terminators = std::move(terminators);
    result.entry_scope = entry_scope_ref;
    return result;
}

CoroGraphInfo coro_graph_split(CoroPreliminaryGraph preliminary) noexcept {
    CoroGraphInfo info;
    if (!preliminary.entry_scope.valid()) {
        info.diagnostics.emplace_back("coro_graph_split: no entry scope in preliminary graph");
        return info;
    }
    ScopeSplitter splitter{preliminary, {}, {}, {}};
    // Create the entry scope (scope 0).
    auto entry_scope = splitter.create_scope();
    // Walk the entry scope body.
    auto &entry_instr = preliminary.instructions[preliminary.entry_scope.index];
    if (entry_instr.tag == CoroInstruction::Tag::ENTRY_SCOPE) {
        // Skip the Entry marker at index 0.
        size_t start = 0;
        if (!entry_instr.body.empty() &&
            preliminary.instructions[entry_instr.body[0].index].tag == CoroInstruction::Tag::ENTRY) {
            start = 1;
        }
        splitter.collect_reachable(entry_instr.body, start, entry_scope, {});
    }
    info.ok = true;
    info.preliminary = std::move(preliminary);
    info.scopes = std::move(splitter.scopes);
    info.token_to_scope = std::move(splitter.token_to_scope);
    info.designated_values = std::move(splitter.designated_values);
    return info;
}

CoroGraphInfo coro_graph_run_on_function(Function *function) noexcept {
    auto pg = coro_preliminary_graph_build(function);
    if (!pg.entry_scope.valid()) {
        CoroGraphInfo info;
        info.diagnostics = std::move(pg.diagnostics);
        return info;
    }
    return coro_graph_split(std::move(pg));
}

}// namespace luisa::compute::xir::coro
