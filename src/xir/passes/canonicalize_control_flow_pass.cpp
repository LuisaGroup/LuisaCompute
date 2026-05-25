#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/Canonicalize_Control_Flow.h>
#include <luisa/xir/passes/reg2mem.h>

namespace luisa::compute::xir {

namespace detail {

struct LoopCollector {
    luisa::unordered_set<BasicBlock *> visited_blocks;
    luisa::vector<LoopInst *> loops;

    void collect_block(BasicBlock *block) noexcept {
        if (block == nullptr || !visited_blocks.emplace(block).second) {
            return;
        }
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: {
                    auto if_inst = static_cast<IfInst *>(inst);
                    collect_block(if_inst->true_block());
                    collect_block(if_inst->false_block());
                    collect_block(if_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(inst);
                    for (auto i = 0u; i < switch_inst->case_count(); i++) {
                        collect_block(switch_inst->case_block(i));
                    }
                    collect_block(switch_inst->default_block());
                    collect_block(switch_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop_inst = static_cast<LoopInst *>(inst);
                    collect_block(loop_inst->prepare_block());
                    collect_block(loop_inst->body_block());
                    collect_block(loop_inst->update_block());
                    collect_block(loop_inst->merge_block());
                    loops.emplace_back(loop_inst);
                    break;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto simple_loop = static_cast<SimpleLoopInst *>(inst);
                    collect_block(simple_loop->body_block());
                    collect_block(simple_loop->merge_block());
                    break;
                }
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto branch = static_cast<ConditionalBranchInst *>(inst);
                    collect_block(branch->true_block());
                    collect_block(branch->false_block());
                    break;
                }
                case DerivedInstructionTag::BRANCH:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE: {
                    auto branch = static_cast<BranchTerminatorInstruction *>(inst);
                    collect_block(branch->target_block());
                    break;
                }
                default: break;
            }
        }
    }
};

[[nodiscard]] static bool is_supported_ast2xir_for_loop_shape(const LoopInst *loop) noexcept {
    if (loop == nullptr) {
        return false;
    }
    auto prepare = loop->prepare_block();
    auto body = loop->body_block();
    auto update = loop->update_block();
    auto merge = loop->merge_block();
    if (prepare == nullptr || body == nullptr || update == nullptr || merge == nullptr) {
        return false;
    }
    if (!prepare->is_terminated() || !update->is_terminated()) {
        return false;
    }
    auto prepare_terminator = prepare->terminator();
    if (!prepare_terminator->isa<ConditionalBranchInst>()) {
        return false;
    }
    auto cond_branch = static_cast<const ConditionalBranchInst *>(prepare_terminator);
    if (cond_branch->condition() == nullptr ||
        cond_branch->true_block() != body ||
        cond_branch->false_block() != merge) {
        return false;
    }
    auto update_terminator = update->terminator();
    if (!update_terminator->isa<BranchInst>()) {
        return false;
    }
    return static_cast<const BranchInst *>(update_terminator)->target_block() == prepare;
}

static void move_prepare_instructions_to_simple_body(BasicBlock *prepare_block,
                                                     BasicBlock *simple_body_block) noexcept {
    XIRBuilder builder;
    builder.set_insertion_point(simple_body_block);
    while (!prepare_block->instructions().empty()) {
        auto inst = prepare_block->instructions().front();
        if (inst->is_terminator()) {
            break;
        }
        builder.append(inst->remove_self());
    }
}

static void lower_loop_to_simple_loop(LoopInst *loop, Canonicalize_Control_Flow_Info &info) noexcept {
    if (!is_supported_ast2xir_for_loop_shape(loop)) {
        info.skipped_loop_count++;
        return;
    }

    auto old_prepare = loop->prepare_block();
    auto old_body = loop->body_block();
    auto old_update = loop->update_block();
    auto old_merge = loop->merge_block();
    auto cond_branch = static_cast<ConditionalBranchInst *>(old_prepare->terminator());
    auto cond = cond_branch->condition();

    LUISA_DEBUG_ASSERT(cond != nullptr, "The loop prepare block must end with a non-null condition.");

    XIRBuilder builder;
    builder.set_insertion_point(loop->prev());
    auto simple_loop = builder.simple_loop();
    auto simple_body = simple_loop->create_body_block();
    simple_loop->set_merge_block(old_merge);

    move_prepare_instructions_to_simple_body(old_prepare, simple_body);

    builder.set_insertion_point(simple_body);
    auto guard = builder.if_(cond);
    guard->set_true_target(old_body);
    auto false_block = guard->create_false_block();
    guard->set_merge_block(old_update);

    builder.set_insertion_point(false_block);
    builder.break_(old_merge);

    auto update_branch = static_cast<BranchInst *>(old_update->terminator());
    update_branch->set_target_block(simple_body);

    loop->remove_self();
    info.lowered_loop_count++;
}

static void move_all_instructions(BasicBlock *from, BasicBlock *to) noexcept {
    LUISA_DEBUG_ASSERT(from != nullptr && to != nullptr, "Invalid basic block.");
    XIRBuilder builder;
    builder.set_insertion_point(to->instructions().head_sentinel());
    while (!from->instructions().empty()) {
        builder.append(from->instructions().front()->remove_self());
    }
}

[[nodiscard]] static BasicBlock *guard_block_entry_with_flag(Function *function,
                                                             AllocaInst *flag,
                                                             BasicBlock *block,
                                                             BasicBlock *skip_target) noexcept {
    LUISA_DEBUG_ASSERT(function != nullptr, "The function owning the guarded block must not be null.");
    LUISA_DEBUG_ASSERT(flag != nullptr, "The control-flow flag must not be null.");
    LUISA_DEBUG_ASSERT(block != nullptr, "The block to guard must not be null.");
    LUISA_DEBUG_ASSERT(skip_target != nullptr, "The guarded block skip target must not be null.");
    if (block == skip_target) {
        return block;
    }
    auto guarded_body = function->create_basic_block();
    move_all_instructions(block, guarded_body);

    XIRBuilder builder;
    builder.set_insertion_point(block->instructions().head_sentinel());
    auto flag_value = builder.load(Type::of<bool>(), flag);
    auto not_flag = builder.call(Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {flag_value});
    builder.cond_br(not_flag, guarded_body, skip_target);
    return guarded_body;
}

struct LoopControlFlags {
    AllocaInst *break_flag{nullptr};
    AllocaInst *continue_flag{nullptr};
};

enum class LoopControlKind : uint32_t {
    BREAK,
    CONTINUE
};

struct LoopControlFlagPreprocess {
    Function *_function;
    FunctionDefinition *_definition;
    luisa::unordered_map<Instruction *, LoopControlFlags> _flags;
    luisa::vector<Instruction *> _loop_stack;

    [[nodiscard]] auto _module() const noexcept { return _function->parent_module(); }

    LoopControlFlags &ensure_flags(Instruction *loop) noexcept {
        return _flags[loop];
    }

    void ensure_flag(Instruction *loop, LoopControlKind kind) noexcept {
        auto &flags = ensure_flags(loop);
        auto &flag = kind == LoopControlKind::BREAK ? flags.break_flag : flags.continue_flag;
        if (flag != nullptr) {
            return;
        }
        XIRBuilder builder;
        builder.set_insertion_point(loop->prev());
        flag = builder.alloca_local(Type::of<bool>());
        flag->add_comment(kind == LoopControlKind::BREAK ? "loop break flag" : "loop continue flag");
        auto const_false = _module()->create_constant_zero(Type::of<bool>());
        builder.store(flag, const_false);
    }

    void traverse_block(BasicBlock *block) noexcept {
        if (block == nullptr) {
            return;
        }
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: {
                    auto if_inst = static_cast<IfInst *>(inst);
                    traverse_block(if_inst->true_block());
                    traverse_block(if_inst->false_block());
                    traverse_block(if_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(inst);
                    for (auto i = 0u; i < switch_inst->case_count(); i++) {
                        traverse_block(switch_inst->case_block(i));
                    }
                    traverse_block(switch_inst->default_block());
                    traverse_block(switch_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop_inst = static_cast<LoopInst *>(inst);
                    _loop_stack.emplace_back(loop_inst);
                    traverse_block(loop_inst->body_block());
                    _loop_stack.pop_back();
                    traverse_block(loop_inst->prepare_block());
                    traverse_block(loop_inst->update_block());
                    traverse_block(loop_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto simple_loop = static_cast<SimpleLoopInst *>(inst);
                    _loop_stack.emplace_back(simple_loop);
                    traverse_block(simple_loop->body_block());
                    _loop_stack.pop_back();
                    traverse_block(simple_loop->merge_block());
                    break;
                }
                case DerivedInstructionTag::BREAK:
                    LUISA_DEBUG_ASSERT(!_loop_stack.empty(), "Break outside of loop.");
                    ensure_flag(_loop_stack.back(), LoopControlKind::BREAK);
                    break;
                case DerivedInstructionTag::CONTINUE:
                    LUISA_DEBUG_ASSERT(!_loop_stack.empty(), "Continue outside of loop.");
                    ensure_flag(_loop_stack.back(), LoopControlKind::CONTINUE);
                    break;
                default: break;
            }
        }
    }

    [[nodiscard]] auto run() noexcept -> luisa::unordered_map<Instruction *, LoopControlFlags> {
        if (_definition != nullptr) {
            traverse_block(_definition->body_block());
        }
        return std::move(_flags);
    }
};

struct StructuredLoopCollector {
    luisa::unordered_set<BasicBlock *> visited_blocks;
    luisa::vector<Instruction *> loops;

    void collect_block(BasicBlock *block) noexcept {
        if (block == nullptr || !visited_blocks.emplace(block).second) {
            return;
        }
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: {
                    auto if_inst = static_cast<IfInst *>(inst);
                    collect_block(if_inst->true_block());
                    collect_block(if_inst->false_block());
                    collect_block(if_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(inst);
                    for (auto i = 0u; i < switch_inst->case_count(); i++) {
                        collect_block(switch_inst->case_block(i));
                    }
                    collect_block(switch_inst->default_block());
                    collect_block(switch_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop_inst = static_cast<LoopInst *>(inst);
                    collect_block(loop_inst->prepare_block());
                    collect_block(loop_inst->body_block());
                    collect_block(loop_inst->update_block());
                    loops.emplace_back(loop_inst);
                    collect_block(loop_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto simple_loop = static_cast<SimpleLoopInst *>(inst);
                    collect_block(simple_loop->body_block());
                    loops.emplace_back(simple_loop);
                    collect_block(simple_loop->merge_block());
                    break;
                }
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto branch = static_cast<ConditionalBranchInst *>(inst);
                    collect_block(branch->true_block());
                    collect_block(branch->false_block());
                    break;
                }
                case DerivedInstructionTag::BRANCH:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE: {
                    auto branch = static_cast<BranchTerminatorInstruction *>(inst);
                    collect_block(branch->target_block());
                    break;
                }
                default: break;
            }
        }
    }
};

class LoopControlLowering {

private:
    Function *_function;
    Instruction *_loop{nullptr};
    AllocaInst *_flag{nullptr};
    LoopControlKind _kind;

private:
    [[nodiscard]] auto _module() const noexcept { return _function->parent_module(); }

    [[nodiscard]] bool matches(Instruction *inst) const noexcept {
        return _kind == LoopControlKind::BREAK ? inst->isa<BreakInst>() : inst->isa<ContinueInst>();
    }

    void reset_flag_at_entry(BasicBlock *block) const noexcept {
        LUISA_DEBUG_ASSERT(block != nullptr, "The loop entry block must not be null.");
        XIRBuilder builder;
        builder.set_insertion_point(block->instructions().head_sentinel());
        auto const_false = _module()->create_constant_zero(Type::of<bool>());
        builder.store(_flag, const_false);
    }

    void rewrite_control(BranchTerminatorInstruction *control_inst, BasicBlock *target) const noexcept {
        LUISA_DEBUG_ASSERT(control_inst != nullptr, "The control terminator to rewrite must not be null.");
        LUISA_DEBUG_ASSERT(target != nullptr, "The rewritten control-flow target must not be null.");
        XIRBuilder builder;
        builder.set_insertion_point(control_inst->prev());
        auto const_true = _module()->create_constant_one(Type::of<bool>());
        builder.store(_flag, const_true);
        auto branch = builder.br(target);
        auto owned_branch = branch->remove_self();
        control_inst->replace_self_with(std::move(owned_branch));
    }

    [[nodiscard]] bool transform_block(BasicBlock *block,
                                       BasicBlock *follow_block,
                                       bool guard_entry) const noexcept {
        LUISA_DEBUG_ASSERT(block != nullptr, "The block to transform must not be null.");
        LUISA_DEBUG_ASSERT(follow_block != nullptr, "The block follow target must not be null.");
        if (guard_entry) {
            block = guard_block_entry_with_flag(_function, _flag, block, follow_block);
        }
        auto terminator = block->terminator();
        switch (terminator->derived_instruction_tag()) {
            case DerivedInstructionTag::BREAK:
            case DerivedInstructionTag::CONTINUE: {
                if (!matches(terminator)) {
                    return false;
                }
                rewrite_control(static_cast<BranchTerminatorInstruction *>(terminator), follow_block);
                return true;
            }
            case DerivedInstructionTag::IF: {
                auto if_inst = static_cast<IfInst *>(terminator);
                auto true_control = transform_block(if_inst->true_block(), if_inst->merge_block(), false);
                auto false_control = transform_block(if_inst->false_block(), if_inst->merge_block(), false);
                auto merge_control = transform_block(if_inst->merge_block(), follow_block, true_control || false_control);
                return true_control || false_control || merge_control;
            }
            case DerivedInstructionTag::SWITCH: {
                auto switch_inst = static_cast<SwitchInst *>(terminator);
                auto branch_control = false;
                for (auto i = 0u; i < switch_inst->case_count(); i++) {
                    branch_control = transform_block(switch_inst->case_block(i), switch_inst->merge_block(), false) || branch_control;
                }
                branch_control = transform_block(switch_inst->default_block(), switch_inst->merge_block(), false) || branch_control;
                auto merge_control = transform_block(switch_inst->merge_block(), follow_block, branch_control);
                return branch_control || merge_control;
            }
            case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto cond_branch = static_cast<ConditionalBranchInst *>(terminator);
                auto true_control = transform_block(cond_branch->true_block(), follow_block, false);
                auto false_control = transform_block(cond_branch->false_block(), follow_block, false);
                return true_control || false_control;
            }
            case DerivedInstructionTag::LOOP: {
                auto loop_inst = static_cast<LoopInst *>(terminator);
                if (loop_inst == _loop) {
                    return false;
                }
                return transform_block(loop_inst->merge_block(), follow_block, false);
            }
            case DerivedInstructionTag::SIMPLE_LOOP: {
                auto simple_loop = static_cast<SimpleLoopInst *>(terminator);
                if (simple_loop == _loop) {
                    return false;
                }
                return transform_block(simple_loop->merge_block(), follow_block, false);
            }
            default: return false;
        }
    }

    void lower_in_loop(LoopInst *loop) const noexcept {
        auto body_block = loop->body_block();
        auto update_block = loop->update_block();
        auto merge_block = loop->merge_block();
        if (body_block == nullptr || update_block == nullptr || merge_block == nullptr) {
            return;
        }
        reset_flag_at_entry(body_block);
        auto lowered = transform_block(body_block, update_block, false);
        if (_kind == LoopControlKind::BREAK && lowered) {
            static_cast<void>(guard_block_entry_with_flag(_function, _flag, update_block, merge_block));
        }
    }

    void lower_in_simple_loop(SimpleLoopInst *loop) const noexcept {
        auto body_block = loop->body_block();
        auto merge_block = loop->merge_block();
        if (body_block == nullptr || merge_block == nullptr) {
            return;
        }
        auto body_merge = body_block->terminator()->control_flow_merge();
        LUISA_DEBUG_ASSERT(body_merge != nullptr && body_merge->merge_block() != nullptr,
                           "The simple loop body must end with a merged control-flow instruction.");
        auto continue_block = body_merge->merge_block();
        reset_flag_at_entry(body_block);
        auto lowered = transform_block(body_block, continue_block, false);
        if (_kind == LoopControlKind::BREAK && lowered) {
            static_cast<void>(guard_block_entry_with_flag(_function, _flag, continue_block, merge_block));
        }
    }

public:
    explicit LoopControlLowering(Function *function, LoopControlKind kind) noexcept
        : _function{function}, _kind{kind} {}

    void run(Instruction *loop, AllocaInst *flag) noexcept {
        if (loop == nullptr || flag == nullptr) {
            return;
        }
        _loop = loop;
        _flag = flag;
        switch (loop->derived_instruction_tag()) {
            case DerivedInstructionTag::LOOP:
                lower_in_loop(static_cast<LoopInst *>(loop));
                break;
            case DerivedInstructionTag::SIMPLE_LOOP:
                lower_in_simple_loop(static_cast<SimpleLoopInst *>(loop));
                break;
            default: break;
        }
    }
};

static void lower_break_continue_in_function(Function *function) noexcept {
    auto definition = function->definition();
    if (definition == nullptr) {
        return;
    }
    auto flags = LoopControlFlagPreprocess{function, definition}.run();
    if (flags.empty()) {
        return;
    }
    StructuredLoopCollector collector;
    collector.collect_block(definition->body_block());
    auto break_lowering = LoopControlLowering{function, LoopControlKind::BREAK};
    auto continue_lowering = LoopControlLowering{function, LoopControlKind::CONTINUE};
    for (auto loop : collector.loops) {
        if (auto iter = flags.find(loop); iter != flags.end()) {
            break_lowering.run(loop, iter->second.break_flag);
            continue_lowering.run(loop, iter->second.continue_flag);
        }
    }
}

[[nodiscard]] static ReturnInst *find_final_return_instruction(FunctionDefinition *definition) noexcept {
    if (definition == nullptr) {
        return nullptr;
    }
    for (auto block = definition->body_block(); block != nullptr;) {
        auto terminator = block->terminator();
        if (terminator->isa<ReturnInst>()) {
            return static_cast<ReturnInst *>(terminator);
        }
        auto merge = terminator->control_flow_merge();
        if (merge == nullptr || merge->merge_block() == nullptr) {
            return nullptr;
        }
        block = merge->merge_block();
    }
    return nullptr;
}

class EarlyReturnLowering {

private:
    Function *_function;
    FunctionDefinition *_definition;
    ReturnInst *_final_return{nullptr};
    AllocaInst *_return_flag{nullptr};
    AllocaInst *_return_value_slot{nullptr};
    BasicBlock *_common_return_block{nullptr};

private:
    [[nodiscard]] auto _module() const noexcept { return _function->parent_module(); }

    [[nodiscard]] bool _has_early_return() const noexcept {
        auto found = false;
        _definition->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ReturnInst>() && inst != _final_return) {
                found = true;
            }
        });
        return found;
    }

    [[nodiscard]] BasicBlock *guard_block_entry(BasicBlock *block, BasicBlock *skip_target) noexcept {
        return guard_block_entry_with_flag(_function, _return_flag, block, skip_target);
    }

    void rewrite_return(ReturnInst *return_inst, BasicBlock *target, bool is_early_return) noexcept {
        LUISA_DEBUG_ASSERT(return_inst != nullptr, "The return instruction to rewrite must not be null.");
        LUISA_DEBUG_ASSERT(target != nullptr, "The rewritten return target must not be null.");
        XIRBuilder builder;
        builder.set_insertion_point(return_inst->prev());
        if (_return_value_slot != nullptr) {
            auto return_value = return_inst->return_value();
            LUISA_DEBUG_ASSERT(return_value != nullptr, "Non-void returns must carry a return value.");
            builder.store(_return_value_slot, return_value);
        }
        if (is_early_return) {
            auto const_true = _module()->create_constant_one(Type::of<bool>());
            builder.store(_return_flag, const_true);
        }
        auto branch = builder.br(target);
        auto owned_branch = branch->remove_self();
        return_inst->replace_self_with(std::move(owned_branch));
    }

    [[nodiscard]] bool transform_block(BasicBlock *block,
                                       BasicBlock *follow_block,
                                       bool guard_entry) noexcept {
        LUISA_DEBUG_ASSERT(block != nullptr, "The block to transform must not be null.");
        LUISA_DEBUG_ASSERT(follow_block != nullptr, "The block follow target must not be null.");
        if (guard_entry) {
            block = guard_block_entry(block, follow_block);
        }
        auto terminator = block->terminator();
        switch (terminator->derived_instruction_tag()) {
            case DerivedInstructionTag::RETURN: {
                auto return_inst = static_cast<ReturnInst *>(terminator);
                if (return_inst == _final_return) {
                    rewrite_return(return_inst, _common_return_block, false);
                    return false;
                }
                rewrite_return(return_inst, follow_block, true);
                return true;
            }
            case DerivedInstructionTag::IF: {
                auto if_inst = static_cast<IfInst *>(terminator);
                auto true_early = transform_block(if_inst->true_block(), if_inst->merge_block(), false);
                auto false_early = transform_block(if_inst->false_block(), if_inst->merge_block(), false);
                auto merge_early = transform_block(if_inst->merge_block(), follow_block, true_early || false_early);
                return true_early || false_early || merge_early;
            }
            case DerivedInstructionTag::SWITCH: {
                auto switch_inst = static_cast<SwitchInst *>(terminator);
                auto branch_early = false;
                for (auto i = 0u; i < switch_inst->case_count(); i++) {
                    branch_early = transform_block(switch_inst->case_block(i), switch_inst->merge_block(), false) || branch_early;
                }
                branch_early = transform_block(switch_inst->default_block(), switch_inst->merge_block(), false) || branch_early;
                auto merge_early = transform_block(switch_inst->merge_block(), follow_block, branch_early);
                return branch_early || merge_early;
            }
            case DerivedInstructionTag::LOOP: {
                auto loop_inst = static_cast<LoopInst *>(terminator);
                auto prepare_early = transform_block(loop_inst->prepare_block(), loop_inst->merge_block(), false);
                auto body_early = transform_block(loop_inst->body_block(), loop_inst->merge_block(), false);
                auto update_early = transform_block(loop_inst->update_block(), loop_inst->merge_block(), false);
                auto merge_early = transform_block(loop_inst->merge_block(), follow_block, prepare_early || body_early || update_early);
                return prepare_early || body_early || update_early || merge_early;
            }
            case DerivedInstructionTag::SIMPLE_LOOP: {
                auto simple_loop = static_cast<SimpleLoopInst *>(terminator);
                auto body_early = transform_block(simple_loop->body_block(), simple_loop->merge_block(), false);
                auto merge_early = transform_block(simple_loop->merge_block(), follow_block, body_early);
                return body_early || merge_early;
            }
            default: return false;
        }
    }

    void initialize_state() noexcept {
        XIRBuilder builder;
        builder.set_insertion_point(_definition->body_block()->instructions().head_sentinel());
        _return_flag = builder.alloca_local(Type::of<bool>());
        _return_flag->add_comment("early return flag");
        auto return_type = _final_return->return_type();
        if (return_type != nullptr) {
            _return_value_slot = builder.alloca_local(return_type);
            _return_value_slot->add_comment("early return value");
        }
        auto const_false = _module()->create_constant_zero(Type::of<bool>());
        builder.store(_return_flag, const_false);

        _common_return_block = _function->create_basic_block();
        builder.set_insertion_point(_common_return_block->instructions().head_sentinel());
        if (return_type != nullptr) {
            auto return_value = builder.load(return_type, _return_value_slot);
            builder.return_(return_value);
        } else {
            builder.return_void();
        }
    }

public:
    explicit EarlyReturnLowering(Function *function) noexcept
        : _function{function}, _definition{function == nullptr ? nullptr : function->definition()} {}

    void run() noexcept {
        if (_definition == nullptr) {
            return;
        }
        _final_return = find_final_return_instruction(_definition);
        if (_final_return == nullptr || !_has_early_return()) {
            return;
        }
        initialize_state();
        static_cast<void>(transform_block(_definition->body_block(), _common_return_block, false));
    }
};

static void lower_early_returns_in_function(Function *function) noexcept {
    EarlyReturnLowering{function}.run();
}

static Canonicalize_Control_Flow_Info run_on_function(Function *function) noexcept {
    Canonicalize_Control_Flow_Info info;
    if (auto definition = function->definition()) {
        LoopCollector collector;
        collector.collect_block(definition->body_block());
        for (auto loop : collector.loops) {
            lower_loop_to_simple_loop(loop, info);
        }
        lower_break_continue_in_function(function);
        lower_early_returns_in_function(function);
    }
    return info;
}

}// namespace detail

Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Function(Function *func) {
    reg2mem_pass_run_on_function(func);
    return detail::run_on_function(func);
}

Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Module(Module *module) {
    reg2mem_pass_run_on_module(module);
    Canonicalize_Control_Flow_Info info;
    for (auto func : module->function_list()) {
        auto function_info = detail::run_on_function(func);
        info.lowered_loop_count += function_info.lowered_loop_count;
        info.skipped_loop_count += function_info.skipped_loop_count;
    }
    return info;
}

}// namespace luisa::compute::xir
