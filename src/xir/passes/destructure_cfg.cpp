#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static DestructureCFGInfo preflight_destructure_input(Function *function) noexcept {
    DestructureCFGInfo info;
    if (function == nullptr) { return info; }
    auto *def = function->definition();
    if (def == nullptr) { return info; }
    auto valid_block = [&](BasicBlock *block) noexcept {
        return block != nullptr && block->parent_function() == function;
    };
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr) { continue; }
        if (!block->is_terminated()) {
            info.leaked_block_count++;
            continue;
        }
        auto *term = block->terminator();
        switch (term->derived_instruction_tag()) {
            case DerivedInstructionTag::IF: {
                auto *if_inst = static_cast<IfInst *>(term);
                auto malformed = if_inst->condition() == nullptr ||
                                 if_inst->condition()->type() != Type::of<bool>() ||
                                 !valid_block(if_inst->true_block()) ||
                                 !valid_block(if_inst->false_block()) ||
                                 (if_inst->merge_block() != nullptr &&
                                  !valid_block(if_inst->merge_block()));
                info.error_count += malformed ? 1u : 0u;
                break;
            }
            case DerivedInstructionTag::LOOP: {
                auto *loop = static_cast<LoopInst *>(term);
                auto malformed = !valid_block(loop->prepare_block()) ||
                                 !valid_block(loop->body_block()) ||
                                 !valid_block(loop->update_block()) ||
                                 !valid_block(loop->merge_block());
                info.error_count += malformed ? 1u : 0u;
                break;
            }
            case DerivedInstructionTag::SIMPLE_LOOP: {
                auto *loop = static_cast<SimpleLoopInst *>(term);
                auto malformed = !valid_block(loop->body_block()) ||
                                 !valid_block(loop->merge_block());
                info.error_count += malformed ? 1u : 0u;
                break;
            }
            case DerivedInstructionTag::BREAK: {
                info.error_count += !valid_block(static_cast<BreakInst *>(term)->target_block()) ? 1u : 0u;
                break;
            }
            case DerivedInstructionTag::CONTINUE: {
                info.error_count += !valid_block(static_cast<ContinueInst *>(term)->target_block()) ? 1u : 0u;
                break;
            }
            case DerivedInstructionTag::RAY_QUERY_LOOP: info.error_count += 1u; break;
            case DerivedInstructionTag::RETURN: {
                auto *return_inst = static_cast<ReturnInst *>(term);
                auto *return_type = function->type();
                auto *value = return_inst->return_value();
                auto malformed = return_type == nullptr ?
                                     value != nullptr :
                                     value == nullptr || value->type() != return_type;
                info.error_count += malformed ? 1u : 0u;
                break;
            }
            case DerivedInstructionTag::BRANCH:
            case DerivedInstructionTag::CONDITIONAL_BRANCH:
            case DerivedInstructionTag::SWITCH:
            case DerivedInstructionTag::AUTODIFF_SCOPE:
            case DerivedInstructionTag::UNREACHABLE:
            case DerivedInstructionTag::RASTER_DISCARD:
            case DerivedInstructionTag::RAY_QUERY_DISPATCH:
            case DerivedInstructionTag::CORO_SUSPEND:
            case DerivedInstructionTag::CORO_TERMINATE: break;
            default: info.error_count += 1u; break;
        }
    }
    return info;
}

static void terminate_leaked_blocks(Function *function, DestructureCFGInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr) { return; }
    luisa::vector<BasicBlock *> leaked;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr) { continue; }
        if (!block->is_terminated()) { leaked.emplace_back(block); }
    }
    if (leaked.empty()) { return; }
    XIRBuilder b;
    for (auto block : leaked) {
        b.set_insertion_point(block);
        b.unreachable_("destructure_cfg: unterminated block patched with unreachable");
        info.leaked_block_count += 1;
    }
}

static void spill_early_returns(Function *function, DestructureCFGInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr) { return; }
    auto body = def->body_block();
    if (body == nullptr) { return; }
    luisa::vector<ReturnInst *> returns;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || !block->is_terminated()) { continue; }
        auto term = block->terminator();
        if (term != nullptr && term->isa<ReturnInst>()) {
            returns.emplace_back(static_cast<ReturnInst *>(term));
        }
    }
    if (returns.size() <= 1) { return; }
    auto ret_type = function->type();
    XIRBuilder b;
    AllocaInst *spill_slot = nullptr;
    if (ret_type != nullptr) {
        b.set_insertion_point(body->instructions().head_sentinel());
        spill_slot = b.alloca_local(ret_type);
        spill_slot->add_comment("early-return spill slot");
    }
    auto exit_block = def->create_basic_block();
    b.set_insertion_point(exit_block);
    if (ret_type == nullptr) {
        b.return_void();
    } else {
        auto loaded = b.load(ret_type, spill_slot);
        b.return_(loaded);
    }
    for (auto r : returns) {
        if (r == nullptr) { continue; }
        auto parent = r->parent_block();
        if (parent == nullptr) {
            LUISA_WARNING_WITH_LOCATION("spill_early_returns: ReturnInst with null parent block.");
            continue;
        }
        auto value = r->return_value();
        if (ret_type != nullptr && value == nullptr) {
            LUISA_WARNING_WITH_LOCATION("spill_early_returns: non-void function has ReturnInst with null value.");
        }
        b.set_insertion_point(parent);
        if (ret_type != nullptr && value != nullptr) { b.store(spill_slot, value); }
        b.br(exit_block);
        r->remove_self();
        info.destructured_early_return_count += 1;
    }
}

static size_t verify_terminators(Function *function) noexcept {
    if (function == nullptr) { return 0u; }
    auto def = function->definition();
    if (def == nullptr) { return 0u; }
    size_t return_count = 0;
    size_t errors = 0u;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr) { continue; }
        if (!block->is_terminated()) {
            LUISA_WARNING_WITH_LOCATION(
                "destructure_cfg: unterminated basic block survived destructuring "
                "(function={}, block={}).",
                static_cast<void *>(function), static_cast<void *>(block));
            ++errors;
            continue;
        }
        auto term = block->terminator();
        if (term == nullptr) { continue; }
        switch (term->derived_instruction_tag()) {
            case DerivedInstructionTag::BRANCH:
            case DerivedInstructionTag::CONDITIONAL_BRANCH:
            case DerivedInstructionTag::SWITCH:
            case DerivedInstructionTag::AUTODIFF_SCOPE:
            case DerivedInstructionTag::UNREACHABLE:
            case DerivedInstructionTag::RASTER_DISCARD:
            case DerivedInstructionTag::RAY_QUERY_DISPATCH:
            case DerivedInstructionTag::CORO_SUSPEND:
            case DerivedInstructionTag::CORO_TERMINATE:
                break;
            case DerivedInstructionTag::RETURN:
                return_count += 1;
                break;
            default:
                LUISA_WARNING_WITH_LOCATION(
                    "destructure_cfg: unexpected terminator tag {} survived destructuring.",
                    static_cast<int>(term->derived_instruction_tag()));
                ++errors;
                break;
        }
    }
    if (return_count > 1) {
        LUISA_WARNING_WITH_LOCATION(
            "destructure_cfg: function still has {} ReturnInsts after early-return spill.",
            return_count);
        ++errors;
    }
    return errors;
}

static void destructure_in_function(Function *function, DestructureCFGInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr) { return; }
    for (;;) {
        luisa::vector<IfInst *> if_insts;
        luisa::vector<LoopInst *> loop_insts;
        luisa::vector<SimpleLoopInst *> simple_loop_insts;
        luisa::vector<BreakInst *> break_insts;
        luisa::vector<ContinueInst *> continue_insts;
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || !block->is_terminated()) { continue; }
            auto term = block->terminator();
            if (term == nullptr) { continue; }
            switch (term->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                    if_insts.emplace_back(static_cast<IfInst *>(term));
                    break;
                case DerivedInstructionTag::LOOP:
                    loop_insts.emplace_back(static_cast<LoopInst *>(term));
                    break;
                case DerivedInstructionTag::SIMPLE_LOOP:
                    simple_loop_insts.emplace_back(static_cast<SimpleLoopInst *>(term));
                    break;
                case DerivedInstructionTag::BREAK:
                    break_insts.emplace_back(static_cast<BreakInst *>(term));
                    break;
                case DerivedInstructionTag::CONTINUE:
                    continue_insts.emplace_back(static_cast<ContinueInst *>(term));
                    break;
                default: break;
            }
        }
        if (if_insts.empty() && loop_insts.empty() && simple_loop_insts.empty() &&
            break_insts.empty() && continue_insts.empty()) {
            break;
        }
        XIRBuilder b;
        auto any_destructured = false;
        for (auto brk : break_insts) {
            if (brk == nullptr) { continue; }
            auto block = brk->parent_block();
            auto target = brk->target_block();
            if (block == nullptr || target == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping break with null parent/target.");
                continue;
            }
            brk->remove_self();
            b.set_insertion_point(block);
            b.br(target);
            ++info.destructured_break_count;
            any_destructured = true;
        }
        for (auto cont : continue_insts) {
            if (cont == nullptr) { continue; }
            auto block = cont->parent_block();
            auto target = cont->target_block();
            if (block == nullptr || target == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping continue with null parent/target.");
                continue;
            }
            cont->remove_self();
            b.set_insertion_point(block);
            b.br(target);
            ++info.destructured_continue_count;
            any_destructured = true;
        }
        for (auto if_inst : if_insts) {
            if (if_inst == nullptr) { continue; }
            auto block = if_inst->parent_block();
            auto cond = if_inst->condition();
            auto true_block = if_inst->true_block();
            auto false_block = if_inst->false_block();
            if (block == nullptr || cond == nullptr || true_block == nullptr || false_block == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping IfInst with null operand.");
                continue;
            }
            if_inst->remove_self();
            b.set_insertion_point(block);
            b.cond_br(cond, true_block, false_block);
            ++info.destructured_if_count;
            any_destructured = true;
        }
        for (auto loop_inst : loop_insts) {
            if (loop_inst == nullptr) { continue; }
            auto block = loop_inst->parent_block();
            auto prepare = loop_inst->prepare_block();
            if (block == nullptr || prepare == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping LoopInst with null parent/prepare block.");
                continue;
            }
            loop_inst->remove_self();
            b.set_insertion_point(block);
            b.br(prepare);
            ++info.destructured_loop_count;
            any_destructured = true;
        }
        for (auto sl : simple_loop_insts) {
            if (sl == nullptr) { continue; }
            auto block = sl->parent_block();
            auto body = sl->body_block();
            if (block == nullptr || body == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping SimpleLoopInst with null parent/body block.");
                continue;
            }
            sl->remove_self();
            b.set_insertion_point(block);
            b.br(body);
            ++info.destructured_simple_loop_count;
            any_destructured = true;
        }
        // Malformed structured terminators are left unchanged after emitting a
        // warning. Do not spin forever or claim they were transformed.
        if (!any_destructured) { break; }
    }
    terminate_leaked_blocks(function, info);
    spill_early_returns(function, info);
    info.error_count += verify_terminators(function);
}

}// namespace detail

DestructureCFGInfo destructure_cfg_pass_run_on_function(Function *function) noexcept {
    DestructureCFGInfo info;
    if (function == nullptr) { return info; }
    info.error_count = detail::preflight_destructure_input(function).error_count;
    if (info.error_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "destructure_cfg: rejecting function with {} malformed or unsupported construct(s).",
            info.error_count);
        return info;
    }
    detail::destructure_in_function(function, info);
    return info;
}

DestructureCFGInfo destructure_cfg_pass_run_on_module(Module *module, PassReport *report) noexcept {
    DestructureCFGInfo info;
    if (module == nullptr) { return info; }
    for (auto *f : module->function_list()) {
        info.error_count += detail::preflight_destructure_input(f).error_count;
    }
    if (info.error_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "destructure_cfg: rejecting module with {} malformed or unsupported construct(s).",
            info.error_count);
        if (report != nullptr) {
            report->set("error", info.error_count);
        }
        return info;
    }
    for (auto f : module->function_list()) {
        detail::destructure_in_function(f, info);
    }
    if (report != nullptr) {
        report->set("destructured_if", info.destructured_if_count);
        report->set("destructured_loop", info.destructured_loop_count);
        report->set("destructured_simple_loop", info.destructured_simple_loop_count);
        report->set("destructured_break", info.destructured_break_count);
        report->set("destructured_continue", info.destructured_continue_count);
        report->set("destructured_early_return", info.destructured_early_return_count);
        report->set("leaked_block", info.leaked_block_count);
        report->set("error", info.error_count);
    }
    return info;
}

DestructureCFGInfo destructure_cfg_pass_preflight_function(Function *function) noexcept {
    return detail::preflight_destructure_input(function);
}

}// namespace luisa::compute::xir
