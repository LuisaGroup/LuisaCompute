#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

// Tunable: cap on instructions hoisted per side. Speculation always executes
// both branches, so converting a large body trades a branch for a lot of work
// that may go unused. 16 keeps the diamond cheap relative to typical kernel
// hot spots.
static constexpr size_t kIfConversionInstructionCap = 16u;

[[nodiscard]] static bool is_speculation_safe(Instruction *inst) noexcept {
    auto info = get_memory_info(inst);
    if (!info.is_pure()) return false;
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC: [[fallthrough]];
        case DerivedInstructionTag::CAST: [[fallthrough]];
        case DerivedInstructionTag::GEP:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static size_t count_predecessors(BasicBlock *block) noexcept {
    size_t n = 0;
    block->traverse_predecessors(false, [&](BasicBlock *) noexcept { ++n; });
    return n;
}

[[nodiscard]] static BasicBlock *single_predecessor(BasicBlock *block) noexcept {
    BasicBlock *result = nullptr;
    size_t n = 0;
    block->traverse_predecessors(false, [&](BasicBlock *p) noexcept {
        result = p;
        ++n;
    });
    return n == 1u ? result : nullptr;
}

// Returns the unique merge block reached by `side` if it satisfies the
// pure-side requirements: single predecessor `b`, exclusively
// speculation-safe non-terminator instructions, and a plain `br M`
// terminator. nullptr means this side is not eligible.
[[nodiscard]] static BasicBlock *eligible_side(BasicBlock *side, BasicBlock *b,
                                               size_t &out_inst_count) noexcept {
    if (side == nullptr) return nullptr;
    if (!side->is_terminated()) return nullptr;
    if (single_predecessor(side) != b) return nullptr;
    auto term = side->terminator();
    if (term->derived_instruction_tag() != DerivedInstructionTag::BRANCH) return nullptr;
    auto br = static_cast<BranchInst *>(term);
    auto m = br->target_block();
    if (m == nullptr) return nullptr;
    size_t inst_count = 0;
    for (auto inst : side->instructions()) {
        if (inst == term) continue;
        if (!is_speculation_safe(inst)) return nullptr;
        if (++inst_count > kIfConversionInstructionCap) return nullptr;
    }
    out_inst_count = inst_count;
    return m;
}

// Build a select for each phi node in `merge` whose incomings come from
// `t_block` and `f_block`. Selects are issued via `builder` (which must be
// pointing at the merge-block-bound parent), and the phi incomings from the
// two sides are replaced by a single incoming from `parent`.
static size_t rewrite_phis_in_merge(BasicBlock *merge, BasicBlock *parent,
                                    BasicBlock *t_block, BasicBlock *f_block,
                                    Value *cond, XIRBuilder &builder) noexcept {
    luisa::vector<PhiInst *> phis;
    for (auto inst : merge->instructions()) {
        if (inst->isa<PhiInst>()) phis.push_back(static_cast<PhiInst *>(inst));
    }
    size_t replaced = 0;
    for (auto phi : phis) {
        Value *t_val = nullptr;
        Value *f_val = nullptr;
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block == t_block) t_val = inc.value;
            else if (inc.block == f_block) f_val = inc.value;
        }
        if (t_val == nullptr || f_val == nullptr) continue;
        Value *merged = nullptr;
        if (t_val == f_val) {
            merged = t_val;
        } else {
            // SELECT operand order is (false_val, true_val, cond).
            merged = builder.call(phi->type(), ArithmeticOp::SELECT,
                                  {f_val, t_val, cond});
        }
        for (size_t i = phi->incoming_count(); i-- > 0;) {
            auto blk = phi->incoming(i).block;
            if (blk == t_block || blk == f_block) {
                phi->remove_incoming(i);
            }
        }
        phi->add_incoming(merged, parent);
        ++replaced;
    }
    return replaced;
}

[[nodiscard]] static bool try_convert_diamond(BasicBlock *b, IfConversionInfo &info) noexcept {
    if (!b->is_terminated()) return false;
    auto term = b->terminator();
    // Skip structured terminators; destructure_cfg lowers IF/LOOP into plain
    // CONDITIONAL_BRANCH for this window of the pipeline, and restructure_cfg
    // expects to rebuild structured frames from those plain branches.
    if (term->derived_instruction_tag() != DerivedInstructionTag::CONDITIONAL_BRANCH) return false;
    auto cond_br = static_cast<ConditionalBranchTerminatorInstruction *>(term);
    auto t_block = cond_br->true_block();
    auto f_block = cond_br->false_block();
    if (t_block == nullptr || f_block == nullptr) return false;
    if (t_block == f_block) return false;
    size_t t_count = 0;
    size_t f_count = 0;
    auto t_merge = eligible_side(t_block, b, t_count);
    if (t_merge == nullptr) return false;
    auto f_merge = eligible_side(f_block, b, f_count);
    if (f_merge == nullptr) return false;
    if (t_merge != f_merge) return false;
    auto merge = t_merge;
    auto cond = cond_br->condition();
    // Hoist non-terminator instructions from each side into b before the
    // current cond_br terminator.
    auto hoist = [&](BasicBlock *side) noexcept {
        auto side_term = side->terminator();
        while (!side->instructions().empty()) {
            auto front = side->instructions().front();
            if (front == side_term) break;
            auto m = front->remove_self();
            term->insert_before_self(std::move(m));
            ++info.hoisted_inst_count;
        }
    };
    hoist(t_block);
    hoist(f_block);
    // The cond_br is removed first so we can emit selects + br at the end of
    // the parent block via set_insertion_point(BasicBlock*), which points the
    // builder at the new last instruction (insert-after semantics).
    term->remove_self();
    XIRBuilder builder;
    builder.set_insertion_point(b);
    auto replaced = rewrite_phis_in_merge(merge, b, t_block, f_block, cond, builder);
    info.replaced_phi_count += replaced;
    builder.br(merge);
    // The two sides now have no predecessors and no live uses.
    t_block->terminator()->remove_self();
    t_block->remove_self();
    f_block->terminator()->remove_self();
    f_block->remove_self();
    ++info.converted_diamond_count;
    return true;
}

static void run_if_conversion_on_function(Function *function, IfConversionInfo &info) noexcept {
    if (function == nullptr || !function->is_definition()) return;
    auto def = function->definition();
    if (def == nullptr || def->body_block() == nullptr) return;
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "If conversion rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    // A successful conversion deletes the two side blocks, so any snapshot of
    // the block list would dangle on the next pass. Re-traverse from the top
    // after each conversion until no eligible diamond remains.
    while (true) {
        BasicBlock *converted = nullptr;
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (converted != nullptr) return;
            if (try_convert_diamond(block, info)) converted = block;
        });
        if (converted == nullptr) break;
    }
}

}// namespace detail

IfConversionInfo if_conversion_pass_run_on_function(Function *function) noexcept {
    IfConversionInfo info;
    detail::run_if_conversion_on_function(function, info);
    return info;
}

IfConversionInfo if_conversion_pass_run_on_module(Module *module, PassReport *report) noexcept {
    IfConversionInfo info;
    for (auto f : module->function_list()) {
        detail::run_if_conversion_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("converted_diamonds", info.converted_diamond_count);
        report->set("hoisted_insts", info.hoisted_inst_count);
        report->set("replaced_phis", info.replaced_phi_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
