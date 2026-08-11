#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/phi.h>

#include <algorithm>
#include <limits>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_speculation_safe(Instruction *inst) noexcept {
    auto info = get_memory_info(inst);
    if (!info.is_pure()) return false;
    // Float-to-integer conversion can produce poison for verifier-valid NaN
    // or out-of-range inputs. Every other verifier-valid scalar/vector static
    // cast is total, including the integer-to-float step conversion commonly
    // found in predicated coordinate updates.
    if (inst->isa<CastInst>()) {
        auto *cast = static_cast<CastInst *>(inst);
        if (cast->op() == CastOp::BITWISE_CAST) { return true; }
        if (cast->op() != CastOp::STATIC_CAST) { return false; }
        auto *source = cast->value() == nullptr ?
                           nullptr :
                           cast->value()->type();
        auto *target = cast->type();
        if (source == nullptr || target == nullptr) { return false; }
        auto source_is_float =
            source->is_float_or_float_vector();
        auto target_is_integer =
            target->is_int_or_int_vector() ||
            target->is_uint_or_uint_vector();
        return !(source_is_float && target_is_integer);
    }
    if (!inst->isa<ArithmeticInst>()) { return false; }
    return is_arithmetic_op_safe_to_speculate(
        static_cast<ArithmeticInst *>(inst)->op());
}

[[nodiscard]] static size_t register_units(
    const Type *type) noexcept {
    if (type == nullptr) { return 0u; }
    return std::max<size_t>(
        1u, (type->size() + sizeof(uint32_t) - 1u) /
                sizeof(uint32_t));
}

[[nodiscard]] static size_t arithmetic_latency_cost(
    ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::POW:
        case ArithmeticOp::POW_INT:
            return 16u;
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_INVERSE:
            return 8u;
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::LERP:
        case ArithmeticOp::SMOOTHSTEP:
        case ArithmeticOp::CROSS:
        case ArithmeticOp::DOT:
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
        case ArithmeticOp::FACEFORWARD:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_LINALG_MUL:
            return 4u;
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::STEP:
        case ArithmeticOp::FMA:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_TRANSPOSE:
            return 2u;
        default: return 1u;
    }
}

[[nodiscard]] static size_t instruction_cost(
    const Instruction *instruction) noexcept {
    auto units = register_units(instruction->type());
    for (auto *operand_use : instruction->operand_uses()) {
        auto *operand = operand_use->value();
        units = std::max(
            units, register_units(
                       operand == nullptr ? nullptr : operand->type()));
    }
    auto latency = instruction->isa<ArithmeticInst>() ?
                       arithmetic_latency_cost(
                           static_cast<const ArithmeticInst *>(instruction)
                               ->op()) :
                       1u;
    if (units != 0u &&
        latency > std::numeric_limits<size_t>::max() / units) {
        return std::numeric_limits<size_t>::max();
    }
    return latency * units;
}

[[nodiscard]] static size_t saturating_add(
    size_t lhs, size_t rhs) noexcept {
    auto maximum = std::numeric_limits<size_t>::max();
    return lhs > maximum - rhs ? maximum : lhs + rhs;
}

[[nodiscard]] static bool can_rewrite_phis(
    BasicBlock *merge, BasicBlock *t_block,
    BasicBlock *f_block, size_t &live_out_units) noexcept {
    live_out_units = 0u;
    for (auto *inst : merge->instructions()) {
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        auto true_count = 0u;
        auto false_count = 0u;
        for (auto i = 0u; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.value == nullptr || incoming.value->type() != phi->type()) { return false; }
            true_count += incoming.block == t_block;
            false_count += incoming.block == f_block;
        }
        if (true_count != 1u || false_count != 1u) { return false; }
        Value *true_value = nullptr;
        Value *false_value = nullptr;
        for (auto i = 0u; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == t_block) {
                true_value = incoming.value;
            } else if (incoming.block == f_block) {
                false_value = incoming.value;
            }
        }
        if (true_value != false_value) {
            live_out_units = saturating_add(
                live_out_units, register_units(phi->type()));
        }
    }
    return true;
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
                                               const IfConversionOptions &options,
                                               size_t &out_inst_count,
                                               size_t &out_cost) noexcept {
    if (side == nullptr) return nullptr;
    // The side block itself is deleted. Unlike its instructions, block-local
    // metadata has no unique owner after both arms are merged into `b`.
    if (!side->metadata_list().empty()) return nullptr;
    if (!side->is_terminated()) return nullptr;
    if (single_predecessor(side) != b) return nullptr;
    auto term = side->terminator();
    if (term->derived_instruction_tag() != DerivedInstructionTag::BRANCH) return nullptr;
    // The two arm-exit branches are both deleted, but there is only one
    // replacement branch. Merging metadata lists can also create duplicate
    // metadata kinds, which is verifier-invalid. Retain annotated exits.
    if (!term->metadata_list().empty()) return nullptr;
    auto br = static_cast<BranchInst *>(term);
    auto m = br->target_block();
    if (m == nullptr) return nullptr;
    size_t inst_count = 0;
    size_t cost = 0u;
    for (auto inst : side->instructions()) {
        if (inst == term) continue;
        if (!is_speculation_safe(inst)) return nullptr;
        if (++inst_count > options.max_arm_instruction_count) {
            return nullptr;
        }
        cost = saturating_add(cost, instruction_cost(inst));
    }
    out_inst_count = inst_count;
    out_cost = cost;
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
            if (inc.block == t_block)
                t_val = inc.value;
            else if (inc.block == f_block)
                f_val = inc.value;
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

[[nodiscard]] static bool try_convert_diamond(
    BasicBlock *b, IfConversionInfo &info,
    const IfConversionOptions &options) noexcept {
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
    auto cond = cond_br->condition();
    if (cond == nullptr || cond->type() == nullptr ||
        !cond->type()->is_bool()) {
        return false;
    }
    if (options.candidate_filter != nullptr &&
        !options.candidate_filter(
            static_cast<const ConditionalBranchInst *>(cond_br),
            options.candidate_filter_context)) {
        return false;
    }
    size_t t_count = 0;
    size_t f_count = 0;
    size_t t_cost = 0u;
    size_t f_cost = 0u;
    auto t_merge = eligible_side(
        t_block, b, options, t_count, t_cost);
    if (t_merge == nullptr) return false;
    auto f_merge = eligible_side(
        f_block, b, options, f_count, f_cost);
    if (f_merge == nullptr) return false;
    if (t_merge != f_merge) return false;
    auto merge = t_merge;
    if (saturating_add(t_count, f_count) >
        options.max_total_instruction_count) {
        return false;
    }
    auto live_out_units = size_t{0u};
    if (!can_rewrite_phis(
            merge, t_block, f_block, live_out_units)) {
        return false;
    }
    if (live_out_units > options.max_live_out_register_units) {
        return false;
    }
    auto speculation_cost = saturating_add(
        saturating_add(t_cost, f_cost), live_out_units);
    if (speculation_cost > options.max_speculation_cost) {
        return false;
    }
    auto clone_metadata = [](const MetadataListMixin &source,
                             MetadataListMixin &target) noexcept {
        for (auto *metadata : source.metadata_list()) {
            target.metadata_list().push_front(metadata->clone());
        }
    };
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
    auto removed_parent_term = term->remove_self();
    XIRBuilder builder;
    builder.set_insertion_point(b);
    auto replaced = rewrite_phis_in_merge(merge, b, t_block, f_block, cond, builder);
    info.replaced_phi_count += replaced;
    auto *replacement_branch = builder.br(merge);
    clone_metadata(*removed_parent_term, *replacement_branch);
    // The two sides now have no predecessors and no live uses.
    t_block->terminator()->remove_self();
    t_block->remove_self();
    f_block->terminator()->remove_self();
    f_block->remove_self();
    ++info.converted_diamond_count;
    return true;
}

static void run_if_conversion_on_function(
    Function *function, IfConversionInfo &info,
    const IfConversionOptions &options) noexcept {
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
            if (try_convert_diamond(block, info, options)) {
                converted = block;
            }
        });
        if (converted == nullptr) break;
    }
}

}// namespace detail

IfConversionInfo if_conversion_pass_run_on_function(Function *function) noexcept {
    return if_conversion_pass_run_on_function(function, {});
}

IfConversionInfo if_conversion_pass_run_on_function(
    Function *function, IfConversionOptions options) noexcept {
    IfConversionInfo info;
    detail::run_if_conversion_on_function(function, info, options);
    return info;
}

IfConversionInfo if_conversion_pass_run_on_module(Module *module, PassReport *report) noexcept {
    return if_conversion_pass_run_on_module(module, {}, report);
}

IfConversionInfo if_conversion_pass_run_on_module(
    Module *module, IfConversionOptions options,
    PassReport *report) noexcept {
    IfConversionInfo info;
    auto set_report = [&]() noexcept {
        if (report == nullptr) { return; }
        report->set("converted_diamonds", info.converted_diamond_count);
        report->set("hoisted_insts", info.hoisted_inst_count);
        report->set("replaced_phis", info.replaced_phi_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    };
    if (module == nullptr) {
        set_report();
        return info;
    }
    for (auto *function : module->function_list()) {
        if (auto *def = function->definition();
            def != nullptr && contains_structured_control_flow(def)) {
            ++info.structured_cfg_error_count;
        }
    }
    if (info.structured_cfg_error_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "If conversion rejected module containing {} structured "
            "function(s); IR was left unchanged.",
            info.structured_cfg_error_count);
        set_report();
        return info;
    }
    for (auto *function : module->function_list()) {
        detail::run_if_conversion_on_function(
            function, info, options);
    }
    set_report();
    return info;
}

}// namespace luisa::compute::xir
