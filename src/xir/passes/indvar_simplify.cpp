#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/gep.h>

#include "helpers.h"
#include "natural_loop.h"

#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] bool is_iv_increment(Instruction *inst, PhiInst *phi) noexcept {
    if (!inst->isa<ArithmeticInst>()) { return false; }
    auto *arith = static_cast<ArithmeticInst *>(inst);
    if (arith->op() != ArithmeticOp::BINARY_ADD) { return false; }
    for (size_t i = 0; i < arith->operand_count(); ++i) {
        if (arith->operand(i) == phi) { return true; }
    }
    return false;
}

struct IVInfo {
    PhiInst *phi;
    Instruction *increment;
    Value *stride_val;
    bool has_constant_stride;
    int64_t stride_const;
};

[[nodiscard]] bool try_build_iv_info(PhiInst *phi, LoopInst *loop, IVInfo &out) noexcept {
    auto *update = loop->update_block();
    if (!update) { return false; }

    Value *start_val = nullptr;
    Value *recur_val = nullptr;
    for (size_t i = 0; i < phi->incoming_count(); ++i) {
        auto inc = phi->incoming(i);
        if (inc.block == update) {
            recur_val = inc.value;
        } else {
            start_val = inc.value;
        }
    }
    if (!start_val || !recur_val) { return false; }
    if (!recur_val->isa<Instruction>()) { return false; }
    auto *recur_inst = static_cast<Instruction *>(recur_val);
    if (!is_iv_increment(recur_inst, phi)) { return false; }

    auto *arith = static_cast<ArithmeticInst *>(recur_inst);
    Value *stride_val = nullptr;
    for (size_t i = 0; i < arith->operand_count(); ++i) {
        if (arith->operand(i) != phi) { stride_val = arith->operand(i); }
    }
    if (!stride_val) { return false; }

    bool has_const_stride = false;
    int64_t stride_const = 0;
    if (stride_val->isa<Constant>()) {
        auto *c = static_cast<Constant *>(stride_val);
        auto *ty = c->type();
        if (ty->is_int32()) {
            has_const_stride = true;
            stride_const = static_cast<int64_t>(c->as<int32_t>());
        } else if (ty->is_uint32()) {
            has_const_stride = true;
            stride_const = static_cast<int64_t>(c->as<uint32_t>());
        } else if (ty->is_int64()) {
            has_const_stride = true;
            stride_const = c->as<int64_t>();
        }
    }

    out.phi = phi;
    out.increment = recur_inst;
    out.stride_val = stride_val;
    out.has_constant_stride = has_const_stride;
    out.stride_const = stride_const;
    return true;
}

void remove_dead_iv(IVInfo &iv, IndVarSimplifyInfo &info) noexcept {
    for (auto *use : iv.phi->use_list()) {
        auto *user = use->user();
        if (user == iv.increment) { continue; }
        if (user == iv.phi) { continue; }
        return;
    }
    // The increment is also a value. If anything other than the recurrence
    // phi observes it, deleting the apparent phi/increment cycle would leave
    // that user dangling.
    for (auto *use : iv.increment->use_list()) {
        if (use->user() != iv.phi) { return; }
    }
    // Break the recurrence through the public phi API before deleting either
    // instruction so both use lists stay valid throughout the mutation.
    for (size_t i = iv.phi->incoming_count(); i-- > 0u;) {
        if (iv.phi->incoming(i).value == iv.increment) {
            iv.phi->remove_incoming(i);
        }
    }
    iv.increment->remove_self();
    iv.phi->remove_self();
    info.removed_dead_iv_count++;
}

void process_loop(LoopInst *loop, IndVarSimplifyInfo &info) noexcept {
    auto *prepare = loop->prepare_block();
    auto *update = loop->update_block();
    if (!prepare || !update) { return; }

    auto *prep_term = prepare->terminator();
    if (!prep_term || !prep_term->isa<ConditionalBranchInst>()) { return; }

    luisa::vector<IVInfo> ivs;
    for (auto *inst : prepare->instructions()) {
        if (inst->is_terminator()) { break; }
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        IVInfo iv;
        if (try_build_iv_info(phi, loop, iv)) {
            ivs.push_back(iv);
        }
    }

    for (auto &iv : ivs) {
        remove_dead_iv(iv, info);
    }
}

void indvar_simplify_on_function_def(FunctionDefinition *def, IndVarSimplifyInfo &info) noexcept {
    static_cast<void>(scev_pass_run_on_function(def));

    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>()) {
            loops.push_back(static_cast<LoopInst *>(inst));
        }
    });

    for (auto *loop : loops) {
        process_loop(loop, info);
    }
}

namespace {

// A scaled-IV expression eligible for strength reduction: either mul(iv, C)
// or add(mul(iv, C), base) with a loop-invariant base. The accumulator
// represents scale * iv (+ base) and is advanced by stride * scale each
// iteration, eliminating the per-iteration multiply.
struct ScaledIVCandidate {
    ArithmeticInst *root;// the instruction replaced by the accumulator
    ArithmeticInst *mul; // the inner multiply
    Value *base;         // loop-invariant addend, nullptr for the bare multiply
    int64_t scale;
};

[[nodiscard]] bool decode_constant_int64(const Value *value, int64_t &out) noexcept {
    if (value == nullptr || !value->isa<Constant>()) { return false; }
    auto *constant = static_cast<const Constant *>(value);
    auto *type = constant->type();
    if (type == nullptr) { return false; }
    if (type->is_int8()) {
        out = constant->as<int8_t>();
    } else if (type->is_uint8()) {
        out = constant->as<uint8_t>();
    } else if (type->is_int16()) {
        out = constant->as<int16_t>();
    } else if (type->is_uint16()) {
        out = constant->as<uint16_t>();
    } else if (type->is_int32()) {
        out = constant->as<int32_t>();
    } else if (type->is_uint32()) {
        out = constant->as<uint32_t>();
    } else if (type->is_int64()) {
        out = constant->as<int64_t>();
    } else if (type->is_uint64()) {
        auto unsigned_value = constant->as<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return false; }
        out = static_cast<int64_t>(unsigned_value);
    } else {
        return false;
    }
    return true;
}

[[nodiscard]] bool match_scaled_iv(ArithmeticInst *inst, PhiInst *iv,
                                   const NaturalLoop &loop,
                                   int64_t &scale_out) noexcept {
    if (inst->operand_count() != 2u) { return false; }
    auto op = inst->op();
    if (op == ArithmeticOp::BINARY_MUL) {
        for (auto i = 0u; i < 2u; ++i) {
            if (inst->operand(i) == iv) {
                if (decode_constant_int64(inst->operand(1u - i), scale_out)) {
                    return true;
                }
            }
        }
        return false;
    }
    if (op == ArithmeticOp::BINARY_SHIFT_LEFT) {
        if (inst->operand(0u) == iv) {
            int64_t shift = 0;
            if (decode_constant_int64(inst->operand(1u), shift) &&
                shift >= 0 && shift < 62) {
                scale_out = int64_t{1} << shift;
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] bool is_loop_invariant_value(const Value *value, const NaturalLoop &loop) noexcept {
    if (value == nullptr) { return false; }
    if (!value->isa<Instruction>()) { return true; }
    auto *block = static_cast<const Instruction *>(value)->parent_block();
    return block == nullptr || !loop.contains(const_cast<BasicBlock *>(block));
}

void strength_reduce_loop_ivs(FunctionDefinition *def, const NaturalLoop &loop,
                              IndVarSimplifyInfo &info) noexcept {
    auto bounds = analyze_loop_bounds(loop);
    if (!bounds.is_valid() || !bounds.stride_is_constant || bounds.stride == 0) { return; }
    auto *iv = bounds.induction_phi;
    auto *header = loop.header;
    auto *preheader = loop.preheader;
    if (preheader == nullptr || loop.latches.size() != 1u) { return; }
    auto *latch = loop.latches.front();

    // Collect candidates from the whole loop body.
    luisa::vector<ScaledIVCandidate> candidates;
    auto scan_block = [&](BasicBlock *block) noexcept {
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (!inst->isa<ArithmeticInst>()) { return; }
            auto *arith = static_cast<ArithmeticInst *>(inst);
            int64_t scale = 0;
            if (match_scaled_iv(arith, iv, loop, scale)) {
                candidates.emplace_back(ScaledIVCandidate{
                    .root = arith, .mul = arith, .base = nullptr, .scale = scale});
                return;
            }
            if (arith->op() == ArithmeticOp::BINARY_ADD && arith->operand_count() == 2u) {
                for (auto i = 0u; i < 2u; ++i) {
                    auto *maybe_mul = arith->operand(i);
                    auto *base = arith->operand(1u - i);
                    if (maybe_mul != nullptr && maybe_mul->isa<ArithmeticInst>() &&
                        is_loop_invariant_value(base, loop)) {
                        auto *mul = static_cast<ArithmeticInst *>(maybe_mul);
                        if (match_scaled_iv(mul, iv, loop, scale)) {
                            candidates.emplace_back(ScaledIVCandidate{
                                .root = arith, .mul = mul, .base = base, .scale = scale});
                            return;
                        }
                    }
                }
            }
        });
    };
    scan_block(header);
    for (auto *block : loop.body_blocks) { scan_block(block); }
    if (candidates.empty()) { return; }

    // Drop bare-multiply candidates whose every in-loop use is a matched
    // add chain; the chain's accumulator already covers the multiply.
    for (auto i = candidates.size(); i-- > 0u;) {
        auto &candidate = candidates[i];
        if (candidate.base != nullptr) { continue; }
        auto covered_by_chain = false;
        auto has_other_uses = false;
        for (auto *use : candidate.mul->use_list()) {
            auto *user = use->user();
            if (user == nullptr || !user->isa<Instruction>()) { continue; }
            auto is_chain_root = false;
            for (auto &other : candidates) {
                if (other.base != nullptr && other.root == user) {
                    is_chain_root = true;
                    break;
                }
            }
            if (is_chain_root) {
                covered_by_chain = true;
            } else if (user != candidate.mul) {
                has_other_uses = true;
            }
        }
        if (covered_by_chain && !has_other_uses) {
            candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(i));
        }
    }
    if (candidates.empty()) { return; }

    // Do not strength-reduce a value that feeds the induction recurrence.
    auto recurrence = static_cast<Instruction *>(nullptr);
    for (auto i = 0u; i < iv->incoming_count(); ++i) {
        if (iv->incoming(i).block == latch &&
            iv->incoming(i).value->isa<Instruction>()) {
            recurrence = static_cast<Instruction *>(iv->incoming(i).value);
        }
    }

    auto *module = def->parent_module();
    auto *iv_type = iv->type();
    for (auto &candidate : candidates) {
        if (candidate.root == recurrence || candidate.mul == recurrence) { continue; }
        auto feeds_iv = false;
        for (auto *use : candidate.root->use_list()) {
            if (use->user() == iv) { feeds_iv = true; }
        }
        if (feeds_iv) { continue; }
        auto increment = bounds.stride * candidate.scale;
        XIRBuilder builder;
        // Start value in the preheader: scale * iv_start (+ base).
        auto *preheader_terminator = preheader->terminator();
        builder.set_insertion_point(preheader_terminator->prev());
        auto scale_bits = candidate.scale;
        auto *scale_const = module->create_constant(iv_type, &scale_bits);
        auto *start = builder.call(iv_type, ArithmeticOp::BINARY_MUL,
                                   {bounds.start_value, scale_const});
        if (candidate.base != nullptr) {
            start = builder.call(iv_type, ArithmeticOp::BINARY_ADD,
                                 {candidate.base, start});
        }
        // Accumulator phi at the end of the header's phi run.
        Instruction *last_phi = nullptr;
        for (auto *inst : header->instructions()) {
            if (!inst->isa<PhiInst>()) { break; }
            last_phi = inst;
        }
        if (last_phi != nullptr) {
            builder.set_insertion_point(last_phi);
        } else {
            builder.set_insertion_point(header);
        }
        auto *acc = builder.phi(iv_type);
        // Advance in the latch: acc + stride * scale.
        auto *latch_terminator = latch->terminator();
        builder.set_insertion_point(latch_terminator->prev());
        auto increment_bits = increment;
        auto *increment_const = module->create_constant(iv_type, &increment_bits);
        auto *acc_next = builder.call(iv_type, ArithmeticOp::BINARY_ADD,
                                      {acc, increment_const});
        acc->add_incoming(start, preheader);
        acc->add_incoming(acc_next, latch);
        // Replace in-loop uses of the candidate root with the accumulator.
        // Collect first: set_value relinks the use lists being traversed.
        luisa::vector<Use *> in_loop_uses;
        for (auto *use : candidate.root->use_list()) {
            auto *user = use->user();
            if (user != nullptr && user->isa<Instruction>()) {
                auto *user_block = static_cast<Instruction *>(user)->parent_block();
                if (user_block != nullptr && loop.contains(user_block)) {
                    in_loop_uses.emplace_back(use);
                }
            }
        }
        for (auto *use : in_loop_uses) {
            auto owned_use = use->remove_self();
            owned_use->set_value(acc);
            acc->use_list().push_front(std::move(owned_use));
        }
        info.simplified_iv_count++;
    }
}

void strength_reduce_indvars_on_plain_cfg(FunctionDefinition *def,
                                          IndVarSimplifyInfo &info) noexcept {
    if (def == nullptr || contains_structured_control_flow(def)) { return; }
    auto dom_tree = compute_dom_tree(def);
    auto loops = discover_natural_loops(def, dom_tree);
    for (auto &loop : loops) {
        strength_reduce_loop_ivs(def, loop, info);
    }
}

}// namespace

}// namespace

IndVarSimplifyInfo indvar_simplify_pass_run_on_function(FunctionDefinition *def) noexcept {
    IndVarSimplifyInfo info;
    indvar_simplify_on_function_def(def, info);
    strength_reduce_indvars_on_plain_cfg(def, info);
    return info;
}

IndVarSimplifyInfo indvar_simplify_pass_run_on_module(Module *module, PassReport *report) noexcept {
    IndVarSimplifyInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            indvar_simplify_on_function_def(def, info);
            strength_reduce_indvars_on_plain_cfg(def, info);
        }
    }
    if (report != nullptr) {
        report->set("simplified_iv", info.simplified_iv_count);
        report->set("removed_dead_iv", info.removed_dead_iv_count);
    }
    return info;
}

}// namespace luisa::compute::xir
