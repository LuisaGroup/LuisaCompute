#include <luisa/core/stl/hash.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

struct GVNLeader {
    Instruction *inst;
    BasicBlock *block;
    uint64_t vn;
};

struct GVNState;

[[nodiscard]] static bool is_commutative_arithmetic(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
            return true;
        default: return false;
    }
}

[[nodiscard]] static uint64_t hash_type(const Type *type, uint64_t seed) noexcept {
    if (type == nullptr) return seed;
    auto th = type->hash();
    return luisa::hash64(&th, sizeof(th), seed);
}

[[nodiscard]] static uint64_t hash_operand_vns(luisa::span<const uint64_t> vns, bool commutative, uint64_t seed) noexcept {
    if (commutative && vns.size() == 2) [[unlikely]] {
        auto a = vns[0];
        auto b = vns[1];
        if (a > b) std::swap(a, b);
        uint64_t pair[2] = {a, b};
        return luisa::hash64(pair, sizeof(pair), seed);
    }
    return luisa::hash64(vns.data(), vns.size() * sizeof(uint64_t), seed);
}

[[nodiscard]] static bool is_structurally_equal(Instruction *a, Instruction *b, GVNState &state) noexcept;

struct GVNState {
    luisa::unordered_map<Value *, uint64_t> value_to_vn;
    luisa::unordered_map<uint64_t, luisa::vector<GVNLeader>> hash_to_leaders;
    uint64_t next_vn = 1;
    const DomTree *dom_tree = nullptr;

    [[nodiscard]] uint64_t get_vn(Value *value) noexcept {
        if (value == nullptr) return 0;
        auto it = value_to_vn.find(value);
        if (it != value_to_vn.end()) return it->second;
        auto vn = next_vn++;
        value_to_vn.emplace(value, vn);
        return vn;
    }

    [[nodiscard]] luisa::fixed_vector<uint64_t, 8> get_operand_vns(Instruction *inst) noexcept {
        luisa::fixed_vector<uint64_t, 8> vns;
        vns.reserve(inst->operand_count());
        for (size_t i = 0; i < inst->operand_count(); ++i) {
            vns.push_back(get_vn(inst->operand(i)));
        }
        return vns;
    }

    [[nodiscard]] Instruction *find_leader(uint64_t hash, BasicBlock *block, Instruction *inst) noexcept {
        auto it = hash_to_leaders.find(hash);
        if (it == hash_to_leaders.end()) return nullptr;
        for (auto &leader : it->second) {
            if (dom_tree->dominates(leader.block, block) && is_structurally_equal(leader.inst, inst, *this)) {
                return leader.inst;
            }
        }
        return nullptr;
    }

    void record_leader(uint64_t hash, uint64_t vn, Instruction *inst, BasicBlock *block) noexcept {
        hash_to_leaders[hash].push_back({inst, block, vn});
    }
};

[[nodiscard]] static bool can_value_number(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::RESOURCE_QUERY:
            return true;
        case DerivedInstructionTag::CALL: {
            // Only value-number calls that are guaranteed pure.
            // Without function attribute analysis, conservatively
            // skip all calls to avoid unsound CSE of impure calls.
            return false;
        }
        // RAY_QUERY_OBJECT_READ reads mutable per-thread state that changes
        // after PROCEED/COMMIT/TERMINATE — not safe to value-number.
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: [[fallthrough]];
        // RESOURCE_READ is disabled: without memory dependency analysis,
        // an intervening write could make two reads non-equivalent.
        case DerivedInstructionTag::RESOURCE_READ: [[fallthrough]];
        case DerivedInstructionTag::LOAD: [[fallthrough]];
        default: return false;
    }
}

[[nodiscard]] static bool is_structurally_equal(Instruction *a, Instruction *b, GVNState &state) noexcept {
    if (a == b) return true;
    if (a->derived_instruction_tag() != b->derived_instruction_tag()) return false;
    if (a->type() != b->type()) return false;
    if (a->operand_count() != b->operand_count()) return false;
    switch (a->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC: {
            auto ari_a = static_cast<ArithmeticInst *>(a);
            auto ari_b = static_cast<ArithmeticInst *>(b);
            if (ari_a->op() != ari_b->op()) return false;
            if (is_commutative_arithmetic(ari_a->op()) && a->operand_count() == 2) {
                auto a0 = state.get_vn(a->operand(0));
                auto a1 = state.get_vn(a->operand(1));
                auto b0 = state.get_vn(b->operand(0));
                auto b1 = state.get_vn(b->operand(1));
                return (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0);
            }
            break;
        }
        case DerivedInstructionTag::CAST: {
            auto cast_a = static_cast<CastInst *>(a);
            auto cast_b = static_cast<CastInst *>(b);
            if (cast_a->op() != cast_b->op()) return false;
            break;
        }
        case DerivedInstructionTag::CALL: {
            auto call_a = static_cast<CallInst *>(a);
            auto call_b = static_cast<CallInst *>(b);
            if (call_a->callee() != call_b->callee()) return false;
            break;
        }
        case DerivedInstructionTag::RESOURCE_QUERY: {
            auto rq_a = static_cast<ResourceQueryInst *>(a);
            auto rq_b = static_cast<ResourceQueryInst *>(b);
            if (rq_a->op() != rq_b->op()) return false;
            break;
        }
        case DerivedInstructionTag::RESOURCE_READ: {
            auto rr_a = static_cast<ResourceReadInst *>(a);
            auto rr_b = static_cast<ResourceReadInst *>(b);
            if (rr_a->op() != rr_b->op()) return false;
            break;
        }
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
            auto rq_a = static_cast<RayQueryObjectReadInst *>(a);
            auto rq_b = static_cast<RayQueryObjectReadInst *>(b);
            if (rq_a->op() != rq_b->op()) return false;
            break;
        }
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::LOAD:
            break;
        default:
            return false;
    }
    for (size_t i = 0; i < a->operand_count(); ++i) {
        if (state.get_vn(a->operand(i)) != state.get_vn(b->operand(i))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static uint64_t compute_instruction_hash(Instruction *inst, GVNState &state) noexcept {
    auto tag = inst->derived_instruction_tag();
    uint64_t h = luisa::hash64(&tag, sizeof(tag), hash64_default_seed);
    h = hash_type(inst->type(), h);
    switch (tag) {
        case DerivedInstructionTag::ARITHMETIC: {
            auto ari = static_cast<ArithmeticInst *>(inst);
            auto op = ari->op();
            h = luisa::hash64(&op, sizeof(op), h);
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, is_commutative_arithmetic(op), h);
            break;
        }
        case DerivedInstructionTag::CAST: {
            auto cast = static_cast<CastInst *>(inst);
            auto op = cast->op();
            h = luisa::hash64(&op, sizeof(op), h);
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        case DerivedInstructionTag::GEP: {
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        case DerivedInstructionTag::LOAD: {
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        case DerivedInstructionTag::CALL: {
            auto call = static_cast<CallInst *>(inst);
            auto callee = call->callee();
            // TODO add by maxwell: is this safe?
            h = luisa::hash64(&callee, sizeof(callee), h);
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        case DerivedInstructionTag::RESOURCE_QUERY: {
            auto rq = static_cast<ResourceQueryInst *>(inst);
            auto op = rq->op();
            h = luisa::hash64(&op, sizeof(op), h);
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        case DerivedInstructionTag::RESOURCE_READ: {
            auto rr = static_cast<ResourceReadInst *>(inst);
            auto op = rr->op();
            h = luisa::hash64(&op, sizeof(op), h);
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
            auto rq = static_cast<RayQueryObjectReadInst *>(inst);
            auto op = rq->op();
            h = luisa::hash64(&op, sizeof(op), h);
            auto vns = state.get_operand_vns(inst);
            h = hash_operand_vns(vns, false, h);
            break;
        }
        default: break;
    }
    return h;
}

static void process_instruction_for_gvn(Instruction *inst, BasicBlock *block, GVNState &state, GVNInfo &info) noexcept {
    if (inst->type() == nullptr) return;
    if (!can_value_number(inst)) return;
    auto hash = compute_instruction_hash(inst, state);
    if (auto leader = state.find_leader(hash, block, inst)) {
        inst->replace_all_uses_with(leader);
        state.value_to_vn[inst] = state.value_to_vn[leader];
        ++info.replaced_inst_count;
    } else {
        auto vn = state.next_vn++;
        state.record_leader(hash, vn, inst, block);
        state.value_to_vn[inst] = vn;
    }
}

[[nodiscard]] static bool is_safe_to_remove(Instruction *inst) noexcept {
    auto info = get_memory_info(inst);
    if (info.is_removable_if_unused()) return true;
    if (inst->derived_instruction_tag() == DerivedInstructionTag::AUTODIFF_INTRINSIC) {
        auto intrinsic = static_cast<AutodiffIntrinsicInst *>(inst);
        return intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_GRADIENT;
    }
    return false;
}

static void gvn_pass_on_function(Function *function, GVNInfo &info) noexcept {
    if (function == nullptr || !function->is_definition()) return;
    auto def = static_cast<FunctionDefinition *>(function);
    if (def->body_block() == nullptr) return;
    auto dom_tree = compute_dom_tree(function);
    GVNState state;
    state.dom_tree = &dom_tree;
    def->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
        luisa::vector<Instruction *> insts;
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (!inst->isa<PhiInst>()) insts.push_back(inst);
        });
        for (auto inst : insts) {
            if (inst->isa<PhiInst>()) continue;
            process_instruction_for_gvn(inst, block, state, info);
        }
    });
    luisa::vector<Instruction *> to_remove;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->type() != nullptr && inst->use_list().empty() && !inst->is_terminator() && is_safe_to_remove(inst)) {
            to_remove.push_back(inst);
        }
    });
    for (auto inst : to_remove) {
        inst->remove_self();
        ++info.removed_inst_count;
    }
    // Coalesce phis that GVN's value-numbering reduced to a single source
    // (typical after mem2reg + GVN finds equivalent incoming values).
    bool changed;
    do {
        changed = false;
        luisa::vector<PhiInst *> phis;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) phis.push_back(static_cast<PhiInst *>(inst));
        });
        for (auto phi : phis) {
            if (simplify_phi_instruction(phi)) {
                ++info.removed_inst_count;
                changed = true;
            }
        }
    } while (changed);
}

}// namespace detail

GVNInfo gvn_pass_run_on_function(Function *function) noexcept {
    GVNInfo info;
    detail::gvn_pass_on_function(function, info);
    return info;
}

GVNInfo gvn_pass_run_on_module(Module *module, PassReport *report) noexcept {
    GVNInfo info;
    if (module == nullptr) return info;
    for (auto f : module->function_list()) {
        auto sub = gvn_pass_run_on_function(f);
        info.replaced_inst_count += sub.replaced_inst_count;
        info.removed_inst_count += sub.removed_inst_count;
    }
    if (report != nullptr) {
        report->set("replaced_inst", info.replaced_inst_count);
        report->set("removed_inst", info.removed_inst_count);
    }
    return info;
}

}// namespace luisa::compute::xir
