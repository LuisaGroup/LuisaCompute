#include <luisa/xir/passes/loop_vectorization.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>

#include "helpers.h"
#include "natural_loop.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

// Vectorization currently packs four (two for 64-bit elements) consecutive
// iterations of a canonical counted loop into vector ALU operations. Memory
// stays scalar: the typed-buffer ABI requires access types to match the
// buffer element type, and local arrays cannot be type-punned, so lanes are
// gathered with scalar loads + AGGREGATE and scattered with EXTRACT +
// scalar stores. Loops with calls, atomics, cross-lane dependencies, or
// non-elementwise operations are rejected.
//
// Reduction recognition: a single loop-carried accumulator of the form
// acc = phi(start, acc <op> x) with <op> in {add, mul, min, max} is kept as
// a scalar phi, while the per-iteration value x is computed vectorially and
// folded horizontally once per packed iteration (avoiding identity-element
// materialization; float add/mul reassociation matches the fast-math
// contract of the surrounding pipeline).
//
// Remainder handling: when the constant trip count is not a multiple of the
// vector factor, the loop bound is tightened to the largest multiple of the
// factor and the trailing iterations are peeled as straight-line scalar
// clones between the loop exit edge and the exit block.

[[nodiscard]] bool is_elementwise_arithmetic(ArithmeticOp op, size_t operand_count) noexcept {
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
        case ArithmeticOp::UNARY_BIT_NOT:
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::ABS:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
        case ArithmeticOp::SIN:
        case ArithmeticOp::COS:
        case ArithmeticOp::TAN:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2: return operand_count == 1u;
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
        case ArithmeticOp::POW:
        case ArithmeticOp::STEP: return operand_count == 2u;
        case ArithmeticOp::SELECT:
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::LERP:
        case ArithmeticOp::FMA: return operand_count == 3u;
        default: return false;
    }
}

struct AffineIndex {
    Value *base;// the address value (buffer or alloca/GEP base)
    Value *index;
    bool from_gep;
};

// Match an address of the form buffer/alloca indexed by iv or iv + const.
[[nodiscard]] bool match_unit_stride_address(Value *address, PhiInst *iv,
                                             const NaturalLoop &loop,
                                             AffineIndex &out) noexcept {
    if (address == nullptr) { return false; }
    if (address->isa<GEPInst>()) {
        auto *gep = static_cast<GEPInst *>(address);
        if (gep->index_count() != 1u) { return false; }
        auto *base = gep->base();
        if (base == nullptr || !base->isa<AllocaInst>()) { return false; }
        out.base = base;
        out.index = gep->index(0u);
        out.from_gep = true;
    } else {
        return false;
    }
    // index must be iv or add(iv, invariant)
    if (out.index == iv) { return true; }
    if (out.index->isa<ArithmeticInst>()) {
        auto *add = static_cast<ArithmeticInst *>(out.index);
        if (add->op() == ArithmeticOp::BINARY_ADD && add->operand_count() == 2u) {
            for (auto i = 0u; i < 2u; ++i) {
                if (add->operand(i) == iv) {
                    auto *other = add->operand(1u - i);
                    auto *other_block = other->isa<Instruction>() ?
                                            static_cast<Instruction *>(other)->parent_block() :
                                            nullptr;
                    if (other_block == nullptr || !loop.contains(other_block)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

struct VectorizationPlan {
    luisa::vector<Instruction *> memory_insts; // ordered loads and stores
    luisa::vector<ArithmeticInst *> arith_insts;
    Instruction *recurrence{nullptr};
};

// A recognized reduction: acc = phi(preheader: start, latch: acc <op> x)
// with an associative/commutative <op>. The accumulator stays scalar; the
// per-iteration value x is vectorized and folded horizontally per packed
// iteration.
struct ReductionInfo {
    PhiInst *phi{nullptr};
    ArithmeticInst *combine{nullptr};
    ArithmeticOp op{ArithmeticOp::BINARY_ADD};
    Value *value{nullptr};// per-lane reduced value x
};

[[nodiscard]] bool is_reduction_op(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX: return true;
        default: return false;
    }
}

[[nodiscard]] bool decode_constant_int(const Value *value, int64_t &out) noexcept {
    if (value == nullptr || !value->isa<Constant>()) { return false; }
    auto constant = static_cast<const Constant *>(value);
    auto type = constant->type();
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

[[nodiscard]] Value *make_int_constant(Module *module, const Type *type,
                                       int64_t value) noexcept {
    if (type->is_int8()) {
        auto v = static_cast<int8_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint8()) {
        auto v = static_cast<uint8_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int16()) {
        auto v = static_cast<int16_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint16()) {
        auto v = static_cast<uint16_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int32()) {
        auto v = static_cast<int32_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint32()) {
        auto v = static_cast<uint32_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int64()) {
        auto v = static_cast<int64_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint64()) {
        auto v = static_cast<uint64_t>(value);
        return module->create_constant(type, &v);
    }
    return nullptr;
}

class PeelValueResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void emplace(const Value *from, Value *to) noexcept { _map.emplace(from, to); }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (auto it = _map.find(value); it != _map.end()) { return it->second; }
        return const_cast<Value *>(value);
    }
};

[[nodiscard]] bool analyze_body(const NaturalLoop &loop, BasicBlock *compute,
                                BasicBlock *latch,
                                PhiInst *iv, Instruction *recurrence,
                                Instruction *reduction_combine,
                                VectorizationPlan &plan) noexcept {
    luisa::unordered_set<Instruction *> body_values;
    for (auto *inst : compute->instructions()) {
        if (inst->is_terminator() || inst == recurrence ||
            inst == reduction_combine) { continue; }
        if (inst->isa<GEPInst>()) {
            // Address producers are fine as long as they only feed the
            // body's loads/stores.
            for (auto *use : inst->use_list()) {
                auto *user = use->user();
                if (user == nullptr || !user->isa<Instruction>()) { return false; }
                auto *user_inst = static_cast<Instruction *>(user);
                if (user_inst->parent_block() != compute ||
                    !(user_inst->isa<LoadInst>() || user_inst->isa<StoreInst>())) {
                    return false;
                }
            }
            continue;
        }
        if (inst->isa<LoadInst>()) {
            AffineIndex addr;
            if (!match_unit_stride_address(inst->operand(0u), iv, loop, addr)) { return false; }
            plan.memory_insts.emplace_back(inst);
            body_values.emplace(inst);
            continue;
        }
        if (inst->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(inst);
            AffineIndex addr;
            if (!match_unit_stride_address(store->variable(), iv, loop, addr)) { return false; }
            plan.memory_insts.emplace_back(inst);
            continue;
        }
        if (inst->isa<ArithmeticInst>()) {
            auto *arith = static_cast<ArithmeticInst *>(inst);
            if (!is_elementwise_arithmetic(arith->op(), arith->operand_count())) { return false; }
            auto *type = arith->type();
            if (type == nullptr || !type->is_scalar()) { return false; }
            // every operand must be loop-invariant, the iv, or produced in
            // the compute block
            for (auto i = 0u; i < arith->operand_count(); ++i) {
                auto *operand = arith->operand(i);
                if (operand == iv) { continue; }
                if (!operand->isa<Instruction>()) { continue; }
                auto *def_inst = static_cast<Instruction *>(operand);
                if (!loop.contains(def_inst->parent_block())) { continue; }
                if (def_inst->parent_block() != compute) { return false; }
            }
            plan.arith_insts.emplace_back(arith);
            body_values.emplace(arith);
            continue;
        }
        return false;
    }
    // Values produced in the body must stay inside the loop's compute and
    // latch blocks (except the recurrence feeding the induction phi and the
    // reduction combine feeding the accumulator phi).
    for (auto *inst : body_values) {
        for (auto *use : inst->use_list()) {
            auto *user = use->user();
            if (user == nullptr || !user->isa<Instruction>()) { return false; }
            auto *user_inst = static_cast<Instruction *>(user);
            if (user_inst == reduction_combine) { continue; }
            auto *user_block = user_inst->parent_block();
            if (user_block != compute && user_block != latch) { return false; }
        }
    }
    // Stores and loads must not alias: every stored base must differ from
    // every loaded base.
    for (auto *inst : plan.memory_insts) {
        if (!inst->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(inst);
        AffineIndex store_addr;
        static_cast<void>(match_unit_stride_address(store->variable(), iv, loop, store_addr));
        for (auto *other : plan.memory_insts) {
            if (!other->isa<LoadInst>()) { continue; }
            AffineIndex load_addr;
            static_cast<void>(match_unit_stride_address(other->operand(0u), iv, loop, load_addr));
            if (store_addr.base == load_addr.base) { return false; }
        }
    }
    plan.recurrence = recurrence;
    return true;
}

[[nodiscard]] uint32_t vector_factor_for(const VectorizationPlan &plan,
                                         const Type *fallback_elem) noexcept {
    // All arithmetic must share one element type family; choose VF by size.
    // A load-only reduction body has no arithmetic and falls back to the
    // accumulator's element type.
    const Type *elem = plan.arith_insts.empty() ?
                           fallback_elem :
                           plan.arith_insts.front()->type();
    if (elem == nullptr) { return 0u; }
    for (auto *arith : plan.arith_insts) {
        if (arith->type() != elem && !arith->type()->is_bool()) { return 0u; }
    }
    if (!elem->is_arithmetic() || elem->is_bool()) { return 0u; }
    return elem->size() >= 8u ? 2u : 4u;
}

namespace {
[[nodiscard]] bool debug_reject_reasons_enabled() noexcept {
    static auto enabled = std::getenv("LUISA_DEBUG_VEC") != nullptr;
    return enabled;
}
}// namespace
#define LUISA_VEC_REJECT(reason)                                     \
    do {                                                             \
        if (debug_reject_reasons_enabled()) {                        \
            LUISA_WARNING_WITH_LOCATION(                             \
                "loop vectorization rejected: {}", reason);         \
        }                                                            \
        return false;                                                \
    } while (false)

[[nodiscard]] bool try_vectorize_loop(FunctionDefinition *def, const NaturalLoop &loop,
                                      LoopVectorizationInfo &info) noexcept {
    if (loop.preheader == nullptr || loop.latches.size() != 1u ||
        loop.exit_blocks.size() != 1u) {
        LUISA_VEC_REJECT("missing preheader/single latch/single exit");
    }
    auto bounds = analyze_loop_bounds(loop);
    if (!bounds.is_valid() || !bounds.stride_is_constant || bounds.stride != 1 ||
        !bounds.trip_count_is_constant) {
        LUISA_VEC_REJECT("no constant unit-stride trip count");
    }
    auto trip_count = bounds.constant_trip_count;
    auto *latch = loop.latches.front();
    // The latch must branch back unconditionally.
    auto *latch_term = latch->terminator();
    if (latch_term == nullptr || !latch_term->isa<BranchInst>() ||
        static_cast<BranchInst *>(latch_term)->target_block() != loop.header) {
        return false;
    }
    // Find the induction recurrence instruction.
    Instruction *recurrence = nullptr;
    for (auto i = 0u; i < bounds.induction_phi->incoming_count(); ++i) {
        auto incoming = bounds.induction_phi->incoming(i);
        if (incoming.block == latch && incoming.value->isa<Instruction>()) {
            recurrence = static_cast<Instruction *>(incoming.value);
        }
    }
    if (recurrence == nullptr) { LUISA_VEC_REJECT("no recurrence in latch"); }
    // Resolve the compute block: either the single-block form (the body is
    // the latch) or the canonical two-block form (compute -> latch), where
    // the latch holds only the recurrence and the back branch and the
    // compute block falls through to it unconditionally.
    BasicBlock *compute = nullptr;
    if (loop.body_blocks.size() == 1u && loop.body_blocks.front() == latch) {
        compute = latch;
    } else if (loop.body_blocks.size() == 2u) {
        auto *other = loop.body_blocks.front() == latch ?
                          loop.body_blocks.back() :
                          loop.body_blocks.front();
        if (other == latch) { LUISA_VEC_REJECT("degenerate body blocks"); }
        auto latch_is_empty = true;
        for (auto *inst : latch->instructions()) {
            if (inst != recurrence && !inst->is_terminator()) {
                latch_is_empty = false;
                break;
            }
        }
        if (!latch_is_empty) { LUISA_VEC_REJECT("non-empty latch in two-block body"); }
        auto *other_term = other->terminator();
        if (other_term == nullptr || !other_term->isa<BranchInst>() ||
            static_cast<BranchInst *>(other_term)->target_block() != latch) {
            LUISA_VEC_REJECT("compute block does not fall through to latch");
        }
        compute = other;
    } else {
        LUISA_VEC_REJECT("multi-block body");
    }
    // Besides the induction phi, at most one loop-carried reduction
    // accumulator is allowed: acc = phi(preheader: start, latch: acc <op> x)
    // with <op> in {add, mul, min, max} and a per-lane value x produced in
    // the body (or the induction variable itself).
    auto *iv = bounds.induction_phi;
    ReductionInfo reduction;
    for (auto *inst : loop.header->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        if (inst == iv) { continue; }
        if (reduction.phi != nullptr) { return false; }// at most one reduction
        auto *acc = static_cast<PhiInst *>(inst);
        auto *type = acc->type();
        if (type == nullptr || !type->is_scalar() || !type->is_arithmetic() ||
            type->is_bool()) {
            return false;
        }
        Value *start = nullptr;
        Value *combine_value = nullptr;
        for (auto i = 0u; i < acc->incoming_count(); ++i) {
            auto incoming = acc->incoming(i);
            if (incoming.block == loop.preheader) { start = incoming.value; }
            if (incoming.block == latch) { combine_value = incoming.value; }
        }
        if (start == nullptr || combine_value == nullptr ||
            !combine_value->isa<ArithmeticInst>()) {
            return false;
        }
        auto *combine = static_cast<ArithmeticInst *>(combine_value);
        if (combine->parent_block() != compute || combine == recurrence ||
            !is_reduction_op(combine->op()) || combine->operand_count() != 2u) {
            return false;
        }
        Value *x = nullptr;
        if (combine->operand(0u) == acc) {
            x = combine->operand(1u);
        } else if (combine->operand(1u) == acc) {
            x = combine->operand(0u);
        } else {
            return false;
        }
        // x must vary per lane: the induction variable or an instruction in
        // the latch that the vectorizer can pack.
        if (x != iv) {
            if (!x->isa<Instruction>()) { return false; }
            auto *x_inst = static_cast<Instruction *>(x);
            if (x_inst->parent_block() != compute ||
                !(x_inst->isa<LoadInst>() || x_inst->isa<ArithmeticInst>())) {
                return false;
            }
        }
        // The combine result must feed only the accumulator phi.
        for (auto *use : combine->use_list()) {
            if (use->user() != acc) { return false; }
        }
        reduction.phi = acc;
        reduction.combine = combine;
        reduction.op = combine->op();
        reduction.value = x;
    }

    VectorizationPlan plan;
    if (!analyze_body(loop, compute, latch, iv, recurrence, reduction.combine, plan)) {
        LUISA_VEC_REJECT("body analysis failed");
    }
    // Nothing to vectorize without arithmetic on loaded data, unless a
    // reduction folds the loaded values directly.
    if (plan.arith_insts.empty() && reduction.phi == nullptr) {
        LUISA_VEC_REJECT("no arithmetic and no reduction");
    }
    auto vf = vector_factor_for(plan, reduction.phi != nullptr ?
                                           reduction.phi->type() :
                                           nullptr);
    if (vf == 0u || trip_count < vf) { LUISA_VEC_REJECT("bad vector factor"); }

    auto *module = def->parent_module();
    auto *elem_type = plan.arith_insts.empty() ?
                          reduction.phi->type() :
                          plan.arith_insts.front()->type();
    auto *vector_type = Type::vector(elem_type, vf);
    if (vector_type == nullptr) { return false; }
    // The reduction accumulator must match the packed element type.
    if (reduction.phi != nullptr && reduction.phi->type() != elem_type) {
        return false;
    }

    // Remainder handling: when the trip count is not a multiple of the
    // vector factor, tighten the loop bound to the largest multiple and
    // peel the trailing iterations as straight-line scalar clones between
    // the header's exit edge and the exit block.
    auto remainder = trip_count % vf;
    if (remainder != 0u) {
        // Peeling a reduction's trailing iterations would require threading
        // the accumulator through the peel chain; rejected for now.
        if (reduction.phi != nullptr) { return false; }
        auto *exit_block = loop.exit_blocks.front();
        // Exit phis may only consume loop-invariant values from the header
        // edge (the peeled clones cannot reproduce loop-defined values on
        // the retargeted edge).
        for (auto *inst : exit_block->instructions()) {
            if (!inst->isa<PhiInst>()) { break; }
            auto *phi = static_cast<PhiInst *>(inst);
            for (auto i = 0u; i < phi->incoming_count(); ++i) {
                auto incoming = phi->incoming(i);
                if (incoming.block != loop.header) { continue; }
                auto *value = incoming.value;
                if (value != nullptr && value->isa<Instruction>() &&
                    loop.contains(static_cast<Instruction *>(value)->parent_block())) {
                    return false;
                }
            }
        }
        // Header non-phi instructions (e.g. index GEPs computed in the
        // header) are referenced by the body and must be cloned into each
        // peel with the constant induction value, so they must be pure.
        luisa::vector<Instruction *> header_scalar_insts;
        {
            auto past_phis = false;
            for (auto *inst : loop.header->instructions()) {
                if (!past_phis && inst->isa<PhiInst>()) { continue; }
                past_phis = true;
                if (inst->is_terminator()) { continue; }
                if (!inst->isa<GEPInst>() && !inst->isa<ArithmeticInst>()) {
                    LUISA_VEC_REJECT("impure header instruction with remainder");
                }
                header_scalar_insts.emplace_back(inst);
            }
        }
        int64_t start_constant = 0;
        if (!decode_constant_int(bounds.start_value, start_constant)) {
            return false;
        }
        auto main_trip_count = trip_count - remainder;
        auto *new_bound = make_int_constant(
            module, iv->type(),
            start_constant + static_cast<int64_t>(main_trip_count));
        if (new_bound == nullptr) { return false; }
        // Tighten the bound in the header comparison.
        {
            auto *header_branch = static_cast<ConditionalBranchInst *>(
                loop.header->terminator());
            auto *condition = header_branch->condition();
            if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
                return false;
            }
            auto *cmp = static_cast<ArithmeticInst *>(condition);
            auto bound_replaced = false;
            for (auto i = 0u; i < cmp->operand_count(); ++i) {
                if (cmp->operand(i) == bounds.bound_value) {
                    cmp->set_operand(i, new_bound);
                    bound_replaced = true;
                }
            }
            if (!bound_replaced) { return false; }
        }
        // Clone the scalar body for each trailing iteration with the
        // induction variable mapped to its constant value. The template is
        // the compute block plus the latch (identical for single-block
        // bodies).
        luisa::vector<BasicBlock *> template_blocks;
        if (compute != latch) { template_blocks.emplace_back(compute); }
        template_blocks.emplace_back(latch);
        struct PeelIteration {
            BasicBlock *entry;
            BasicBlock *back;
        };
        luisa::vector<PeelIteration> peel_blocks;
        peel_blocks.reserve(remainder);
        for (auto k = 0u; k < remainder; ++k) {
            PeelValueResolver resolver;
            luisa::unordered_map<BasicBlock *, BasicBlock *> block_map;
            for (auto *tb : template_blocks) {
                auto *clone = def->create_basic_block();
                block_map.emplace(tb, clone);
                resolver.emplace(tb, clone);
            }
            auto iv_value = start_constant +
                            static_cast<int64_t>(main_trip_count + k);
            auto *iv_constant = make_int_constant(module, iv->type(), iv_value);
            if (iv_constant == nullptr) { return false; }
            resolver.emplace(iv, iv_constant);
            // Clone the header's scalar instructions (index expressions
            // using the induction variable) into this peel's entry so the
            // body clones resolve them to the constant-induction versions.
            {
                XIRBuilder header_builder;
                header_builder.set_insertion_point(block_map[template_blocks.front()]);
                for (auto *inst : header_scalar_insts) {
                    auto *cloned = inst->clone_with_metadata(header_builder, resolver);
                    resolver.emplace(inst, cloned);
                }
            }
            for (auto *tb : template_blocks) {
                XIRBuilder peel_builder;
                peel_builder.set_insertion_point(block_map[tb]);
                for (auto *inst : tb->instructions()) {
                    auto *cloned = inst->clone_with_metadata(peel_builder, resolver);
                    resolver.emplace(inst, cloned);
                }
            }
            peel_blocks.emplace_back(PeelIteration{
                .entry = block_map[template_blocks.front()],
                .back = block_map[template_blocks.back()]});
        }
        // Rewire: header exit arm -> peel[0] -> ... -> peel[r-1] -> exit.
        {
            auto *header_branch = static_cast<ConditionalBranchInst *>(
                loop.header->terminator());
            if (header_branch->true_block() == exit_block) {
                header_branch->set_true_target(peel_blocks.front().entry);
            } else {
                LUISA_ASSERT(header_branch->false_block() == exit_block,
                             "Loop vectorization lost the header exit edge.");
                header_branch->set_false_target(peel_blocks.front().entry);
            }
            for (auto k = 0u; k + 1u < remainder; ++k) {
                static_cast<BranchInst *>(peel_blocks[k].back->terminator())
                    ->set_target_block(peel_blocks[k + 1u].entry);
            }
            static_cast<BranchInst *>(peel_blocks.back().back->terminator())
                ->set_target_block(exit_block);
        }
        // Exit phis' header edge now comes from the last peel block.
        for (auto *inst : exit_block->instructions()) {
            if (!inst->isa<PhiInst>()) { break; }
            auto *phi = static_cast<PhiInst *>(inst);
            for (auto i = 0u; i < phi->incoming_count(); ++i) {
                if (phi->incoming(i).block == loop.header) {
                    phi->set_incoming(i, phi->incoming(i).value,
                                      peel_blocks.back().back);
                }
            }
        }
    }

    XIRBuilder builder;
    builder.set_insertion_point(compute->instructions().front()->prev());

    luisa::unordered_map<const Value *, Value *> scalar_to_vector;
    // Per-lane index offsets, cached by (index, lane): keying on the index
    // alone would alias every lane > 0 to the same offset.
    luisa::unordered_map<const Value *, luisa::unordered_map<uint32_t, Value *>> lane_index;
    auto index_plus_lane = [&](Value *index, uint32_t lane) noexcept -> Value * {
        if (lane == 0u) { return index; }
        if (auto it = lane_index.find(index); it != lane_index.end()) {
            if (auto jt = it->second.find(lane); jt != it->second.end()) {
                return jt->second;
            }
        }
        uint32_t k = lane;
        auto *kc = module->create_constant(index->type(), &k);
        auto *added = builder.call(index->type(), ArithmeticOp::BINARY_ADD, {index, kc});
        lane_index[index].emplace(lane, added);
        return added;
    };
    auto broadcast = [&](Value *scalar) noexcept -> Value * {
        if (auto it = scalar_to_vector.find(scalar); it != scalar_to_vector.end()) {
            return it->second;
        }
        luisa::vector<Value *> lanes;
        lanes.reserve(vf);
        for (auto k = 0u; k < vf; ++k) { lanes.emplace_back(scalar); }
        auto *vec = builder.call(Type::vector(scalar->type(), vf),
                                 ArithmeticOp::AGGREGATE, lanes);
        scalar_to_vector.emplace(scalar, vec);
        return vec;
    };

    // Vectorized induction variable for data uses of iv.
    {
        luisa::vector<Value *> lanes;
        lanes.reserve(vf);
        for (auto k = 0u; k < vf; ++k) {
            lanes.emplace_back(index_plus_lane(iv, k));
        }
        auto *vec_iv = builder.call(Type::vector(iv->type(), vf),
                                    ArithmeticOp::AGGREGATE, lanes);
        scalar_to_vector.emplace(iv, vec_iv);
    }

    // Gather loads into vectors.
    for (auto *inst : plan.memory_insts) {
        if (!inst->isa<LoadInst>()) { continue; }
        AffineIndex addr;
        static_cast<void>(match_unit_stride_address(inst->operand(0u), iv, loop, addr));
        luisa::vector<Value *> lanes;
        lanes.reserve(vf);
        for (auto k = 0u; k < vf; ++k) {
            auto *lane_index_value = index_plus_lane(addr.index, k);
            auto *gep = builder.gep(elem_type, addr.base, {lane_index_value});
            lanes.emplace_back(builder.load(elem_type, gep));
        }
        auto *vec = builder.call(Type::vector(inst->type(), vf),
                                 ArithmeticOp::AGGREGATE, lanes);
        scalar_to_vector.emplace(inst, vec);
    }

    // Vector arithmetic.
    for (auto *arith : plan.arith_insts) {
        luisa::vector<Value *> operands;
        operands.reserve(arith->operand_count());
        for (auto i = 0u; i < arith->operand_count(); ++i) {
            auto *operand = arith->operand(i);
            if (auto it = scalar_to_vector.find(operand); it != scalar_to_vector.end()) {
                operands.emplace_back(it->second);
            } else {
                operands.emplace_back(broadcast(operand));
            }
        }
        auto *result_type = arith->type()->is_bool() ?
                                Type::vector(arith->type(), vf) :
                                vector_type;
        auto *vec = builder.call(result_type, arith->op(), operands);
        scalar_to_vector.emplace(arith, vec);
        info.created_vector_inst_count++;
    }

    // Scatter stores lane by lane.
    luisa::vector<Instruction *> erase_list;
    for (auto *inst : plan.memory_insts) {
        if (!inst->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(inst);
        AffineIndex addr;
        static_cast<void>(match_unit_stride_address(store->variable(), iv, loop, addr));
        auto *value = store->value();
        for (auto k = 0u; k < vf; ++k) {
            auto *lane_index_value = index_plus_lane(addr.index, k);
            auto *gep = builder.gep(elem_type, addr.base, {lane_index_value});
            Value *lane_value = value;
            if (auto it = scalar_to_vector.find(value); it != scalar_to_vector.end()) {
                uint32_t lane = k;
                auto *lane_const = module->create_constant(Type::of<uint32_t>(), &lane);
                lane_value = builder.call(elem_type, ArithmeticOp::EXTRACT,
                                          {it->second, lane_const});
            }
            builder.store(gep, lane_value);
        }
    }

    // Reduction: fold the packed lanes horizontally with the reduction op,
    // then combine with the scalar accumulator. The accumulator phi's latch
    // incoming is rewired to the new combine; the original combine is erased
    // below. Keeping the accumulator scalar avoids materializing identity
    // elements for arbitrary element types.
    if (reduction.phi != nullptr) {
        auto it = scalar_to_vector.find(reduction.value);
        LUISA_ASSERT(it != scalar_to_vector.end(),
                     "Loop vectorization lost the reduction value.");
        auto *vec_x = it->second;
        Value *folded = nullptr;
        for (auto k = 0u; k < vf; ++k) {
            auto *lane_const = module->create_constant(Type::of<uint32_t>(), &k);
            auto *lane = builder.call(elem_type, ArithmeticOp::EXTRACT,
                                      {vec_x, lane_const});
            folded = folded == nullptr ?
                         lane :
                         builder.call(elem_type, reduction.op, {folded, lane});
        }
        auto *new_combine = builder.call(elem_type, reduction.op,
                                         {reduction.phi, folded});
        for (auto i = 0u; i < reduction.phi->incoming_count(); ++i) {
            if (reduction.phi->incoming(i).block == latch) {
                reduction.phi->set_incoming(i, new_combine, latch);
            }
        }
    }

    // New recurrence: iv + VF.
    auto vf_value = static_cast<uint32_t>(vf);
    auto *vf_const = module->create_constant(iv->type(), &vf_value);
    auto *new_recurrence = builder.call(iv->type(), ArithmeticOp::BINARY_ADD,
                                        {iv, vf_const});
    for (auto i = 0u; i < iv->incoming_count(); ++i) {
        if (iv->incoming(i).block == latch) {
            iv->set_incoming(i, new_recurrence, latch);
        }
    }

    // Erase the original scalar body, users before definitions so use lists
    // stay valid throughout the mutation: stores, then the reduction combine
    // (a user of the packed value), then arithmetic, then loads, then the
    // old recurrence.
    for (auto *inst : plan.memory_insts) {
        if (inst->isa<StoreInst>()) { erase_list.emplace_back(inst); }
    }
    if (reduction.combine != nullptr) { erase_list.emplace_back(reduction.combine); }
    for (auto *arith : plan.arith_insts) { erase_list.emplace_back(arith); }
    for (auto *inst : plan.memory_insts) {
        if (inst->isa<LoadInst>()) { erase_list.emplace_back(inst); }
    }
    erase_list.emplace_back(recurrence);
    for (auto *inst : erase_list) {
        static_cast<void>(inst->remove_self());
    }
    info.vectorized_loop_count++;
    return true;
}

}// namespace

static void run(Function *function, LoopVectorizationInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto *def = function->definition();
    if (def == nullptr) { return; }
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop vectorization rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    auto dom_tree = compute_dom_tree(def);
    auto loops = discover_natural_loops(def, dom_tree);
    for (auto &loop : loops) {
        static_cast<void>(try_vectorize_loop(def, loop, info));
    }
}

}// namespace detail

LoopVectorizationInfo loop_vectorization_pass_run_on_function(Function *function) noexcept {
    LoopVectorizationInfo info;
    detail::run(function, info);
    return info;
}

LoopVectorizationInfo loop_vectorization_pass_run_on_module(Module *module,
                                                            PassReport *report) noexcept {
    LoopVectorizationInfo info;
    for (auto *function : module->function_list()) {
        detail::run(function, info);
    }
    if (report != nullptr) {
        report->set("vectorized_loop_count", info.vectorized_loop_count);
        report->set("created_vector_inst_count", info.created_vector_inst_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
