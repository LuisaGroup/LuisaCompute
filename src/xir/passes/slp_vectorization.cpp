#include <luisa/xir/passes/slp_vectorization.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/alloca.h>
#include <limits>
#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static unsigned get_slp_vector_factor(const Type *elem_type) noexcept {
    auto size = elem_type->size();
    return size >= 8u ? 2u : 4u;
}

[[nodiscard]] static bool get_constant_int_value(Constant *constant, int64_t &value) noexcept {
    if (constant == nullptr || constant->type() == nullptr) { return false; }
    auto type = constant->type();
    if (type->is_int8()) {
        value = constant->as<int8_t>();
    } else if (type->is_uint8()) {
        value = constant->as<uint8_t>();
    } else if (type->is_int16()) {
        value = constant->as<int16_t>();
    } else if (type->is_uint16()) {
        value = constant->as<uint16_t>();
    } else if (type->is_int32()) {
        value = constant->as<int32_t>();
    } else if (type->is_uint32()) {
        value = constant->as<uint32_t>();
    } else if (type->is_int64()) {
        value = constant->as<int64_t>();
    } else if (type->is_uint64()) {
        auto unsigned_value = constant->as<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return false; }
        value = static_cast<int64_t>(unsigned_value);
    } else {
        return false;
    }
    return true;
}

[[nodiscard]] static bool get_gep_constant_offset(GEPInst *gep, int64_t &out) noexcept {
    if (gep == nullptr || gep->index_count() != 1) return false;
    auto *index = gep->index(0);
    if (index == nullptr || !index->isa<Constant>()) return false;
    return get_constant_int_value(static_cast<Constant *>(index), out) && out >= 0;
}

[[nodiscard]] static luisa::vector<luisa::vector<StoreInst *>> collect_store_seeds(FunctionDefinition *def) noexcept {
    luisa::vector<luisa::vector<StoreInst *>> seeds;
    for (auto *bb : def->basic_blocks()) {
        luisa::vector<StoreInst *> run;
        const AllocaInst *current_alloca = nullptr;
        const Type *current_elem_type = nullptr;
        int64_t current_base_offset = 0;
        unsigned max_factor = 0;
        auto flush_run = [&]() noexcept {
            if (run.size() >= 2) {
                seeds.emplace_back(run.begin(), run.end());
            }
            run.clear();
            current_alloca = nullptr;
            current_elem_type = nullptr;
            current_base_offset = 0;
            max_factor = 0;
        };
        for (auto *inst : bb->instructions()) {
            StoreInst *store = nullptr;
            if (inst->isa<StoreInst>()) { store = static_cast<StoreInst *>(inst); }
            if (store != nullptr) {
                auto *value = store->value();
                auto *value_type = value != nullptr ? value->type() : nullptr;
                if (value_type != nullptr && value_type->is_scalar() && value_type->is_arithmetic()) {
                    auto *ptr = store->variable();
                    GEPInst *gep = nullptr;
                    if (ptr != nullptr && ptr->isa<GEPInst>()) { gep = static_cast<GEPInst *>(ptr); }
                    int64_t offset = 0;
                    if (gep != nullptr && get_gep_constant_offset(gep, offset)) {
                        auto *base_alloca = trace_pointer_base_local_alloca_inst(gep);
                        if (base_alloca != nullptr) {
                            bool can_extend = !run.empty() &&
                                              base_alloca == current_alloca &&
                                              value_type == current_elem_type &&
                                              static_cast<uint64_t>(offset) ==
                                                  static_cast<uint64_t>(current_base_offset) + run.size() &&
                                              run.size() < max_factor;
                            if (!run.empty() && !can_extend) {
                                flush_run();
                            }
                            if (run.empty()) {
                                current_alloca = base_alloca;
                                current_elem_type = value_type;
                                current_base_offset = offset;
                                max_factor = get_slp_vector_factor(value_type);
                            }
                            run.emplace_back(store);
                            if (run.size() == max_factor) {
                                flush_run();
                            }
                            continue;
                        }
                    }
                }
            }
            if (!run.empty()) { flush_run(); }
        }
        if (!run.empty()) { flush_run(); }
    }
    return seeds;
}

[[nodiscard]] static size_t elementwise_arithmetic_operand_count(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
        case ArithmeticOp::UNARY_BIT_NOT:
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::ABS:
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
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
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT: return 1u;
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
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
        case ArithmeticOp::STEP:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::POW:
        case ArithmeticOp::POW_INT:
        case ArithmeticOp::COPYSIGN: return 2u;
        case ArithmeticOp::SELECT:
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::LERP:
        case ArithmeticOp::SMOOTHSTEP:
        case ArithmeticOp::FMA:
            return 3u;
        default: return 0u;
    }
}

[[nodiscard]] static bool has_single_store_use(Instruction *instruction, StoreInst *store) noexcept {
    size_t use_count = 0u;
    for (auto *use : instruction->use_list()) {
        if (use->user() != store) { return false; }
        use_count++;
    }
    return use_count == 1u;
}

[[nodiscard]] static bool collect_vectorizable_roots(
    FunctionDefinition *def,
    const luisa::vector<StoreInst *> &stores,
    luisa::vector<Instruction *> &roots) noexcept {
    if (stores.size() < 2u || stores.size() > 4u) { return false; }
    roots.clear();
    roots.reserve(stores.size());
    luisa::unordered_set<Instruction *> unique_roots;
    for (auto *store : stores) {
        auto *value = store->value();
        if (value == nullptr || !value->isa<Instruction>()) { return false; }
        auto *root = static_cast<Instruction *>(value);
        if (!root->is_linked() || root->parent_function() != def ||
            !unique_roots.emplace(root).second ||
            !has_single_store_use(root, store)) {
            return false;
        }
        roots.emplace_back(root);
    }
    auto *first = roots.front();
    auto tag = first->derived_instruction_tag();
    if (tag != DerivedInstructionTag::ARITHMETIC && tag != DerivedInstructionTag::CAST) { return false; }
    auto *result_type = first->type();
    if (result_type == nullptr || !result_type->is_arithmetic()) { return false; }
    for (auto *root : roots) {
        if (root->derived_instruction_tag() != tag ||
            root->type() != result_type ||
            root->operand_count() != first->operand_count()) {
            return false;
        }
    }
    if (tag == DerivedInstructionTag::ARITHMETIC) {
        auto op = static_cast<ArithmeticInst *>(first)->op();
        auto operand_count = elementwise_arithmetic_operand_count(op);
        if (operand_count == 0u || first->operand_count() != operand_count) { return false; }
        for (auto *root : roots) {
            if (static_cast<ArithmeticInst *>(root)->op() != op) { return false; }
        }
    } else {
        if (first->operand_count() != 1u) { return false; }
        auto op = static_cast<CastInst *>(first)->op();
        if (op != CastOp::STATIC_CAST && op != CastOp::BITWISE_CAST) { return false; }
        for (auto *root : roots) {
            if (static_cast<CastInst *>(root)->op() != op) { return false; }
        }
    }
    for (size_t operand_index = 0u; operand_index < first->operand_count(); operand_index++) {
        auto *operand_type = first->operand(operand_index)->type();
        if (operand_type == nullptr || !operand_type->is_scalar()) { return false; }
        for (auto *root : roots) {
            auto *operand = root->operand(operand_index);
            if (operand == nullptr || operand->is_lvalue() || operand->type() != operand_type) { return false; }
        }
    }
    return true;
}

[[nodiscard]] static bool vectorize_store_seed(
    FunctionDefinition *def,
    const luisa::vector<StoreInst *> &stores) noexcept {
    luisa::vector<Instruction *> roots;
    if (!collect_vectorizable_roots(def, stores, roots)) { return false; }
    auto *first = roots.front();
    auto lane_count = stores.size();
    auto *vector_type = Type::vector(first->type(), lane_count);
    if (vector_type == nullptr) { return false; }
    luisa::vector<const Type *> operand_vector_types;
    operand_vector_types.reserve(first->operand_count());
    for (size_t operand_index = 0u; operand_index < first->operand_count(); operand_index++) {
        auto *operand_vector_type = Type::vector(first->operand(operand_index)->type(), lane_count);
        if (operand_vector_type == nullptr) { return false; }
        operand_vector_types.emplace_back(operand_vector_type);
    }
    XIRBuilder builder;
    builder.set_insertion_point(stores.front()->prev());
    luisa::vector<Value *> vector_operands;
    vector_operands.reserve(first->operand_count());
    for (size_t operand_index = 0u; operand_index < first->operand_count(); operand_index++) {
        luisa::vector<Value *> scalar_operands;
        scalar_operands.reserve(lane_count);
        for (auto *root : roots) {
            scalar_operands.emplace_back(root->operand(operand_index));
        }
        vector_operands.emplace_back(
            builder.call(operand_vector_types[operand_index], ArithmeticOp::AGGREGATE, scalar_operands));
    }
    Instruction *vectorized = nullptr;
    if (first->isa<ArithmeticInst>()) {
        vectorized = builder.call(
            vector_type, static_cast<ArithmeticInst *>(first)->op(), vector_operands);
    } else {
        vectorized = builder.cast_(
            vector_type, static_cast<CastInst *>(first)->op(), vector_operands.front());
    }
    auto *module = def->parent_module();
    for (size_t lane = 0u; lane < lane_count; lane++) {
        auto lane_index = static_cast<uint32_t>(lane);
        auto *index = module->create_constant(Type::of<uint32_t>(), &lane_index);
        auto *extract = builder.call(
            first->type(), ArithmeticOp::EXTRACT, {vectorized, index});
        stores[lane]->set_value(extract);
    }
    for (auto *root : roots) {
        static_cast<void>(root->remove_self());
    }
    return true;
}

}// namespace detail

SLPVectorizationInfo slp_vectorization_pass_run_on_function(Function *function) noexcept {
    auto def = function->definition();
    if (!def) return {};
    SLPVectorizationInfo info;
    auto seeds = detail::collect_store_seeds(def);
    for (auto &chain : seeds) {
        if (detail::vectorize_store_seed(def, chain)) {
            info.vectorized_tree_count++;
            info.vectorized_inst_count += chain.size();
        } else {
            info.rejected_candidate_count++;
        }
    }
    return info;
}

SLPVectorizationInfo slp_vectorization_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SLPVectorizationInfo info;
    for (auto *f : module->function_list()) {
        auto func_info = slp_vectorization_pass_run_on_function(f);
        info.vectorized_tree_count += func_info.vectorized_tree_count;
        info.vectorized_inst_count += func_info.vectorized_inst_count;
        info.rejected_candidate_count += func_info.rejected_candidate_count;
    }
    if (report) {
        report->set("vectorized_tree_count", info.vectorized_tree_count);
        report->set("vectorized_inst_count", info.vectorized_inst_count);
        report->set("rejected_candidate_count", info.rejected_candidate_count);
    }
    return info;
}

}// namespace luisa::compute::xir
