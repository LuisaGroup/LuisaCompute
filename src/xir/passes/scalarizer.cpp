#include <luisa/xir/passes/scalarizer.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/constant.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir {

namespace detail {

// Key for caching scalarized component values.
struct ScalarKey {
    const Value *vector_value;
    uint32_t component;

    bool operator==(const ScalarKey &other) const noexcept {
        return vector_value == other.vector_value && component == other.component;
    }
};

struct ScalarKeyHash {
    size_t operator()(const ScalarKey &k) const noexcept {
        auto h = std::hash<const Value *>{}(k.vector_value);
        h ^= std::hash<uint32_t>{}(k.component) + 0x9e3779b9ULL + (h << 6U) + (h >> 2U);
        return h;
    }
};

// Check if an instruction is a candidate: vector-typed ArithmeticInst or CastInst.
[[nodiscard]] static bool is_vector_arith_or_cast(const Instruction *inst) noexcept {
    return inst->type()->is_vector() &&
           (inst->isa<ArithmeticInst>() || inst->isa<CastInst>());
}

// Check if an ArithmeticOp is safe to scalarize per-component.
// Some ops like CROSS, NORMALIZE, REFLECT, etc. do not decompose
// into independent per-component operations.
[[nodiscard]] static bool is_per_component_op(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::CROSS:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::FACEFORWARD:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_LINALG_MUL:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE:
        case ArithmeticOp::AGGREGATE:
        case ArithmeticOp::SHUFFLE:
        case ArithmeticOp::INSERT:
        case ArithmeticOp::EXTRACT:
            return false;
        default:
            return true;
    }
}

// Compute the set of scalarizable instructions using fixed-point iteration.
// An instruction is scalarizable if:
// - It is an ArithmeticInst or CastInst with vector type, AND
// - Its op is per-component safe (for ArithmeticInst), AND
// - All uses are from instructions that are also scalarizable, OR it has no uses.
[[nodiscard]] static luisa::unordered_set<const Instruction *>
compute_scalarizable_set(FunctionDefinition *def) noexcept {

    // Collect all candidate instructions.
    luisa::vector<Instruction *> candidates;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        auto ty = inst->type();
        if (ty == nullptr || !ty->is_vector()) return;
        if (inst->isa<ArithmeticInst>()) {
            auto arith = static_cast<ArithmeticInst *>(inst);
            if (is_per_component_op(arith->op())) {
                candidates.push_back(inst);
            }
        } else if (inst->isa<CastInst>()) {
            candidates.push_back(inst);
        }
    });

    // Fixed-point iteration: an instruction is scalarizable if
    // all its users are also scalarizable (or it has no users).
    luisa::unordered_set<const Instruction *> scalarizable;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto inst : candidates) {
            if (scalarizable.count(inst)) continue;

            bool all_users_scalarizable = true;
            bool has_use = false;
            for (auto &&use : inst->use_list()) {
                has_use = true;
                auto user = use->user();
                if (user == nullptr || !user->isa<Instruction>()) {
                    all_users_scalarizable = false;
                    break;
                }
                auto user_inst = static_cast<const Instruction *>(user);
                if (!scalarizable.count(user_inst)) {
                    all_users_scalarizable = false;
                    break;
                }
            }
            if (!has_use || all_users_scalarizable) {
                scalarizable.insert(inst);
                changed = true;
            }
        }
    }

    return scalarizable;
}

// Create a scalar constant for an index value.
[[nodiscard]] static Constant *make_index_constant(Module *module, uint32_t idx) noexcept {
    return module->create_constant(Type::of<uint32_t>(), &idx);
}

// Get or create the scalar value for a given component of a vector value.
[[nodiscard]] static Value *get_or_create_scalar_component(
    Value *vector_val, uint32_t component,
    luisa::unordered_map<ScalarKey, Value *, ScalarKeyHash> &scalar_map,
    XIRBuilder &builder, Module *module) noexcept {

    ScalarKey key{vector_val, component};
    auto it = scalar_map.find(key);
    if (it != scalar_map.end()) {
        return it->second;
    }

    // Create an extract instruction to get the component.
    auto elem_type = vector_val->type()->element();
    auto idx_const = make_index_constant(module, component);
    auto extract = builder.call(elem_type, ArithmeticOp::EXTRACT,
                                {vector_val, idx_const});
    scalar_map[key] = extract;
    return extract;
}

// Build scalar operands for a given component of a vector instruction.
[[nodiscard]] static luisa::vector<Value *>
build_scalar_operands(Instruction *inst, uint32_t component,
                      luisa::unordered_map<ScalarKey, Value *, ScalarKeyHash> &scalar_map,
                      XIRBuilder &builder, Module *module) noexcept {

    luisa::vector<Value *> scalars;
    size_t n = inst->operand_count();
    scalars.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        auto op = inst->operand(i);
        if (op->type()->is_vector()) {
            scalars.push_back(
                get_or_create_scalar_component(op, component, scalar_map, builder, module));
        } else {
            // Scalar operand (e.g., shift amount for BINARY_SHIFT_LEFT).
            scalars.push_back(op);
        }
    }
    return scalars;
}

static void scalarize_on_function(FunctionDefinition *def, ScalarizerInfo &info) noexcept {
    auto module = def->parent_module();
    if (module == nullptr) return;
    auto scalarizable = compute_scalarizable_set(def);
    if (scalarizable.empty()) return;

    XIRBuilder builder;
    luisa::unordered_map<ScalarKey, Value *, ScalarKeyHash> scalar_map;

    // Gather scalarizable instructions in traversal order.
    luisa::vector<Instruction *> worklist;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (scalarizable.count(inst)) {
            worklist.push_back(inst);
        }
    });

    // Phase 1: Create scalar versions. Process in traversal order;
    // if an operand hasn't been scalarized yet, we create extracts
    // from the original vector value (which is still alive).
    for (auto inst : worklist) {
        auto vec_type = inst->type();
        auto elem_type = vec_type->element();
        auto dim = vec_type->dimension();

        // Dead instructions: skip scalar creation, just remove later.
        if (inst->use_list().empty()) continue;

        builder.set_insertion_point(inst);

        for (uint32_t c = 0; c < dim; ++c) {
            auto scalar_ops = build_scalar_operands(inst, c, scalar_map, builder, module);

            Value *scalar_result = nullptr;
            if (inst->isa<ArithmeticInst>()) {
                auto arith = static_cast<ArithmeticInst *>(inst);
                scalar_result = builder.call(elem_type, arith->op(), scalar_ops);
            } else if (inst->isa<CastInst>()) {
                auto cast = static_cast<CastInst *>(inst);
                LUISA_ASSERT(scalar_ops.size() == 1,
                             "CastInst should have exactly one operand.");
                scalar_result = builder.cast_(elem_type, cast->op(), scalar_ops[0]);
            }

            ScalarKey key{inst, c};
            scalar_map[key] = scalar_result;
        }
        info.scalarized_inst_count++;
    }

    // Phase 2: Remove original vector instructions.
    // Keep them alive via ManagedPtr so that extracts created in Phase 1
    // (which may reference them via raw operand pointers) remain valid.
    luisa::vector<ManagedPtr<Instruction>> kept_alive;
    for (auto inst : worklist) {
        kept_alive.push_back(inst->remove_self());
    }
}

} // namespace detail

ScalarizerInfo scalarizer_pass_run_on_function(FunctionDefinition *def) noexcept {
    ScalarizerInfo info;
    detail::scalarize_on_function(def, info);
    return info;
}

ScalarizerInfo scalarizer_pass_run_on_module(Module *module, PassReport *report) noexcept {
    ScalarizerInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            detail::scalarize_on_function(def, info);
        }
    }
    if (report != nullptr) {
        report->set("scalarized_inst", info.scalarized_inst_count);
    }
    return info;
}

} // namespace luisa::compute::xir
