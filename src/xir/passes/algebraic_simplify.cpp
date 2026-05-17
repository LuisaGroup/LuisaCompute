#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/constant.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir {

namespace detail {

// Check if a Constant has a specific value
[[nodiscard]] static bool is_const_value(const Value *v, int32_t expected) noexcept {
    if (!v->isa<Constant>()) return false;
    auto c = static_cast<const Constant *>(v);
    if (c->type()->is_int32()) return c->as<int32_t>() == expected;
    if (c->type()->is_uint32()) return static_cast<int32_t>(c->as<uint32_t>()) == expected;
    if (c->type()->is_float32()) return c->as<float>() == static_cast<float>(expected);
    return false;
}

[[nodiscard]] static bool is_const_zero(const Value *v) noexcept { return is_const_value(v, 0); }
[[nodiscard]] static bool is_const_one(const Value *v) noexcept { return is_const_value(v, 1); }

[[nodiscard]] static bool is_float_type(const Type *t) noexcept {
    if (t->is_float32() || t->is_float64()) return true;
    if (t->is_vector()) return t->element()->is_float32() || t->element()->is_float64();
    return false;
}

// Try to simplify an arithmetic instruction. Returns replacement Value or nullptr.
[[nodiscard]] static Value *try_simplify(ArithmeticInst *inst, Module *module) noexcept {
    auto op = inst->op();
    auto type = inst->type();
    if (type == nullptr) return nullptr;

    switch (op) {
        case ArithmeticOp::BINARY_ADD: {
            // x + 0 → x  (integer only; float: +0 may change sign)
            if (!is_float_type(type)) {
                if (is_const_zero(inst->operand(1))) return inst->operand(0);
                if (is_const_zero(inst->operand(0))) return inst->operand(1);
            }
            break;
        }
        case ArithmeticOp::BINARY_SUB: {
            // x - 0 → x  (integer only)
            if (!is_float_type(type)) {
                if (is_const_zero(inst->operand(1))) return inst->operand(0);
                // x - x → 0  (integer only)
                if (inst->operand(0) == inst->operand(1)) {
                    return module->create_constant_zero(type);
                }
            }
            break;
        }
        case ArithmeticOp::BINARY_MUL: {
            // x * 1 → x  (safe for all types: IEEE754 preserves this for normal values)
            if (is_const_one(inst->operand(1))) return inst->operand(0);
            if (is_const_one(inst->operand(0))) return inst->operand(1);
            // x * 0 → 0  (integer only; float: NaN * 0 = NaN)
            if (!is_float_type(type)) {
                if (is_const_zero(inst->operand(0)) || is_const_zero(inst->operand(1))) {
                    return module->create_constant_zero(type);
                }
            }
            break;
        }
        case ArithmeticOp::BINARY_DIV: {
            // x / 1 → x  (safe for all types)
            if (is_const_one(inst->operand(1))) return inst->operand(0);
            // 0 / x → 0  (integer only; float: 0/0 = NaN)
            if (!is_float_type(type)) {
                if (is_const_zero(inst->operand(0))) return module->create_constant_zero(type);
            }
            break;
        }
        case ArithmeticOp::BINARY_BIT_AND: {
            // x & 0 → 0
            if (is_const_zero(inst->operand(0)) || is_const_zero(inst->operand(1)))
                return module->create_constant_zero(type);
            // x & -1 (all bits) → x (for uint32: 0xFFFFFFFF)
            break;
        }
        case ArithmeticOp::BINARY_BIT_OR: {
            // x | 0 → x
            if (is_const_zero(inst->operand(1))) return inst->operand(0);
            if (is_const_zero(inst->operand(0))) return inst->operand(1);
            break;
        }
        case ArithmeticOp::BINARY_BIT_XOR: {
            // x ^ 0 → x
            if (is_const_zero(inst->operand(1))) return inst->operand(0);
            if (is_const_zero(inst->operand(0))) return inst->operand(1);
            break;
        }
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT: {
            // x << 0 → x, x >> 0 → x
            if (is_const_zero(inst->operand(1))) return inst->operand(0);
            break;
        }
        case ArithmeticOp::UNARY_MINUS: {
            // -0 → 0 (for non-float)
            if (is_const_zero(inst->operand(0)) && !type->is_float())
                return inst->operand(0);
            break;
        }
        default:
            break;
    }
    return nullptr;
}

static void algebraic_simplify_on_function(Function *function, AlgebraicSimplifyInfo &info) noexcept {
    auto def = function->definition();
    if (!def) return;
    auto module = function->parent_module();

    luisa::vector<ArithmeticInst *> to_simplify;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ArithmeticInst>()) {
            to_simplify.push_back(static_cast<ArithmeticInst *>(inst));
        }
    });

    for (auto inst : to_simplify) {
        auto replacement = try_simplify(inst, module);
        if (replacement != nullptr) {
            inst->replace_all_uses_with(replacement);
            inst->remove_self();
            info.simplified_inst_count++;
        }
    }
}

}// namespace detail

AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_function(Function *function) noexcept {
    AlgebraicSimplifyInfo info;
    detail::algebraic_simplify_on_function(function, info);
    return info;
}

AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_module(Module *module) noexcept {
    AlgebraicSimplifyInfo info;
    for (auto f : module->function_list()) {
        detail::algebraic_simplify_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
