#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/undefined.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir {

namespace detail {

// Check if a Constant has a specific value
[[nodiscard]] static bool is_const_value(const Value *v, int32_t expected) noexcept {
    if (!v->isa<Constant>()) return false;
    auto c = static_cast<const Constant *>(v);
    auto t = c->type();
    auto check_scalar = [expected](const Type *st, const void *data) noexcept {
        if (st->is_int32()) return *static_cast<const int32_t *>(data) == expected;
        if (st->is_uint32()) return static_cast<int32_t>(*static_cast<const uint32_t *>(data)) == expected;
        if (st->is_float32()) return *static_cast<const float *>(data) == static_cast<float>(expected);
        return false;
    };
    if (t->is_scalar()) return check_scalar(t, c->data());
    if (t->is_vector()) {
        auto elem = t->element();
        auto stride = elem->size();
        auto base = static_cast<const std::byte *>(c->data());
        for (size_t i = 0; i < t->dimension(); ++i) {
            if (!check_scalar(elem, base + i * stride)) return false;
        }
        return true;
    }
    return false;
}

[[nodiscard]] static bool is_const_zero(const Value *v) noexcept { return is_const_value(v, 0); }
[[nodiscard]] static bool is_const_one(const Value *v) noexcept { return is_const_value(v, 1); }

// Try to simplify an arithmetic instruction. Returns replacement Value or nullptr.
[[nodiscard]] static Value *try_simplify(ArithmeticInst *inst, Module *module, XIRBuilder &builder) noexcept {
    auto op = inst->op();
    auto type = inst->type();
    if (type == nullptr) return nullptr;

    switch (op) {
        case ArithmeticOp::BINARY_ADD: {
            // x + 0 → x  (integer only; float: +0 may change sign)
            if (!type->is_float_or_float_vector()) {
                if (is_const_zero(inst->operand(1))) return inst->operand(0);
                if (is_const_zero(inst->operand(0))) return inst->operand(1);
            }
            break;
        }
        case ArithmeticOp::BINARY_SUB: {
            // x - 0 → x  (integer only)
            if (!type->is_float_or_float_vector()) {
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
            if (!type->is_float_or_float_vector()) {
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
            if (!type->is_float_or_float_vector()) {
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
        case ArithmeticOp::EXTRACT: {
            auto idx = inst->operand(1);
            if (!idx->isa<Constant>()) break;
            auto idx_val = static_cast<const Constant *>(idx)->as<uint32_t>();
            auto src = inst->operand(0);
            while (src->isa<Instruction>()) {
                auto src_inst = static_cast<Instruction *>(src);
                if (!src_inst->isa<ArithmeticInst>()) break;
                auto src_arith = static_cast<ArithmeticInst *>(src_inst);
                if (src_arith->op() == ArithmeticOp::AGGREGATE) {
                    if (idx_val < src_arith->operand_count()) {
                        return src_arith->operand(idx_val);
                    }
                    break;
                }
                if (src_arith->op() == ArithmeticOp::INSERT) {
                    auto insert_idx = src_arith->operand(2);
                    if (!insert_idx->isa<Constant>()) break;
                    auto insert_idx_val = static_cast<const Constant *>(insert_idx)->as<uint32_t>();
                    if (insert_idx_val == idx_val) {
                        return src_arith->operand(1);
                    }
                    src = src_arith->operand(0);
                    continue;
                }
                break;
            }
            if (src != inst->operand(0)) {
                inst->set_operand(0, src);
            }
            break;
        }
        case ArithmeticOp::AGGREGATE: {
            if (inst->operand_count() == 0) break;
            auto first_op = inst->operand(0);
            bool all_same = true;
            for (size_t i = 1; i < inst->operand_count(); ++i) {
                if (inst->operand(i) != first_op) { all_same = false; break; }
            }
            if (all_same && first_op->isa<Instruction>()) {
                auto first_inst = static_cast<Instruction *>(first_op);
                if (first_inst->isa<ArithmeticInst>()) {
                    auto first_arith = static_cast<ArithmeticInst *>(first_inst);
                    if (first_arith->op() == ArithmeticOp::EXTRACT &&
                        first_arith->operand(0)->type() == inst->type()) {
                        return first_arith->operand(0);
                    }
                }
            }
            if (!first_op->isa<Instruction>()) break;
            auto first_inst = static_cast<Instruction *>(first_op);
            if (!first_inst->isa<ArithmeticInst>()) break;
            auto first_arith = static_cast<ArithmeticInst *>(first_inst);
            if (first_arith->op() != ArithmeticOp::EXTRACT) break;
            if (first_arith->operand_count() < 2) break;
            auto common_src = first_arith->operand(0);
            if (common_src->type() != inst->type()) break;
            auto first_idx = first_arith->operand(1);
            if (!first_idx->isa<Constant>()) break;
            if (static_cast<const Constant *>(first_idx)->as<uint32_t>() != 0u) break;
            bool all_match = true;
            for (size_t i = 1; i < inst->operand_count(); ++i) {
                auto op_i = inst->operand(i);
                if (!op_i->isa<Instruction>()) { all_match = false; break; }
                auto op_inst = static_cast<Instruction *>(op_i);
                if (!op_inst->isa<ArithmeticInst>()) { all_match = false; break; }
                auto op_arith = static_cast<ArithmeticInst *>(op_inst);
                if (op_arith->op() != ArithmeticOp::EXTRACT) { all_match = false; break; }
                if (op_arith->operand(0) != common_src) { all_match = false; break; }
                auto op_idx = op_arith->operand(1);
                if (!op_idx->isa<Constant>()) { all_match = false; break; }
                if (static_cast<const Constant *>(op_idx)->as<uint32_t>() != static_cast<uint32_t>(i)) { all_match = false; break; }
            }
            if (all_match) return common_src;
            break;
        }
        case ArithmeticOp::INSERT: {
            auto base = inst->operand(0);
            auto val = inst->operand(1);
            auto idx = inst->operand(2);
            if (!idx->isa<Constant>()) break;
            auto idx_val = static_cast<const Constant *>(idx)->as<uint32_t>();
            if (base->isa<Instruction>()) {
                auto base_inst = static_cast<Instruction *>(base);
                if (base_inst->isa<ArithmeticInst>()) {
                    auto base_arith = static_cast<ArithmeticInst *>(base_inst);
                    if (base_arith->op() == ArithmeticOp::INSERT) {
                        auto inner_idx = base_arith->operand(2);
                        if (inner_idx->isa<Constant>()) {
                            auto inner_idx_val = static_cast<const Constant *>(inner_idx)->as<uint32_t>();
                            if (inner_idx_val == idx_val) {
                                inst->set_operand(0, base_arith->operand(0));
                                return nullptr;
                            }
                        }
                    }
                }
            }
            if (inst->type() != nullptr && (inst->type()->is_vector() || inst->type()->is_array())) {
                auto dim = inst->type()->dimension();
                if (idx_val == dim - 1u) {
                    luisa::vector<Value *> elems(dim, nullptr);
                    elems[idx_val] = val;
                    auto cur = base;
                    bool valid = true;
                    for (auto slot = static_cast<int32_t>(dim) - 2; slot >= 0; --slot) {
                        if (cur->isa<Undefined>()) {
                            valid = false;
                            break;
                        }
                        if (!cur->isa<Instruction>()) { valid = false; break; }
                        auto cur_inst = static_cast<Instruction *>(cur);
                        if (!cur_inst->isa<ArithmeticInst>()) { valid = false; break; }
                        auto cur_arith = static_cast<ArithmeticInst *>(cur_inst);
                        if (cur_arith->op() != ArithmeticOp::INSERT) { valid = false; break; }
                        auto ci = cur_arith->operand(2);
                        if (!ci->isa<Constant>()) { valid = false; break; }
                        auto ci_val = static_cast<const Constant *>(ci)->as<uint32_t>();
                        if (ci_val != static_cast<uint32_t>(slot)) { valid = false; break; }
                        elems[slot] = cur_arith->operand(1);
                        cur = cur_arith->operand(0);
                    }
                    if (valid && cur->isa<Undefined>()) {
                        bool all_filled = true;
                        for (auto e : elems) { if (!e) { all_filled = false; break; } }
                        if (all_filled) {
                            builder.set_insertion_point(inst);
                            return builder.call(inst->type(), ArithmeticOp::AGGREGATE, elems);
                        }
                    }
                }
            }
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
    XIRBuilder builder;

    luisa::vector<ArithmeticInst *> to_simplify;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ArithmeticInst>()) {
            to_simplify.push_back(static_cast<ArithmeticInst *>(inst));
        }
    });

    for (auto inst : to_simplify) {
        auto replacement = try_simplify(inst, module, builder);
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
