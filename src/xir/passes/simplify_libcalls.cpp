#include <luisa/xir/passes/simplify_libcalls.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/constant.h>

namespace luisa::compute::xir {

namespace detail {

// Check if a Constant has a specific float value (scalar or uniform vector).
[[nodiscard]] static bool is_const_float_value(const Value *v, float expected) noexcept {
    if (!v->isa<Constant>()) return false;
    auto c = static_cast<const Constant *>(v);
    auto t = c->type();
    if (t->is_float32()) return *static_cast<const float *>(c->data()) == expected;
    if (t->is_float64()) return *static_cast<const double *>(c->data()) == static_cast<double>(expected);
    if (t->is_vector()) {
        auto elem = t->element();
        auto stride = elem->size();
        auto base = static_cast<const std::byte *>(c->data());
        if (elem->is_float32()) {
            for (size_t i = 0; i < t->dimension(); ++i) {
                if (*reinterpret_cast<const float *>(base + i * stride) != expected) return false;
            }
            return true;
        }
        if (elem->is_float64()) {
            auto de = static_cast<double>(expected);
            for (size_t i = 0; i < t->dimension(); ++i) {
                if (*reinterpret_cast<const double *>(base + i * stride) != de) return false;
            }
            return true;
        }
    }
    return false;
}

[[nodiscard]] static bool is_const_float_zero(const Value *v) noexcept { return is_const_float_value(v, 0.0f); }
[[nodiscard]] static bool is_const_float_one(const Value *v) noexcept { return is_const_float_value(v, 1.0f); }

/// Attempt to simplify a single ArithmeticInst.
/// Returns a replacement Value* if simplification applies, nullptr otherwise.
[[nodiscard]] static Value *try_simplify(ArithmeticInst *inst, Module *module, XIRBuilder &builder) noexcept {
    auto op = inst->op();
    auto type = inst->type();
    if (type == nullptr) return nullptr;

    switch (op) {

        case ArithmeticOp::LERP: {
            // LERP(x, y, 0.0) → x
            // LERP(x, y, 1.0) → y
            if (inst->operand_count() < 3) break;
            auto s = inst->operand(2);
            if (is_const_float_zero(s)) return inst->operand(0);
            if (is_const_float_one(s)) return inst->operand(1);
            break;
        }

        case ArithmeticOp::CLAMP: {
            // CLAMP(x, lo, hi) where lo == 0.0 and hi == 1.0 → SATURATE(x)
            if (inst->operand_count() < 3) break;
            auto lo = inst->operand(1);
            auto hi = inst->operand(2);
            if (is_const_float_zero(lo) && is_const_float_one(hi)) {
                builder.set_insertion_point(inst);
                return builder.call(type, ArithmeticOp::SATURATE, {inst->operand(0)});
            }
            break;
        }

        case ArithmeticOp::STEP: {
            // STEP(edge, x): returns (x >= edge) ? 1 : 0.
            // If edge is 0.0, then for all float x, x >= 0.0 → 1.0
            // (IEEE 754: -0.0 >= 0.0 is true)
            if (inst->operand_count() < 2) break;
            auto edge = inst->operand(0);
            if (is_const_float_zero(edge)) {
                return module->create_constant_one(type);
            }
            break;
        }

        case ArithmeticOp::ABS: {
            // ABS(x) where x is unsigned → x (no-op)
            // For uint types, abs is identity.
            if (inst->operand_count() < 1) break;
            auto x_type = inst->operand(0)->type();
            if (x_type != nullptr && (x_type->is_uint32() || x_type->is_uint64() ||
                                      x_type->is_uint16() || x_type->is_uint8())) {
                return inst->operand(0);
            }
            // If x_type is a uint vector, also identity.
            if (x_type != nullptr && x_type->is_vector()) {
                auto elem = x_type->element();
                if (elem->is_uint8() || elem->is_uint16() || elem->is_uint32() || elem->is_uint64()) {
                    return inst->operand(0);
                }
            }
            break;
        }

        case ArithmeticOp::SELECT: {
            // SELECT(cond, x, x) → x
            if (inst->operand_count() < 3) break;
            if (inst->operand(0) == inst->operand(1)) return inst->operand(0);
            break;
        }

        default:
            break;
    }

    return nullptr;
}

static void simplify_libcalls_on_function(Function *function, SimplifyLibCallsInfo &info) noexcept {
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
            info.simplified_count++;
        }
    }
}

}// namespace detail

SimplifyLibCallsInfo simplify_libcalls_pass_run_on_function(FunctionDefinition *def) noexcept {
    SimplifyLibCallsInfo info;
    if (def == nullptr || def->body_block() == nullptr) return info;
    detail::simplify_libcalls_on_function(def, info);
    return info;
}

SimplifyLibCallsInfo simplify_libcalls_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SimplifyLibCallsInfo info;
    if (module == nullptr) return info;
    for (auto f : module->function_list()) {
        detail::simplify_libcalls_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("simplified_count", info.simplified_count);
    }
    return info;
}

}// namespace luisa::compute::xir
