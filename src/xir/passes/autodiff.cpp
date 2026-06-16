#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/autodiff.h>
#include <algorithm>
#include <cmath>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] auto is_differentiable_type(const Type *type) noexcept -> bool {
    switch (type->tag()) {
        case Type::Tag::FLOAT16:
        case Type::Tag::FLOAT32:
        case Type::Tag::FLOAT64:
            return true;
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
            return is_differentiable_type(type->element());
        case Type::Tag::ARRAY:
            return is_differentiable_type(type->element());
        case Type::Tag::STRUCTURE: {
            for (auto m : type->members()) {
                if (is_differentiable_type(m)) { return true; }
            }
            return false;
        }
        default:
            return false;
    }
}

[[nodiscard]] auto bool_type_of(const Type *type) noexcept -> const Type * {
    return type->is_vector() ? Type::vector(Type::of<bool>(), type->dimension()) : Type::of<bool>();
}

struct TransformAdScope {
    Function *function{};
    FunctionDefinition *definition{};
    Module *module{};
    AutodiffScopeInst *scope{};
    luisa::unordered_map<Value *, AllocaInst *> grads;
    luisa::unordered_set<Value *> detached;
    luisa::unordered_set<Value *> forward_reachable;
    luisa::unordered_set<Value *> backward_reachable;
    luisa::vector<Instruction *> forward_instructions;
    luisa::vector<Instruction *> backward_emit_instructions;
    luisa::unordered_map<IfInst *, luisa::vector<Instruction *>> if_true_backward_emit_instructions;
    luisa::unordered_map<IfInst *, luisa::vector<Instruction *>> if_false_backward_emit_instructions;
    luisa::vector<AutodiffIntrinsicInst *> removable_intrinsics;
    luisa::vector<std::pair<Value *, Value *>> seeds;
    AutodiffIntrinsicInst *backward_marker{};
    BasicBlock *backward_marker_block{};
    BasicBlock *epilogue_block{};
    size_t changed_count{0u};

    [[nodiscard]] auto index(uint32_t i) noexcept -> Constant * {
        return module->create_constant(Type::of<uint32_t>(), &i);
    }

    [[nodiscard]] auto zero(const Type *type) noexcept -> Constant * {
        return module->create_constant_zero(type);
    }

    [[nodiscard]] auto one(const Type *type) noexcept -> Constant * {
        return module->create_constant_one(type);
    }

    [[nodiscard]] auto fp(XIRBuilder &b, const Type *type, double x) noexcept -> Value * {
        switch (type->tag()) {
            case Type::Tag::FLOAT32: {
                auto v = static_cast<float>(x);
                return module->create_constant(type, &v);
            }
            case Type::Tag::FLOAT64:
                return module->create_constant(type, &x);
            case Type::Tag::FLOAT16: {
                auto v = static_cast<half>(x);
                return module->create_constant(type, &v);
            }
            case Type::Tag::VECTOR:
            case Type::Tag::MATRIX: {
                auto s = fp(b, type->element(), x);
                return broadcast(b, type, s);
            }
            default:
                LUISA_ERROR_WITH_LOCATION("Invalid floating-point constant type {}.", type->description());
        }
    }

    [[nodiscard]] auto broadcast(XIRBuilder &b, const Type *type, Value *value) noexcept -> Value * {
        if (type == value->type()) { return value; }
        LUISA_ASSERT(type->is_vector() || type->is_matrix(), "Invalid broadcast target type {}.", type->description());
        luisa::fixed_vector<Value *, 16u> args;
        if (type->is_vector()) {
            args.reserve(type->dimension());
            value = b.static_cast_if_necessary(type->element(), value);
            for (auto i = 0u; i < type->dimension(); i++) { args.emplace_back(value); }
        } else {
            auto column_type = Type::vector(type->element(), type->dimension());
            auto column = broadcast(b, column_type, value);
            args.reserve(type->dimension());
            for (auto i = 0u; i < type->dimension(); i++) { args.emplace_back(column); }
        }
        return b.call(type, ArithmeticOp::AGGREGATE, args);
    }

    [[nodiscard]] auto add(XIRBuilder &b, const Type *type, Value *lhs, Value *rhs) noexcept -> Value * {
        auto op = type->is_matrix() ? ArithmeticOp::MATRIX_COMP_ADD : ArithmeticOp::BINARY_ADD;
        return b.call(type, op, {lhs, rhs});
    }

    [[nodiscard]] auto sub(XIRBuilder &b, const Type *type, Value *lhs, Value *rhs) noexcept -> Value * {
        auto op = type->is_matrix() ? ArithmeticOp::MATRIX_COMP_SUB : ArithmeticOp::BINARY_SUB;
        return b.call(type, op, {lhs, rhs});
    }

    [[nodiscard]] auto mul(XIRBuilder &b, const Type *type, Value *lhs, Value *rhs) noexcept -> Value * {
        auto op = type->is_matrix() ? ArithmeticOp::MATRIX_COMP_MUL : ArithmeticOp::BINARY_MUL;
        return b.call(type, op, {lhs, rhs});
    }

    [[nodiscard]] auto div(XIRBuilder &b, const Type *type, Value *lhs, Value *rhs) noexcept -> Value * {
        auto op = type->is_matrix() ? ArithmeticOp::MATRIX_COMP_DIV : ArithmeticOp::BINARY_DIV;
        return b.call(type, op, {lhs, rhs});
    }

    [[nodiscard]] auto neg(XIRBuilder &b, const Type *type, Value *value) noexcept -> Value * {
        auto op = type->is_matrix() ? ArithmeticOp::MATRIX_COMP_NEG : ArithmeticOp::UNARY_MINUS;
        return b.call(type, op, {value});
    }

    [[nodiscard]] auto select(XIRBuilder &b, const Type *type, Value *cond, Value *a, Value *z) noexcept -> Value * {
        return b.call(type, ArithmeticOp::SELECT, {z, a, cond});
    }

    [[nodiscard]] auto extract(XIRBuilder &b, const Type *type, Value *value, uint32_t i) noexcept -> Value * {
        return b.call(type, ArithmeticOp::EXTRACT, {value, index(i)});
    }

    [[nodiscard]] auto insert(XIRBuilder &b, const Type *type, Value *aggregate, Value *elem, uint32_t i) noexcept -> Value * {
        return b.call(type, ArithmeticOp::INSERT, {aggregate, elem, index(i)});
    }

    [[nodiscard]] auto grad_slot(Value *value) noexcept -> AllocaInst * {
        if (!is_differentiable_type(value->type())) { return nullptr; }
        if (value->isa<GEPInst>()) { return nullptr; }
        if (auto iter = grads.find(value); iter != grads.end()) { return iter->second; }
        XIRBuilder b;
        b.set_insertion_point(definition->body_block()->instructions().head_sentinel());
        auto slot = b.alloca_local(value->type());
        b.store(slot, zero(value->type()));
        grads.emplace(value, slot);
        return slot;
    }

    [[nodiscard]] auto load_grad(XIRBuilder &b, Value *value) noexcept -> Value * {
        if (value->isa<GEPInst>()) {
            auto gep = static_cast<GEPInst *>(value);
            auto base_grad = load_grad(b, gep->base());
            luisa::fixed_vector<Value *, 8u> args;
            args.emplace_back(base_grad);
            for (auto use : gep->index_uses()) { args.emplace_back(use->value()); }
            return b.call(value->type(), ArithmeticOp::EXTRACT, args);
        }
        auto slot = grad_slot(value);
        return slot == nullptr ? static_cast<Value *>(zero(value->type())) : b.load(value->type(), slot);
    }

    void accumulate_grad(XIRBuilder &b, Value *value, Value *grad) noexcept {
        if (value == nullptr || grad == nullptr || !is_differentiable_type(value->type())) { return; }
        LUISA_ASSERT(value->type() == grad->type(), "Gradient type mismatch: {} vs {}.", value->type()->description(), grad->type()->description());
        if (value->isa<GEPInst>()) {
            auto gep = static_cast<GEPInst *>(value);
            luisa::fixed_vector<Value *, 8u> indices;
            for (auto use : gep->index_uses()) { indices.emplace_back(use->value()); }
            auto base_grad = aggregate_zero_with_element(b, gep->base()->type(), grad, indices);
            accumulate_grad(b, gep->base(), base_grad);
            return;
        }
        auto slot = grad_slot(value);
        if (slot == nullptr) { return; }
        accumulate_into_lvalue(b, slot, grad);
    }

    void overwrite_grad(XIRBuilder &b, Value *value, Value *grad) noexcept {
        if (value == nullptr || grad == nullptr || !is_differentiable_type(value->type())) { return; }
        LUISA_ASSERT(value->type() == grad->type(), "Gradient type mismatch: {} vs {}.", value->type()->description(), grad->type()->description());
        if (value->isa<GEPInst>()) {
            auto gep = static_cast<GEPInst *>(value);
            luisa::fixed_vector<Value *, 8u> args;
            args.emplace_back(load_grad(b, gep->base()));
            args.emplace_back(grad);
            for (auto use : gep->index_uses()) { args.emplace_back(use->value()); }
            overwrite_grad(b, gep->base(), b.call(gep->base()->type(), ArithmeticOp::INSERT, args));
            return;
        }
        auto slot = grad_slot(value);
        if (slot != nullptr) { b.store(slot, grad); }
    }

    void clear_grad(XIRBuilder &b, Value *value) noexcept {
        overwrite_grad(b, value, zero(value->type()));
    }

    void accumulate_into_lvalue(XIRBuilder &b, Value *lvalue, Value *grad) noexcept {
        auto type = lvalue->type();
        LUISA_ASSERT(type == grad->type(), "Gradient type mismatch: {} vs {}.", type->description(), grad->type()->description());
        switch (type->tag()) {
            case Type::Tag::FLOAT16:
            case Type::Tag::FLOAT32:
            case Type::Tag::FLOAT64:
            case Type::Tag::VECTOR:
            case Type::Tag::MATRIX: {
                auto old = b.load(type, lvalue);
                auto next = add(b, type, old, grad);
                b.store(lvalue, next);
                break;
            }
            case Type::Tag::ARRAY: {
                auto elem_type = type->element();
                for (auto i = 0u; i < type->dimension(); i++) {
                    auto elem_lvalue = b.gep(elem_type, lvalue, {index(i)});
                    auto elem_grad = extract(b, elem_type, grad, i);
                    accumulate_into_lvalue(b, elem_lvalue, elem_grad);
                }
                break;
            }
            case Type::Tag::STRUCTURE: {
                auto members = type->members();
                for (auto i = 0u; i < members.size(); i++) {
                    if (!is_differentiable_type(members[i])) { continue; }
                    auto elem_lvalue = b.gep(members[i], lvalue, {index(static_cast<uint32_t>(i))});
                    auto elem_grad = extract(b, members[i], grad, static_cast<uint32_t>(i));
                    accumulate_into_lvalue(b, elem_lvalue, elem_grad);
                }
                break;
            }
            default:
                break;
        }
    }

    [[nodiscard]] auto aggregate_zero_with_element(XIRBuilder &b, const Type *type, Value *elem, luisa::span<Value *const> indices) noexcept -> Value * {
        luisa::fixed_vector<Value *, 8u> args;
        args.emplace_back(zero(type));
        args.emplace_back(elem);
        for (auto i : indices) { args.emplace_back(i); }
        return b.call(type, ArithmeticOp::INSERT, args);
    }

    void mark_backward_reachable(Value *value) noexcept {
        if (value == nullptr || !is_differentiable_type(value->type())) { return; }
        backward_reachable.emplace(value);
        if (value->isa<GEPInst>()) {
            auto gep = static_cast<GEPInst *>(value);
            mark_backward_reachable(gep->base());
        } else {
            static_cast<void>(grad_slot(value));
        }
    }

    [[nodiscard]] auto lvalue_backward_reachable(Value *value) const noexcept -> bool {
        if (backward_reachable.contains(value)) { return true; }
        if (value != nullptr && value->isa<GEPInst>()) {
            return lvalue_backward_reachable(static_cast<GEPInst *>(value)->base());
        }
        return false;
    }

    void process_forward_instruction(Instruction *inst) noexcept {
        if (auto intrinsic = inst->isa<AutodiffIntrinsicInst>() ? static_cast<AutodiffIntrinsicInst *>(inst) : nullptr) {
            switch (intrinsic->op()) {
                case AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT:
                    LUISA_ASSERT(intrinsic->operand_count() == 1u, "Invalid requires_gradient operand count.");
                    forward_reachable.emplace(intrinsic->operand(0));
                    static_cast<void>(grad_slot(intrinsic->operand(0)));
                    removable_intrinsics.emplace_back(intrinsic);
                    break;
                case AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER:
                    LUISA_ASSERT(intrinsic->operand_count() == 2u, "Invalid gradient_marker operand count.");
                    seeds.emplace_back(intrinsic->operand(0), intrinsic->operand(1));
                    mark_backward_reachable(intrinsic->operand(0));
                    removable_intrinsics.emplace_back(intrinsic);
                    break;
                case AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT:
                    removable_intrinsics.emplace_back(intrinsic);
                    break;
                case AutodiffIntrinsicOp::AUTODIFF_BACKWARD:
                    LUISA_ASSERT(backward_marker == nullptr, "Multiple backward calls in one autodiff scope.");
                    backward_marker = intrinsic;
                    backward_marker_block = intrinsic->parent_block();
                    removable_intrinsics.emplace_back(intrinsic);
                    break;
                case AutodiffIntrinsicOp::AUTODIFF_GRADIENT:
                case AutodiffIntrinsicOp::AUTODIFF_DETACH:
                    break;
            }
        }
        if (inst->isa<AutodiffIntrinsicInst>()) {
            auto intrinsic = static_cast<AutodiffIntrinsicInst *>(inst);
            if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_DETACH) {
                detached.emplace(intrinsic);
                intrinsic->replace_all_uses_with(intrinsic->operand(0));
                removable_intrinsics.emplace_back(intrinsic);
                return;
            }
        }
        if (auto gep = inst->isa<GEPInst>() ? static_cast<GEPInst *>(inst) : nullptr) {
            if (forward_reachable.contains(gep->base()) && is_differentiable_type(gep->type())) {
                forward_reachable.emplace(gep);
            }
        } else if (auto store = inst->isa<StoreInst>() ? static_cast<StoreInst *>(inst) : nullptr) {
            if (forward_reachable.contains(store->value()) && is_differentiable_type(store->variable()->type())) {
                forward_reachable.emplace(store->variable());
                static_cast<void>(grad_slot(store->variable()));
            }
        } else if (auto load = inst->isa<LoadInst>() ? static_cast<LoadInst *>(inst) : nullptr) {
            if (forward_reachable.contains(load->variable()) && is_differentiable_type(load->type())) {
                forward_reachable.emplace(load);
                static_cast<void>(grad_slot(load));
            }
        } else if (auto arith = inst->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst) : nullptr) {
            if (!detached.contains(arith) && is_differentiable_type(arith->type())) {
                for (auto use : arith->operand_uses()) {
                    if (forward_reachable.contains(use->value())) {
                        forward_reachable.emplace(arith);
                        static_cast<void>(grad_slot(arith));
                        break;
                    }
                }
            }
        }
    }

    [[nodiscard]] auto collect_forward(BasicBlock *block, BasicBlock *merge, luisa::unordered_set<BasicBlock *> &visited, luisa::vector<Instruction *> &emit_instructions) noexcept -> bool {
        if (block == merge || !visited.emplace(block).second) { return false; }
        for (auto inst : block->instructions()) {
            forward_instructions.emplace_back(inst);
            process_forward_instruction(inst);
            emit_instructions.emplace_back(inst);
            if (inst == backward_marker) { return true; }
        }
        if (auto if_inst = block->terminator(); if_inst != nullptr && if_inst->isa<IfInst>()) {
            auto structured_if = static_cast<IfInst *>(if_inst);
            auto &true_emit = if_true_backward_emit_instructions[structured_if];
            auto &false_emit = if_false_backward_emit_instructions[structured_if];
            auto found = collect_forward(structured_if->true_block(), structured_if->merge_block(), visited, true_emit);
            found |= collect_forward(structured_if->false_block(), structured_if->merge_block(), visited, false_emit);
            if (found) { return true; }
            return collect_forward(structured_if->merge_block(), merge, visited, emit_instructions);
        }
        if (block->terminator() != nullptr) {
            bool found = false;
            block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                if (!found) { found = collect_forward(succ, merge, visited, emit_instructions); }
            });
            if (found) { return true; }
        }
        return false;
    }

    void collect_backward() noexcept {
        for (auto it = forward_instructions.rbegin(); it != forward_instructions.rend(); ++it) {
            auto inst = *it;
            if (auto arith = inst->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst) : nullptr) {
                if (backward_reachable.contains(arith)) {
                    for (auto use : arith->operand_uses()) {
                        auto value = use->value();
                        if (forward_reachable.contains(value) && is_differentiable_type(value->type())) {
                            mark_backward_reachable(value);
                        }
                    }
                }
            } else if (auto load = inst->isa<LoadInst>() ? static_cast<LoadInst *>(inst) : nullptr) {
                if (backward_reachable.contains(load) && forward_reachable.contains(load->variable())) {
                    mark_backward_reachable(load->variable());
                }
            } else if (auto store = inst->isa<StoreInst>() ? static_cast<StoreInst *>(inst) : nullptr) {
                if (lvalue_backward_reachable(store->variable()) && forward_reachable.contains(store->value())) {
                    mark_backward_reachable(store->value());
                }
            }
        }
    }

    void lower_epilogue_gradients(BasicBlock *block, BasicBlock *merge, luisa::unordered_set<BasicBlock *> &visited) noexcept {
        if (block == merge || !visited.emplace(block).second) { return; }
        luisa::vector<AutodiffIntrinsicInst *> gradients;
        for (auto inst : block->instructions()) {
            if (inst->isa<AutodiffIntrinsicInst>()) {
                auto intrinsic = static_cast<AutodiffIntrinsicInst *>(inst);
                if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_GRADIENT) {
                    gradients.emplace_back(intrinsic);
                } else if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_DETACH) {
                    intrinsic->replace_all_uses_with(intrinsic->operand(0));
                    gradients.emplace_back(intrinsic);
                }
            }
            if (auto if_inst = inst->isa<IfInst>() ? static_cast<IfInst *>(inst) : nullptr) {
                lower_epilogue_gradients(if_inst->true_block(), if_inst->merge_block(), visited);
                lower_epilogue_gradients(if_inst->false_block(), if_inst->merge_block(), visited);
                lower_epilogue_gradients(if_inst->merge_block(), merge, visited);
            }
        }
        for (auto intrinsic : gradients) {
            XIRBuilder b;
            b.set_insertion_point(intrinsic->prev());
            if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_GRADIENT) {
                auto value = intrinsic->operand(0);
                auto grad = load_grad(b, value);
                intrinsic->replace_all_uses_with(grad);
            }
            intrinsic->remove_self();
            changed_count++;
        }
    }

    void seed_gradients(XIRBuilder &b) noexcept {
        for (auto [value, grad] : seeds) {
            accumulate_grad(b, value, grad);
        }
    }

    void backward_inst(XIRBuilder &b, Instruction *inst) noexcept {
        if (auto arith = inst->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst) : nullptr) {
            backward_arithmetic(b, arith);
        } else if (auto load = inst->isa<LoadInst>() ? static_cast<LoadInst *>(inst) : nullptr) {
            if (backward_reachable.contains(load)) {
                accumulate_grad(b, load->variable(), load_grad(b, load));
            }
        } else if (auto store = inst->isa<StoreInst>() ? static_cast<StoreInst *>(inst) : nullptr) {
            if (lvalue_backward_reachable(store->variable())) {
                auto g = load_grad(b, store->variable());
                accumulate_grad(b, store->value(), g);
                clear_grad(b, store->variable());
            }
        } else if (auto intrinsic = inst->isa<AutodiffIntrinsicInst>() ? static_cast<AutodiffIntrinsicInst *>(inst) : nullptr) {
            if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT) {
                accumulate_grad(b, intrinsic->operand(0), intrinsic->operand(1));
            }
        }
    }

    void backward_arithmetic(XIRBuilder &b, ArithmeticInst *inst) noexcept {
        if (!backward_reachable.contains(inst) || !is_differentiable_type(inst->type())) { return; }
        auto type = inst->type();
        auto out_grad = load_grad(b, inst);
        auto arg = [&](size_t i) noexcept { return inst->operand(i); };
        auto accum = [&](Value *value, Value *grad) noexcept {
            if (forward_reachable.contains(value)) { accumulate_grad(b, value, grad); }
        };
        switch (inst->op()) {
            case ArithmeticOp::BINARY_ADD:
            case ArithmeticOp::MATRIX_COMP_ADD:
                accum(arg(0), out_grad);
                accum(arg(1), out_grad);
                break;
            case ArithmeticOp::BINARY_SUB:
            case ArithmeticOp::MATRIX_COMP_SUB:
                accum(arg(0), out_grad);
                accum(arg(1), neg(b, arg(1)->type(), out_grad));
                break;
            case ArithmeticOp::UNARY_MINUS:
            case ArithmeticOp::MATRIX_COMP_NEG:
                accum(arg(0), neg(b, arg(0)->type(), out_grad));
                break;
            case ArithmeticOp::BINARY_MUL:
            case ArithmeticOp::MATRIX_COMP_MUL:
                accum(arg(0), mul(b, arg(0)->type(), out_grad, arg(1)));
                accum(arg(1), mul(b, arg(1)->type(), out_grad, arg(0)));
                break;
            case ArithmeticOp::MATRIX_LINALG_MUL: {
                if (arg(0)->type()->is_matrix() && arg(1)->type()->is_vector()) {
                    auto lhs_grad = b.call(arg(0)->type(), ArithmeticOp::OUTER_PRODUCT, {out_grad, arg(1)});
                    auto lhs_t = b.call(arg(0)->type(), ArithmeticOp::MATRIX_TRANSPOSE, {arg(0)});
                    auto rhs_grad = b.call(arg(1)->type(), ArithmeticOp::MATRIX_LINALG_MUL, {lhs_t, out_grad});
                    accum(arg(0), lhs_grad);
                    accum(arg(1), rhs_grad);
                } else if (arg(0)->type()->is_matrix() && arg(1)->type()->is_matrix()) {
                    auto rhs_t = b.call(arg(1)->type(), ArithmeticOp::MATRIX_TRANSPOSE, {arg(1)});
                    auto lhs_grad = b.call(arg(0)->type(), ArithmeticOp::MATRIX_LINALG_MUL, {out_grad, rhs_t});
                    auto lhs_t = b.call(arg(0)->type(), ArithmeticOp::MATRIX_TRANSPOSE, {arg(0)});
                    auto rhs_grad = b.call(arg(1)->type(), ArithmeticOp::MATRIX_LINALG_MUL, {lhs_t, out_grad});
                    accum(arg(0), lhs_grad);
                    accum(arg(1), rhs_grad);
                }
                break;
            }
            case ArithmeticOp::BINARY_DIV:
            case ArithmeticOp::MATRIX_COMP_DIV: {
                auto lhs_grad = div(b, arg(0)->type(), out_grad, arg(1));
                auto neg_lhs = neg(b, arg(0)->type(), arg(0));
                auto sqr_rhs = mul(b, arg(1)->type(), arg(1), arg(1));
                auto rhs_factor = div(b, arg(1)->type(), neg_lhs, sqr_rhs);
                auto rhs_grad = mul(b, arg(1)->type(), out_grad, rhs_factor);
                accum(arg(0), lhs_grad);
                accum(arg(1), rhs_grad);
                break;
            }
            case ArithmeticOp::SELECT: {
                auto cond = arg(2);
                auto zero_value = zero(type);
                accum(arg(0), select(b, type, cond, zero_value, out_grad));
                accum(arg(1), select(b, type, cond, out_grad, zero_value));
                break;
            }
            case ArithmeticOp::MIN: {
                auto cond = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), arg(1)});
                auto zero_value = zero(type);
                accum(arg(0), select(b, type, cond, out_grad, zero_value));
                accum(arg(1), select(b, type, cond, zero_value, out_grad));
                break;
            }
            case ArithmeticOp::MAX: {
                auto cond = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {arg(0), arg(1)});
                auto zero_value = zero(type);
                accum(arg(0), select(b, type, cond, out_grad, zero_value));
                accum(arg(1), select(b, type, cond, zero_value, out_grad));
                break;
            }
            case ArithmeticOp::CLAMP: {
                auto max_x_a = b.call(type, ArithmeticOp::MAX, {arg(0), arg(1)});
                auto cond_min = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {max_x_a, arg(2)});
                auto zero_value = zero(type);
                auto min_grad = select(b, type, cond_min, out_grad, zero_value);
                auto b_grad = select(b, type, cond_min, zero_value, out_grad);
                auto cond_max = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {arg(0), arg(1)});
                auto x_grad = select(b, type, cond_max, min_grad, zero_value);
                auto a_grad = select(b, type, cond_max, zero_value, min_grad);
                accum(arg(0), x_grad);
                accum(arg(1), a_grad);
                accum(arg(2), b_grad);
                break;
            }
            case ArithmeticOp::SATURATE: {
                auto zero_value = zero(type);
                auto one_value = one(type);
                auto gt_zero = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {arg(0), zero_value});
                auto lt_one = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), one_value});
                auto between = b.call(bool_type_of(type), ArithmeticOp::BINARY_BIT_AND, {gt_zero, lt_one});
                accum(arg(0), select(b, type, between, out_grad, zero_value));
                break;
            }
            case ArithmeticOp::LERP: {
                auto b_minus_a = sub(b, type, arg(1), arg(0));
                auto b_minus_a_grad = mul(b, type, out_grad, arg(2));
                auto t_grad = mul(b, type, out_grad, b_minus_a);
                auto b_grad = b_minus_a_grad;
                auto a_grad = add(b, type, neg(b, type, b_minus_a_grad), out_grad);
                accum(arg(0), a_grad);
                accum(arg(1), b_grad);
                accum(arg(2), t_grad);
                break;
            }
            case ArithmeticOp::ABS: {
                auto zero_value = zero(type);
                auto cond = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER_EQUAL, {arg(0), zero_value});
                accum(arg(0), select(b, type, cond, out_grad, neg(b, type, out_grad)));
                break;
            }
            case ArithmeticOp::SIN: {
                auto c = b.call(type, ArithmeticOp::COS, {arg(0)});
                accum(arg(0), mul(b, type, c, out_grad));
                break;
            }
            case ArithmeticOp::COS: {
                auto s = b.call(type, ArithmeticOp::SIN, {arg(0)});
                accum(arg(0), mul(b, type, neg(b, type, s), out_grad));
                break;
            }
            case ArithmeticOp::TAN: {
                auto c = b.call(type, ArithmeticOp::COS, {arg(0)});
                auto c2 = mul(b, type, c, c);
                accum(arg(0), div(b, type, out_grad, c2));
                break;
            }
            case ArithmeticOp::SINH: {
                auto c = b.call(type, ArithmeticOp::COSH, {arg(0)});
                accum(arg(0), mul(b, type, c, out_grad));
                break;
            }
            case ArithmeticOp::COSH: {
                auto s = b.call(type, ArithmeticOp::SINH, {arg(0)});
                accum(arg(0), mul(b, type, s, out_grad));
                break;
            }
            case ArithmeticOp::TANH: {
                auto c = b.call(type, ArithmeticOp::COSH, {arg(0)});
                auto c2 = mul(b, type, c, c);
                accum(arg(0), div(b, type, out_grad, c2));
                break;
            }
            case ArithmeticOp::ASIN: {
                auto x2 = mul(b, type, arg(0), arg(0));
                auto denom = b.call(type, ArithmeticOp::SQRT, {sub(b, type, one(type), x2)});
                accum(arg(0), div(b, type, out_grad, denom));
                break;
            }
            case ArithmeticOp::ACOS: {
                auto x2 = mul(b, type, arg(0), arg(0));
                auto denom = b.call(type, ArithmeticOp::SQRT, {sub(b, type, one(type), x2)});
                accum(arg(0), neg(b, type, div(b, type, out_grad, denom)));
                break;
            }
            case ArithmeticOp::ATAN: {
                auto x2 = mul(b, type, arg(0), arg(0));
                accum(arg(0), div(b, type, out_grad, add(b, type, one(type), x2)));
                break;
            }
            case ArithmeticOp::ASINH: {
                auto x2 = mul(b, type, arg(0), arg(0));
                auto denom = b.call(type, ArithmeticOp::SQRT, {add(b, type, one(type), x2)});
                accum(arg(0), div(b, type, out_grad, denom));
                break;
            }
            case ArithmeticOp::ACOSH: {
                auto x2 = mul(b, type, arg(0), arg(0));
                auto denom = b.call(type, ArithmeticOp::SQRT, {sub(b, type, x2, one(type))});
                accum(arg(0), div(b, type, out_grad, denom));
                break;
            }
            case ArithmeticOp::ATANH: {
                auto x2 = mul(b, type, arg(0), arg(0));
                accum(arg(0), div(b, type, out_grad, sub(b, type, one(type), x2)));
                break;
            }
            case ArithmeticOp::ATAN2: {
                auto y = arg(0);
                auto x = arg(1);
                auto xx = mul(b, x->type(), x, x);
                auto yy = mul(b, y->type(), y, y);
                auto sum = add(b, x->type(), xx, yy);
                accum(y, mul(b, y->type(), div(b, y->type(), x, sum), out_grad));
                accum(x, mul(b, x->type(), div(b, x->type(), neg(b, x->type(), y), sum), out_grad));
                break;
            }
            case ArithmeticOp::EXP:
                accum(arg(0), mul(b, type, inst, out_grad));
                break;
            case ArithmeticOp::EXP2: {
                auto factor = mul(b, type, fp(b, type, std::log(2.0)), inst);
                accum(arg(0), mul(b, type, factor, out_grad));
                break;
            }
            case ArithmeticOp::EXP10: {
                auto factor = mul(b, type, fp(b, type, std::log(10.0)), inst);
                accum(arg(0), mul(b, type, factor, out_grad));
                break;
            }
            case ArithmeticOp::LOG:
                accum(arg(0), div(b, type, out_grad, arg(0)));
                break;
            case ArithmeticOp::LOG2: {
                auto scale = fp(b, type, 1.0 / std::log(2.0));
                accum(arg(0), div(b, type, mul(b, type, out_grad, scale), arg(0)));
                break;
            }
            case ArithmeticOp::LOG10: {
                auto scale = fp(b, type, 1.0 / std::log(10.0));
                accum(arg(0), div(b, type, mul(b, type, out_grad, scale), arg(0)));
                break;
            }
            case ArithmeticOp::POW: {
                auto b_minus_one = sub(b, type, arg(1), one(type));
                auto pow_a = b.call(type, ArithmeticOp::POW, {arg(0), b_minus_one});
                auto a_grad = mul(b, type, mul(b, type, arg(1), pow_a), out_grad);
                auto log_a = b.call(type, ArithmeticOp::LOG, {arg(0)});
                auto b_grad = mul(b, type, mul(b, type, inst, log_a), out_grad);
                accum(arg(0), a_grad);
                accum(arg(1), b_grad);
                break;
            }
            case ArithmeticOp::SQRT: {
                auto denom = add(b, type, inst, inst);
                accum(arg(0), div(b, type, out_grad, denom));
                break;
            }
            case ArithmeticOp::RSQRT: {
                auto sqrt_x = b.call(type, ArithmeticOp::SQRT, {arg(0)});
                auto twice_x = add(b, type, arg(0), arg(0));
                auto denom = mul(b, type, twice_x, sqrt_x);
                accum(arg(0), div(b, type, neg(b, type, out_grad), denom));
                break;
            }
            case ArithmeticOp::FRACT:
                accum(arg(0), out_grad);
                break;
            case ArithmeticOp::FMA: {
                accum(arg(0), mul(b, type, arg(1), out_grad));
                accum(arg(1), mul(b, type, arg(0), out_grad));
                accum(arg(2), out_grad);
                break;
            }
            case ArithmeticOp::COPYSIGN: {
                auto zero_value = zero(type);
                auto x_neg = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), zero_value});
                auto y_neg = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(1), zero_value});
                auto same = b.call(bool_type_of(type), ArithmeticOp::BINARY_EQUAL, {x_neg, y_neg});
                accum(arg(0), select(b, type, same, out_grad, neg(b, type, out_grad)));
                break;
            }
            case ArithmeticOp::DOT: {
                auto lhs_mul_rhs = mul(b, arg(0)->type(), arg(0), arg(1));
                auto d_ab = broadcast(b, arg(0)->type(), out_grad);
                static_cast<void>(lhs_mul_rhs);
                accum(arg(0), mul(b, arg(0)->type(), d_ab, arg(1)));
                accum(arg(1), mul(b, arg(1)->type(), d_ab, arg(0)));
                break;
            }
            case ArithmeticOp::CROSS:
                accum(arg(0), b.call(arg(0)->type(), ArithmeticOp::CROSS, {arg(1), out_grad}));
                accum(arg(1), b.call(arg(1)->type(), ArithmeticOp::CROSS, {out_grad, arg(0)}));
                break;
            case ArithmeticOp::LENGTH: {
                auto n = b.call(arg(0)->type(), ArithmeticOp::NORMALIZE, {arg(0)});
                auto g = broadcast(b, arg(0)->type(), out_grad);
                accum(arg(0), mul(b, arg(0)->type(), n, g));
                break;
            }
            case ArithmeticOp::LENGTH_SQUARED: {
                auto twice_x = add(b, arg(0)->type(), arg(0), arg(0));
                auto g = broadcast(b, arg(0)->type(), out_grad);
                accum(arg(0), mul(b, arg(0)->type(), twice_x, g));
                break;
            }
            case ArithmeticOp::NORMALIZE: {
                auto n = b.call(type, ArithmeticOp::NORMALIZE, {arg(0)});
                auto dot = b.call(type->element(), ArithmeticOp::DOT, {n, out_grad});
                auto dot_vec = broadcast(b, type, dot);
                auto dot_times_n = mul(b, type, dot_vec, n);
                auto numer = sub(b, type, out_grad, dot_times_n);
                auto len = b.call(type->element(), ArithmeticOp::LENGTH, {arg(0)});
                accum(arg(0), div(b, type, numer, broadcast(b, type, len)));
                break;
            }
            case ArithmeticOp::REDUCE_SUM:
                accum(arg(0), broadcast(b, arg(0)->type(), out_grad));
                break;
            case ArithmeticOp::REDUCE_PRODUCT: {
                auto value_type = arg(0)->type();
                if (value_type->is_vector()) {
                    auto elem_type = value_type->element();
                    auto x_grad = static_cast<Value *>(zero(value_type));
                    for (auto i = 0u; i < value_type->dimension(); i++) {
                        auto elem_grad = out_grad;
                        for (auto j = 0u; j < value_type->dimension(); j++) {
                            if (i == j) { continue; }
                            elem_grad = mul(b, elem_type, elem_grad, extract(b, elem_type, arg(0), j));
                        }
                        x_grad = insert(b, value_type, x_grad, elem_grad, i);
                    }
                    accum(arg(0), x_grad);
                }
                break;
            }
            case ArithmeticOp::REDUCE_MIN:
            case ArithmeticOp::REDUCE_MAX: {
                auto value_type = arg(0)->type();
                auto reduced = b.call(type, inst->op(), {arg(0)});
                auto cond = b.call(bool_type_of(value_type), ArithmeticOp::BINARY_EQUAL, {reduced, arg(0)});
                auto g = select(b, value_type, cond, broadcast(b, value_type, out_grad), zero(value_type));
                accum(arg(0), g);
                break;
            }
            case ArithmeticOp::EXTRACT: {
                luisa::fixed_vector<Value *, 4u> indices;
                for (auto i = 1u; i < inst->operand_count(); i++) { indices.emplace_back(arg(i)); }
                auto g = aggregate_zero_with_element(b, arg(0)->type(), out_grad, indices);
                accum(arg(0), g);
                break;
            }
            case ArithmeticOp::INSERT: {
                luisa::fixed_vector<Value *, 4u> indices;
                for (auto i = 2u; i < inst->operand_count(); i++) { indices.emplace_back(arg(i)); }
                auto base_grad = aggregate_zero_with_element(b, arg(0)->type(), zero(arg(1)->type()), indices);
                base_grad = sub(b, arg(0)->type(), out_grad, base_grad);
                auto elem_grad = [&] {
                    luisa::fixed_vector<Value *, 8u> args;
                    args.emplace_back(out_grad);
                    for (auto idx : indices) { args.emplace_back(idx); }
                    return b.call(arg(1)->type(), ArithmeticOp::EXTRACT, args);
                }();
                accum(arg(0), base_grad);
                accum(arg(1), elem_grad);
                break;
            }
            case ArithmeticOp::SHUFFLE: {
                auto v_grad = static_cast<Value *>(zero(arg(0)->type()));
                for (auto out_i = 0u; out_i + 1u < inst->operand_count(); out_i++) {
                    auto out_grad_elem = extract(b, arg(0)->type()->element(), out_grad, out_i);
                    luisa::fixed_vector<Value *, 8u> extract_args{v_grad, arg(out_i + 1u)};
                    auto old = b.call(arg(0)->type()->element(), ArithmeticOp::EXTRACT, extract_args);
                    auto next = add(b, old->type(), old, out_grad_elem);
                    v_grad = b.call(arg(0)->type(), ArithmeticOp::INSERT, {v_grad, next, arg(out_i + 1u)});
                }
                accum(arg(0), v_grad);
                break;
            }
            case ArithmeticOp::AGGREGATE: {
                for (auto i = 0u; i < inst->operand_count(); i++) {
                    auto member_grad = extract(b, arg(i)->type(), out_grad, i);
                    accum(arg(i), member_grad);
                }
                break;
            }
            case ArithmeticOp::MATRIX_TRANSPOSE:
                accum(arg(0), b.call(arg(0)->type(), ArithmeticOp::MATRIX_TRANSPOSE, {out_grad}));
                break;
            case ArithmeticOp::MATRIX_INVERSE: {
                auto mt = b.call(type, ArithmeticOp::MATRIX_TRANSPOSE, {arg(0)});
                auto mt_g = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {mt, out_grad});
                auto mt_g_mt = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {mt_g, mt});
                accum(arg(0), neg(b, type, mt_g_mt));
                break;
            }
            case ArithmeticOp::MATRIX_DETERMINANT: {
                auto out_times_grad = mul(b, type, inst, out_grad);
                auto inv = b.call(arg(0)->type(), ArithmeticOp::MATRIX_INVERSE, {arg(0)});
                auto inv_t = b.call(arg(0)->type(), ArithmeticOp::MATRIX_TRANSPOSE, {inv});
                auto g = mul(b, arg(0)->type(), broadcast(b, arg(0)->type(), out_times_grad), inv_t);
                accum(arg(0), g);
                break;
            }
            case ArithmeticOp::OUTER_PRODUCT: {
                auto a_grad = b.call(arg(0)->type(), ArithmeticOp::MATRIX_LINALG_MUL, {out_grad, arg(1)});
                auto out_grad_t = b.call(out_grad->type(), ArithmeticOp::MATRIX_TRANSPOSE, {out_grad});
                auto b_grad = b.call(arg(1)->type(), ArithmeticOp::MATRIX_LINALG_MUL, {out_grad_t, arg(0)});
                accum(arg(0), a_grad);
                accum(arg(1), b_grad);
                break;
            }
            default:
                break;
        }
    }

    void emit_backward_if(XIRBuilder &b, IfInst *inst) noexcept {
        auto backward_if = b.if_(inst->condition());
        auto merge_block = backward_if->create_merge_block();
        {
            b.set_insertion_point(backward_if->create_true_block());
            emit_backward_instructions(b, if_true_backward_emit_instructions[inst]);
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        {
            b.set_insertion_point(backward_if->create_false_block());
            emit_backward_instructions(b, if_false_backward_emit_instructions[inst]);
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        b.set_insertion_point(merge_block);
    }

    void emit_backward_instructions(XIRBuilder &b, luisa::span<Instruction *const> instructions) noexcept {
        for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
            auto inst = *it;
            if (auto if_inst = inst->isa<IfInst>() ? static_cast<IfInst *>(inst) : nullptr) {
                emit_backward_if(b, if_inst);
                continue;
            }
            if (inst->is_terminator()) { continue; }
            backward_inst(b, inst);
        }
    }

    void emit_backward(BasicBlock *backward_block, BasicBlock *epilogue) noexcept {
        XIRBuilder b;
        b.set_insertion_point(backward_block);
        seed_gradients(b);
        emit_backward_instructions(b, backward_emit_instructions);
        b.br(epilogue);
    }

    void remove_ad_intrinsics() noexcept {
        std::sort(removable_intrinsics.begin(), removable_intrinsics.end());
        removable_intrinsics.erase(std::unique(removable_intrinsics.begin(), removable_intrinsics.end()), removable_intrinsics.end());
        for (auto inst : removable_intrinsics) {
            if (inst->is_linked()) {
                inst->remove_self();
                changed_count++;
            }
        }
    }

    void split_at_backward(BasicBlock *merge) noexcept {
        LUISA_ASSERT(backward_marker != nullptr, "Autodiff scope requires a backward() call.");
        LUISA_ASSERT(backward_marker_block != nullptr, "Invalid backward marker block.");
        epilogue_block = function->create_basic_block();
        luisa::vector<Instruction *> to_move;
        auto past_backward = false;
        for (auto inst : backward_marker_block->instructions()) {
            if (inst == backward_marker) {
                past_backward = true;
                continue;
            }
            if (past_backward) { to_move.emplace_back(inst); }
        }
        XIRBuilder b;
        b.set_insertion_point(epilogue_block);
        for (auto inst : to_move) {
            b.append(inst->remove_self());
        }
        if (!epilogue_block->is_terminated()) {
            b.set_insertion_point(epilogue_block);
            b.br(merge);
        }
    }

    [[nodiscard]] auto run() noexcept -> size_t {
        auto entry = scope->entry_block();
        auto merge = scope->merge_block();
        LUISA_ASSERT(entry != nullptr && merge != nullptr, "Invalid autodiff scope.");
        {
            luisa::unordered_set<BasicBlock *> visited;
            LUISA_ASSERT(collect_forward(entry, merge, visited, backward_emit_instructions), "Autodiff scope requires a backward() call.");
        }
        split_at_backward(merge);
        collect_backward();
        {
            luisa::unordered_set<BasicBlock *> visited;
            lower_epilogue_gradients(epilogue_block, merge, visited);
        }
        auto backward_block = function->create_basic_block();
        emit_backward(backward_block, epilogue_block);
        remove_ad_intrinsics();
        {
            XIRBuilder b;
            b.set_insertion_point(backward_marker_block);
            b.br(backward_block);
        }
        {
            auto parent = scope->parent_block();
            scope->remove_self();
            XIRBuilder b;
            b.set_insertion_point(parent);
            b.br(entry);
        }
        return changed_count + 1u;
    }
};

struct AutodiffPass {
    Function *function{};
    AutodiffOptions options{};

    [[nodiscard]] auto locate_autodiff_scopes() const noexcept {
        luisa::vector<AutodiffScopeInst *> scopes;
        if (auto def = function->definition()) {
            def->traverse_instructions([&](Instruction *inst) noexcept {
                if (inst->isa<AutodiffScopeInst>()) {
                    scopes.emplace_back(static_cast<AutodiffScopeInst *>(inst));
                }
            });
        }
        return scopes;
    }

    [[nodiscard]] auto run() noexcept -> AutodiffInfo {
        AutodiffInfo info;
        if (!function->definition() || !options.run_backward) { return info; }
        auto scopes = locate_autodiff_scopes();
        for (auto scope : scopes) {
            TransformAdScope transform{.function = function,
                                       .definition = function->definition(),
                                       .module = function->parent_module(),
                                       .scope = scope};
            info.transformed_scope_count++;
            info.removed_instruction_count += transform.run();
        }
        return info;
    }
};

}// namespace

LUISA_XIR_API AutodiffInfo autodiff_pass_run_on_function(Function *function, const AutodiffOptions &options) noexcept {
    AutodiffPass pass{function, options};
    return pass.run();
}

LUISA_XIR_API AutodiffInfo autodiff_pass_run_on_module(Module *module, const AutodiffOptions &options) noexcept {
    AutodiffInfo info;
    for (auto func : module->function_list()) {
        auto f_info = autodiff_pass_run_on_function(func, options);
        info.transformed_scope_count += f_info.transformed_scope_count;
        info.removed_instruction_count += f_info.removed_instruction_count;
    }
    return info;
}

}// namespace luisa::compute::xir
