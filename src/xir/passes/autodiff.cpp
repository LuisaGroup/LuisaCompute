#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/lower_switch.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace luisa::compute::xir {

namespace {

static constexpr auto max_ad_loop_unroll_count = 64u;

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

[[nodiscard]] auto constant_i64(Value *value) noexcept -> luisa::optional<int64_t> {
    if (value == nullptr || !value->isa<Constant>()) { return luisa::nullopt; }
    auto c = static_cast<Constant *>(value);
    switch (c->type()->tag()) {
        case Type::Tag::INT8: return static_cast<int64_t>(c->as<int8_t>());
        case Type::Tag::UINT8: return static_cast<int64_t>(c->as<uint8_t>());
        case Type::Tag::INT16: return static_cast<int64_t>(c->as<int16_t>());
        case Type::Tag::UINT16: return static_cast<int64_t>(c->as<uint16_t>());
        case Type::Tag::INT32: return static_cast<int64_t>(c->as<int32_t>());
        case Type::Tag::UINT32: return static_cast<int64_t>(c->as<uint32_t>());
        case Type::Tag::INT64: return c->as<int64_t>();
        case Type::Tag::UINT64: {
            auto v = c->as<uint64_t>();
            if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return luisa::nullopt; }
            return static_cast<int64_t>(v);
        }
        default: return luisa::nullopt;
    }
}

[[nodiscard]] auto is_integer_lvalue(Value *value) noexcept {
    auto type = value == nullptr ? nullptr : value->type();
    return type != nullptr && (type->is_int8() || type->is_uint8() ||
                               type->is_int16() || type->is_uint16() ||
                               type->is_int32() || type->is_uint32() ||
                               type->is_int64() || type->is_uint64());
}

[[nodiscard]] auto find_store_before(BasicBlock *block, Instruction *before, Value *variable) noexcept -> StoreInst * {
    StoreInst *store = nullptr;
    if (block == nullptr || before == nullptr || variable == nullptr) { return nullptr; }
    for (auto inst : block->instructions()) {
        if (inst == before) { break; }
        if (auto s = inst->isa<StoreInst>() ? static_cast<StoreInst *>(inst) : nullptr) {
            if (s->variable() == variable) { store = s; }
        }
    }
    return store;
}

[[nodiscard]] auto constant_i64_before(Value *value, BasicBlock *block, Instruction *before,
                                       luisa::unordered_set<Value *> &visiting) noexcept -> luisa::optional<int64_t> {
    if (auto c = constant_i64(value)) { return c; }
    auto load = value != nullptr && value->isa<LoadInst>() ? static_cast<LoadInst *>(value) : nullptr;
    if (load == nullptr || !is_integer_lvalue(load->variable())) { return luisa::nullopt; }
    if (!visiting.emplace(load->variable()).second) { return luisa::nullopt; }
    auto store = find_store_before(block, before, load->variable());
    auto result = store == nullptr ? luisa::optional<int64_t>{} :
                                     constant_i64_before(store->value(), block, before, visiting);
    visiting.erase(load->variable());
    return result;
}

[[nodiscard]] auto constant_i64_before(Value *value, BasicBlock *block, Instruction *before) noexcept -> luisa::optional<int64_t> {
    luisa::unordered_set<Value *> visiting;
    return constant_i64_before(value, block, before, visiting);
}

void reject_loop_autodiff() noexcept {
    LUISA_ERROR_WITH_LOCATION("Reverse-mode autodiff over dynamic loops is not supported by XIR autodiff yet. "
                              "Use a fixed-trip for-loop that XIR can unroll inside the autodiff scope, "
                              "or move the dynamic loop outside the autodiff scope.");
}

void retarget_terminator(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr || from == nullptr || to == nullptr) { return; }
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto br = static_cast<BranchInst *>(term);
            if (br->target_block() == from) { br->set_target_block(to); }
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto cb = static_cast<ConditionalBranchInst *>(term);
            if (cb->true_block() == from) { cb->set_true_target(to); }
            if (cb->false_block() == from) { cb->set_false_target(to); }
            break;
        }
        case DerivedInstructionTag::IF: {
            auto if_inst = static_cast<IfInst *>(term);
            if (if_inst->true_block() == from) { if_inst->set_true_target(to); }
            if (if_inst->false_block() == from) { if_inst->set_false_target(to); }
            if (if_inst->merge_block() == from) { if_inst->set_merge_block(to); }
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto sw = static_cast<SwitchInst *>(term);
            if (sw->default_block() == from) { sw->set_default_block(to); }
            if (sw->merge_block() == from) { sw->set_merge_block(to); }
            for (auto i = 0u; i < sw->case_count(); i++) {
                if (sw->case_block(i) == from) { sw->set_case_block(i, to); }
            }
            break;
        }
        case DerivedInstructionTag::LOOP: {
            auto loop = static_cast<LoopInst *>(term);
            if (loop->prepare_block() == from) { loop->set_prepare_block(to); }
            if (loop->body_block() == from) { loop->set_body_block(to); }
            if (loop->update_block() == from) { loop->set_update_block(to); }
            if (loop->merge_block() == from) { loop->set_merge_block(to); }
            break;
        }
        case DerivedInstructionTag::SIMPLE_LOOP: {
            auto loop = static_cast<SimpleLoopInst *>(term);
            if (loop->body_block() == from) { loop->set_body_block(to); }
            if (loop->merge_block() == from) { loop->set_merge_block(to); }
            break;
        }
        default: break;
    }
}

template<typename Visit>
void traverse_loop_region_successors(BasicBlock *block, const luisa::unordered_set<BasicBlock *> &owned,
                                     Visit &&visit) noexcept {
    if (block == nullptr || !block->is_terminated()) { return; }
    auto term = block->terminator();
    for (auto use : term->operand_uses()) {
        auto value = use->value();
        for (auto succ : owned) {
            if (static_cast<Value *>(succ) == value) {
                visit(succ);
                break;
            }
        }
    }
    if (auto merge = term->control_flow_merge(); merge != nullptr) {
        if (auto merge_block = merge->merge_block(); merge_block != nullptr && owned.contains(merge_block)) {
            visit(merge_block);
        }
    }
    if (term->isa<LoopInst>()) {
        auto loop = static_cast<LoopInst *>(term);
        if (auto body = loop->body_block(); body != nullptr && owned.contains(body)) { visit(body); }
        if (auto update = loop->update_block(); update != nullptr && owned.contains(update)) { visit(update); }
    }
}

struct LoopTripCount {
    Value *variable{};
    int64_t start{};
    int64_t step{};
    int64_t trip_count{};
};

struct LoopCondition {
    ArithmeticInst *compare{};
    bool inverted{};
};

[[nodiscard]] auto is_integer_comparison_op(ArithmeticOp op) noexcept {
    return op == ArithmeticOp::BINARY_LESS ||
           op == ArithmeticOp::BINARY_LESS_EQUAL ||
           op == ArithmeticOp::BINARY_GREATER ||
           op == ArithmeticOp::BINARY_GREATER_EQUAL;
}

[[nodiscard]] auto constant_bool_before(Value *value, BasicBlock *block, Instruction *before) noexcept -> luisa::optional<bool> {
    if (value == nullptr) { return luisa::nullopt; }
    if (auto c = value->isa<Constant>() ? static_cast<Constant *>(value) : nullptr) {
        if (c->type()->is_bool()) { return c->as<bool>(); }
    }
    if (!value->isa<ArithmeticInst>()) { return luisa::nullopt; }
    auto inst = static_cast<ArithmeticInst *>(value);
    if (inst->operand_count() != 2u) { return luisa::nullopt; }
    auto lhs = constant_i64_before(inst->operand(0), block, before);
    auto rhs = constant_i64_before(inst->operand(1), block, before);
    if (!lhs || !rhs) { return luisa::nullopt; }
    switch (inst->op()) {
        case ArithmeticOp::BINARY_LESS: return *lhs < *rhs;
        case ArithmeticOp::BINARY_LESS_EQUAL: return *lhs <= *rhs;
        case ArithmeticOp::BINARY_GREATER: return *lhs > *rhs;
        case ArithmeticOp::BINARY_GREATER_EQUAL: return *lhs >= *rhs;
        case ArithmeticOp::BINARY_EQUAL: return *lhs == *rhs;
        case ArithmeticOp::BINARY_NOT_EQUAL: return *lhs != *rhs;
        default: return luisa::nullopt;
    }
}

[[nodiscard]] auto analyze_loop_condition(Value *condition, BasicBlock *block,
                                          Instruction *before) noexcept -> luisa::optional<LoopCondition> {
    auto inst = condition != nullptr && condition->isa<ArithmeticInst>() ?
                    static_cast<ArithmeticInst *>(condition) :
                    nullptr;
    if (inst == nullptr || inst->operand_count() != 2u) { return luisa::nullopt; }
    if (is_integer_comparison_op(inst->op())) {
        return LoopCondition{.compare = inst, .inverted = false};
    }
    if (inst->op() != ArithmeticOp::BINARY_BIT_XOR || !inst->type()->is_bool()) { return luisa::nullopt; }
    for (auto i = 0u; i < 2u; i++) {
        auto cmp = inst->operand(i)->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst->operand(i)) : nullptr;
        if (cmp == nullptr || !is_integer_comparison_op(cmp->op())) { continue; }
        auto inv = constant_bool_before(inst->operand(1u - i), block, before);
        if (inv) { return LoopCondition{.compare = cmp, .inverted = *inv}; }
    }
    return luisa::nullopt;
}

[[nodiscard]] auto analyze_simple_counted_loop(LoopInst *loop) noexcept -> luisa::optional<LoopTripCount> {
    auto preheader = loop->parent_block();
    auto prepare = loop->prepare_block();
    auto body = loop->body_block();
    auto update = loop->update_block();
    auto merge = loop->merge_block();
    if (preheader == nullptr || prepare == nullptr || body == nullptr || update == nullptr || merge == nullptr) { return luisa::nullopt; }
    auto cond_br = prepare->terminator();
    if (cond_br == nullptr || !cond_br->isa<ConditionalBranchInst>()) { return luisa::nullopt; }
    auto prepare_branch = static_cast<ConditionalBranchInst *>(cond_br);
    if (prepare_branch->true_block() != body || prepare_branch->false_block() != merge) { return luisa::nullopt; }
    auto loop_cond = analyze_loop_condition(prepare_branch->condition(), preheader, loop);
    if (!loop_cond) { return luisa::nullopt; }
    auto cmp = loop_cond->compare;
    auto op = cmp->op();
    auto load = cmp->operand(0)->isa<LoadInst>() ? static_cast<LoadInst *>(cmp->operand(0)) : nullptr;
    auto bound = constant_i64_before(cmp->operand(1), preheader, loop);
    if (load == nullptr || !bound) { return luisa::nullopt; }
    auto variable = load->variable();
    if (!is_integer_lvalue(variable)) { return luisa::nullopt; }
    auto update_br = update->terminator();
    if (update_br == nullptr || !update_br->isa<BranchInst>() ||
        static_cast<BranchInst *>(update_br)->target_block() != prepare) {
        return luisa::nullopt;
    }
    auto init_store = find_store_before(preheader, loop, variable);
    if (init_store == nullptr) { return luisa::nullopt; }
    auto start = constant_i64_before(init_store->value(), preheader, loop);
    if (!start) { return luisa::nullopt; }
    StoreInst *update_store = nullptr;
    for (auto inst : update->instructions()) {
        if (inst->is_terminator()) { break; }
        if (inst->isa<StoreInst>()) {
            auto store = static_cast<StoreInst *>(inst);
            if (store->variable() == variable) {
                if (update_store != nullptr) { return luisa::nullopt; }
                update_store = store;
            }
        }
    }
    if (update_store == nullptr) { return luisa::nullopt; }
    auto step_value = update_store->value();
    if (step_value == nullptr || !step_value->isa<ArithmeticInst>()) { return luisa::nullopt; }
    auto add = static_cast<ArithmeticInst *>(step_value);
    if (add->op() != ArithmeticOp::BINARY_ADD || add->operand_count() != 2u) { return luisa::nullopt; }
    auto update_load = add->operand(0)->isa<LoadInst>() ? static_cast<LoadInst *>(add->operand(0)) : nullptr;
    auto step = constant_i64_before(add->operand(1), preheader, loop);
    if (update_load == nullptr || update_load->variable() != variable || !step) { return luisa::nullopt; }
    if (*step == 0) { return luisa::nullopt; }
    auto compare = [&](int64_t v) noexcept {
        auto result = false;
        switch (op) {
            case ArithmeticOp::BINARY_LESS: result = v < *bound; break;
            case ArithmeticOp::BINARY_LESS_EQUAL: result = v <= *bound; break;
            case ArithmeticOp::BINARY_GREATER: result = v > *bound; break;
            case ArithmeticOp::BINARY_GREATER_EQUAL: result = v >= *bound; break;
            default: break;
        }
        return loop_cond->inverted ? !result : result;
    };
    auto v = *start;
    int64_t trips = 0;
    while (compare(v)) {
        trips++;
        if (trips > max_ad_loop_unroll_count) { return luisa::nullopt; }
        if ((*step > 0 && v > std::numeric_limits<int64_t>::max() - *step) ||
            (*step < 0 && v < std::numeric_limits<int64_t>::min() - *step)) {
            return luisa::nullopt;
        }
        v += *step;
    }
    return LoopTripCount{.variable = variable, .start = *start, .step = *step, .trip_count = trips};
}

struct CloneRemap final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> map;
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (auto iter = map.find(value); iter != map.end()) { return iter->second; }
        return const_cast<Value *>(value);
    }
};

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
    luisa::unordered_map<IfInst *, AllocaInst *> if_condition_slots;
    luisa::unordered_map<IfInst *, luisa::vector<Instruction *>> if_true_backward_emit_instructions;
    luisa::unordered_map<IfInst *, luisa::vector<Instruction *>> if_false_backward_emit_instructions;
    luisa::unordered_map<SwitchInst *, AllocaInst *> switch_value_slots;
    luisa::unordered_map<SwitchInst *, luisa::vector<Instruction *>> switch_default_backward_emit_instructions;
    luisa::unordered_map<SwitchInst *, luisa::vector<luisa::vector<Instruction *>>> switch_case_backward_emit_instructions;
    luisa::vector<AutodiffIntrinsicInst *> removable_intrinsics;
    luisa::vector<std::pair<Value *, Value *>> seeds;
    AutodiffIntrinsicInst *backward_marker{};
    BasicBlock *backward_marker_block{};
    BasicBlock *epilogue_block{};
    size_t changed_count{0u};
    bool unrolled_early_exit_loop{false};

    [[nodiscard]] auto owns_block(BasicBlock *block) const noexcept {
        if (block == nullptr) { return false; }
        for (auto owned : definition->basic_blocks()) {
            if (owned == block) { return true; }
        }
        return false;
    }

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

    [[nodiscard]] auto reduce_sum_all(XIRBuilder &b, Value *value) noexcept -> Value * {
        auto type = value->type();
        if (type->is_scalar()) { return value; }
        if (type->is_vector()) {
            return b.call(type->element(), ArithmeticOp::REDUCE_SUM, {value});
        }
        if (type->is_matrix()) {
            auto elem_type = type->element();
            auto column_type = Type::vector(elem_type, type->dimension());
            auto sum = static_cast<Value *>(zero(elem_type));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto column = extract(b, column_type, value, i);
                auto column_sum = b.call(elem_type, ArithmeticOp::REDUCE_SUM, {column});
                sum = add(b, elem_type, sum, column_sum);
            }
            return sum;
        }
        LUISA_ERROR_WITH_LOCATION("Cannot reduce gradient of type {} to a scalar.", type->description());
    }

    [[nodiscard]] auto project_grad_to_type(XIRBuilder &b, const Type *type, Value *grad) noexcept -> Value * {
        if (type == grad->type()) { return grad; }
        if (type->is_scalar()) {
            return b.static_cast_if_necessary(type, reduce_sum_all(b, grad));
        }
        if ((type->is_vector() || type->is_matrix()) && grad->type()->is_scalar()) {
            return broadcast(b, type, grad);
        }
        if (type->is_vector() && grad->type()->is_vector() && type->dimension() == grad->type()->dimension()) {
            auto result = static_cast<Value *>(zero(type));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto elem = extract(b, grad->type()->element(), grad, i);
                result = insert(b, type, result, b.static_cast_if_necessary(type->element(), elem), i);
            }
            return result;
        }
        if (type->is_matrix() && grad->type()->is_matrix() && type->dimension() == grad->type()->dimension()) {
            auto column_type = Type::vector(type->element(), type->dimension());
            auto grad_column_type = Type::vector(grad->type()->element(), grad->type()->dimension());
            auto result = static_cast<Value *>(zero(type));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto column = extract(b, grad_column_type, grad, i);
                auto projected_column = project_grad_to_type(b, column_type, column);
                result = insert(b, type, result, projected_column, i);
            }
            return result;
        }
        LUISA_ERROR_WITH_LOCATION("Cannot project gradient from type {} to {}.", grad->type()->description(), type->description());
    }

    [[nodiscard]] auto lift_value_to_type(XIRBuilder &b, const Type *type, Value *value) noexcept -> Value * {
        if (type == value->type()) { return value; }
        if ((type->is_vector() || type->is_matrix()) && value->type()->is_scalar()) {
            return broadcast(b, type, value);
        }
        return value;
    }

    [[nodiscard]] auto collect_loop_unroll_region(LoopInst *loop,
                                                  luisa::unordered_set<BasicBlock *> &region,
                                                  luisa::vector<BasicBlock *> &ordered) noexcept -> bool {
        auto prepare = loop->prepare_block();
        auto body = loop->body_block();
        auto update = loop->update_block();
        auto merge = loop->merge_block();
        if (prepare == nullptr || body == nullptr || update == nullptr || merge == nullptr) { return false; }
        luisa::unordered_set<BasicBlock *> owned;
        for (auto block : definition->basic_blocks()) { owned.emplace(block); }
        region.clear();
        ordered.clear();
        luisa::vector<BasicBlock *> work{body};
        while (!work.empty()) {
            auto block = work.back();
            work.pop_back();
            if (block == prepare || block == merge || region.contains(block)) { continue; }
            region.emplace(block);
            ordered.emplace_back(block);
            traverse_loop_region_successors(block, owned, [&](BasicBlock *succ) noexcept {
                if (succ != prepare && succ != merge && !region.contains(succ)) {
                    work.emplace_back(succ);
                }
            });
        }
        if (!region.contains(body) || !region.contains(update)) { return false; }
        auto valid = true;
        for (auto block : ordered) {
            for (auto inst : block->instructions()) {
                if (auto intrinsic = inst->isa<AutodiffIntrinsicInst>() ? static_cast<AutodiffIntrinsicInst *>(inst) : nullptr) {
                    if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_BACKWARD) { valid = false; }
                }
                switch (inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::LOOP:
                    case DerivedInstructionTag::SIMPLE_LOOP:
                    case DerivedInstructionTag::AUTODIFF_SCOPE:
                        valid = false;
                        break;
                    default:
                        break;
                }
            }
            traverse_loop_region_successors(block, owned, [&](BasicBlock *succ) noexcept {
                auto term = block->terminator();
                if (succ == merge && term != nullptr && term->isa<BreakInst>()) {
                    auto break_inst = static_cast<BreakInst *>(term);
                    valid &= break_inst->target_block() == merge;
                } else if (succ == merge) {
                    valid = false;
                } else if (succ == prepare) {
                    valid &= block == update;
                } else if (!region.contains(succ)) {
                    valid = false;
                }
            });
        }
        return valid;
    }

    void retarget_unrolled_early_exits(BasicBlock *block, BasicBlock *break_target, BasicBlock *continue_target) noexcept {
        if (!block->is_terminated()) { return; }
        auto term = block->terminator();
        if (auto break_inst = term->isa<BreakInst>() ? static_cast<BreakInst *>(term) : nullptr) {
            if (break_inst->target_block() == nullptr || break_inst->target_block() == break_target) {
                break_inst->remove_self();
                XIRBuilder b;
                b.set_insertion_point(block);
                b.br(break_target);
                unrolled_early_exit_loop = true;
            }
        } else if (auto continue_inst = term->isa<ContinueInst>() ? static_cast<ContinueInst *>(term) : nullptr) {
            if (continue_inst->target_block() == nullptr || continue_inst->target_block() == continue_target) {
                continue_inst->remove_self();
                XIRBuilder b;
                b.set_insertion_point(block);
                b.br(continue_target);
                unrolled_early_exit_loop = true;
            }
        }
    }

    void unroll_fixed_trip_loop(LoopInst *loop, const LoopTripCount &trip) noexcept {
        auto preheader = loop->parent_block();
        auto merge = loop->merge_block();
        LUISA_ASSERT(preheader != nullptr && merge != nullptr, "Invalid loop.");
        if (trip.trip_count == 0) {
            loop->remove_self();
            XIRBuilder b;
            b.set_insertion_point(preheader);
            b.br(merge);
            changed_count++;
            return;
        }
        luisa::unordered_set<BasicBlock *> region;
        luisa::vector<BasicBlock *> ordered;
        if (!collect_loop_unroll_region(loop, region, ordered)) { reject_loop_autodiff(); }
        auto trips = static_cast<size_t>(trip.trip_count);
        luisa::vector<CloneRemap> remaps;
        remaps.resize(trips);
        for (auto iter = 0u; iter < trips; iter++) {
            for (auto block : ordered) {
                remaps[iter].map[block] = definition->create_basic_block();
            }
        }
        XIRBuilder b;
        for (auto iter = 0u; iter < trips; iter++) {
            auto &remap = remaps[iter];
            for (auto old_block : ordered) {
                auto new_block = static_cast<BasicBlock *>(remap.map[old_block]);
                b.set_insertion_point(new_block);
                if (old_block == loop->body_block()) {
                    for (auto old_inst : loop->prepare_block()->instructions()) {
                        if (old_inst->is_terminator()) { break; }
                        auto new_inst = old_inst->clone_with_metadata(b, remap);
                        remap.map[old_inst] = new_inst;
                    }
                }
                for (auto old_inst : old_block->instructions()) {
                    auto new_inst = old_inst->clone_with_metadata(b, remap);
                    remap.map[old_inst] = new_inst;
                }
            }
        }
        for (auto iter = 0u; iter < trips; iter++) {
            auto &remap = remaps[iter];
            auto next = iter + 1u < trips ?
                            static_cast<BasicBlock *>(remaps[iter + 1u].map[loop->body_block()]) :
                            merge;
            auto iteration_update = static_cast<BasicBlock *>(remap.map[loop->update_block()]);
            for (auto old_block : ordered) {
                auto new_block = static_cast<BasicBlock *>(remap.map[old_block]);
                if (new_block->is_terminated()) {
                    retarget_terminator(new_block->terminator(), loop->prepare_block(), next);
                    retarget_unrolled_early_exits(new_block, merge, iteration_update);
                }
            }
        }
        auto first = static_cast<BasicBlock *>(remaps.front().map[loop->body_block()]);
        loop->remove_self();
        b.set_insertion_point(preheader);
        b.br(first);
        changed_count++;
    }

    [[nodiscard]] auto clone_loop_prepare_condition(XIRBuilder &b, BasicBlock *prepare, ConditionalBranchInst *branch,
                                                    CloneRemap &remap, BasicBlock *target) noexcept -> Value * {
        b.set_insertion_point(target);
        for (auto old_inst : prepare->instructions()) {
            if (old_inst->is_terminator()) { break; }
            auto new_inst = old_inst->clone_with_metadata(b, remap);
            remap.map[old_inst] = new_inst;
        }
        return remap.resolve(branch->condition());
    }

    [[nodiscard]] auto validate_dynamic_loop_shape(LoopInst *loop) noexcept -> bool {
        auto prepare = loop->prepare_block();
        auto body = loop->body_block();
        auto update = loop->update_block();
        auto merge = loop->merge_block();
        if (prepare == nullptr || body == nullptr || update == nullptr || merge == nullptr) { return false; }
        auto cond_br = prepare->terminator();
        if (cond_br == nullptr || !cond_br->isa<ConditionalBranchInst>()) { return false; }
        auto branch = static_cast<ConditionalBranchInst *>(cond_br);
        if (branch->true_block() != body || branch->false_block() != merge) { return false; }
        auto update_br = update->terminator();
        if (update_br == nullptr || !update_br->isa<BranchInst>() ||
            static_cast<BranchInst *>(update_br)->target_block() != prepare) {
            return false;
        }
        for (auto inst : prepare->instructions()) {
            if (inst->is_terminator()) { break; }
            if (inst->isa<PhiInst>()) { return false; }
        }
        return true;
    }

    void retarget_dynamic_unrolled_early_exits(BasicBlock *block, BasicBlock *break_target,
                                               BasicBlock *continue_target, AllocaInst *done_slot,
                                               BasicBlock *exit_target) noexcept {
        if (!block->is_terminated()) { return; }
        auto term = block->terminator();
        if (auto break_inst = term->isa<BreakInst>() ? static_cast<BreakInst *>(term) : nullptr) {
            if (break_inst->target_block() == nullptr || break_inst->target_block() == break_target) {
                break_inst->remove_self();
                XIRBuilder b;
                b.set_insertion_point(block);
                b.store(done_slot, one(Type::of<bool>()));
                b.br(exit_target);
            }
        } else if (auto continue_inst = term->isa<ContinueInst>() ? static_cast<ContinueInst *>(term) : nullptr) {
            if (continue_inst->target_block() == nullptr || continue_inst->target_block() == continue_target) {
                continue_inst->remove_self();
                XIRBuilder b;
                b.set_insertion_point(block);
                b.br(continue_target);
            }
        }
    }

    void unroll_bounded_dynamic_loop(LoopInst *loop) noexcept {
        auto preheader = loop->parent_block();
        auto prepare = loop->prepare_block();
        auto merge = loop->merge_block();
        LUISA_ASSERT(preheader != nullptr && prepare != nullptr && merge != nullptr, "Invalid loop.");
        if (!validate_dynamic_loop_shape(loop)) { reject_loop_autodiff(); }
        auto prepare_branch = static_cast<ConditionalBranchInst *>(prepare->terminator());
        luisa::unordered_set<BasicBlock *> region;
        luisa::vector<BasicBlock *> ordered;
        if (!collect_loop_unroll_region(loop, region, ordered)) { reject_loop_autodiff(); }
        auto done_slot = create_snapshot_slot(Type::of<bool>());
        luisa::vector<CloneRemap> remaps;
        remaps.resize(max_ad_loop_unroll_count);
        luisa::vector<BasicBlock *> gate_blocks;
        luisa::vector<BasicBlock *> eval_blocks;
        luisa::vector<BasicBlock *> inactive_blocks;
        luisa::vector<BasicBlock *> condition_false_blocks;
        luisa::vector<BasicBlock *> condition_merge_blocks;
        luisa::vector<BasicBlock *> iteration_merge_blocks;
        gate_blocks.reserve(max_ad_loop_unroll_count);
        eval_blocks.reserve(max_ad_loop_unroll_count);
        inactive_blocks.reserve(max_ad_loop_unroll_count);
        condition_false_blocks.reserve(max_ad_loop_unroll_count);
        condition_merge_blocks.reserve(max_ad_loop_unroll_count);
        iteration_merge_blocks.reserve(max_ad_loop_unroll_count);
        for (auto iter = 0u; iter < max_ad_loop_unroll_count; iter++) {
            gate_blocks.emplace_back(definition->create_basic_block());
            eval_blocks.emplace_back(definition->create_basic_block());
            inactive_blocks.emplace_back(definition->create_basic_block());
            condition_false_blocks.emplace_back(definition->create_basic_block());
            condition_merge_blocks.emplace_back(definition->create_basic_block());
            iteration_merge_blocks.emplace_back(definition->create_basic_block());
            for (auto block : ordered) {
                remaps[iter].map[block] = definition->create_basic_block();
            }
        }
        XIRBuilder b;
        for (auto iter = 0u; iter < max_ad_loop_unroll_count; iter++) {
            auto &remap = remaps[iter];
            b.set_insertion_point(gate_blocks[iter]);
            auto active = b.call(Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {b.load(Type::of<bool>(), done_slot)});
            auto active_if = b.if_(active);
            active_if->set_true_target(eval_blocks[iter]);
            active_if->set_false_target(inactive_blocks[iter]);
            active_if->set_merge_block(iteration_merge_blocks[iter]);
            b.set_insertion_point(inactive_blocks[iter]);
            b.br(iteration_merge_blocks[iter]);
            auto loop_condition = clone_loop_prepare_condition(b, prepare, prepare_branch, remap, eval_blocks[iter]);
            auto condition_if = b.if_(loop_condition);
            condition_if->set_true_target(static_cast<BasicBlock *>(remap.map[loop->body_block()]));
            condition_if->set_false_target(condition_false_blocks[iter]);
            condition_if->set_merge_block(condition_merge_blocks[iter]);
            b.set_insertion_point(condition_false_blocks[iter]);
            b.store(done_slot, one(Type::of<bool>()));
            b.br(condition_merge_blocks[iter]);
            for (auto old_block : ordered) {
                auto new_block = static_cast<BasicBlock *>(remap.map[old_block]);
                b.set_insertion_point(new_block);
                for (auto old_inst : old_block->instructions()) {
                    auto new_inst = old_inst->clone_with_metadata(b, remap);
                    remap.map[old_inst] = new_inst;
                }
            }
            b.set_insertion_point(condition_merge_blocks[iter]);
            b.br(iteration_merge_blocks[iter]);
        }
        auto overflow_check = definition->create_basic_block();
        auto overflow_eval = definition->create_basic_block();
        auto overflow_inactive = definition->create_basic_block();
        auto overflow_condition_false = definition->create_basic_block();
        auto overflow_condition_merge = definition->create_basic_block();
        auto overflow_merge = definition->create_basic_block();
        auto overflow = definition->create_basic_block();
        for (auto iter = 0u; iter < max_ad_loop_unroll_count; iter++) {
            auto &remap = remaps[iter];
            auto next = iter + 1u < max_ad_loop_unroll_count ? gate_blocks[iter + 1u] : overflow_check;
            auto iteration_update = static_cast<BasicBlock *>(remap.map[loop->update_block()]);
            for (auto old_block : ordered) {
                auto new_block = static_cast<BasicBlock *>(remap.map[old_block]);
                if (new_block->is_terminated()) {
                    retarget_terminator(new_block->terminator(), loop->prepare_block(), condition_merge_blocks[iter]);
                    retarget_dynamic_unrolled_early_exits(new_block, merge, iteration_update, done_slot,
                                                          condition_merge_blocks[iter]);
                }
            }
            b.set_insertion_point(iteration_merge_blocks[iter]);
            b.br(next);
        }
        CloneRemap overflow_remap;
        b.set_insertion_point(overflow_check);
        auto overflow_active = b.call(Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {b.load(Type::of<bool>(), done_slot)});
        auto overflow_active_if = b.if_(overflow_active);
        overflow_active_if->set_true_target(overflow_eval);
        overflow_active_if->set_false_target(overflow_inactive);
        overflow_active_if->set_merge_block(overflow_merge);
        b.set_insertion_point(overflow_inactive);
        b.br(overflow_merge);
        auto overflow_condition = clone_loop_prepare_condition(b, prepare, prepare_branch, overflow_remap, overflow_eval);
        auto overflow_condition_if = b.if_(overflow_condition);
        overflow_condition_if->set_true_target(overflow);
        overflow_condition_if->set_false_target(overflow_condition_false);
        overflow_condition_if->set_merge_block(overflow_condition_merge);
        b.set_insertion_point(overflow_condition_false);
        b.store(done_slot, one(Type::of<bool>()));
        b.br(overflow_condition_merge);
        b.set_insertion_point(overflow);
        b.unreachable_("XIR autodiff dynamic loop exceeded bounded unroll limit.");
        b.set_insertion_point(overflow_condition_merge);
        b.br(overflow_merge);
        b.set_insertion_point(overflow_merge);
        b.br(merge);
        auto first = gate_blocks.front();
        loop->remove_self();
        b.set_insertion_point(preheader);
        b.br(first);
        changed_count++;
    }

    void collect_first_level_loops(BasicBlock *block, BasicBlock *merge,
                                   luisa::unordered_set<BasicBlock *> &visited,
                                   luisa::vector<LoopInst *> &loops) noexcept {
        if (block == merge || !visited.emplace(block).second) { return; }
        auto term = block->terminator();
        if (term == nullptr) { return; }
        if (auto loop = term->isa<LoopInst>() ? static_cast<LoopInst *>(term) : nullptr) {
            loops.emplace_back(loop);
            return;
        }
        if (term->isa<SimpleLoopInst>()) { reject_loop_autodiff(); }
        if (auto if_inst = term->isa<IfInst>() ? static_cast<IfInst *>(term) : nullptr) {
            collect_first_level_loops(if_inst->true_block(), if_inst->merge_block(), visited, loops);
            collect_first_level_loops(if_inst->false_block(), if_inst->merge_block(), visited, loops);
            collect_first_level_loops(if_inst->merge_block(), merge, visited, loops);
            return;
        }
        if (auto switch_inst = term->isa<SwitchInst>() ? static_cast<SwitchInst *>(term) : nullptr) {
            collect_first_level_loops(switch_inst->default_block(), switch_inst->merge_block(), visited, loops);
            for (auto i = 0u; i < switch_inst->case_count(); i++) {
                collect_first_level_loops(switch_inst->case_block(i), switch_inst->merge_block(), visited, loops);
            }
            collect_first_level_loops(switch_inst->merge_block(), merge, visited, loops);
            return;
        }
        block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            collect_first_level_loops(succ, merge, visited, loops);
        });
    }

    void unroll_fixed_trip_loops(BasicBlock *entry, BasicBlock *merge) noexcept {
        for (;;) {
            luisa::vector<LoopInst *> loops;
            luisa::unordered_set<BasicBlock *> visited;
            collect_first_level_loops(entry, merge, visited, loops);
            if (loops.empty()) { break; }
            for (auto loop : loops) {
                auto trip = analyze_simple_counted_loop(loop);
                if (trip) {
                    unroll_fixed_trip_loop(loop, *trip);
                } else {
                    unroll_bounded_dynamic_loop(loop);
                }
            }
        }
    }

    void normalize_cfg_after_early_exit_unrolls() noexcept {
        if (!unrolled_early_exit_loop) { return; }
        [[maybe_unused]] auto lower_switch_info = lower_switch_pass_run_on_function(function);
        [[maybe_unused]] auto destructure_info = destructure_cfg_pass_run_on_function(function);
        [[maybe_unused]] auto simplify_info = simplify_cfg_pass_run_on_function(function);
        [[maybe_unused]] auto reg2mem_pre_info = reg2mem_pass_run_on_function(function);
        [[maybe_unused]] auto restructure_info = restructure_cfg_pass_run_on_function(function);
        [[maybe_unused]] auto dce_info = dce_pass_run_on_function(function);
        [[maybe_unused]] auto reg2mem_post_info = reg2mem_pass_run_on_function(function);
        changed_count++;
        unrolled_early_exit_loop = false;
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

    [[nodiscard]] auto cast_to_matching_shape(XIRBuilder &b, const Type *type, Value *value) noexcept -> Value * {
        if (value->type() == type) { return value; }
        if (type->is_scalar()) {
            LUISA_ASSERT(value->type()->is_scalar(), "Invalid scalar cast.");
            return b.static_cast_(type, value);
        }
        LUISA_ASSERT(type->is_vector(), "Invalid target type.");
        if (value->type()->is_scalar()) {
            return broadcast(b, type, b.static_cast_(type->element(), value));
        }
        LUISA_ASSERT(value->type()->is_vector() && value->type()->dimension() == type->dimension(),
                     "Invalid vector cast.");
        auto result = static_cast<Value *>(zero(type));
        for (auto i = 0u; i < type->dimension(); i++) {
            auto elem = extract(b, value->type()->element(), value, i);
            result = insert(b, type, result, b.static_cast_(type->element(), elem), i);
        }
        return result;
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

    [[nodiscard]] auto create_snapshot_slot(const Type *type) noexcept -> AllocaInst * {
        XIRBuilder b;
        b.set_insertion_point(definition->body_block()->instructions().head_sentinel());
        auto slot = b.alloca_local(type);
        b.store(slot, zero(type));
        return slot;
    }

    void snapshot_if_condition(IfInst *inst) noexcept {
        auto [iter, inserted] = if_condition_slots.try_emplace(inst, nullptr);
        if (!inserted) { return; }
        auto slot = create_snapshot_slot(inst->condition()->type());
        iter->second = slot;
        XIRBuilder b;
        b.set_insertion_point(inst->prev());
        b.store(slot, inst->condition());
        changed_count++;
    }

    void snapshot_switch_value(SwitchInst *inst) noexcept {
        auto [iter, inserted] = switch_value_slots.try_emplace(inst, nullptr);
        if (!inserted) { return; }
        auto slot = create_snapshot_slot(inst->value()->type());
        iter->second = slot;
        XIRBuilder b;
        b.set_insertion_point(inst->prev());
        b.store(slot, inst->value());
        changed_count++;
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
                    LUISA_ASSERT(intrinsic->operand_count() == 2u, "Invalid accumulate_gradient operand count.");
                    mark_backward_reachable(intrinsic->operand(0));
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
                case AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT:
                case AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT:
                    LUISA_ERROR_WITH_LOCATION("Forward-mode autodiff intrinsics cannot appear in a reverse-mode autodiff scope.");
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
        } else if (auto cast = inst->isa<CastInst>() ? static_cast<CastInst *>(inst) : nullptr) {
            if (!detached.contains(cast) &&
                is_differentiable_type(cast->type()) &&
                forward_reachable.contains(cast->value()) &&
                is_differentiable_type(cast->value()->type())) {
                forward_reachable.emplace(cast);
                static_cast<void>(grad_slot(cast));
            }
        } else if (auto call = inst->isa<CallInst>() ? static_cast<CallInst *>(inst) : nullptr) {
            auto relevant = call->type() != nullptr && is_differentiable_type(call->type());
            for (auto use : call->argument_uses()) {
                auto value = use->value();
                if (forward_reachable.contains(value) && is_differentiable_type(value->type())) {
                    relevant = true;
                    break;
                }
            }
            if (relevant) {
                forward_reachable.emplace(call);
                if (call->type() != nullptr) { static_cast<void>(grad_slot(call)); }
            }
        }
    }

    [[nodiscard]] auto collect_forward(BasicBlock *block, BasicBlock *merge, luisa::unordered_set<BasicBlock *> &visited, luisa::vector<Instruction *> &emit_instructions) noexcept -> bool {
        LUISA_ASSERT(owns_block(block), "Invalid XIR autodiff CFG block.");
        LUISA_ASSERT(merge == nullptr || owns_block(merge), "Invalid XIR autodiff CFG merge block.");
        if (block == merge || !visited.emplace(block).second) { return false; }
        luisa::vector<Instruction *> instructions;
        for (auto inst : block->instructions()) {
            instructions.emplace_back(inst);
        }
        for (auto inst : instructions) {
            forward_instructions.emplace_back(inst);
            process_forward_instruction(inst);
            emit_instructions.emplace_back(inst);
            if (inst == backward_marker) { return true; }
        }
        if (auto if_inst = block->terminator(); if_inst != nullptr && if_inst->isa<IfInst>()) {
            auto structured_if = static_cast<IfInst *>(if_inst);
            LUISA_ASSERT(owns_block(structured_if->true_block()) &&
                             owns_block(structured_if->false_block()) &&
                             owns_block(structured_if->merge_block()),
                         "Invalid XIR if region in autodiff scope.");
            snapshot_if_condition(structured_if);
            luisa::vector<Instruction *> true_emit;
            luisa::vector<Instruction *> false_emit;
            auto found = collect_forward(structured_if->true_block(), structured_if->merge_block(), visited, true_emit);
            found |= collect_forward(structured_if->false_block(), structured_if->merge_block(), visited, false_emit);
            if_true_backward_emit_instructions[structured_if] = std::move(true_emit);
            if_false_backward_emit_instructions[structured_if] = std::move(false_emit);
            if (found) { return true; }
            return collect_forward(structured_if->merge_block(), merge, visited, emit_instructions);
        }
        if (auto switch_inst = block->terminator(); switch_inst != nullptr && switch_inst->isa<SwitchInst>()) {
            auto structured_switch = static_cast<SwitchInst *>(switch_inst);
            LUISA_ASSERT(owns_block(structured_switch->default_block()) &&
                             owns_block(structured_switch->merge_block()),
                         "Invalid XIR switch region in autodiff scope.");
            for (auto i = 0u; i < structured_switch->case_count(); i++) {
                LUISA_ASSERT(owns_block(structured_switch->case_block(i)),
                             "Invalid XIR switch case block in autodiff scope.");
            }
            snapshot_switch_value(structured_switch);
            luisa::vector<Instruction *> default_emit;
            luisa::vector<luisa::vector<Instruction *>> case_emits;
            case_emits.resize(structured_switch->case_count());
            auto found = collect_forward(structured_switch->default_block(), structured_switch->merge_block(), visited, default_emit);
            for (auto i = 0u; i < structured_switch->case_count(); i++) {
                found |= collect_forward(structured_switch->case_block(i), structured_switch->merge_block(), visited, case_emits[i]);
            }
            switch_default_backward_emit_instructions[structured_switch] = std::move(default_emit);
            switch_case_backward_emit_instructions[structured_switch] = std::move(case_emits);
            if (found) { return true; }
            return collect_forward(structured_switch->merge_block(), merge, visited, emit_instructions);
        }
        if (auto loop_inst = block->terminator(); loop_inst != nullptr &&
                                                (loop_inst->isa<LoopInst>() || loop_inst->isa<SimpleLoopInst>())) {
            reject_loop_autodiff();
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
            } else if (auto cast = inst->isa<CastInst>() ? static_cast<CastInst *>(inst) : nullptr) {
                if (backward_reachable.contains(cast) &&
                    forward_reachable.contains(cast->value()) &&
                    is_differentiable_type(cast->value()->type())) {
                    mark_backward_reachable(cast->value());
                }
            } else if (auto load = inst->isa<LoadInst>() ? static_cast<LoadInst *>(inst) : nullptr) {
                if (backward_reachable.contains(load) && forward_reachable.contains(load->variable())) {
                    mark_backward_reachable(load->variable());
                }
            } else if (auto store = inst->isa<StoreInst>() ? static_cast<StoreInst *>(inst) : nullptr) {
                if (lvalue_backward_reachable(store->variable()) && forward_reachable.contains(store->value())) {
                    mark_backward_reachable(store->value());
                }
            } else if (auto call = inst->isa<CallInst>() ? static_cast<CallInst *>(inst) : nullptr) {
                if (backward_reachable.contains(call)) {
                    for (auto use : call->argument_uses()) {
                        auto value = use->value();
                        if (forward_reachable.contains(value) && is_differentiable_type(value->type())) {
                            mark_backward_reachable(value);
                        }
                    }
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
            } else if (auto switch_inst = inst->isa<SwitchInst>() ? static_cast<SwitchInst *>(inst) : nullptr) {
                lower_epilogue_gradients(switch_inst->default_block(), switch_inst->merge_block(), visited);
                for (auto i = 0u; i < switch_inst->case_count(); i++) {
                    lower_epilogue_gradients(switch_inst->case_block(i), switch_inst->merge_block(), visited);
                }
                lower_epilogue_gradients(switch_inst->merge_block(), merge, visited);
            } else if (inst->isa<LoopInst>() || inst->isa<SimpleLoopInst>()) {
                reject_loop_autodiff();
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

    void backward_cast(XIRBuilder &b, CastInst *inst) noexcept {
        if (!backward_reachable.contains(inst) || !is_differentiable_type(inst->type())) { return; }
        auto value = inst->value();
        if (!forward_reachable.contains(value) || !is_differentiable_type(value->type())) { return; }
        switch (inst->op()) {
            case CastOp::STATIC_CAST: {
                auto out_grad = load_grad(b, inst);
                accumulate_grad(b, value, project_grad_to_type(b, value->type(), out_grad));
                break;
            }
            case CastOp::BITWISE_CAST:
                LUISA_ERROR_WITH_LOCATION("Reverse-mode autodiff over differentiable bitwise casts is not supported.");
        }
    }

    void backward_inst(XIRBuilder &b, Instruction *inst) noexcept {
        if (auto arith = inst->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst) : nullptr) {
            backward_arithmetic(b, arith);
        } else if (auto cast = inst->isa<CastInst>() ? static_cast<CastInst *>(inst) : nullptr) {
            backward_cast(b, cast);
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
        } else if (auto call = inst->isa<CallInst>() ? static_cast<CallInst *>(inst) : nullptr) {
            if (forward_reachable.contains(call) || backward_reachable.contains(call)) {
                LUISA_ERROR_WITH_LOCATION("Reverse-mode autodiff over callable calls requires XIR inlining before autodiff.");
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
        auto accum_component = [&](Value *value, Value *grad) noexcept {
            if (forward_reachable.contains(value)) {
                accumulate_grad(b, value, project_grad_to_type(b, value->type(), grad));
            }
        };
        switch (inst->op()) {
            case ArithmeticOp::BINARY_ADD:
                accum(arg(0), out_grad);
                accum(arg(1), out_grad);
                break;
            case ArithmeticOp::MATRIX_COMP_ADD:
                accum_component(arg(0), out_grad);
                accum_component(arg(1), out_grad);
                break;
            case ArithmeticOp::BINARY_SUB:
                accum(arg(0), out_grad);
                accum(arg(1), neg(b, arg(1)->type(), out_grad));
                break;
            case ArithmeticOp::MATRIX_COMP_SUB:
                accum_component(arg(0), out_grad);
                accum_component(arg(1), neg(b, type, out_grad));
                break;
            case ArithmeticOp::UNARY_MINUS:
            case ArithmeticOp::MATRIX_COMP_NEG:
                accum(arg(0), neg(b, arg(0)->type(), out_grad));
                break;
            case ArithmeticOp::BINARY_MUL:
                accum(arg(0), mul(b, arg(0)->type(), out_grad, arg(1)));
                accum(arg(1), mul(b, arg(1)->type(), out_grad, arg(0)));
                break;
            case ArithmeticOp::MATRIX_COMP_MUL:
                accum_component(arg(0), mul(b, type, out_grad, lift_value_to_type(b, type, arg(1))));
                accum_component(arg(1), mul(b, type, out_grad, lift_value_to_type(b, type, arg(0))));
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
            case ArithmeticOp::BINARY_DIV: {
                auto lhs_grad = div(b, arg(0)->type(), out_grad, arg(1));
                auto neg_lhs = neg(b, arg(0)->type(), arg(0));
                auto sqr_rhs = mul(b, arg(1)->type(), arg(1), arg(1));
                auto rhs_factor = div(b, arg(1)->type(), neg_lhs, sqr_rhs);
                auto rhs_grad = mul(b, arg(1)->type(), out_grad, rhs_factor);
                accum(arg(0), lhs_grad);
                accum(arg(1), rhs_grad);
                break;
            }
            case ArithmeticOp::MATRIX_COMP_DIV: {
                auto lhs = lift_value_to_type(b, type, arg(0));
                auto rhs = lift_value_to_type(b, type, arg(1));
                auto lhs_grad = div(b, type, out_grad, rhs);
                auto sqr_rhs = mul(b, type, rhs, rhs);
                auto rhs_factor = div(b, type, neg(b, type, lhs), sqr_rhs);
                auto rhs_grad = mul(b, type, out_grad, rhs_factor);
                accum_component(arg(0), lhs_grad);
                accum_component(arg(1), rhs_grad);
                break;
            }
            case ArithmeticOp::BINARY_MOD: {
                auto quotient = div(b, type, arg(0), arg(1));
                auto truncated = b.call(type, ArithmeticOp::TRUNC, {quotient});
                accum(arg(0), out_grad);
                accum(arg(1), neg(b, type, mul(b, type, out_grad, truncated)));
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
            case ArithmeticOp::SMOOTHSTEP: {
                auto edge0 = arg(0);
                auto edge1 = arg(1);
                auto x = arg(2);
                auto denom = sub(b, type, edge1, edge0);
                auto numer = sub(b, type, x, edge0);
                auto t_raw = div(b, type, numer, denom);
                auto t = b.call(type, ArithmeticOp::SATURATE, {t_raw});
                auto t_one_minus_t = mul(b, type, t, sub(b, type, one(type), t));
                auto t_grad = mul(b, type, out_grad, mul(b, type, fp(b, type, 6.0), t_one_minus_t));
                auto denom_sq = mul(b, type, denom, denom);
                accum(edge0, mul(b, type, t_grad, div(b, type, sub(b, type, x, edge1), denom_sq)));
                accum(edge1, mul(b, type, t_grad, div(b, type, neg(b, type, numer), denom_sq)));
                accum(x, div(b, type, t_grad, denom));
                break;
            }
            case ArithmeticOp::STEP:
            case ArithmeticOp::CEIL:
            case ArithmeticOp::FLOOR:
            case ArithmeticOp::TRUNC:
            case ArithmeticOp::ROUND:
            case ArithmeticOp::RINT:
                break;
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
            case ArithmeticOp::POW_INT: {
                auto exp_minus_one = sub(b, arg(1)->type(), arg(1), one(arg(1)->type()));
                auto pow_a = b.call(type, ArithmeticOp::POW_INT, {arg(0), exp_minus_one});
                auto exp = cast_to_matching_shape(b, type, arg(1));
                auto a_grad = mul(b, type, mul(b, type, exp, pow_a), out_grad);
                accum(arg(0), a_grad);
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
            case ArithmeticOp::FACEFORWARD: {
                auto dot = b.call(type->element(), ArithmeticOp::DOT, {arg(2), arg(1)});
                auto cond_scalar = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {dot, zero(type->element())});
                auto cond = type->is_vector() ? broadcast(b, bool_type_of(type), cond_scalar) : cond_scalar;
                accum(arg(0), select(b, type, cond, out_grad, neg(b, type, out_grad)));
                break;
            }
            case ArithmeticOp::REFLECT: {
                auto scale_type = type->element();
                auto two = fp(b, scale_type, 2.0);
                auto n_dot_g = b.call(scale_type, ArithmeticOp::DOT, {arg(1), out_grad});
                auto n_dot_i = b.call(scale_type, ArithmeticOp::DOT, {arg(1), arg(0)});
                auto i_grad_factor = broadcast(b, type, mul(b, scale_type, two, n_dot_g));
                auto n_grad_from_i = mul(b, type, arg(0), broadcast(b, type, n_dot_g));
                auto n_grad_from_out = mul(b, type, out_grad, broadcast(b, type, n_dot_i));
                auto n_grad_sum = add(b, type, n_grad_from_i, n_grad_from_out);
                auto i_grad = sub(b, type, out_grad, mul(b, type, arg(1), i_grad_factor));
                auto n_grad = neg(b, type, mul(b, type, broadcast(b, type, two), n_grad_sum));
                accum(arg(0), i_grad);
                accum(arg(1), n_grad);
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
                auto base_grad = [&] {
                    luisa::fixed_vector<Value *, 8u> args;
                    args.emplace_back(out_grad);
                    args.emplace_back(zero(arg(1)->type()));
                    for (auto idx : indices) { args.emplace_back(idx); }
                    return b.call(arg(0)->type(), ArithmeticOp::INSERT, args);
                }();
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
                auto inv_t = b.call(type, ArithmeticOp::MATRIX_TRANSPOSE, {inst});
                auto inv_t_g = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {inv_t, out_grad});
                auto inv_t_g_inv_t = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {inv_t_g, inv_t});
                accum(arg(0), neg(b, type, inv_t_g_inv_t));
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
                LUISA_ERROR_WITH_LOCATION("Unsupported reverse-mode autodiff arithmetic operation {}.", xir::to_string(inst->op()));
        }
    }

    void emit_backward_if(XIRBuilder &b, IfInst *inst) noexcept {
        auto slot_iter = if_condition_slots.find(inst);
        LUISA_ASSERT(slot_iter != if_condition_slots.end() && slot_iter->second != nullptr,
                     "Missing saved autodiff branch condition.");
        auto condition = b.load(inst->condition()->type(), slot_iter->second);
        auto backward_if = b.if_(condition);
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

    void emit_backward_switch(XIRBuilder &b, SwitchInst *inst) noexcept {
        auto slot_iter = switch_value_slots.find(inst);
        LUISA_ASSERT(slot_iter != switch_value_slots.end() && slot_iter->second != nullptr,
                     "Missing saved autodiff switch value.");
        auto value = b.load(inst->value()->type(), slot_iter->second);
        auto backward_switch = b.switch_(value);
        auto merge_block = backward_switch->create_merge_block();
        {
            b.set_insertion_point(backward_switch->create_default_block());
            emit_backward_instructions(b, switch_default_backward_emit_instructions[inst]);
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        auto &case_emits = switch_case_backward_emit_instructions[inst];
        LUISA_ASSERT(case_emits.size() == inst->case_count(), "Invalid switch case backward instruction list.");
        for (auto i = 0u; i < inst->case_count(); i++) {
            b.set_insertion_point(backward_switch->create_case_block(inst->case_value(i)));
            emit_backward_instructions(b, case_emits[i]);
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
            if (auto switch_inst = inst->isa<SwitchInst>() ? static_cast<SwitchInst *>(inst) : nullptr) {
                emit_backward_switch(b, switch_inst);
                continue;
            }
            if (inst->isa<LoopInst>() || inst->isa<SimpleLoopInst>()) {
                reject_loop_autodiff();
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
        unroll_fixed_trip_loops(entry, merge);
        normalize_cfg_after_early_exit_unrolls();
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

struct TransformForwardAdScope {
    Function *function{};
    FunctionDefinition *definition{};
    Module *module{};
    AutodiffScopeInst *scope{};
    luisa::unordered_map<Value *, luisa::vector<AllocaInst *>> grads;
    luisa::vector<AutodiffIntrinsicInst *> removable_intrinsics;
    size_t changed_count{0u};

    [[nodiscard]] auto n_grads() const noexcept { return scope->n_forward_grads(); }

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

    [[nodiscard]] auto cast_to_matching_shape(XIRBuilder &b, const Type *type, Value *value) noexcept -> Value * {
        if (value->type() == type) { return value; }
        if (type->is_scalar()) {
            LUISA_ASSERT(value->type()->is_scalar(), "Invalid scalar cast.");
            return b.static_cast_(type, value);
        }
        LUISA_ASSERT(type->is_vector(), "Invalid target type.");
        if (value->type()->is_scalar()) {
            return broadcast(b, type, b.static_cast_(type->element(), value));
        }
        LUISA_ASSERT(value->type()->is_vector() && value->type()->dimension() == type->dimension(),
                     "Invalid vector cast.");
        auto result = static_cast<Value *>(zero(type));
        for (auto i = 0u; i < type->dimension(); i++) {
            auto elem = extract(b, value->type()->element(), value, i);
            result = insert(b, type, result, b.static_cast_(type->element(), elem), i);
        }
        return result;
    }

    [[nodiscard]] auto extract(XIRBuilder &b, const Type *type, Value *value, uint32_t i) noexcept -> Value * {
        return b.call(type, ArithmeticOp::EXTRACT, {value, index(i)});
    }

    [[nodiscard]] auto insert(XIRBuilder &b, const Type *type, Value *aggregate, Value *elem, uint32_t i) noexcept -> Value * {
        return b.call(type, ArithmeticOp::INSERT, {aggregate, elem, index(i)});
    }

    [[nodiscard]] auto reduce_sum_all(XIRBuilder &b, Value *value) noexcept -> Value * {
        auto type = value->type();
        if (type->is_scalar()) { return value; }
        if (type->is_vector()) {
            return b.call(type->element(), ArithmeticOp::REDUCE_SUM, {value});
        }
        if (type->is_matrix()) {
            auto elem_type = type->element();
            auto column_type = Type::vector(elem_type, type->dimension());
            auto sum = static_cast<Value *>(zero(elem_type));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto column = extract(b, column_type, value, i);
                auto column_sum = b.call(elem_type, ArithmeticOp::REDUCE_SUM, {column});
                sum = add(b, elem_type, sum, column_sum);
            }
            return sum;
        }
        LUISA_ERROR_WITH_LOCATION("Cannot reduce gradient of type {} to a scalar.", type->description());
    }

    [[nodiscard]] auto project_grad_to_type(XIRBuilder &b, const Type *type, Value *grad) noexcept -> Value * {
        if (type == grad->type()) { return grad; }
        if (type->is_scalar()) {
            return b.static_cast_if_necessary(type, reduce_sum_all(b, grad));
        }
        if ((type->is_vector() || type->is_matrix()) && grad->type()->is_scalar()) {
            return broadcast(b, type, grad);
        }
        if (type->is_vector() && grad->type()->is_vector() && type->dimension() == grad->type()->dimension()) {
            auto result = static_cast<Value *>(zero(type));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto elem = extract(b, grad->type()->element(), grad, i);
                result = insert(b, type, result, b.static_cast_if_necessary(type->element(), elem), i);
            }
            return result;
        }
        if (type->is_matrix() && grad->type()->is_matrix() && type->dimension() == grad->type()->dimension()) {
            auto column_type = Type::vector(type->element(), type->dimension());
            auto grad_column_type = Type::vector(grad->type()->element(), grad->type()->dimension());
            auto result = static_cast<Value *>(zero(type));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto column = extract(b, grad_column_type, grad, i);
                auto projected_column = project_grad_to_type(b, column_type, column);
                result = insert(b, type, result, projected_column, i);
            }
            return result;
        }
        LUISA_ERROR_WITH_LOCATION("Cannot project gradient from type {} to {}.", grad->type()->description(), type->description());
    }

    [[nodiscard]] auto lift_value_to_type(XIRBuilder &b, const Type *type, Value *value) noexcept -> Value * {
        if (type == value->type()) { return value; }
        if ((type->is_vector() || type->is_matrix()) && value->type()->is_scalar()) {
            return broadcast(b, type, value);
        }
        return value;
    }

    [[nodiscard]] auto ensure_grad_slots(Value *value) noexcept -> luisa::vector<AllocaInst *> & {
        LUISA_ASSERT(value != nullptr && is_differentiable_type(value->type()), "Invalid forward gradient value.");
        LUISA_ASSERT(!value->isa<GEPInst>(), "GEP gradient slots are represented by their base.");
        if (auto iter = grads.find(value); iter != grads.end()) { return iter->second; }
        XIRBuilder b;
        b.set_insertion_point(definition->body_block()->instructions().head_sentinel());
        luisa::vector<AllocaInst *> slots;
        slots.reserve(n_grads());
        for (auto i = 0u; i < n_grads(); i++) {
            auto slot = b.alloca_local(value->type());
            b.store(slot, zero(value->type()));
            slots.emplace_back(slot);
        }
        auto [iter, inserted] = grads.emplace(value, std::move(slots));
        LUISA_DEBUG_ASSERT(inserted, "Forward gradient slots already exist.");
        return iter->second;
    }

    [[nodiscard]] auto grad_lvalue(XIRBuilder &b, Value *value, size_t i) noexcept -> Value * {
        LUISA_ASSERT(i < n_grads(), "Forward gradient index out of range.");
        if (value->isa<GEPInst>()) {
            auto gep = static_cast<GEPInst *>(value);
            auto base = grad_lvalue(b, gep->base(), i);
            luisa::fixed_vector<Value *, 8u> indices;
            for (auto use : gep->index_uses()) { indices.emplace_back(use->value()); }
            return b.gep(value->type(), base, indices);
        }
        return ensure_grad_slots(value)[i];
    }

    [[nodiscard]] auto load_grad(XIRBuilder &b, Value *value, size_t i) noexcept -> Value * {
        if (value == nullptr || !is_differentiable_type(value->type())) { return nullptr; }
        return b.load(value->type(), grad_lvalue(b, value, i));
    }

    [[nodiscard]] auto load_grad_or_zero(XIRBuilder &b, Value *value, size_t i, const Type *type) noexcept -> Value * {
        auto grad = load_grad(b, value, i);
        return grad == nullptr ? static_cast<Value *>(zero(type)) : grad;
    }

    void store_grad(XIRBuilder &b, Value *value, size_t i, Value *grad) noexcept {
        if (value == nullptr || grad == nullptr || !is_differentiable_type(value->type())) { return; }
        LUISA_ASSERT(value->type() == grad->type(), "Forward gradient type mismatch: {} vs {}.", value->type()->description(), grad->type()->description());
        b.store(grad_lvalue(b, value, i), grad);
    }

    void set_grads(XIRBuilder &b, Value *value, luisa::span<Value *const> values) noexcept {
        if (value == nullptr || !is_differentiable_type(value->type())) { return; }
        LUISA_ASSERT(values.size() == n_grads(), "Invalid forward gradient count.");
        for (auto i = 0u; i < values.size(); i++) { store_grad(b, value, i, values[i]); }
    }

    void zero_grads(XIRBuilder &b, Value *value) noexcept {
        if (value == nullptr || !is_differentiable_type(value->type())) { return; }
        for (auto i = 0u; i < n_grads(); i++) { store_grad(b, value, i, zero(value->type())); }
    }

    [[nodiscard]] auto intrinsic_index(Value *value) noexcept -> size_t {
        auto idx = constant_i64(value);
        LUISA_ASSERT(idx && *idx >= 0 && static_cast<size_t>(*idx) < n_grads(), "Invalid forward gradient index.");
        return static_cast<size_t>(*idx);
    }

    void transform_intrinsic(XIRBuilder &b, AutodiffIntrinsicInst *inst) noexcept {
        switch (inst->op()) {
            case AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT: {
                LUISA_ASSERT(inst->operand_count() == n_grads() + 1u, "Invalid propagate_gradient operand count.");
                luisa::fixed_vector<Value *, 16u> values;
                values.reserve(n_grads());
                for (auto i = 0u; i < n_grads(); i++) {
                    auto grad = inst->operand(i + 1u);
                    LUISA_ASSERT(grad->type() == inst->operand(0)->type(), "Invalid propagate_gradient operand type.");
                    values.emplace_back(grad);
                }
                set_grads(b, inst->operand(0), values);
                removable_intrinsics.emplace_back(inst);
                break;
            }
            case AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT: {
                LUISA_ASSERT(inst->operand_count() == 2u, "Invalid output_gradient operand count.");
                auto idx = intrinsic_index(inst->operand(1));
                auto grad = load_grad(b, inst->operand(0), idx);
                if (grad == nullptr) { grad = zero(inst->type()); }
                inst->replace_all_uses_with(grad);
                removable_intrinsics.emplace_back(inst);
                break;
            }
            case AutodiffIntrinsicOp::AUTODIFF_DETACH:
                LUISA_ASSERT(inst->operand_count() == 1u, "Invalid detach operand count.");
                zero_grads(b, inst);
                removable_intrinsics.emplace_back(inst);
                break;
            case AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT:
            case AutodiffIntrinsicOp::AUTODIFF_GRADIENT:
            case AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER:
            case AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT:
            case AutodiffIntrinsicOp::AUTODIFF_BACKWARD:
                LUISA_ERROR_WITH_LOCATION("Reverse-mode autodiff intrinsic {} cannot appear in a forward-mode autodiff scope.", xir::to_string(inst->op()));
        }
    }

    void transform_cast(XIRBuilder &b, CastInst *inst) noexcept {
        if (!is_differentiable_type(inst->type())) { return; }
        switch (inst->op()) {
            case CastOp::STATIC_CAST: break;
            case CastOp::BITWISE_CAST:
                LUISA_ERROR_WITH_LOCATION("Forward-mode autodiff over differentiable bitwise casts is not supported.");
        }
        luisa::fixed_vector<Value *, 16u> values;
        values.reserve(n_grads());
        for (auto i = 0u; i < n_grads(); i++) {
            auto grad = load_grad_or_zero(b, inst->value(), i, inst->value()->type());
            values.emplace_back(project_grad_to_type(b, inst->type(), grad));
        }
        set_grads(b, inst, values);
    }

    void transform_arithmetic(XIRBuilder &b, ArithmeticInst *inst) noexcept {
        if (!is_differentiable_type(inst->type())) { return; }
        auto type = inst->type();
        auto arg = [&](size_t i) noexcept { return inst->operand(i); };
        auto g = [&](Value *value, size_t i) noexcept { return load_grad_or_zero(b, value, i, value->type()); };
        auto component_g = [&](Value *value, size_t i) noexcept {
            return lift_value_to_type(b, type, g(value, i));
        };
        auto component_v = [&](Value *value) noexcept {
            return lift_value_to_type(b, type, value);
        };
        luisa::fixed_vector<Value *, 16u> values;
        values.reserve(n_grads());
        for (auto i = 0u; i < n_grads(); i++) {
            Value *grad = nullptr;
            switch (inst->op()) {
                case ArithmeticOp::BINARY_ADD:
                    grad = add(b, type, g(arg(0), i), g(arg(1), i));
                    break;
                case ArithmeticOp::MATRIX_COMP_ADD:
                    grad = add(b, type, component_g(arg(0), i), component_g(arg(1), i));
                    break;
                case ArithmeticOp::BINARY_SUB:
                    grad = sub(b, type, g(arg(0), i), g(arg(1), i));
                    break;
                case ArithmeticOp::MATRIX_COMP_SUB:
                    grad = sub(b, type, component_g(arg(0), i), component_g(arg(1), i));
                    break;
                case ArithmeticOp::UNARY_MINUS:
                case ArithmeticOp::MATRIX_COMP_NEG:
                    grad = neg(b, type, g(arg(0), i));
                    break;
                case ArithmeticOp::BINARY_MUL: {
                    auto lhs = mul(b, type, g(arg(0), i), arg(1));
                    auto rhs = mul(b, type, g(arg(1), i), arg(0));
                    grad = add(b, type, lhs, rhs);
                    break;
                }
                case ArithmeticOp::MATRIX_COMP_MUL: {
                    auto lhs = mul(b, type, component_g(arg(0), i), component_v(arg(1)));
                    auto rhs = mul(b, type, component_g(arg(1), i), component_v(arg(0)));
                    grad = add(b, type, lhs, rhs);
                    break;
                }
                case ArithmeticOp::BINARY_DIV: {
                    auto lhs = mul(b, type, g(arg(0), i), arg(1));
                    auto rhs = mul(b, type, g(arg(1), i), arg(0));
                    auto numer = sub(b, type, lhs, rhs);
                    auto denom = mul(b, type, arg(1), arg(1));
                    grad = div(b, type, numer, denom);
                    break;
                }
                case ArithmeticOp::MATRIX_COMP_DIV: {
                    auto lhs_value = component_v(arg(0));
                    auto rhs_value = component_v(arg(1));
                    auto lhs = mul(b, type, component_g(arg(0), i), rhs_value);
                    auto rhs = mul(b, type, component_g(arg(1), i), lhs_value);
                    auto numer = sub(b, type, lhs, rhs);
                    auto denom = mul(b, type, rhs_value, rhs_value);
                    grad = div(b, type, numer, denom);
                    break;
                }
                case ArithmeticOp::BINARY_MOD: {
                    auto quotient = div(b, type, arg(0), arg(1));
                    auto truncated = b.call(type, ArithmeticOp::TRUNC, {quotient});
                    grad = sub(b, type, g(arg(0), i), mul(b, type, truncated, g(arg(1), i)));
                    break;
                }
                case ArithmeticOp::SELECT:
                    grad = select(b, type, arg(2), g(arg(1), i), g(arg(0), i));
                    break;
                case ArithmeticOp::MIN: {
                    auto cond = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), arg(1)});
                    grad = select(b, type, cond, g(arg(0), i), g(arg(1), i));
                    break;
                }
                case ArithmeticOp::MAX: {
                    auto cond = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {arg(0), arg(1)});
                    grad = select(b, type, cond, g(arg(0), i), g(arg(1), i));
                    break;
                }
                case ArithmeticOp::CLAMP: {
                    auto lt_min = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), arg(1)});
                    auto gt_max = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {arg(0), arg(2)});
                    auto lo = select(b, type, lt_min, g(arg(1), i), g(arg(0), i));
                    grad = select(b, type, gt_max, g(arg(2), i), lo);
                    break;
                }
                case ArithmeticOp::SATURATE: {
                    auto gt_zero = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {arg(0), zero(type)});
                    auto lt_one = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), one(type)});
                    auto between = b.call(bool_type_of(type), ArithmeticOp::BINARY_BIT_AND, {gt_zero, lt_one});
                    grad = select(b, type, between, g(arg(0), i), zero(type));
                    break;
                }
                case ArithmeticOp::LERP: {
                    auto one_minus_t = sub(b, type, one(type), arg(2));
                    auto da = mul(b, type, g(arg(0), i), one_minus_t);
                    auto db = mul(b, type, g(arg(1), i), arg(2));
                    auto dt = mul(b, type, g(arg(2), i), sub(b, type, arg(1), arg(0)));
                    grad = add(b, type, add(b, type, da, db), dt);
                    break;
                }
                case ArithmeticOp::SMOOTHSTEP: {
                    auto edge0 = arg(0);
                    auto edge1 = arg(1);
                    auto x = arg(2);
                    auto denom = sub(b, type, edge1, edge0);
                    auto numer = sub(b, type, x, edge0);
                    auto t_raw = div(b, type, numer, denom);
                    auto t = b.call(type, ArithmeticOp::SATURATE, {t_raw});
                    auto d_numer = sub(b, type, g(x, i), g(edge0, i));
                    auto d_denom = sub(b, type, g(edge1, i), g(edge0, i));
                    auto d_t_raw = div(b, type, sub(b, type, mul(b, type, d_numer, denom), mul(b, type, numer, d_denom)), mul(b, type, denom, denom));
                    auto gt_zero = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER, {t_raw, zero(type)});
                    auto lt_one = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {t_raw, one(type)});
                    auto between = b.call(bool_type_of(type), ArithmeticOp::BINARY_BIT_AND, {gt_zero, lt_one});
                    auto d_t = select(b, type, between, d_t_raw, zero(type));
                    auto factor = mul(b, type, fp(b, type, 6.0), mul(b, type, t, sub(b, type, one(type), t)));
                    grad = mul(b, type, factor, d_t);
                    break;
                }
                case ArithmeticOp::STEP:
                case ArithmeticOp::CEIL:
                case ArithmeticOp::FLOOR:
                case ArithmeticOp::TRUNC:
                case ArithmeticOp::ROUND:
                case ArithmeticOp::RINT:
                    grad = zero(type);
                    break;
                case ArithmeticOp::ABS: {
                    auto cond = b.call(bool_type_of(type), ArithmeticOp::BINARY_GREATER_EQUAL, {arg(0), zero(type)});
                    grad = select(b, type, cond, g(arg(0), i), neg(b, type, g(arg(0), i)));
                    break;
                }
                case ArithmeticOp::SIN:
                    grad = mul(b, type, b.call(type, ArithmeticOp::COS, {arg(0)}), g(arg(0), i));
                    break;
                case ArithmeticOp::COS:
                    grad = mul(b, type, neg(b, type, b.call(type, ArithmeticOp::SIN, {arg(0)})), g(arg(0), i));
                    break;
                case ArithmeticOp::TAN: {
                    auto c = b.call(type, ArithmeticOp::COS, {arg(0)});
                    grad = div(b, type, g(arg(0), i), mul(b, type, c, c));
                    break;
                }
                case ArithmeticOp::SINH:
                    grad = mul(b, type, b.call(type, ArithmeticOp::COSH, {arg(0)}), g(arg(0), i));
                    break;
                case ArithmeticOp::COSH:
                    grad = mul(b, type, b.call(type, ArithmeticOp::SINH, {arg(0)}), g(arg(0), i));
                    break;
                case ArithmeticOp::TANH: {
                    auto c = b.call(type, ArithmeticOp::COSH, {arg(0)});
                    grad = div(b, type, g(arg(0), i), mul(b, type, c, c));
                    break;
                }
                case ArithmeticOp::ASIN: {
                    auto x2 = mul(b, type, arg(0), arg(0));
                    auto denom = b.call(type, ArithmeticOp::SQRT, {sub(b, type, one(type), x2)});
                    grad = div(b, type, g(arg(0), i), denom);
                    break;
                }
                case ArithmeticOp::ACOS: {
                    auto x2 = mul(b, type, arg(0), arg(0));
                    auto denom = b.call(type, ArithmeticOp::SQRT, {sub(b, type, one(type), x2)});
                    grad = neg(b, type, div(b, type, g(arg(0), i), denom));
                    break;
                }
                case ArithmeticOp::ATAN: {
                    auto x2 = mul(b, type, arg(0), arg(0));
                    grad = div(b, type, g(arg(0), i), add(b, type, one(type), x2));
                    break;
                }
                case ArithmeticOp::ASINH: {
                    auto x2 = mul(b, type, arg(0), arg(0));
                    auto denom = b.call(type, ArithmeticOp::SQRT, {add(b, type, one(type), x2)});
                    grad = div(b, type, g(arg(0), i), denom);
                    break;
                }
                case ArithmeticOp::ACOSH: {
                    auto x2 = mul(b, type, arg(0), arg(0));
                    auto denom = b.call(type, ArithmeticOp::SQRT, {sub(b, type, x2, one(type))});
                    grad = div(b, type, g(arg(0), i), denom);
                    break;
                }
                case ArithmeticOp::ATANH: {
                    auto x2 = mul(b, type, arg(0), arg(0));
                    grad = div(b, type, g(arg(0), i), sub(b, type, one(type), x2));
                    break;
                }
                case ArithmeticOp::ATAN2: {
                    auto y = arg(0);
                    auto x = arg(1);
                    auto xx = mul(b, x->type(), x, x);
                    auto yy = mul(b, y->type(), y, y);
                    auto sum = add(b, type, xx, yy);
                    auto y_part = mul(b, type, div(b, type, x, sum), g(y, i));
                    auto x_part = mul(b, type, div(b, type, neg(b, type, y), sum), g(x, i));
                    grad = add(b, type, y_part, x_part);
                    break;
                }
                case ArithmeticOp::EXP:
                    grad = mul(b, type, inst, g(arg(0), i));
                    break;
                case ArithmeticOp::EXP2:
                    grad = mul(b, type, mul(b, type, fp(b, type, std::log(2.0)), inst), g(arg(0), i));
                    break;
                case ArithmeticOp::EXP10:
                    grad = mul(b, type, mul(b, type, fp(b, type, std::log(10.0)), inst), g(arg(0), i));
                    break;
                case ArithmeticOp::LOG:
                    grad = div(b, type, g(arg(0), i), arg(0));
                    break;
                case ArithmeticOp::LOG2:
                    grad = div(b, type, mul(b, type, g(arg(0), i), fp(b, type, 1.0 / std::log(2.0))), arg(0));
                    break;
                case ArithmeticOp::LOG10:
                    grad = div(b, type, mul(b, type, g(arg(0), i), fp(b, type, 1.0 / std::log(10.0))), arg(0));
                    break;
                case ArithmeticOp::POW: {
                    auto log_a = b.call(type, ArithmeticOp::LOG, {arg(0)});
                    auto lhs = mul(b, type, g(arg(1), i), log_a);
                    auto rhs = div(b, type, mul(b, type, arg(1), g(arg(0), i)), arg(0));
                    grad = mul(b, type, inst, add(b, type, lhs, rhs));
                    break;
                }
                case ArithmeticOp::POW_INT: {
                    auto exp_minus_one = sub(b, arg(1)->type(), arg(1), one(arg(1)->type()));
                    auto pow_a = b.call(type, ArithmeticOp::POW_INT, {arg(0), exp_minus_one});
                    auto exp = cast_to_matching_shape(b, type, arg(1));
                    grad = mul(b, type, mul(b, type, exp, pow_a), g(arg(0), i));
                    break;
                }
                case ArithmeticOp::SQRT:
                    grad = div(b, type, g(arg(0), i), add(b, type, inst, inst));
                    break;
                case ArithmeticOp::RSQRT: {
                    auto sqrt_x = b.call(type, ArithmeticOp::SQRT, {arg(0)});
                    auto twice_x = add(b, type, arg(0), arg(0));
                    auto denom = mul(b, type, twice_x, sqrt_x);
                    grad = div(b, type, neg(b, type, g(arg(0), i)), denom);
                    break;
                }
                case ArithmeticOp::FRACT:
                    grad = g(arg(0), i);
                    break;
                case ArithmeticOp::FMA: {
                    auto a = mul(b, type, g(arg(0), i), arg(1));
                    auto c = mul(b, type, arg(0), g(arg(1), i));
                    grad = add(b, type, add(b, type, a, c), g(arg(2), i));
                    break;
                }
                case ArithmeticOp::COPYSIGN: {
                    auto x_neg = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(0), zero(type)});
                    auto y_neg = b.call(bool_type_of(type), ArithmeticOp::BINARY_LESS, {arg(1), zero(type)});
                    auto same = b.call(bool_type_of(type), ArithmeticOp::BINARY_EQUAL, {x_neg, y_neg});
                    grad = select(b, type, same, g(arg(0), i), neg(b, type, g(arg(0), i)));
                    break;
                }
                case ArithmeticOp::DOT: {
                    auto lhs = b.call(type, ArithmeticOp::DOT, {g(arg(0), i), arg(1)});
                    auto rhs = b.call(type, ArithmeticOp::DOT, {arg(0), g(arg(1), i)});
                    grad = add(b, type, lhs, rhs);
                    break;
                }
                case ArithmeticOp::CROSS: {
                    auto lhs = b.call(type, ArithmeticOp::CROSS, {g(arg(0), i), arg(1)});
                    auto rhs = b.call(type, ArithmeticOp::CROSS, {arg(0), g(arg(1), i)});
                    grad = add(b, type, lhs, rhs);
                    break;
                }
                case ArithmeticOp::LENGTH: {
                    auto n = b.call(arg(0)->type(), ArithmeticOp::NORMALIZE, {arg(0)});
                    grad = b.call(type, ArithmeticOp::DOT, {n, g(arg(0), i)});
                    break;
                }
                case ArithmeticOp::LENGTH_SQUARED: {
                    auto twice_x = add(b, arg(0)->type(), arg(0), arg(0));
                    grad = b.call(type, ArithmeticOp::DOT, {twice_x, g(arg(0), i)});
                    break;
                }
                case ArithmeticOp::NORMALIZE: {
                    auto n = b.call(type, ArithmeticOp::NORMALIZE, {arg(0)});
                    auto dot = b.call(type->element(), ArithmeticOp::DOT, {n, g(arg(0), i)});
                    auto numer = sub(b, type, g(arg(0), i), mul(b, type, n, broadcast(b, type, dot)));
                    auto len = b.call(type->element(), ArithmeticOp::LENGTH, {arg(0)});
                    grad = div(b, type, numer, broadcast(b, type, len));
                    break;
                }
                case ArithmeticOp::FACEFORWARD: {
                    auto dot = b.call(type->element(), ArithmeticOp::DOT, {arg(2), arg(1)});
                    auto cond_scalar = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {dot, zero(type->element())});
                    auto cond = type->is_vector() ? broadcast(b, bool_type_of(type), cond_scalar) : cond_scalar;
                    grad = select(b, type, cond, g(arg(0), i), neg(b, type, g(arg(0), i)));
                    break;
                }
                case ArithmeticOp::REFLECT: {
                    auto scale_type = type->element();
                    auto two = fp(b, scale_type, 2.0);
                    auto n_dot_i = b.call(scale_type, ArithmeticOp::DOT, {arg(1), arg(0)});
                    auto dn_dot_i = b.call(scale_type, ArithmeticOp::DOT, {g(arg(1), i), arg(0)});
                    auto n_dot_di = b.call(scale_type, ArithmeticOp::DOT, {arg(1), g(arg(0), i)});
                    auto d_dot = add(b, scale_type, dn_dot_i, n_dot_di);
                    auto a = mul(b, type, g(arg(1), i), broadcast(b, type, n_dot_i));
                    auto c = mul(b, type, arg(1), broadcast(b, type, d_dot));
                    grad = sub(b, type, g(arg(0), i), mul(b, type, broadcast(b, type, two), add(b, type, a, c)));
                    break;
                }
                case ArithmeticOp::REDUCE_SUM:
                    grad = b.call(type, ArithmeticOp::REDUCE_SUM, {g(arg(0), i)});
                    break;
                case ArithmeticOp::REDUCE_PRODUCT: {
                    auto value_type = arg(0)->type();
                    if (!value_type->is_vector()) {
                        grad = zero(type);
                        break;
                    }
                    auto elem_type = value_type->element();
                    grad = zero(type);
                    for (auto out_i = 0u; out_i < value_type->dimension(); out_i++) {
                        auto term = extract(b, elem_type, g(arg(0), i), out_i);
                        for (auto j = 0u; j < value_type->dimension(); j++) {
                            if (j != out_i) { term = mul(b, elem_type, term, extract(b, elem_type, arg(0), j)); }
                        }
                        grad = add(b, type, grad, term);
                    }
                    break;
                }
                case ArithmeticOp::REDUCE_MIN:
                case ArithmeticOp::REDUCE_MAX: {
                    auto value_type = arg(0)->type();
                    auto reduced = b.call(type, inst->op(), {arg(0)});
                    auto cond = b.call(bool_type_of(value_type), ArithmeticOp::BINARY_EQUAL, {reduced, arg(0)});
                    auto masked = select(b, value_type, cond, g(arg(0), i), zero(value_type));
                    grad = b.call(type, ArithmeticOp::REDUCE_SUM, {masked});
                    break;
                }
                case ArithmeticOp::EXTRACT: {
                    luisa::fixed_vector<Value *, 8u> args;
                    args.emplace_back(g(arg(0), i));
                    for (auto j = 1u; j < inst->operand_count(); j++) { args.emplace_back(arg(j)); }
                    grad = b.call(type, ArithmeticOp::EXTRACT, args);
                    break;
                }
                case ArithmeticOp::INSERT: {
                    luisa::fixed_vector<Value *, 8u> args;
                    args.emplace_back(g(arg(0), i));
                    args.emplace_back(g(arg(1), i));
                    for (auto j = 2u; j < inst->operand_count(); j++) { args.emplace_back(arg(j)); }
                    grad = b.call(type, ArithmeticOp::INSERT, args);
                    break;
                }
                case ArithmeticOp::SHUFFLE: {
                    luisa::fixed_vector<Value *, 8u> args;
                    args.reserve(inst->operand_count());
                    args.emplace_back(g(arg(0), i));
                    for (auto j = 1u; j < inst->operand_count(); j++) { args.emplace_back(arg(j)); }
                    grad = b.call(type, ArithmeticOp::SHUFFLE, args);
                    break;
                }
                case ArithmeticOp::AGGREGATE: {
                    luisa::fixed_vector<Value *, 16u> args;
                    args.reserve(inst->operand_count());
                    for (auto j = 0u; j < inst->operand_count(); j++) {
                        args.emplace_back(load_grad_or_zero(b, arg(j), i, arg(j)->type()));
                    }
                    grad = b.call(type, ArithmeticOp::AGGREGATE, args);
                    break;
                }
                case ArithmeticOp::MATRIX_TRANSPOSE:
                    grad = b.call(type, ArithmeticOp::MATRIX_TRANSPOSE, {g(arg(0), i)});
                    break;
                case ArithmeticOp::MATRIX_LINALG_MUL: {
                    auto lhs = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {g(arg(0), i), arg(1)});
                    auto rhs = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {arg(0), g(arg(1), i)});
                    grad = add(b, type, lhs, rhs);
                    break;
                }
                case ArithmeticOp::MATRIX_DETERMINANT: {
                    auto inv = b.call(arg(0)->type(), ArithmeticOp::MATRIX_INVERSE, {arg(0)});
                    auto inv_t = b.call(arg(0)->type(), ArithmeticOp::MATRIX_TRANSPOSE, {inv});
                    auto hadamard = mul(b, arg(0)->type(), inv_t, g(arg(0), i));
                    auto trace = reduce_sum_all(b, hadamard);
                    grad = mul(b, type, inst, trace);
                    break;
                }
                case ArithmeticOp::MATRIX_INVERSE: {
                    auto inv = inst;
                    auto lhs = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {inv, g(arg(0), i)});
                    auto rhs = b.call(type, ArithmeticOp::MATRIX_LINALG_MUL, {lhs, inv});
                    grad = neg(b, type, rhs);
                    break;
                }
                case ArithmeticOp::OUTER_PRODUCT: {
                    auto lhs = b.call(type, ArithmeticOp::OUTER_PRODUCT, {g(arg(0), i), arg(1)});
                    auto rhs = b.call(type, ArithmeticOp::OUTER_PRODUCT, {arg(0), g(arg(1), i)});
                    grad = add(b, type, lhs, rhs);
                    break;
                }
                default:
                    LUISA_ERROR_WITH_LOCATION("Unsupported forward-mode autodiff arithmetic operation {}.", xir::to_string(inst->op()));
            }
            values.emplace_back(grad);
        }
        set_grads(b, inst, values);
    }

    void transform_instruction(Instruction *inst) noexcept {
        XIRBuilder b;
        b.set_insertion_point(inst);
        if (auto intrinsic = inst->isa<AutodiffIntrinsicInst>() ? static_cast<AutodiffIntrinsicInst *>(inst) : nullptr) {
            transform_intrinsic(b, intrinsic);
        } else if (auto arith = inst->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst) : nullptr) {
            transform_arithmetic(b, arith);
        } else if (auto cast = inst->isa<CastInst>() ? static_cast<CastInst *>(inst) : nullptr) {
            transform_cast(b, cast);
        } else if (auto load = inst->isa<LoadInst>() ? static_cast<LoadInst *>(inst) : nullptr) {
            if (is_differentiable_type(load->type())) {
                luisa::fixed_vector<Value *, 16u> values;
                values.reserve(n_grads());
                for (auto i = 0u; i < n_grads(); i++) {
                    values.emplace_back(b.load(load->type(), grad_lvalue(b, load->variable(), i)));
                }
                set_grads(b, load, values);
            }
        } else if (auto store = inst->isa<StoreInst>() ? static_cast<StoreInst *>(inst) : nullptr) {
            if (is_differentiable_type(store->variable()->type())) {
                for (auto i = 0u; i < n_grads(); i++) {
                    auto grad = load_grad_or_zero(b, store->value(), i, store->value()->type());
                    b.store(grad_lvalue(b, store->variable(), i), grad);
                }
            }
        } else if (auto call = inst->isa<CallInst>() ? static_cast<CallInst *>(inst) : nullptr) {
            auto relevant = call->type() != nullptr && is_differentiable_type(call->type());
            for (auto use : call->argument_uses()) {
                auto value = use->value();
                relevant |= value != nullptr && is_differentiable_type(value->type());
            }
            if (relevant) {
                LUISA_ERROR_WITH_LOCATION("Forward-mode autodiff over callable calls requires XIR inlining before autodiff.");
            }
        } else if (inst->isa<PhiInst>() && is_differentiable_type(inst->type())) {
            LUISA_ERROR_WITH_LOCATION("Forward-mode autodiff over differentiable PHI nodes requires running XIR autodiff before mem2reg.");
        } else if (inst->isa<AutodiffScopeInst>()) {
            LUISA_ERROR_WITH_LOCATION("Nested autodiff scopes are not supported by XIR forward-mode autodiff.");
        }
    }

    void transform_region(BasicBlock *block, BasicBlock *merge, luisa::unordered_set<BasicBlock *> &visited) noexcept {
        if (block == nullptr || block == merge || !visited.emplace(block).second) { return; }
        luisa::vector<Instruction *> instructions;
        for (auto inst : block->instructions()) {
            instructions.emplace_back(inst);
        }
        for (auto inst : instructions) {
            transform_instruction(inst);
        }
        auto term = block->terminator();
        if (term == nullptr) { return; }
        if (auto if_inst = term->isa<IfInst>() ? static_cast<IfInst *>(term) : nullptr) {
            transform_region(if_inst->true_block(), if_inst->merge_block(), visited);
            transform_region(if_inst->false_block(), if_inst->merge_block(), visited);
            transform_region(if_inst->merge_block(), merge, visited);
            return;
        }
        if (auto switch_inst = term->isa<SwitchInst>() ? static_cast<SwitchInst *>(term) : nullptr) {
            transform_region(switch_inst->default_block(), switch_inst->merge_block(), visited);
            for (auto i = 0u; i < switch_inst->case_count(); i++) {
                transform_region(switch_inst->case_block(i), switch_inst->merge_block(), visited);
            }
            transform_region(switch_inst->merge_block(), merge, visited);
            return;
        }
        if (auto loop = term->isa<LoopInst>() ? static_cast<LoopInst *>(term) : nullptr) {
            transform_region(loop->prepare_block(), loop->merge_block(), visited);
            transform_region(loop->merge_block(), merge, visited);
            return;
        }
        if (auto loop = term->isa<SimpleLoopInst>() ? static_cast<SimpleLoopInst *>(term) : nullptr) {
            transform_region(loop->body_block(), loop->merge_block(), visited);
            transform_region(loop->merge_block(), merge, visited);
            return;
        }
        block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            transform_region(succ, merge, visited);
        });
    }

    void remove_intrinsics() noexcept {
        std::sort(removable_intrinsics.begin(), removable_intrinsics.end());
        removable_intrinsics.erase(std::unique(removable_intrinsics.begin(), removable_intrinsics.end()), removable_intrinsics.end());
        for (auto inst : removable_intrinsics) {
            if (!inst->is_linked()) { continue; }
            if (inst->op() == AutodiffIntrinsicOp::AUTODIFF_DETACH) {
                inst->replace_all_uses_with(inst->operand(0));
            }
            inst->remove_self();
            changed_count++;
        }
    }

    [[nodiscard]] auto run() noexcept -> size_t {
        LUISA_ASSERT(scope->is_forward(), "Invalid forward autodiff scope.");
        auto entry = scope->entry_block();
        auto merge = scope->merge_block();
        LUISA_ASSERT(entry != nullptr && merge != nullptr, "Invalid autodiff scope.");
        luisa::unordered_set<BasicBlock *> visited;
        transform_region(entry, merge, visited);
        remove_intrinsics();
        auto parent = scope->parent_block();
        scope->remove_self();
        XIRBuilder b;
        b.set_insertion_point(parent);
        b.br(entry);
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
        if (!function->definition()) { return info; }
        auto scopes = locate_autodiff_scopes();
        for (auto scope : scopes) {
            if (scope->is_forward()) {
                if (!options.run_forward) { continue; }
                TransformForwardAdScope transform{.function = function,
                                                  .definition = function->definition(),
                                                  .module = function->parent_module(),
                                                  .scope = scope};
                info.transformed_scope_count++;
                info.removed_instruction_count += transform.run();
                continue;
            }
            if (!options.run_backward) { continue; }
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
