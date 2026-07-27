#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/undefined.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>

#include <cmath>
#include <limits>
#include <bit>
#include <cstring>

namespace luisa::compute::xir {

namespace detail {

template<typename T>
[[nodiscard]] static T load_constant_scalar(const void *data) noexcept {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

// Check if a Constant has a specific value
[[nodiscard]] static bool is_const_value(const Value *v, int32_t expected) noexcept {
    if (!v->isa<Constant>()) return false;
    auto c = static_cast<const Constant *>(v);
    auto t = c->type();
    auto check_scalar = [expected](const Type *st, const void *data) noexcept {
        if (st->is_int32()) {
            return load_constant_scalar<int32_t>(data) == expected;
        }
        if (st->is_uint32()) {
            return static_cast<int32_t>(
                       load_constant_scalar<uint32_t>(data)) == expected;
        }
        if (st->is_float32()) {
            return load_constant_scalar<float>(data) ==
                   static_cast<float>(expected);
        }
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

// Unlike equality with zero, this distinguishes +0 from -0. The identity
// x - (+0) == x preserves signed zero, while x - (-0) does not.
[[nodiscard]] static bool is_const_positive_float_zero(const Value *v) noexcept {
    if (!v->isa<Constant>()) return false;
    auto c = static_cast<const Constant *>(v);
    auto t = c->type();
    auto check_scalar = [](const Type *st, const void *data) noexcept {
        if (st->is_float32()) {
            auto x = load_constant_scalar<float>(data);
            return x == 0.0f && !std::signbit(x);
        }
        if (st->is_float64()) {
            auto x = load_constant_scalar<double>(data);
            return x == 0.0 && !std::signbit(x);
        }
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

[[nodiscard]] static bool is_float_like(const Type *type) noexcept {
    return type != nullptr &&
           (type->is_float_or_float_vector() ||
            (type->is_matrix() && type->element()->is_float()));
}

[[nodiscard]] static luisa::optional<size_t> decode_constant_index(const Value *value) noexcept {
    uint64_t index = 0u;
    if (!try_decode_constant_nonnegative_integer(value, index)) {
        return luisa::nullopt;
    }
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
        if (index > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return luisa::nullopt;
        }
    }
    return static_cast<size_t>(index);
}

[[nodiscard]] static bool indices_equal(Value *a, Value *b) noexcept {
    if (a == nullptr || b == nullptr) { return false; }
    if (a == b && !a->isa<Constant>()) {
        return a->type() != nullptr && (a->type()->is_int() || a->type()->is_uint());
    }
    auto lhs = decode_constant_index(a);
    auto rhs = decode_constant_index(b);
    return lhs.has_value() && rhs.has_value() && *lhs == *rhs;
}

[[nodiscard]] static bool is_power_of_two(uint64_t v) noexcept {
    return v != 0u && (v & (v - 1u)) == 0u;
}

[[nodiscard]] static uint64_t floor_log2(uint64_t v) noexcept {
    return std::bit_width(v) - 1u;
}

[[nodiscard]] static bool is_unsigned_integer_type(const Type *type) noexcept {
    if (type == nullptr) return false;
    if (type->is_uint32() || type->is_uint64()) return true;
    if (type->is_vector()) {
        auto *elem = type->element();
        return elem != nullptr && (elem->is_uint32() || elem->is_uint64());
    }
    return false;
}

[[nodiscard]] static bool decode_uniform_unsigned_constant(
    const Value *value, uint64_t &result) noexcept {
    if (value == nullptr || !value->isa<Constant>() ||
        !is_unsigned_integer_type(value->type())) {
        return false;
    }
    auto *constant = static_cast<const Constant *>(value);
    auto *type = value->type();
    auto decode_lane = [](const Type *lane_type,
                          const std::byte *data) noexcept -> uint64_t {
        if (lane_type->is_uint32()) {
            uint32_t lane = 0u;
            std::memcpy(&lane, data, sizeof(lane));
            return lane;
        }
        uint64_t lane = 0u;
        std::memcpy(&lane, data, sizeof(lane));
        return lane;
    };
    if (type->is_scalar()) {
        result = decode_lane(
            type, static_cast<const std::byte *>(constant->data()));
        return true;
    }
    auto *element = type->element();
    auto *bytes = static_cast<const std::byte *>(constant->data());
    auto first = decode_lane(element, bytes);
    for (auto lane = 1u; lane < type->dimension(); ++lane) {
        if (decode_lane(element, bytes + lane * element->size()) != first) {
            return false;
        }
    }
    result = first;
    return true;
}

[[nodiscard]] static bool is_proven_nonzero_integer_constant(
    const Value *value) noexcept {
    if (value == nullptr || !value->isa<Constant>() ||
        value->type() == nullptr) {
        return false;
    }
    auto *type = value->type();
    auto *element = type->is_vector() ? type->element() : type;
    if (element == nullptr ||
        !(element->is_int() || element->is_uint())) {
        return false;
    }
    auto lane_count = type->is_vector() ? type->dimension() : 1u;
    auto lane_size = element->size();
    auto *bytes = static_cast<const std::byte *>(
        static_cast<const Constant *>(value)->data());
    for (auto lane = 0u; lane < lane_count; ++lane) {
        auto nonzero = false;
        for (auto byte = 0u; byte < lane_size; ++byte) {
            nonzero |= bytes[lane * lane_size + byte] != std::byte{0};
        }
        if (!nonzero) { return false; }
    }
    return true;
}

// Create a scalar or vector constant where all elements have the same numeric value.
// The created constant has the same type as `like_type` (unsigned integer
// scalar/vector). Typed host values, rather than hand-written little-endian
// bytes, preserve the Constant ABI on every supported host byte order.
[[nodiscard]] static Constant *create_broadcast_constant(
    Module *module, const Type *like_type, uint64_t value) noexcept {
    if (like_type->is_scalar()) {
        if (like_type->is_uint32()) {
            auto lane = static_cast<uint32_t>(value);
            return module->create_constant(like_type, &lane);
        }
        if (like_type->is_uint64()) {
            return module->create_constant(like_type, &value);
        }
        return nullptr;
    }
    if (like_type->is_vector()) {
        auto dim = like_type->dimension();
        auto *element = like_type->element();
        if (element == nullptr ||
            !(element->is_uint32() || element->is_uint64())) {
            return nullptr;
        }
        auto elem_size = element->size();
        luisa::vector<std::byte> elem_data(elem_size, std::byte{0});
        if (element->is_uint32()) {
            auto lane = static_cast<uint32_t>(value);
            std::memcpy(elem_data.data(), &lane, sizeof(lane));
        } else {
            std::memcpy(elem_data.data(), &value, sizeof(value));
        }
        luisa::vector<std::byte> data(dim * elem_size);
        for (size_t i = 0u; i < dim; ++i) {
            std::memcpy(data.data() + i * elem_size, elem_data.data(), elem_size);
        }
        return module->create_constant(like_type, data.data());
    }
    return nullptr;
}

[[nodiscard]] static bool indices_provably_different(Value *a, Value *b) noexcept {
    auto lhs = decode_constant_index(a);
    auto rhs = decode_constant_index(b);
    return lhs.has_value() && rhs.has_value() && *lhs != *rhs;
}

[[nodiscard]] static Value *try_simplify(
    ArithmeticInst *inst, Module *module, XIRBuilder &builder,
    AlgebraicSimplifyOptions options, bool &changed_in_place) noexcept {
    changed_in_place = false;
    auto op = inst->op();
    auto type = inst->type();
    if (type == nullptr) return nullptr;

    switch (op) {
        case ArithmeticOp::BINARY_ADD: {
            if (!is_float_like(type)) {
                if (is_const_zero(inst->operand(1))) return inst->operand(0);
                if (is_const_zero(inst->operand(0))) return inst->operand(1);
            }
            break;
        }
        case ArithmeticOp::BINARY_SUB: {
            if ((!is_float_like(type) && is_const_zero(inst->operand(1))) ||
                (is_float_like(type) && is_const_positive_float_zero(inst->operand(1)))) {
                return inst->operand(0);
            }
            if (inst->operand(0) == inst->operand(1) &&
                (!is_float_like(type) || options.enable_fast_math)) {
                return module->create_constant_zero(type);
            }
            break;
        }
        case ArithmeticOp::BINARY_MUL: {
            if (is_const_one(inst->operand(1))) return inst->operand(0);
            if (is_const_one(inst->operand(0))) return inst->operand(1);
            if (!is_float_like(type) && (is_const_zero(inst->operand(0)) || is_const_zero(inst->operand(1)))) {
                return module->create_constant_zero(type);
            }
            break;
        }
        case ArithmeticOp::BINARY_DIV: {
            if (is_const_one(inst->operand(1))) return inst->operand(0);
            if (!is_float_like(type) && is_const_zero(inst->operand(0))) {
                if (is_proven_nonzero_integer_constant(
                        inst->operand(1))) {
                    return module->create_constant_zero(type);
                }
            }
            // Unsigned integer division by power-of-two constant:
            //   x / pow2 → x >> log2(pow2)
            if (is_unsigned_integer_type(type)) {
                uint64_t divisor = 0u;
                if (decode_uniform_unsigned_constant(
                        inst->operand(1), divisor) &&
                    divisor > 0u && is_power_of_two(divisor)) {
                    auto shift = floor_log2(divisor);
                    builder.set_insertion_point(inst);
                    auto *shift_const = create_broadcast_constant(
                        module, inst->operand(0)->type(), shift);
                    if (shift_const != nullptr) {
                        return builder.call(
                            type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                            {inst->operand(0), shift_const});
                    }
                }
            }
            break;
        }
        case ArithmeticOp::BINARY_MOD: {
            // Unsigned integer modulo by power-of-two constant:
            //   x % pow2 → x & (pow2 - 1)
            if (is_unsigned_integer_type(type)) {
                uint64_t divisor = 0u;
                if (decode_uniform_unsigned_constant(
                        inst->operand(1), divisor) &&
                    divisor > 0u && is_power_of_two(divisor)) {
                    auto mask = divisor - 1u;
                    builder.set_insertion_point(inst);
                    auto *mask_const = create_broadcast_constant(
                        module, inst->operand(0)->type(), mask);
                    if (mask_const != nullptr) {
                        return builder.call(
                            type, ArithmeticOp::BINARY_BIT_AND,
                            {inst->operand(0), mask_const});
                    }
                }
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
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR: {
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
            if (is_const_zero(inst->operand(0)) && !is_float_like(type))
                return inst->operand(0);
            break;
        }
        case ArithmeticOp::EXTRACT: {
            auto idx = inst->operand(1);
            auto idx_val = decode_constant_index(idx);
            if (!idx_val.has_value()) break;
            auto src = inst->operand(0);
            while (src->isa<Instruction>()) {
                auto src_inst = static_cast<Instruction *>(src);
                if (!src_inst->isa<ArithmeticInst>()) break;
                auto src_arith = static_cast<ArithmeticInst *>(src_inst);
                if (src_arith->op() == ArithmeticOp::AGGREGATE) {
                    if (inst->operand_count() == 2u && *idx_val < src_arith->operand_count()) {
                        return src_arith->operand(*idx_val);
                    }
                    break;
                }
                if (src_arith->op() == ArithmeticOp::INSERT) {
                    auto extract_index_count = inst->operand_count() - 1u;
                    auto insert_index_count = src_arith->operand_count() - 2u;
                    auto index_count = std::min(extract_index_count, insert_index_count);
                    auto all_indices_match = extract_index_count == insert_index_count;
                    auto indices_differ = false;
                    for (size_t i = 0u; i < index_count; ++i) {
                        auto extract_index = inst->operand(i + 1u);
                        auto insert_index = src_arith->operand(i + 2u);
                        all_indices_match &= indices_equal(extract_index, insert_index);
                        indices_differ |= indices_provably_different(extract_index, insert_index);
                        if (indices_differ) { break; }
                    }
                    if (all_indices_match) {
                        return src_arith->operand(1);
                    }
                    if (!indices_differ) { break; }
                    src = src_arith->operand(0);
                    continue;
                }
                break;
            }
            if (src != inst->operand(0)) {
                inst->set_operand(0, src);
                changed_in_place = true;
            }
            break;
        }
        case ArithmeticOp::SELECT: {
            auto cond = inst->operand(2);
            if (cond->isa<Constant>()) {
                auto c = static_cast<const Constant *>(cond);
                if (c->type()->is_bool()) {
                    return c->as<bool>() ? inst->operand(1) : inst->operand(0);
                }
            }
            if (inst->operand(0) == inst->operand(1)) return inst->operand(0);
            break;
        }
        case ArithmeticOp::AGGREGATE: {
            if (inst->operand_count() == 0) break;
            if (!inst->type()->is_vector()) break;
            auto first_op = inst->operand(0);
            if (!first_op->isa<Instruction>()) break;
            auto first_inst = static_cast<Instruction *>(first_op);
            if (!first_inst->isa<ArithmeticInst>()) break;
            auto first_arith = static_cast<ArithmeticInst *>(first_inst);
            if (first_arith->op() != ArithmeticOp::EXTRACT) break;
            if (first_arith->operand_count() < 2) break;
            auto common_src = first_arith->operand(0);
            if (!common_src->type()->is_vector()) break;
            if (common_src->type()->element() != inst->type()->element()) break;
            auto first_idx = first_arith->operand(1);
            auto src_dim = common_src->type()->dimension();
            luisa::vector<Value *> shuffle_operands;
            shuffle_operands.reserve(inst->operand_count() + 1);
            shuffle_operands.emplace_back(common_src);
            auto first_idx_val = decode_constant_index(first_idx);
            if (!first_idx_val.has_value() || *first_idx_val >= src_dim) break;
            shuffle_operands.emplace_back(first_idx);
            bool all_match = true;
            bool identity = common_src->type() == inst->type() && *first_idx_val == 0u;
            for (size_t i = 1; i < inst->operand_count(); ++i) {
                auto op_i = inst->operand(i);
                if (!op_i->isa<Instruction>()) {
                    all_match = false;
                    break;
                }
                auto op_inst = static_cast<Instruction *>(op_i);
                if (!op_inst->isa<ArithmeticInst>()) {
                    all_match = false;
                    break;
                }
                auto op_arith = static_cast<ArithmeticInst *>(op_inst);
                if (op_arith->op() != ArithmeticOp::EXTRACT) {
                    all_match = false;
                    break;
                }
                if (op_arith->operand(0) != common_src) {
                    all_match = false;
                    break;
                }
                auto op_idx = op_arith->operand(1);
                auto op_idx_val = decode_constant_index(op_idx);
                if (!op_idx_val.has_value() || *op_idx_val >= src_dim) {
                    all_match = false;
                    break;
                }
                identity &= *op_idx_val == i;
                shuffle_operands.emplace_back(op_idx);
            }
            if (all_match) {
                if (identity) return common_src;
                builder.set_insertion_point(inst);
                return builder.call(inst->type(), ArithmeticOp::SHUFFLE, shuffle_operands);
            }
            break;
        }
        case ArithmeticOp::INSERT: {
            auto base = inst->operand(0);
            auto val = inst->operand(1);
            auto idx = inst->operand(2);
            auto idx_val = decode_constant_index(idx);
            if (!idx_val.has_value()) break;
            if (base->isa<Instruction>()) {
                auto base_inst = static_cast<Instruction *>(base);
                if (base_inst->isa<ArithmeticInst>()) {
                    auto base_arith = static_cast<ArithmeticInst *>(base_inst);
                    if (base_arith->op() == ArithmeticOp::AGGREGATE &&
                        inst->operand_count() == 3u && *idx_val < base_arith->operand_count()) {
                        luisa::vector<Value *> elems;
                        elems.reserve(base_arith->operand_count());
                        for (size_t i = 0; i < base_arith->operand_count(); ++i) {
                            elems.emplace_back(i == *idx_val ? val : base_arith->operand(i));
                        }
                        builder.set_insertion_point(inst);
                        return builder.call(inst->type(), ArithmeticOp::AGGREGATE, elems);
                    }
                    if (base_arith->op() == ArithmeticOp::INSERT) {
                        bool all_indices_match = inst->operand_count() == base_arith->operand_count();
                        for (size_t i = 2; all_indices_match && i < inst->operand_count(); ++i) {
                            auto outer_idx = inst->operand(i);
                            auto inner_idx = base_arith->operand(i);
                            if (!indices_equal(outer_idx, inner_idx)) {
                                all_indices_match = false;
                                break;
                            }
                        }
                        if (all_indices_match) {
                            inst->set_operand(0, base_arith->operand(0));
                            changed_in_place = true;
                            return nullptr;
                        }
                    }
                }
            }
            if (inst->operand_count() == 3u && inst->type() != nullptr &&
                (inst->type()->is_vector() || inst->type()->is_array())) {
                auto dim = inst->type()->dimension();
                // Zero-sized arrays are valid XIR types but have no valid
                // INSERT index. Guard the subtraction and element access even
                // for malformed input so the pass remains a conservative
                // no-op instead of indexing an empty reconstruction vector.
                if (dim != 0u && *idx_val < dim &&
                    *idx_val == dim - 1u) {
                    luisa::vector<Value *> elems(dim, nullptr);
                    elems[*idx_val] = val;
                    auto cur = base;
                    bool valid = true;
                    for (auto slot = static_cast<int32_t>(dim) - 2; slot >= 0; --slot) {
                        if (cur->isa<Undefined>()) {
                            valid = false;
                            break;
                        }
                        if (!cur->isa<Instruction>()) {
                            valid = false;
                            break;
                        }
                        auto cur_inst = static_cast<Instruction *>(cur);
                        if (!cur_inst->isa<ArithmeticInst>()) {
                            valid = false;
                            break;
                        }
                        auto cur_arith = static_cast<ArithmeticInst *>(cur_inst);
                        if (cur_arith->op() != ArithmeticOp::INSERT || cur_arith->operand_count() != 3u) {
                            valid = false;
                            break;
                        }
                        auto ci = cur_arith->operand(2);
                        auto ci_val = decode_constant_index(ci);
                        if (!ci_val.has_value() || *ci_val != static_cast<size_t>(slot)) {
                            valid = false;
                            break;
                        }
                        elems[slot] = cur_arith->operand(1);
                        cur = cur_arith->operand(0);
                    }
                    if (valid && cur->isa<Undefined>()) {
                        bool all_filled = true;
                        for (auto e : elems) {
                            if (!e) {
                                all_filled = false;
                                break;
                            }
                        }
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

static void algebraic_simplify_on_function(Function *function, AlgebraicSimplifyInfo &info, AlgebraicSimplifyOptions options) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr || def->body_block() == nullptr) { return; }
    auto module = function->parent_module();
    XIRBuilder builder;

    luisa::vector<ArithmeticInst *> to_simplify;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ArithmeticInst>()) {
            to_simplify.push_back(static_cast<ArithmeticInst *>(inst));
        }
    });

    for (auto inst : to_simplify) {
        // try_simplify may either return a shared operand/pooled constant or
        // create a new instruction. Without a uniform unique replacement
        // owner, preserve annotated instructions conservatively in place.
        if (!inst->metadata_list().empty()) { continue; }
        auto changed_in_place = false;
        auto replacement = try_simplify(
            inst, module, builder, options, changed_in_place);
        if (replacement != nullptr) {
            inst->replace_all_uses_with(replacement);
            inst->remove_self();
            info.simplified_inst_count++;
        } else if (changed_in_place) {
            info.simplified_inst_count++;
        }
    }
}

}// namespace detail

AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_function(Function *function, AlgebraicSimplifyOptions options) noexcept {
    AlgebraicSimplifyInfo info;
    detail::algebraic_simplify_on_function(function, info, options);
    return info;
}

AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_module(Module *module, AlgebraicSimplifyOptions options, PassReport *report) noexcept {
    AlgebraicSimplifyInfo info;
    if (module == nullptr) {
        if (report != nullptr) { report->set("simplified_inst", 0u); }
        return info;
    }
    for (auto f : module->function_list()) {
        detail::algebraic_simplify_on_function(f, info, options);
    }
    if (report != nullptr) {
        report->set("simplified_inst", info.simplified_inst_count);
    }
    return info;
}

}// namespace luisa::compute::xir
