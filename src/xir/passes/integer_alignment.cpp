#include <luisa/xir/passes/integer_alignment.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <luisa/ast/type.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/phi.h>

namespace luisa::compute::xir {
namespace {

class IntegerAlignmentAnalysis {

private:
    size_t _maximum_alignment;
    std::unordered_map<const Value *, size_t> _cache;
    std::unordered_set<const Value *> _active;

private:
    template<typename T>
    [[nodiscard]] size_t _constant_alignment(
        const Constant *constant) const noexcept {
        using U = std::make_unsigned_t<T>;
        auto bits = static_cast<U>(constant->as<T>());
        return static_cast<size_t>(
            std::gcd(static_cast<uint64_t>(bits),
                     static_cast<uint64_t>(_maximum_alignment)));
    }

    [[nodiscard]] size_t _constant_alignment(
        const Constant *constant) const noexcept {
        switch (constant->type()->tag()) {
            case Type::Tag::INT8:
                return _constant_alignment<int8_t>(constant);
            case Type::Tag::UINT8:
                return _constant_alignment<uint8_t>(constant);
            case Type::Tag::INT16:
                return _constant_alignment<int16_t>(constant);
            case Type::Tag::UINT16:
                return _constant_alignment<uint16_t>(constant);
            case Type::Tag::INT32:
                return _constant_alignment<int32_t>(constant);
            case Type::Tag::UINT32:
                return _constant_alignment<uint32_t>(constant);
            case Type::Tag::INT64:
                return _constant_alignment<int64_t>(constant);
            case Type::Tag::UINT64:
                return _constant_alignment<uint64_t>(constant);
            default: return 1u;
        }
    }

    [[nodiscard]] size_t _operand_alignment(
        const ArithmeticInst *inst, size_t index) noexcept {
        return index < inst->operand_count() ?
                   _analyze(inst->operand(index)) :
                   1u;
    }

    [[nodiscard]] size_t _product_alignment(
        size_t lhs, size_t rhs) const noexcept {
        return lhs >= _maximum_alignment / rhs ?
                   _maximum_alignment :
                   lhs * rhs;
    }

    [[nodiscard]] size_t _left_shift_alignment(
        size_t alignment, uint64_t shift) const noexcept {
        auto result = alignment;
        while (shift-- != 0u && result < _maximum_alignment) {
            result = std::min(_maximum_alignment, result * 2u);
        }
        return result;
    }

    [[nodiscard]] size_t _right_shift_alignment(
        size_t alignment, uint64_t shift) const noexcept {
        while (shift-- != 0u && alignment > 1u) {
            alignment /= 2u;
        }
        return alignment;
    }

    [[nodiscard]] size_t _analyze_arithmetic(
        const ArithmeticInst *inst) noexcept {
        auto a0 = [&]() noexcept { return _operand_alignment(inst, 0u); };
        auto a1 = [&]() noexcept { return _operand_alignment(inst, 1u); };
        auto minimum = [&](size_t count) noexcept {
            auto result = _maximum_alignment;
            for (auto i = 0u; i < count; ++i) {
                result = std::min(result, _operand_alignment(inst, i));
            }
            return result;
        };
        switch (inst->op()) {
            case ArithmeticOp::UNARY_MINUS:
            case ArithmeticOp::ABS: return a0();
            case ArithmeticOp::BINARY_ADD:
            case ArithmeticOp::BINARY_SUB:
            case ArithmeticOp::BINARY_MOD:
            case ArithmeticOp::BINARY_BIT_OR:
            case ArithmeticOp::BINARY_BIT_XOR:
            case ArithmeticOp::MIN:
            case ArithmeticOp::MAX:
                return std::min(a0(), a1());
            case ArithmeticOp::BINARY_MUL:
                return _product_alignment(a0(), a1());
            case ArithmeticOp::BINARY_BIT_AND:
                // Every low bit proved zero by either operand is zero in the
                // result, so the stronger of the two divisibility facts wins.
                return std::max(a0(), a1());
            case ArithmeticOp::BINARY_SHIFT_LEFT: {
                uint64_t shift = 0u;
                if (inst->operand_count() >= 2u &&
                    try_decode_constant_nonnegative_integer(
                        inst->operand(1u), shift)) {
                    auto bit_width = inst->type()->size() * 8u;
                    return shift < bit_width ?
                               _left_shift_alignment(a0(), shift) :
                               1u;
                }
                // A defined left shift cannot introduce a non-zero bit below
                // an already-known zero suffix.
                return a0();
            }
            case ArithmeticOp::BINARY_SHIFT_RIGHT: {
                uint64_t shift = 0u;
                if (inst->operand_count() >= 2u &&
                    try_decode_constant_nonnegative_integer(
                        inst->operand(1u), shift)) {
                    auto bit_width = inst->type()->size() * 8u;
                    return shift < bit_width ?
                               _right_shift_alignment(a0(), shift) :
                               1u;
                }
                return 1u;
            }
            case ArithmeticOp::SELECT:
                // XIR SELECT is (false_value, true_value, condition).
                return std::min(a0(), a1());
            case ArithmeticOp::CLAMP: return minimum(3u);
            default: return 1u;
        }
    }

    [[nodiscard]] size_t _analyze_cast(const CastInst *cast) noexcept {
        auto *source = cast->value();
        auto *source_type = source == nullptr ? nullptr : source->type();
        auto *target_type = cast->type();
        if (source_type == nullptr || target_type == nullptr ||
            !(source_type->is_int() || source_type->is_uint()) ||
            !(target_type->is_int() || target_type->is_uint())) {
            return 1u;
        }
        auto alignment = _analyze(source);
        // Integer conversion is reduction modulo 2^N followed by an optional
        // extension. It preserves every 2^k divisor with k <= N.
        auto target_bits = target_type->size() * 8u;
        if (target_bits < std::numeric_limits<size_t>::digits) {
            auto target_modulus = size_t{1u} << target_bits;
            alignment = std::min(alignment, target_modulus);
        }
        return alignment;
    }

    [[nodiscard]] size_t _analyze_phi(const PhiInst *phi) noexcept {
        if (phi->incoming_count() == 0u) { return 1u; }
        auto alignment = _maximum_alignment;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            alignment = std::min(
                alignment, _analyze(phi->incoming(i).value));
        }
        return alignment;
    }

    [[nodiscard]] size_t _analyze(const Value *value) noexcept {
        if (value == nullptr || value->type() == nullptr ||
            !(value->type()->is_int() || value->type()->is_uint())) {
            return 1u;
        }
        if (auto iter = _cache.find(value); iter != _cache.end()) {
            return iter->second;
        }
        // A loop-carried recurrence needs a fixed-point analysis to recover a
        // stronger invariant. Returning bottom here is deliberately
        // conservative and prevents a cyclic definition from proving itself.
        if (!_active.emplace(value).second) { return 1u; }
        auto alignment = size_t{1u};
        if (value->isa<Constant>()) {
            alignment = _constant_alignment(
                static_cast<const Constant *>(value));
        } else if (value->isa<ArithmeticInst>()) {
            alignment = _analyze_arithmetic(
                static_cast<const ArithmeticInst *>(value));
        } else if (value->isa<CastInst>()) {
            alignment = _analyze_cast(
                static_cast<const CastInst *>(value));
        } else if (value->isa<PhiInst>()) {
            alignment = _analyze_phi(
                static_cast<const PhiInst *>(value));
        }
        _active.erase(value);
        alignment = std::clamp(
            std::bit_floor(alignment), size_t{1u},
            _maximum_alignment);
        _cache.emplace(value, alignment);
        return alignment;
    }

public:
    explicit IntegerAlignmentAnalysis(size_t maximum_alignment) noexcept
        : _maximum_alignment{maximum_alignment} {}

    [[nodiscard]] size_t analyze(const Value *value) noexcept {
        return _analyze(value);
    }
};

}// namespace

size_t integer_value_guaranteed_alignment(
    const Value *value, size_t maximum_alignment) noexcept {
    maximum_alignment = std::bit_floor(maximum_alignment);
    if (maximum_alignment == 0u) { maximum_alignment = 1u; }
    return IntegerAlignmentAnalysis{maximum_alignment}.analyze(value);
}

}// namespace luisa::compute::xir
