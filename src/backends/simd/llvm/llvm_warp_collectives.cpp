#include "llvm_warp_collectives.h"

#include <bit>
#include <vector>

#include <llvm/ADT/APInt.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::simd {

LLVMWarpCollectives::LLVMWarpCollectives(uint32_t width) noexcept
    : _width{width} {
    if (width == 0u || width > 128u) {
        _error = "LLVM warp collective width must be in [1, 128]";
    }
}

void LLVMWarpCollectives::_fail(const char *message) noexcept {
    if (_error.empty()) { _error = message; }
}

bool LLVMWarpCollectives::_validate_vector(::llvm::Value *value,
                                          const char *role) noexcept {
    if (!_error.empty()) { return false; }
    auto *type = value == nullptr ?
                     nullptr :
                     ::llvm::dyn_cast<::llvm::FixedVectorType>(
                         value->getType());
    if (type == nullptr || type->getNumElements() != _width) {
        _fail(role);
        return false;
    }
    return true;
}

bool LLVMWarpCollectives::_validate_mask(::llvm::Value *value,
                                        const char *role) noexcept {
    if (!_validate_vector(value, role)) { return false; }
    auto *type = ::llvm::cast<::llvm::FixedVectorType>(value->getType());
    if (!type->getElementType()->isIntegerTy(1u)) {
        _fail(role);
        return false;
    }
    return true;
}

::llvm::Value *LLVMWarpCollectives::_active_values(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants, bool product) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *element = ::llvm::cast<::llvm::VectorType>(values->getType())
                        ->getElementType();
    ::llvm::Constant *identity = nullptr;
    if (element->isFloatingPointTy()) {
        identity = ::llvm::ConstantFP::get(element, product ? 1.0 : 0.0);
    } else if (element->isIntegerTy()) {
        identity = ::llvm::ConstantInt::get(element, product ? 1u : 0u);
    } else {
        _fail("warp arithmetic only supports scalar integer and float lanes");
        return nullptr;
    }
    auto *identities = builder.CreateVectorSplat(_width, identity);
    return builder.CreateSelect(participants, values, identities);
}

::llvm::Value *LLVMWarpCollectives::_mask_bits(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *mask) noexcept {
    if (!_validate_mask(mask, "mask bit packing requires an i1 lane vector")) {
        return nullptr;
    }
    auto *integer = ::llvm::IntegerType::get(builder.getContext(), _width);
    // This backend JITs for the process host. LLVM's little-endian
    // vector-to-integer bitcast is the cheapest mask pack and maps Luisa lane
    // 0 to bit 0. Keep the explicit lane-wise spelling on any other host so
    // physical lane order never depends on a target data layout convention.
    if constexpr (std::endian::native == std::endian::little) {
        return builder.CreateBitCast(mask, integer);
    }
    auto *wide_vector = ::llvm::FixedVectorType::get(integer, _width);
    auto *extended = builder.CreateZExt(mask, wide_vector);
    std::vector<::llvm::Constant *> shifts;
    shifts.reserve(_width);
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        shifts.emplace_back(::llvm::ConstantInt::get(integer, lane));
    }
    return builder.CreateOrReduce(
        builder.CreateShl(extended, ::llvm::ConstantVector::get(shifts)));
}

::llvm::Value *LLVMWarpCollectives::first_active_lane(
    ::llvm::IRBuilder<> &builder,
    ::llvm::Value *participants) noexcept {
    auto *bits = _mask_bits(builder, participants);
    if (bits == nullptr) { return nullptr; }
    auto *declaration =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
        builder.GetInsertBlock()->getModule(), ::llvm::Intrinsic::cttz,
        {bits->getType()});
    auto *first = builder.CreateCall(
        declaration, {bits, builder.getFalse()});
    return builder.CreateZExtOrTrunc(first, builder.getInt32Ty());
}

::llvm::Value *LLVMWarpCollectives::is_first_active_lane(
    ::llvm::IRBuilder<> &builder,
    ::llvm::Value *participants) noexcept {
    auto *first = first_active_lane(builder, participants);
    if (first == nullptr) { return nullptr; }
    std::vector<::llvm::Constant *> lane_constants;
    lane_constants.reserve(_width);
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        lane_constants.emplace_back(builder.getInt32(lane));
    }
    auto *lanes = ::llvm::ConstantVector::get(lane_constants);
    auto *is_first = builder.CreateICmpEQ(lanes, builder.CreateVectorSplat(
                                                     _width, first));
    return builder.CreateAnd(participants, is_first);
}

::llvm::Value *LLVMWarpCollectives::active_count_bits(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
    ::llvm::Value *participants) noexcept {
    if (!_validate_mask(predicate, "warp predicate must be an i1 lane vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *active = builder.CreateAnd(predicate, participants);
    auto *counts = builder.CreateZExt(
        active, ::llvm::FixedVectorType::get(builder.getInt32Ty(), _width));
    return builder.CreateAddReduce(counts);
}

::llvm::Value *LLVMWarpCollectives::active_all(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
    ::llvm::Value *participants) noexcept {
    if (!_validate_mask(predicate, "warp predicate must be an i1 lane vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *ones = ::llvm::Constant::getAllOnesValue(predicate->getType());
    return builder.CreateAndReduce(
        builder.CreateSelect(participants, predicate, ones));
}

::llvm::Value *LLVMWarpCollectives::active_any(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
    ::llvm::Value *participants) noexcept {
    if (!_validate_mask(predicate, "warp predicate must be an i1 lane vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    return builder.CreateOrReduce(
        builder.CreateAnd(predicate, participants));
}

::llvm::Value *LLVMWarpCollectives::active_all_equal(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto first = read_first_active_lane(builder, values, participants);
    if (first.values == nullptr) { return nullptr; }
    auto *element = ::llvm::cast<::llvm::VectorType>(values->getType())
                        ->getElementType();
    ::llvm::Value *equal = nullptr;
    if (element->isFloatingPointTy()) {
        equal = builder.CreateFCmpOEQ(values, first.values);
    } else if (element->isIntegerTy()) {
        equal = builder.CreateICmpEQ(values, first.values);
    } else {
        _fail("warp equality only supports scalar integer and float lanes");
        return nullptr;
    }
    auto *ones = ::llvm::Constant::getAllOnesValue(participants->getType());
    return builder.CreateAndReduce(
        builder.CreateSelect(participants, equal, ones));
}

::llvm::Value *LLVMWarpCollectives::active_bit_mask(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
    ::llvm::Value *participants) noexcept {
    if (!_validate_mask(predicate, "warp predicate must be an i1 lane vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *bits = _mask_bits(
        builder, builder.CreateAnd(predicate, participants));
    if (bits == nullptr) { return nullptr; }
    auto *result_type = ::llvm::FixedVectorType::get(
        builder.getInt32Ty(), 4u);
    ::llvm::Value *result = ::llvm::PoisonValue::get(result_type);
    for (auto word = uint32_t{0u}; word < 4u; word++) {
        auto offset = word * 32u;
        ::llvm::Value *value = nullptr;
        if (offset >= _width) {
            value = builder.getInt32(0u);
        } else {
            value = bits;
            if (offset != 0u) {
                value = builder.CreateLShr(value, offset);
            }
            value = builder.CreateZExtOrTrunc(value, builder.getInt32Ty());
        }
        result = builder.CreateInsertElement(result, value, word);
    }
    return result;
}

::llvm::Value *LLVMWarpCollectives::active_sum(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    auto *active = _active_values(builder, values, participants, false);
    if (active == nullptr) { return nullptr; }
    auto *element = ::llvm::cast<::llvm::VectorType>(active->getType())
                        ->getElementType();
    if (element->isFloatingPointTy()) {
        return builder.CreateFAddReduce(
            ::llvm::ConstantFP::get(element, 0.0), active);
    }
    return builder.CreateAddReduce(active);
}

::llvm::Value *LLVMWarpCollectives::active_product(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    auto *active = _active_values(builder, values, participants, true);
    if (active == nullptr) { return nullptr; }
    auto *element = ::llvm::cast<::llvm::VectorType>(active->getType())
                        ->getElementType();
    if (element->isFloatingPointTy()) {
        return builder.CreateFMulReduce(
            ::llvm::ConstantFP::get(element, 1.0), active);
    }
    return builder.CreateMulReduce(active);
}

::llvm::Value *LLVMWarpCollectives::_active_extreme(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants, bool maximum,
    bool signed_integer) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *element = ::llvm::cast<::llvm::VectorType>(values->getType())
                        ->getElementType();
    ::llvm::Constant *identity = nullptr;
    if (element->isFloatingPointTy()) {
        // max uses -infinity; min uses +infinity.
        identity = ::llvm::ConstantFP::getInfinity(element, maximum);
    } else if (auto *integer =
                   ::llvm::dyn_cast<::llvm::IntegerType>(element)) {
        auto bits = integer->getBitWidth();
        auto value = maximum ?
                         (signed_integer ?
                              ::llvm::APInt::getSignedMinValue(bits) :
                              ::llvm::APInt::getMinValue(bits)) :
                         (signed_integer ?
                              ::llvm::APInt::getSignedMaxValue(bits) :
                              ::llvm::APInt::getMaxValue(bits));
        identity = ::llvm::ConstantInt::get(integer, value);
    } else {
        _fail("warp min/max only supports scalar integer and float lanes");
        return nullptr;
    }
    auto *active = builder.CreateSelect(
        participants, values, builder.CreateVectorSplat(_width, identity));
    if (element->isFloatingPointTy()) {
        return maximum ? builder.CreateFPMaxReduce(active) :
                         builder.CreateFPMinReduce(active);
    }
    return maximum ? builder.CreateIntMaxReduce(active, signed_integer) :
                     builder.CreateIntMinReduce(active, signed_integer);
}

::llvm::Value *LLVMWarpCollectives::active_min(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants, bool signed_integer) noexcept {
    return _active_extreme(
        builder, values, participants, false, signed_integer);
}

::llvm::Value *LLVMWarpCollectives::active_max(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants, bool signed_integer) noexcept {
    return _active_extreme(
        builder, values, participants, true, signed_integer);
}

::llvm::Value *LLVMWarpCollectives::active_bit_and(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *element = ::llvm::cast<::llvm::VectorType>(values->getType())
                        ->getElementType();
    if (!element->isIntegerTy()) {
        _fail("warp bit reductions require integer lanes");
        return nullptr;
    }
    auto *ones = ::llvm::Constant::getAllOnesValue(values->getType());
    return builder.CreateAndReduce(
        builder.CreateSelect(participants, values, ones));
}

::llvm::Value *LLVMWarpCollectives::active_bit_or(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *element = ::llvm::cast<::llvm::VectorType>(values->getType())
                        ->getElementType();
    if (!element->isIntegerTy()) {
        _fail("warp bit reductions require integer lanes");
        return nullptr;
    }
    auto *zeros = ::llvm::Constant::getNullValue(values->getType());
    return builder.CreateOrReduce(
        builder.CreateSelect(participants, values, zeros));
}

::llvm::Value *LLVMWarpCollectives::active_bit_xor(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *element = ::llvm::cast<::llvm::VectorType>(values->getType())
                        ->getElementType();
    if (!element->isIntegerTy()) {
        _fail("warp bit reductions require integer lanes");
        return nullptr;
    }
    auto *zeros = ::llvm::Constant::getNullValue(values->getType());
    return builder.CreateXorReduce(
        builder.CreateSelect(participants, values, zeros));
}

::llvm::Value *LLVMWarpCollectives::_prefix(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants, bool product) noexcept {
    auto *scan = _active_values(builder, values, participants, product);
    if (scan == nullptr) { return nullptr; }
    auto *element = ::llvm::cast<::llvm::VectorType>(scan->getType())
                        ->getElementType();
    auto binary = [&](::llvm::Value *lhs,
                      ::llvm::Value *rhs) noexcept -> ::llvm::Value * {
        if (element->isFloatingPointTy()) {
            return product ? builder.CreateFMul(lhs, rhs) :
                             builder.CreateFAdd(lhs, rhs);
        }
        return product ? builder.CreateMul(lhs, rhs) :
                         builder.CreateAdd(lhs, rhs);
    };
    for (auto offset = uint32_t{1u}; offset < _width; offset <<= 1u) {
        std::vector<int> shuffle;
        std::vector<::llvm::Constant *> enabled;
        shuffle.reserve(_width);
        enabled.reserve(_width);
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            shuffle.emplace_back(lane >= offset ?
                                     static_cast<int>(lane - offset) :
                                     -1);
            enabled.emplace_back(builder.getInt1(lane >= offset));
        }
        auto *shifted = builder.CreateShuffleVector(scan, shuffle);
        auto *combined = binary(scan, shifted);
        scan = builder.CreateSelect(
            ::llvm::ConstantVector::get(enabled), combined, scan);
    }

    ::llvm::Constant *identity = element->isFloatingPointTy() ?
                                    static_cast<::llvm::Constant *>(
                                        ::llvm::ConstantFP::get(
                                            element, product ? 1.0 : 0.0)) :
                                    static_cast<::llvm::Constant *>(
                                        ::llvm::ConstantInt::get(
                                            element, product ? 1u : 0u));
    std::vector<int> exclusive_shuffle;
    exclusive_shuffle.reserve(_width);
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        exclusive_shuffle.emplace_back(
            lane == 0u ? -1 : static_cast<int>(lane - 1u));
    }
    auto *exclusive = builder.CreateShuffleVector(scan, exclusive_shuffle);
    exclusive = builder.CreateInsertElement(
        exclusive, identity, uint64_t{0u});
    auto *inactive = ::llvm::Constant::getNullValue(values->getType());
    return builder.CreateSelect(participants, exclusive, inactive);
}

::llvm::Value *LLVMWarpCollectives::prefix_count_bits(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
    ::llvm::Value *participants) noexcept {
    if (!_validate_mask(predicate, "warp predicate must be an i1 lane vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return nullptr;
    }
    auto *active = builder.CreateAnd(predicate, participants);
    auto *values = builder.CreateZExt(
        active, ::llvm::FixedVectorType::get(builder.getInt32Ty(), _width));
    return _prefix(builder, values, participants, false);
}

::llvm::Value *LLVMWarpCollectives::prefix_sum(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    return _prefix(builder, values, participants, false);
}

::llvm::Value *LLVMWarpCollectives::prefix_product(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    return _prefix(builder, values, participants, true);
}

LLVMReadLaneResult LLVMWarpCollectives::read_lane(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *source_lanes,
    ::llvm::Value *participants) noexcept {
    if (!_validate_vector(values, "warp value must be a fixed-width vector") ||
        !_validate_vector(source_lanes,
                          "source lanes must be a fixed-width vector") ||
        !_validate_mask(participants,
                        "participant mask must be an i1 lane vector")) {
        return {};
    }
    auto *source_type = ::llvm::cast<::llvm::VectorType>(
                            source_lanes->getType())
                            ->getElementType();
    if (!source_type->isIntegerTy()) {
        _fail("source lanes must contain integers");
        return {};
    }
    auto *result_type = values->getType();
    auto *invalid_type = participants->getType();
    ::llvm::Value *result = ::llvm::PoisonValue::get(result_type);
    ::llvm::Value *invalid = ::llvm::PoisonValue::get(invalid_type);
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        auto *source = builder.CreateExtractElement(source_lanes, lane);
        auto *in_range = builder.CreateICmpULT(
            source, ::llvm::ConstantInt::get(source_type, _width));
        auto *safe_source = builder.CreateSelect(
            in_range, source, ::llvm::ConstantInt::get(source_type, 0u));
        auto *source_active = builder.CreateExtractElement(
            participants, safe_source);
        auto *destination_active = builder.CreateExtractElement(
            participants, lane);
        auto *valid = builder.CreateAnd(
            destination_active,
            builder.CreateAnd(in_range, source_active));
        auto *read = builder.CreateExtractElement(values, safe_source);
        auto *zero = ::llvm::Constant::getNullValue(
            ::llvm::cast<::llvm::VectorType>(result_type)->getElementType());
        result = builder.CreateInsertElement(
            result, builder.CreateSelect(valid, read, zero), lane);
        invalid = builder.CreateInsertElement(
            invalid, builder.CreateAnd(destination_active,
                                       builder.CreateNot(valid)), lane);
    }
    return {.values = result, .invalid_lanes = invalid};
}

LLVMReadLaneResult LLVMWarpCollectives::read_first_active_lane(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
    ::llvm::Value *participants) noexcept {
    auto *first = first_active_lane(builder, participants);
    if (first == nullptr) { return {}; }
    auto *source_type = ::llvm::FixedVectorType::get(
        builder.getInt32Ty(), _width);
    return read_lane(builder, values,
                     builder.CreateVectorSplat(
                         source_type->getElementCount(), first),
                     participants);
}

}// namespace luisa::compute::simd
