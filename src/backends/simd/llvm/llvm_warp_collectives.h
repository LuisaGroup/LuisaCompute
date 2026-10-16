#pragma once

#include <cstdint>
#include <string>

#include <llvm/IR/IRBuilder.h>

namespace llvm {
class Value;
}// namespace llvm

namespace luisa::compute::simd {

struct LLVMReadLaneResult {
    ::llvm::Value *values{nullptr};
    ::llvm::Value *invalid_lanes{nullptr};
};

// Target-independent fixed-vector lowering for the core warp operations. All
// methods preserve physical lane order and take the dynamic participant mask
// explicitly; no operation consults an implicit process-global mask.
class LLVMWarpCollectives {

private:
    uint32_t _width{0u};
    std::string _error;

private:
    [[nodiscard]] bool _validate_vector(
        ::llvm::Value *value, const char *role) noexcept;
    [[nodiscard]] bool _validate_mask(
        ::llvm::Value *value, const char *role) noexcept;
    [[nodiscard]] ::llvm::Value *_active_values(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants, bool product) noexcept;
    [[nodiscard]] ::llvm::Value *_prefix(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants, bool product) noexcept;
    [[nodiscard]] ::llvm::Value *_active_extreme(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants, bool maximum,
        bool signed_integer) noexcept;
    [[nodiscard]] ::llvm::Value *_mask_bits(
        ::llvm::IRBuilder<> &builder,
        ::llvm::Value *mask) noexcept;
    void _fail(const char *message) noexcept;

public:
    explicit LLVMWarpCollectives(uint32_t width) noexcept;

    [[nodiscard]] ::llvm::Value *first_active_lane(
        ::llvm::IRBuilder<> &builder,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *is_first_active_lane(
        ::llvm::IRBuilder<> &builder,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_count_bits(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_all(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_any(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_all_equal(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_bit_mask(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_sum(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_product(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_min(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants, bool signed_integer) noexcept;
    [[nodiscard]] ::llvm::Value *active_max(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants, bool signed_integer) noexcept;
    [[nodiscard]] ::llvm::Value *active_bit_and(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_bit_or(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *active_bit_xor(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *prefix_count_bits(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *predicate,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *prefix_sum(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] ::llvm::Value *prefix_product(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] LLVMReadLaneResult read_lane(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *source_lanes,
        ::llvm::Value *participants) noexcept;
    [[nodiscard]] LLVMReadLaneResult read_first_active_lane(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *values,
        ::llvm::Value *participants) noexcept;

    [[nodiscard]] uint32_t width() const noexcept { return _width; }
    [[nodiscard]] bool succeeded() const noexcept { return _error.empty(); }
    [[nodiscard]] const std::string &error() const noexcept { return _error; }
};

}// namespace luisa::compute::simd
