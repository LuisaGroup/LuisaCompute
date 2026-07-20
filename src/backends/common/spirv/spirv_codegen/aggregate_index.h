#pragma once

#include <cstdint>

#include <luisa/ast/type.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/value.h>

namespace lc::spirv {

using luisa::compute::Type;

struct SpirvAggregateIndexConstant {
    uint64_t value{0u};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostic.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Decodes every integer scalar representation admitted by XIR without first
// narrowing through uint32_t or size_t. Negative signed constants fail with a
// diagnostic; non-constant and non-integer values are not silently accepted.
[[nodiscard]] SpirvAggregateIndexConstant
decode_spirv_aggregate_index_constant(
    const luisa::compute::xir::Value *value) noexcept;

enum class SpirvAggregateIndexKind : uint8_t {
    SEQUENCE_ELEMENT,
    STRUCTURE_MEMBER,
};

struct SpirvAggregateIndexStep {
    const Type *aggregate_type{nullptr};
    const Type *indexed_type{nullptr};
    const luisa::compute::xir::Value *index{nullptr};
    uint64_t constant_index{0u};
    SpirvAggregateIndexKind kind{
        SpirvAggregateIndexKind::SEQUENCE_ELEMENT};
    bool is_constant{false};
};

struct SpirvAggregateIndexPlan {
    luisa::vector<SpirvAggregateIndexStep> steps;
    const Type *indexed_type{nullptr};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostic.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
    [[nodiscard]] bool all_constant() const noexcept;
};

// Plans the logical type walk before any SPIR-V IDs are emitted. Buffer roots
// are treated as an unbounded sequence whose element is the buffer element;
// arrays, vectors, and matrices retain legal dynamic indices. Structure
// members must be non-negative integer constants, fit SPIR-V's 32-bit member
// operand, and name an existing member.
[[nodiscard]] SpirvAggregateIndexPlan plan_spirv_aggregate_indices(
    const Type *aggregate_type,
    luisa::span<const luisa::compute::xir::Value *const> indices) noexcept;

}// namespace lc::spirv
