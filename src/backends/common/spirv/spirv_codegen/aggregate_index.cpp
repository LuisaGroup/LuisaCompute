#include "aggregate_index.h"

#include <limits>

#include <luisa/core/stl/format.h>
#include <luisa/xir/constant.h>

namespace lc::spirv {

namespace xir = luisa::compute::xir;

namespace {

[[nodiscard]] luisa::string_view type_description(
    const Type *type) noexcept {
    return type == nullptr ? luisa::string_view{"<null>"} :
                             type->description();
}

}// namespace

SpirvAggregateIndexConstant
decode_spirv_aggregate_index_constant(
    const xir::Value *value) noexcept {
    if (value == nullptr || !value->isa<xir::Constant>() ||
        value->type() == nullptr ||
        (!value->type()->is_int() && !value->type()->is_uint())) {
        return {
            .diagnostic = luisa::format(
                "Aggregate index constant must be an integer scalar constant, got {}.",
                value == nullptr ? "<null>" :
                                   type_description(value->type()))};
    }
    auto constant = static_cast<const xir::Constant *>(value);
    switch (value->type()->tag()) {
        case Type::Tag::INT8: {
            auto decoded = constant->as<int8_t>();
            if (decoded < 0) {
                return {.diagnostic = luisa::format(
                            "Aggregate index constant {} is negative.",
                            static_cast<int32_t>(decoded))};
            }
            return {.value = static_cast<uint8_t>(decoded)};
        }
        case Type::Tag::UINT8:
            return {.value = constant->as<uint8_t>()};
        case Type::Tag::INT16: {
            auto decoded = constant->as<int16_t>();
            if (decoded < 0) {
                return {.diagnostic = luisa::format(
                            "Aggregate index constant {} is negative.",
                            static_cast<int32_t>(decoded))};
            }
            return {.value = static_cast<uint16_t>(decoded)};
        }
        case Type::Tag::UINT16:
            return {.value = constant->as<uint16_t>()};
        case Type::Tag::INT32: {
            auto decoded = constant->as<int32_t>();
            if (decoded < 0) {
                return {.diagnostic = luisa::format(
                            "Aggregate index constant {} is negative.",
                            decoded)};
            }
            return {.value = static_cast<uint32_t>(decoded)};
        }
        case Type::Tag::UINT32:
            return {.value = constant->as<uint32_t>()};
        case Type::Tag::INT64: {
            auto decoded = constant->as<int64_t>();
            if (decoded < 0) {
                return {.diagnostic = luisa::format(
                            "Aggregate index constant {} is negative.",
                            decoded)};
            }
            return {.value = static_cast<uint64_t>(decoded)};
        }
        case Type::Tag::UINT64:
            return {.value = constant->as<uint64_t>()};
        default:
            return {.diagnostic = luisa::format(
                        "Aggregate index constant must be an integer scalar constant, got {}.",
                        value->type()->description())};
    }
}

bool SpirvAggregateIndexPlan::all_constant() const noexcept {
    for (auto &&step : steps) {
        if (!step.is_constant) { return false; }
    }
    return true;
}

SpirvAggregateIndexPlan plan_spirv_aggregate_indices(
    const Type *aggregate_type,
    luisa::span<const xir::Value *const> indices) noexcept {
    SpirvAggregateIndexPlan plan;
    plan.indexed_type = aggregate_type;
    plan.steps.reserve(indices.size());
    if (aggregate_type == nullptr) {
        plan.diagnostic = "Aggregate index plan has no root type.";
        return plan;
    }
    for (auto i = 0u; i < indices.size(); ++i) {
        auto current = plan.indexed_type;
        auto index = indices[i];
        if (current == nullptr) {
            plan.diagnostic = luisa::format(
                "Aggregate index {} has no container type.", i);
            return plan;
        }
        if (index == nullptr || index->type() == nullptr ||
            (!index->type()->is_int() && !index->type()->is_uint())) {
            plan.diagnostic = luisa::format(
                "Aggregate index {} into {} must be an integer scalar, got {}.",
                i, current->description(),
                index == nullptr ? "<null>" :
                                   type_description(index->type()));
            return plan;
        }

        auto is_constant = index->isa<xir::Constant>();
        SpirvAggregateIndexConstant decoded;
        if (is_constant) {
            decoded = decode_spirv_aggregate_index_constant(index);
            if (!decoded) {
                plan.diagnostic = luisa::format(
                    "Aggregate index {} into {} is invalid: {}",
                    i, current->description(), decoded.diagnostic);
                return plan;
            }
        }

        SpirvAggregateIndexStep step{
            .aggregate_type = current,
            .index = index,
            .constant_index = decoded.value,
            .is_constant = is_constant};
        switch (current->tag()) {
            case Type::Tag::BUFFER: {
                if (current->element() == nullptr) {
                    plan.diagnostic = luisa::format(
                        "Cannot aggregate-index untyped buffer {}.",
                        current->description());
                    return plan;
                }
                step.indexed_type = current->element();
                break;
            }
            case Type::Tag::ARRAY:
            case Type::Tag::VECTOR: {
                if (is_constant &&
                    decoded.value >= current->dimension()) {
                    plan.diagnostic = luisa::format(
                        "Aggregate index {} is out of bounds for {} (dimension {}).",
                        decoded.value, current->description(),
                        current->dimension());
                    return plan;
                }
                step.indexed_type = current->element();
                break;
            }
            case Type::Tag::MATRIX: {
                if (is_constant &&
                    decoded.value >= current->dimension()) {
                    plan.diagnostic = luisa::format(
                        "Aggregate index {} is out of bounds for {} (dimension {}).",
                        decoded.value, current->description(),
                        current->dimension());
                    return plan;
                }
                step.indexed_type = Type::vector(
                    current->element(), current->dimension());
                break;
            }
            case Type::Tag::STRUCTURE: {
                step.kind = SpirvAggregateIndexKind::STRUCTURE_MEMBER;
                if (!is_constant) {
                    plan.diagnostic = luisa::format(
                        "Structure index {} into {} must be a compile-time integer constant.",
                        i, current->description());
                    return plan;
                }
                if (decoded.value >
                    std::numeric_limits<uint32_t>::max()) {
                    plan.diagnostic = luisa::format(
                        "Structure member index {} cannot be represented by SPIR-V's 32-bit member operand for {}.",
                        decoded.value, current->description());
                    return plan;
                }
                auto members = current->members();
                if (decoded.value >= members.size()) {
                    plan.diagnostic = luisa::format(
                        "Structure member index {} is out of bounds for {} ({} members).",
                        decoded.value, current->description(), members.size());
                    return plan;
                }
                step.indexed_type = members[decoded.value];
                break;
            }
            default:
                plan.diagnostic = luisa::format(
                    "Cannot apply aggregate index {} to non-aggregate type {}.",
                    i, current->description());
                return plan;
        }
        plan.steps.emplace_back(step);
        plan.indexed_type = step.indexed_type;
    }
    return plan;
}

}// namespace lc::spirv
