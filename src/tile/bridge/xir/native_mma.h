#pragma once

#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/ir.h>
#include <luisa/xir/metadata/strided_mma.h>

namespace luisa::compute::tile::bridge::xir::detail {

// A physical realization of typed MMA, not an attention/GEMM recognizer.
// All addresses below refer to immutable, row-major logical Tile snapshots;
// they say nothing about the source Buffer's physical layout. Admission is
// pure and nonrecursive so producer/carry storage and emission agree.
[[nodiscard]] inline luisa::optional<compute::xir::StridedMmaDescriptor> native_mma_plan(
    const Operation &op, const LowerOptions &options) noexcept {
    auto width = options.native_mma_vector_width;
    if (width < 2u || (width & (width - 1u)) != 0u || options.local_lanes != 1u ||
        options.max_unrolled_tile_elements == 0u || op.kind() != OperationKind::MMA ||
        op.operand_count() != 3u || op.result_count() != 1u) { return {}; }
    auto valid = [](const Type &type) noexcept {
        if (!type.is_tile() || type.scalar_type() != ScalarType::FLOAT32) { return false; }
        auto volume = uint64_t{1u};
        for (auto &axis : type.index_space()->axes()) {
            if (!axis.extent.is_constant()) { return false; }
            auto extent = axis.extent.constant_value();
            if (extent == 0u || volume > static_cast<uint64_t>(INT64_MAX) / extent) { return false; }
            volume *= extent;
        }
        return true;
    };
    if (!valid(op.result(0u)->type())) { return {}; }
    for (size_t i = 0u; i < op.operand_count(); i++) {
        if (!valid(op.operand(i)->type())) { return {}; }
    }
    auto &output = *op.result(0u)->type().index_space();
    auto &lhs = *op.operand(0u)->type().index_space();
    auto &rhs = *op.operand(1u)->type().index_space();
    luisa::optional<Dim> contraction;
    uint64_t terms = 0u;
    for (auto &axis : lhs.axes()) {
        if (!output.contains(axis.dimension)) {
            if (contraction) { return {}; }
            contraction = axis.dimension;
            terms = axis.extent.constant_value();
        }
    }
    if (!contraction) { return {}; }
    auto rhs_axis = rhs.axis_index(*contraction);
    if (!rhs_axis || rhs.axis(*rhs_axis).extent.constant_value() != terms) { return {}; }
    for (auto &axis : rhs.axes()) {
        if (!output.contains(axis.dimension) && axis.dimension != *contraction) { return {}; }
    }
    auto stride = [](const IndexSpace &space, Dim dimension) noexcept {
        auto index = space.axis_index(dimension);
        if (!index || space.axis(*index).extent.constant_value() == 1u) { return uint64_t{0u}; }
        auto result = uint64_t{1u};
        for (auto i = *index + 1u; i < space.rank(); i++) { result *= space.axis(i).extent.constant_value(); }
        return result;
    };
    compute::xir::StridedMmaDescriptor descriptor;
    descriptor.contraction_extent = terms;
    descriptor.lhs_contraction_stride = stride(lhs, *contraction);
    descriptor.rhs_contraction_stride = stride(rhs, *contraction);
    descriptor.vector_width = width;
    descriptor.allow_reassociation = op.mma_policy().allow_reassociation;
    for (auto &axis : output.axes()) {
        descriptor.output_extents.emplace_back(axis.extent.constant_value());
        descriptor.lhs_output_strides.emplace_back(stride(lhs, axis.dimension));
        descriptor.rhs_output_strides.emplace_back(stride(rhs, axis.dimension));
    }
    if (descriptor.output_extents.empty()) {
        descriptor.output_extents.emplace_back(1u);
        descriptor.lhs_output_strides.emplace_back(0u);
        descriptor.rhs_output_strides.emplace_back(0u);
    }
    auto inner = output.rank();
    while (inner != 0u && descriptor.output_extents[inner - 1u] == 1u) { inner--; }
    if (inner != 0u && descriptor.output_extents[inner - 1u] >= width &&
        ((descriptor.lhs_output_strides[inner - 1u] == 0u && descriptor.rhs_output_strides[inner - 1u] == 1u) ||
         (descriptor.rhs_output_strides[inner - 1u] == 0u && descriptor.lhs_output_strides[inner - 1u] == 1u))) {
        descriptor.vectorization = compute::xir::StridedMmaVectorization::OUTPUT;
        return descriptor;
    }
    if (terms >= width && descriptor.lhs_contraction_stride == 1u && descriptor.rhs_contraction_stride == 1u &&
        descriptor.allow_reassociation) {
        descriptor.vectorization = compute::xir::StridedMmaVectorization::CONTRACTION;
        return descriptor;
    }
    return {};
}

[[nodiscard]] inline bool native_mma_snapshot(const Value *value, const LowerOptions &options) noexcept {
    if (options.native_mma_vector_width == 0u || !value->type().is_tile()) { return false; }
    if (auto producer = value->defining_operation(); producer && native_mma_plan(*producer, options)) { return true; }
    for (auto use : value->use_list()) {
        if (native_mma_plan(*use->user(), options)) { return true; }
    }
    return false;
}

}// namespace luisa::compute::tile::bridge::xir::detail
