#include <array>
#include <limits>
#include <optional>
#include <stdexcept>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/tile/bridge/tirx/layout.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

using Axes = std::array<tvm::tirx::PrimVar, 3u>;
using Coordinates = tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>;

struct AffineIndex {
    tvm::PrimExpr base{tvm::IntImm::Int64(0)};
    std::array<uint64_t, 3u> strides{};
};

[[nodiscard]] bool accumulate_stride(uint64_t &value, uint64_t addend, uint64_t scale = 1u) {
    constexpr auto limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (addend != 0u && scale > (limit - value) / addend) { return false; }
    value += addend * scale;
    return true;
}

// Prove a positive strided matrix projection, rather than guessing it from
// buffer rank or dimension names. Uniform pipeline-slot coordinates remain
// symbolic. Nonlinear/reversed element maps conservatively keep the loop.
[[nodiscard]] std::optional<AffineIndex> affine_index(const tvm::PrimExpr &expression, const Axes &axes) {
    for (auto i = 0u; i < axes.size(); i++) {
        if (expression.same_as(axes[i])) {
            AffineIndex result;
            result.strides[i] = 1u;
            return result;
        }
    }
    auto uniform = true;
    auto pure = true;
    tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
        for (auto &&axis : axes) { uniform &= !node.same_as(axis); }
        pure &= node.as<tvm::tirx::BufferLoadNode>() == nullptr && node.as<tvm::CallNode>() == nullptr;
    });
    if (!pure) { return {}; }
    if (uniform) { return AffineIndex{expression, {}}; }
    if (auto add = expression.as<tvm::tirx::AddNode>()) {
        auto a = affine_index(add->a, axes);
        auto b = affine_index(add->b, axes);
        if (!a || !b) { return {}; }
        a->base = a->base + b->base;
        for (auto i = 0u; i < axes.size(); i++) {
            if (!accumulate_stride(a->strides[i], b->strides[i])) { return {}; }
        }
        return a;
    }
    if (auto sub = expression.as<tvm::tirx::SubNode>()) {
        auto a = affine_index(sub->a, axes);
        auto b = affine_index(sub->b, axes);
        if (!a || !b || b->strides != std::array<uint64_t, 3u>{}) { return {}; }
        a->base = a->base - b->base;
        return a;
    }
    if (auto mul = expression.as<tvm::tirx::MulNode>()) {
        auto scale = mul->a.as<tvm::IntImmNode>();
        auto operand = mul->b;
        if (scale == nullptr) {
            scale = mul->b.as<tvm::IntImmNode>();
            operand = mul->a;
        }
        if (scale == nullptr || scale->value < 0) { return {}; }
        auto result = affine_index(operand, axes);
        if (!result) { return {}; }
        result->base = result->base * tvm::IntImm::Int64(scale->value);
        for (auto &stride : result->strides) {
            auto product = uint64_t{0u};
            if (!accumulate_stride(product, stride, static_cast<uint64_t>(scale->value))) { return {}; }
            stride = product;
        }
        return result;
    }
    return {};
}

struct MatrixView {
    tvm::tirx::BufferVar buffer;
    tvm::ffi::Array<tvm::PrimExpr> indices;
    uint64_t stride;
    bool transpose;
    tvm::tirx::BufferVar source;
};

[[nodiscard]] std::optional<MatrixView> matrix_view(
    const tvm::tirx::BufferLoadNode *load, const Axes &axes,
    uint32_t row_axis, uint32_t column_axis, uint64_t rows, uint64_t columns,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer) {
    if (load == nullptr || load->predicate || load->buffer->dtype != tvm::PrimType::Float(32)) { return {}; }
    auto buffer = map_buffer(load->buffer);
    if (!buffer.defined()) { return {}; }
    auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
    // Compiler-owned shared allocations cannot alias one another. Do not
    // extend that assumption to external views or opaque placed allocations.
    if (buffer.scope() != "shared" || !buffer->strides.empty() || buffer->layout ||
        !buffer->allocated_addr.empty() || offset == nullptr || offset->value != 0 ||
        buffer->shape.size() != load->indices.size()) { return {}; }
    AffineIndex linear;
    for (auto i = 0u; i < load->indices.size(); i++) {
        auto extent = buffer->shape[i].as<tvm::IntImmNode>();
        auto index = affine_index(load->indices[i], axes);
        if (extent == nullptr || extent->value <= 0 || !index) { return {}; }
        linear.base = linear.base * buffer->shape[i] + index->base;
        for (auto j = 0u; j < axes.size(); j++) {
            auto stride = index->strides[j];
            if (!accumulate_stride(stride, linear.strides[j], static_cast<uint64_t>(extent->value))) { return {}; }
            linear.strides[j] = stride;
        }
    }
    for (auto i = 0u; i < axes.size(); i++) {
        if (i != row_axis && i != column_axis && linear.strides[i] != 0u) { return {}; }
    }
    auto row_stride = linear.strides[row_axis];
    auto column_stride = linear.strides[column_axis];
    if (column_stride == 1u && row_stride >= columns) {
        return MatrixView{std::move(buffer), load->indices, row_stride, false, load->buffer};
    }
    if (row_stride == 1u && column_stride >= rows) {
        return MatrixView{std::move(buffer), load->indices, column_stride, true, load->buffer};
    }
    return {};
}

[[nodiscard]] tvm::Expr matrix_address(const MatrixView &view, const Coordinates &coordinates) {
    auto indices = tvm::tirx::Substitute(view.indices, coordinates);
    return tvm::Call{view.buffer.DataPointerType(), tvm::tirx::builtin::address_of(), {tvm::tirx::BufferLoad{view.buffer, std::move(indices)}}};
}

[[nodiscard]] tvm::tirx::Stmt matrix_transfer(
    const tvm::tirx::BufferVar &fragment, const MatrixView &view,
    const Coordinates &coordinates, bool store, int32_t fragment_index = 0) {
    static const auto load_op = tvm::Op::Get("tirx.simdgroup_load");
    static const auto store_op = tvm::Op::Get("tirx.simdgroup_store");
    return tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), store ? store_op : load_op, {fragment, tvm::IntImm::Int32(fragment_index), matrix_address(view, coordinates), tvm::IntImm::Int64(static_cast<int64_t>(view.stride)), tvm::IntImm::Int32(8), tvm::IntImm::Int32(8), tvm::IntImm::Bool(view.transpose)}}};
}

[[nodiscard]] int64_t matrix_extent(const tvm::tirx::ForNode *loop) {
    if (loop == nullptr || loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding ||
        loop->loop_var.ty() != tvm::PrimType::Int(64)) { return 0; }
    auto extent = loop->extent.as<tvm::IntImmNode>();
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
    if (extent == nullptr || minimum == nullptr || minimum->value != 0 ||
        (loop->step && (step == nullptr || step->value != 1)) ||
        extent->value <= 0 || extent->value % 8 != 0) { return 0; }
    return extent->value;
}

struct MatchedMatrix {
    Axes axes;
    MatrixView a, b, d;
    std::optional<MatrixView> c;
    tvm::PrimExpr initial;
    int64_t m, n, k;
};

[[gnu::noinline]] int32_t native_fragment_index(const tvm::tirx::Layout &layout,
                                                const tvm::ffi::Array<tvm::PrimExpr> &shape, int64_t row, int64_t column) {
    auto placement = layout->Apply({tvm::IntImm::Int64(row), tvm::IntImm::Int64(column)}, shape);
    auto index = placement.Get("m");
    auto constant = index ? index->as<tvm::IntImmNode>() : nullptr;
    if (constant == nullptr || constant->value < 0 || constant->value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error{"native matrix distribution did not resolve a static fragment ordinal"};
    }
    return static_cast<int32_t>(constant->value);
}

[[nodiscard]] std::optional<MatchedMatrix> match_metal_matrix(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer) {
    auto permission = loop->annotations.Get(mma_annotation);
    auto independent = loop->annotations.Get(independent_elements_annotation);
    if (!permission || !independent || loop->annotations.size() != 2u) { return {}; }
    auto reassociate = permission.value().as<tvm::IntImmNode>();
    auto rank = independent.value().as<tvm::IntImmNode>();
    if (reassociate == nullptr || reassociate->value != 1 || rank == nullptr || rank->value != 2) { return {}; }
    auto column_loop = loop->body.as<tvm::tirx::ForNode>();
    auto m = matrix_extent(loop.get());
    auto n = matrix_extent(column_loop);
    if (m == 0 || n == 0 || !column_loop->annotations.empty()) { return {}; }
    auto body = column_loop->body.as<tvm::tirx::SeqStmtNode>();
    if (body == nullptr || body->seq.size() != 2u) { return {}; }
    auto init = body->seq[0].as<tvm::tirx::BufferStoreNode>();
    auto contraction = body->seq[1].as<tvm::tirx::ForNode>();
    auto k = matrix_extent(contraction);
    if (init == nullptr || init->predicate || k == 0 || !contraction->annotations.empty()) { return {}; }
    auto update = contraction->body.as<tvm::tirx::BufferStoreNode>();
    if (update == nullptr || update->predicate || !update->buffer.same_as(init->buffer) ||
        !tvm::ffi::StructuralEqual{}(update->indices, init->indices)) { return {}; }
    auto sum = update->value.as<tvm::tirx::AddNode>();
    if (sum == nullptr || update->value.ty() != tvm::PrimType::Float(32)) { return {}; }
    auto accumulator = sum->a.as<tvm::tirx::BufferLoadNode>();
    auto product = sum->b.as<tvm::tirx::MulNode>();
    if (accumulator == nullptr || product == nullptr || accumulator->predicate ||
        !accumulator->buffer.same_as(update->buffer) || !tvm::ffi::StructuralEqual{}(accumulator->indices, update->indices)) { return {}; }
    auto a_load = product->a.as<tvm::tirx::BufferLoadNode>();
    auto b_load = product->b.as<tvm::tirx::BufferLoadNode>();
    Axes axes{loop->loop_var, column_loop->loop_var, contraction->loop_var};
    auto a = matrix_view(a_load, axes, 0u, 2u, m, k, map_buffer);
    auto b = matrix_view(b_load, axes, 2u, 1u, k, n, map_buffer);
    auto d = matrix_view(accumulator, axes, 0u, 1u, m, n, map_buffer);
    if (!a || !b || !d || d->buffer.same_as(a->buffer) || d->buffer.same_as(b->buffer)) { return {}; }
    // Initialization is either one uniform literal or an independent C tile.
    auto fill = init->value.as<tvm::FloatImmNode>();
    auto c = matrix_view(init->value.as<tvm::tirx::BufferLoadNode>(), axes, 0u, 1u, m, n, map_buffer);
    if ((fill == nullptr || init->value.ty() != tvm::PrimType::Float(32)) && !c) { return {}; }
    if (c && d->buffer.same_as(c->buffer)) { return {}; }
    return MatchedMatrix{axes, *a, *b, *d, c, init->value, m, n, k};
}

[[nodiscard]] tvm::tirx::Stmt rectangular_matrix(
    const MatchedMatrix &matrix, const MatrixDistribution &distribution,
    const tvm::tirx::PrimVar &thread, MatrixLoopEmission *loop_emission) {
    auto suffix = matrix.axes[0]->name;
    auto rows = static_cast<int64_t>(distribution.atom_rows);
    auto columns = static_cast<int64_t>(distribution.atom_columns);
    auto subgroup = tvm::floordiv(thread, tvm::IntImm::Int64(32));
    MatrixWorkload workload{static_cast<uint64_t>(matrix.m), static_cast<uint64_t>(matrix.n), static_cast<uint64_t>(matrix.k)};
    auto layout = matrix_distribution_layout(workload, distribution);
    if (!layout) { throw std::runtime_error{layout.error.c_str()}; }
    tvm::ffi::Array<tvm::PrimExpr> atom_shape{tvm::IntImm::Int64(matrix.m / 8), tvm::IntImm::Int64(matrix.n / 8)};
    auto fragment_index = [&](int64_t i, int64_t j) { return native_fragment_index(layout.value, atom_shape, i, j); };
    auto af = tvm::tirx::decl_buffer({tvm::IntImm::Int64(rows * 64)}, tvm::PrimType::Float(32), suffix + "_mma_a", "metal.simdgroup");
    auto bf = tvm::tirx::decl_buffer({tvm::IntImm::Int64(columns * 64)}, tvm::PrimType::Float(32), suffix + "_mma_b", "metal.simdgroup");
    auto cf = tvm::tirx::decl_buffer({tvm::IntImm::Int64(rows * columns * 64)}, tvm::PrimType::Float(32), suffix + "_mma_c", "metal.simdgroup");
    auto reduction = tvm::tirx::PrimVar{suffix + "_mma_k", tvm::PrimType::Int(64)};
    auto coordinates = [&](int64_t i, int64_t j) {
        auto mapped = matrix_atom_coordinates(workload, distribution, subgroup, tvm::IntImm::Int64(i * columns + j));
        if (!mapped) { throw std::runtime_error{mapped.error.c_str()}; }
        return Coordinates{{matrix.axes[0], mapped.value[0] * tvm::IntImm::Int64(8)},
                           {matrix.axes[1], mapped.value[1] * tvm::IntImm::Int64(8)},
                           {matrix.axes[2], reduction * tvm::IntImm::Int64(8)}};
    };
    tvm::ffi::Array<tvm::tirx::Stmt> initial{tvm::tirx::AllocBuffer{cf}};
    tvm::ffi::Array<tvm::tirx::Stmt> statements{tvm::tirx::AllocBuffer{af}, tvm::tirx::AllocBuffer{bf}};
    tvm::ffi::Array<tvm::tirx::Stmt> final;
    static const auto fill_op = tvm::Op::Get("tirx.make_filled_simdgroup_matrix");
    for (auto i = int64_t{0}; i < rows; i++) {
        for (auto j = int64_t{0}; j < columns; j++) {
            auto index = fragment_index(i, j);
            if (matrix.c) {
                initial.push_back(matrix_transfer(cf, *matrix.c, coordinates(i, j), false, index));
            } else {
                initial.push_back(tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), fill_op, {cf, tvm::IntImm::Int32(index), matrix.initial, tvm::IntImm::Int32(8), tvm::IntImm::Int32(8)}}});
            }
        }
    }
    // The contraction is outside the local output-fragment grid. Each A/B
    // fragment is loaded once and reused by all applicable accumulators.
    tvm::ffi::Array<tvm::tirx::Stmt> step;
    for (auto i = int64_t{0}; i < rows; i++) { step.push_back(matrix_transfer(af, matrix.a, coordinates(i, 0), false, static_cast<int32_t>(i))); }
    for (auto j = int64_t{0}; j < columns; j++) { step.push_back(matrix_transfer(bf, matrix.b, coordinates(0, j), false, static_cast<int32_t>(j))); }
    static const auto mma_op = tvm::Op::Get("tirx.simdgroup_multiply_accumulate");
    for (auto i = int64_t{0}; i < rows; i++) {
        for (auto j = int64_t{0}; j < columns; j++) {
            auto index = tvm::IntImm::Int32(fragment_index(i, j));
            step.push_back(tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), mma_op, {cf, index, af, tvm::IntImm::Int32(static_cast<int32_t>(i)), bf, tvm::IntImm::Int32(static_cast<int32_t>(j)), cf, index}}});
        }
    }
    statements.push_back(tvm::tirx::For{reduction, tvm::IntImm::Int64(0), tvm::IntImm::Int64(matrix.k / 8),
                                        tvm::tirx::ForKind::kSerial, tvm::tirx::SeqStmt::Flatten(step)});
    for (auto i = int64_t{0}; i < rows; i++) {
        for (auto j = int64_t{0}; j < columns; j++) {
            auto &destination = loop_emission == nullptr ? matrix.d : *matrix.c;
            final.push_back(matrix_transfer(cf, destination, coordinates(i, j), true, fragment_index(i, j)));
        }
    }
    if (loop_emission != nullptr) {
        loop_emission->before = tvm::tirx::SeqStmt::Flatten(initial);
        loop_emission->after = tvm::tirx::SeqStmt::Flatten(final);
        return tvm::tirx::SeqStmt::Flatten(statements);
    }
    initial.push_back(tvm::tirx::SeqStmt::Flatten(statements));
    initial.push_back(tvm::tirx::SeqStmt::Flatten(final));
    return tvm::tirx::SeqStmt::Flatten(initial);
}

}// namespace

std::optional<MatrixWorkload> metal_matrix_workload(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer) {
    if (auto matched = match_metal_matrix(loop, map_buffer)) {
        return MatrixWorkload{static_cast<uint64_t>(matched->m), static_cast<uint64_t>(matched->n), static_cast<uint64_t>(matched->k)};
    }
    return {};
}

std::optional<MatrixCarry> metal_matrix_carry(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer) {
    auto matrix = match_metal_matrix(loop, map_buffer);
    if (!matrix || !matrix->c) { return {}; }
    for (auto view : {&*matrix->c, &matrix->d}) {
        if (view->transpose || view->stride != static_cast<uint64_t>(matrix->n) || view->indices.size() != 2u ||
            view->source->shape.size() != 2u || !view->indices[0].same_as(matrix->axes[0]) || !view->indices[1].same_as(matrix->axes[1])) { return {}; }
        auto m = view->source->shape[0].as<tvm::IntImmNode>();
        auto n = view->source->shape[1].as<tvm::IntImmNode>();
        if (m == nullptr || n == nullptr || m->value != matrix->m || n->value != matrix->n) { return {}; }
    }
    return MatrixCarry{matrix->c->source, matrix->d.source, static_cast<uint64_t>(matrix->m), static_cast<uint64_t>(matrix->n)};
}

tvm::tirx::Stmt try_metal_matrix(
    const tvm::tirx::For &loop, const tvm::tirx::PrimVar &thread, uint64_t threads,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    const MatrixDistribution &distribution, MatrixLoopEmission *loop_emission) {
    if (threads < 32u || threads % 32u != 0u) { return {}; }
    auto matched = match_metal_matrix(loop, map_buffer);
    if (!matched) { return {}; }
    auto &[axes, a_view, b_view, d_view, c, initial, m, n, k] = *matched;
    auto a = &a_view;
    auto b = &b_view;
    auto d = &d_view;
    if (distribution.rectangular()) {
        MatrixWorkload workload{static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k)};
        workload.accumulator_iterations = loop_emission == nullptr ? 0u : 1u;
        if (threads > std::numeric_limits<uint32_t>::max() ||
            !verify_matrix_distribution(workload, distribution, static_cast<uint32_t>(threads), 32u)) { return {}; }
        if (loop_emission != nullptr && (!distribution.persistent_accumulator || !metal_matrix_carry(loop, map_buffer))) { return {}; }
        return rectangular_matrix(*matched, distribution, thread, loop_emission);
    }

    auto suffix = loop->loop_var->name;
    auto af = tvm::tirx::decl_buffer({tvm::IntImm::Int64(64)}, tvm::PrimType::Float(32), suffix + "_mma_a", "metal.simdgroup");
    auto bf = tvm::tirx::decl_buffer({tvm::IntImm::Int64(64)}, tvm::PrimType::Float(32), suffix + "_mma_b", "metal.simdgroup");
    auto cf = tvm::tirx::decl_buffer({tvm::IntImm::Int64(64)}, tvm::PrimType::Float(32), suffix + "_mma_c", "metal.simdgroup");
    auto wave = tvm::tirx::PrimVar{suffix + "_mma_wave", tvm::PrimType::Int(64)};
    auto reduction = tvm::tirx::PrimVar{suffix + "_mma_k", tvm::PrimType::Int(64)};
    auto groups = static_cast<int64_t>(threads / 32u);
    auto tiles_n = n / 8;
    if (m / 8 > std::numeric_limits<int64_t>::max() / tiles_n) { return {}; }
    auto tiles = (m / 8) * tiles_n;
    auto job = wave * tvm::IntImm::Int64(groups) + tvm::floordiv(thread, tvm::IntImm::Int64(32));
    auto row = tvm::floordiv(job, tvm::IntImm::Int64(tiles_n)) * tvm::IntImm::Int64(8);
    auto column = tvm::floormod(job, tvm::IntImm::Int64(tiles_n)) * tvm::IntImm::Int64(8);
    Coordinates coordinates{{axes[0], row}, {axes[1], column}, {axes[2], reduction * tvm::IntImm::Int64(8)}};
    tvm::ffi::Array<tvm::tirx::Stmt> statements{tvm::tirx::AllocBuffer{af}, tvm::tirx::AllocBuffer{bf}, tvm::tirx::AllocBuffer{cf}};
    if (c) {
        statements.push_back(matrix_transfer(cf, *c, coordinates, false));
    } else {
        static const auto fill_op = tvm::Op::Get("tirx.make_filled_simdgroup_matrix");
        statements.push_back(tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), fill_op, {cf, tvm::IntImm::Int32(0), initial, tvm::IntImm::Int32(8), tvm::IntImm::Int32(8)}}});
    }
    static const auto mma_op = tvm::Op::Get("tirx.simdgroup_multiply_accumulate");
    auto multiply = tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), mma_op, {cf, tvm::IntImm::Int32(0), af, tvm::IntImm::Int32(0), bf, tvm::IntImm::Int32(0), cf, tvm::IntImm::Int32(0)}}};
    auto reduction_body = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
        matrix_transfer(af, *a, coordinates, false), matrix_transfer(bf, *b, coordinates, false), std::move(multiply)});
    statements.push_back(tvm::tirx::For{reduction, tvm::IntImm::Int64(0), tvm::IntImm::Int64(k / 8), tvm::tirx::ForKind::kSerial, std::move(reduction_body)});
    statements.push_back(matrix_transfer(cf, *d, coordinates, true));
    auto result = tvm::tirx::SeqStmt::Flatten(statements);
    if (tiles % groups != 0) {
        // Uniform for every complete SIMD group; never predicate individual
        // lanes around a cooperative matrix instruction.
        result = tvm::tirx::IfThenElse{job < tvm::IntImm::Int64(tiles), std::move(result)};
    }
    auto waves = tiles / groups + (tiles % groups != 0);
    return tvm::tirx::For{wave, tvm::IntImm::Int64(0), tvm::IntImm::Int64(waves), tvm::tirx::ForKind::kSerial, std::move(result)};
}

}// namespace luisa::compute::tile::bridge::tirx::detail
