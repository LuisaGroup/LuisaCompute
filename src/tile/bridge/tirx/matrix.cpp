#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

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

[[nodiscard]] bool is_positive_zero(const tvm::PrimExpr &expression) noexcept {
    auto value = expression.as<tvm::FloatImmNode>();
    return value != nullptr && expression.ty() == tvm::PrimType::Float(32) &&
           value->value == 0.0 && !std::signbit(value->value);
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
    // Present only for a proved zero-padded K suffix. M/N remain in bounds.
    tvm::PrimExpr reduction_length;
};

[[nodiscard]] std::optional<MatrixView> matrix_projection(
    tvm::tirx::BufferVar buffer, const tvm::ffi::Array<tvm::PrimExpr> &indices, const Axes &axes,
    uint32_t row_axis, uint32_t column_axis, uint64_t rows, uint64_t columns,
    tvm::tirx::BufferVar source) {
    if (!buffer.defined() || buffer->dtype != tvm::PrimType::Float(32)) { return {}; }
    auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
    if (!buffer->strides.empty() || buffer->layout ||
        !buffer->allocated_addr.empty() || offset == nullptr || offset->value != 0 ||
        buffer->shape.size() != indices.size()) { return {}; }
    AffineIndex linear;
    for (auto i = 0u; i < indices.size(); i++) {
        auto extent = buffer->shape[i].as<tvm::IntImmNode>();
        auto index = affine_index(indices[i], axes);
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
        return MatrixView{std::move(buffer), indices, row_stride, false, std::move(source)};
    }
    if (row_stride == 1u && column_stride >= rows) {
        return MatrixView{std::move(buffer), indices, column_stride, true, std::move(source)};
    }
    return {};
}

[[nodiscard]] std::optional<MatrixView> matrix_view(
    const tvm::tirx::BufferLoadNode *load, const Axes &axes,
    uint32_t row_axis, uint32_t column_axis, uint64_t rows, uint64_t columns,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer, bool writable = false) {
    if (load == nullptr || load->predicate || load->buffer->dtype != tvm::PrimType::Float(32)) { return {}; }
    auto buffer = map_buffer(load->buffer);
    // The caller authorizes compiler-owned shared allocations and explicitly
    // proved immutable noalias inputs. A global scope label alone is never
    // authority. Writable accumulators still require owned shared storage.
    if (!buffer.defined() || (buffer.scope() != "shared" && (writable || buffer.scope() != "global"))) { return {}; }
    return matrix_projection(std::move(buffer), load->indices, axes, row_axis, column_axis, rows, columns, load->buffer);
}

[[nodiscard]] std::optional<MatrixView> matrix_input(
    const tvm::PrimExpr &value, const Axes &axes, uint32_t row_axis, uint32_t column_axis,
    uint64_t rows, uint64_t columns,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    bool bounded_k, luisa::span<const tvm::tirx::ForNode *const> domain) {
    if (auto direct = matrix_view(value.as<tvm::tirx::BufferLoadNode>(), axes, row_axis, column_axis, rows, columns, map_buffer)) { return direct; }
    auto conditional = value.as<tvm::CallNode>();
    if (!bounded_k || conditional == nullptr || !conditional->op.same_as(tvm::tirx::builtin::if_then_else()) ||
        conditional->args.size() != 3u || !is_positive_zero(conditional->args[2].as_or_throw<tvm::PrimExpr>())) { return {}; }
    auto capability = tvm::ffi::Function::GetGlobal("target.metal.mpp_bounded_k_contract_version");
    if (!capability || (*capability)().cast<int64_t>() != 1) { return {}; }
    auto load = conditional->args[1].as<tvm::tirx::BufferLoadNode>();
    if (load == nullptr || load->predicate || load->indices.size() != 2u || load->buffer->shape.size() != 2u) { return {}; }
    auto buffer = map_buffer(load->buffer);
    // Bounded memory inputs must be explicitly authorized immutable globals;
    // this is not permission to omit padding of shared/manual storage.
    if (!buffer.defined() || buffer.scope() != "global") { return {}; }
    tvm::PrimExpr bounds = tvm::IntImm::Bool(true);
    tvm::PrimExpr length;
    auto outer = row_axis == 2u ? column_axis : row_axis;
    auto outer_extent = row_axis == 2u ? columns : rows;
    auto reduction_extent = row_axis == 2u ? rows : columns;
    auto outer_dimensions = 0u;
    for (auto i = 0u; i < 2u; i++) {
        auto index = affine_index(load->indices[i], axes);
        auto extent = buffer->shape[i].as<tvm::IntImmNode>();
        if (!index || extent == nullptr || extent->value <= 0 || extent->value > std::numeric_limits<int32_t>::max()) { return {}; }
        std::array<uint64_t, 3u> k_stride{}, outer_stride{};
        k_stride[2u] = 1u;
        outer_stride[outer] = 1u;
        if (index->strides == k_stride) {
            if (length.defined() || !prove_in_loop_domain(index->base >= 0 && index->base < buffer->shape[i], domain)) { return {}; }
            length = tvm::min(buffer->shape[i] - index->base, tvm::IntImm::Int64(static_cast<int64_t>(reduction_extent)));
        } else if (index->strides == outer_stride) {
            outer_dimensions++;
            auto last = index->base + tvm::IntImm::Int64(static_cast<int64_t>(outer_extent));
            if (!prove_in_loop_domain(index->base >= 0 && last <= buffer->shape[i], domain)) { return {}; }
        } else {
            return {};
        }
        bounds = bounds && load->indices[i] >= 0 && load->indices[i] < buffer->shape[i];
    }
    if (!length.defined() || outer_dimensions != 1u ||
        !prove_in_loop_domain(tvm::equal(conditional->args[0].as_or_throw<tvm::PrimExpr>(), bounds), domain)) { return {}; }
    // The actual K suffix can be shorter than the nominal tile/leading stride.
    // Only that dimension relaxes the full-rectangle projection check.
    auto result = matrix_projection(buffer, load->indices, axes, row_axis, column_axis,
                                    row_axis == 2u ? 1u : rows, column_axis == 2u ? 1u : columns, load->buffer);
    if (result) { result->reduction_length = std::move(length); }
    return result;
}

}// namespace

bool prove_in_loop_domain(tvm::PrimExpr predicate, luisa::span<const tvm::tirx::ForNode *const> domain) {
    auto pure = true;
    tvm::tirx::PostOrderVisit(predicate, [&](const tvm::ffi::ObjectRef &node) {
        pure &= node.as<tvm::tirx::BufferLoadNode>() == nullptr && node.as<tvm::CallNode>() == nullptr;
    });
    if (!pure) { return false; }
    // Use TVMx's public native simplifier as a proof query. This borrowed
    // function never executes and never rewrites the actual program/markers.
    // An observable Boolean store prevents a discarded pure expression from
    // being mistaken for a proof. Every surviving store must be literal true.
    auto result = tvm::tirx::decl_buffer({tvm::IntImm::Int64(1)}, tvm::PrimType::Bool(), "bounds_proof", "global");
    tvm::tirx::Stmt body = tvm::tirx::BufferStore{result, std::move(predicate), {tvm::IntImm::Int64(0)}};
    for (auto i = domain.size(); i != 0u; i--) {
        auto loop = domain[i - 1u];
        auto extent = loop->extent.as<tvm::IntImmNode>();
        auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
        if (extent == nullptr || extent->value <= 0 || loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding ||
            (loop->step && (step == nullptr || step->value != 1))) { return false; }
        body = tvm::tirx::For{loop->loop_var, loop->min, loop->extent, tvm::tirx::ForKind::kSerial, std::move(body)};
    }
    auto function = tvm::tirx::PrimFunc{tvm::tirx::UndefinedVars(body, {}), body};
    auto global = tvm::GlobalVar{"tile_bounds_proof"};
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> functions{{global, std::move(function)}};
    static auto make_module = tvm::ffi::Function::GetGlobalRequired("ir.IRModule");
    static auto run_pass = tvm::ffi::Function::GetGlobalRequired("transform.RunPass");
    auto module = make_module(std::move(functions), tvm::DictAttrs{},
                              tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>{})
                      .cast<tvm::IRModule>();
    module = run_pass(tvm::tirx::transform::StmtSimplify(), std::move(module)).cast<tvm::IRModule>();
    auto simplified = module->functions.at(global).as<tvm::tirx::PrimFunc>().value();
    auto stores = 0u;
    auto proven = true;
    tvm::tirx::PostOrderVisit(simplified->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto store = node.as<tvm::tirx::BufferStoreNode>()) {
            auto literal = store->value.as<tvm::IntImmNode>();
            proven &= store->buffer.same_as(result) && !store->predicate && literal != nullptr && literal->value == 1;
            stores++;
        }
    });
    return proven && stores != 0u;
}

namespace {

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
    tvm::PrimExpr reduction_length;
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
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    bool bounded_k = false, luisa::span<const tvm::tirx::ForNode *const> ancestors = {}) {
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
    Axes axes{loop->loop_var, column_loop->loop_var, contraction->loop_var};
    luisa::vector<const tvm::tirx::ForNode *> domain{ancestors.begin(), ancestors.end()};
    for (auto axis : {loop.get(), column_loop, contraction}) {
        if (std::find(domain.begin(), domain.end(), axis) == domain.end()) { domain.emplace_back(axis); }
    }
    auto a = matrix_input(product->a, axes, 0u, 2u, m, k, map_buffer, bounded_k, domain);
    auto b = matrix_input(product->b, axes, 2u, 1u, k, n, map_buffer, bounded_k, domain);
    auto d = matrix_view(accumulator, axes, 0u, 1u, m, n, map_buffer, true);
    if (!a || !b || !d || d->buffer.same_as(a->buffer) || d->buffer.same_as(b->buffer)) { return {}; }
    tvm::PrimExpr reduction_length;
    if (a->reduction_length.defined() || b->reduction_length.defined()) {
        auto ak = a->reduction_length.defined() ? a->reduction_length : tvm::IntImm::Int64(k);
        auto bk = b->reduction_length.defined() ? b->reduction_length : tvm::IntImm::Int64(k);
        // Omit only a common zero*zero suffix, never 0*an unmasked operand.
        if (!prove_in_loop_domain(tvm::equal(ak, bk), domain)) { return {}; }
        reduction_length = std::move(ak);
    }
    // Initialization is either one uniform literal or an independent C tile.
    auto fill = init->value.as<tvm::FloatImmNode>();
    auto c = matrix_view(init->value.as<tvm::tirx::BufferLoadNode>(), axes, 0u, 1u, m, n, map_buffer);
    if ((fill == nullptr || init->value.ty() != tvm::PrimType::Float(32)) && !c) { return {}; }
    if (c && d->buffer.same_as(c->buffer)) { return {}; }
    return MatchedMatrix{axes, *a, *b, *d, c, init->value, m, n, k, std::move(reduction_length)};
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
    auto direct = loop_emission != nullptr && loop_emission->output.has_value();
    for (auto i = int64_t{0}; i < rows; i++) {
        for (auto j = int64_t{0}; j < columns; j++) {
            auto index = fragment_index(i, j);
            if (matrix.c && !direct) {
                initial.push_back(matrix_transfer(cf, *matrix.c, coordinates(i, j), false, index));
            } else {
                auto value = direct ? loop_emission->initial : matrix.initial;
                initial.push_back(tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), fill_op, {cf, tvm::IntImm::Int32(index), value, tvm::IntImm::Int32(8), tvm::IntImm::Int32(8)}}});
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
    auto destination = loop_emission == nullptr ? matrix.d : *matrix.c;
    if (direct) {
        auto &output = *loop_emission->output;
        auto indices = tvm::tirx::Substitute(output.indices, Coordinates{{output.row, matrix.axes[0]}, {output.column, matrix.axes[1]}});
        destination = MatrixView{output.buffer, std::move(indices), output.stride, output.transpose, output.buffer};
    }
    for (auto i = int64_t{0}; i < rows; i++) {
        for (auto j = int64_t{0}; j < columns; j++) {
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

[[nodiscard]] tvm::tirx::Stmt mpp_matrix(
    const MatchedMatrix &matrix, const MatrixDistribution &distribution,
    const tvm::tirx::PrimVar &thread, MatrixLoopEmission *loop_emission) {
    // Use the already verified contiguous subgroup rectangle. A/B remain
    // memory views read by one MPP operation; only C is materialized. This
    // delegates internal K scheduling to MPP without extending fragment lives
    // or bypassing any TileIR ownership, bounds, or recurrence proof.
    auto m = static_cast<int64_t>(distribution.atom_rows * 8u);
    auto n = static_cast<int64_t>(distribution.atom_columns * 8u);
    auto k = matrix.k;
    auto subgroup = tvm::floordiv(thread, tvm::IntImm::Int64(32));
    auto sg_n = tvm::IntImm::Int64(distribution.subgroups_n);
    Coordinates coordinates{{matrix.axes[0], tvm::floordiv(subgroup, sg_n) * tvm::IntImm::Int64(m)},
                            {matrix.axes[1], tvm::floormod(subgroup, sg_n) * tvm::IntImm::Int64(n)},
                            {matrix.axes[2], tvm::IntImm::Int64(0)}};
    auto cf = tvm::tirx::decl_buffer({tvm::IntImm::Int64(m * n)}, tvm::PrimType::Float(32),
                                     matrix.axes[0]->name + "_mpp_c", "metal.cooperative_tensor");
    auto zero = tvm::IntImm::Int32(0);
    auto transfer = [&](const MatrixView &view, bool store) {
        static const auto load_op = tvm::Op::Get("tirx.cooperative_tensor_load");
        static const auto store_op = tvm::Op::Get("tirx.cooperative_tensor_store");
        return tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), store ? store_op : load_op, {cf, zero, matrix_address(view, coordinates), tvm::IntImm::Int64(static_cast<int64_t>(view.stride)), tvm::IntImm::Int64(m), tvm::IntImm::Int64(n), tvm::IntImm::Bool(view.transpose), tvm::IntImm::Int64(m), tvm::IntImm::Int64(n), tvm::IntImm::Int64(k), tvm::IntImm::Int32(2)}}};
    };
    auto direct = loop_emission != nullptr && loop_emission->output.has_value();
    auto overwrite = direct ? loop_emission->overwrite_accumulator : !matrix.c && is_positive_zero(matrix.initial);
    tvm::ffi::Array<tvm::tirx::Stmt> initial{tvm::tirx::AllocBuffer{cf}};
    // MPP multiply mode defines D = A * B, so no destination
    // initialization is required or observable.
    if (!overwrite) {
        if (matrix.c && !direct) {
            initial.push_back(transfer(*matrix.c, false));
        } else {
            static const auto fill_op = tvm::Op::Get("tirx.cooperative_tensor_fill");
            initial.push_back(tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), fill_op, {cf, zero, direct ? loop_emission->initial : matrix.initial, tvm::IntImm::Int64(m), tvm::IntImm::Int64(n)}}});
        }
    }
    // Resolve only after compile() has checked the extension capability. This
    // keeps ordinary SIMD-group builds link-compatible with unpatched TVMx.
    const auto &mma_op = tvm::Op::Get(overwrite ? "tirx.cooperative_tensor_multiply_from_memory" :
                                                  "tirx.cooperative_tensor_multiply_accumulate_from_memory");
    tvm::ffi::Array<tvm::Expr> mma_args{
        cf, zero,
        matrix_address(matrix.a, coordinates), tvm::IntImm::Int64(static_cast<int64_t>(matrix.a.stride)),
        matrix_address(matrix.b, coordinates), tvm::IntImm::Int64(static_cast<int64_t>(matrix.b.stride))};
    if (!overwrite) {
        mma_args.push_back(cf);
        mma_args.push_back(zero);
    }
    mma_args.push_back(tvm::IntImm::Int64(m));
    mma_args.push_back(tvm::IntImm::Int64(n));
    mma_args.push_back(tvm::IntImm::Int64(k));
    mma_args.push_back(tvm::IntImm::Bool(matrix.a.transpose));
    mma_args.push_back(tvm::IntImm::Bool(matrix.b.transpose));
    if (matrix.reduction_length.defined()) { mma_args.push_back(matrix.reduction_length); }
    auto multiply = tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), mma_op, std::move(mma_args)}};
    auto destination = loop_emission == nullptr ? matrix.d : *matrix.c;
    if (direct) {
        auto &output = *loop_emission->output;
        auto indices = tvm::tirx::Substitute(output.indices, Coordinates{{output.row, matrix.axes[0]}, {output.column, matrix.axes[1]}});
        destination = MatrixView{output.buffer, std::move(indices), output.stride, output.transpose, output.buffer};
    }
    auto final = transfer(destination, true);
    if (loop_emission != nullptr) {
        loop_emission->before = tvm::tirx::SeqStmt::Flatten(initial);
        loop_emission->after = std::move(final);
        if (direct) {
            loop_emission->subgroup_inputs = std::array{matrix.a.buffer, matrix.b.buffer};
            loop_emission->subgroup_step = multiply;
        }
        return multiply;
    }
    initial.push_back(std::move(multiply));
    initial.push_back(std::move(final));
    return tvm::tirx::SeqStmt::Flatten(initial);
}

}// namespace

std::optional<MatrixWorkload> metal_matrix_workload(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    bool bounded_k, luisa::span<const tvm::tirx::ForNode *const> ancestors) {
    if (auto matched = match_metal_matrix(loop, map_buffer, bounded_k, ancestors)) {
        return MatrixWorkload{static_cast<uint64_t>(matched->m), static_cast<uint64_t>(matched->n), static_cast<uint64_t>(matched->k)};
    }
    return {};
}

std::optional<MatrixCarry> metal_matrix_carry(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    bool bounded_k, luisa::span<const tvm::tirx::ForNode *const> ancestors) {
    auto matrix = match_metal_matrix(loop, map_buffer, bounded_k, ancestors);
    if (!matrix || !matrix->c) { return {}; }
    // A valid MMA may use C as A or B. Such a recurrence needs its newly
    // produced elements visible to the next multiply, not only to CF. Keeping
    // CF resident while leaving a stale shared C would silently change it.
    if (matrix->c->buffer.same_as(matrix->a.buffer) || matrix->c->buffer.same_as(matrix->b.buffer)) { return {}; }
    for (auto view : {&*matrix->c, &matrix->d}) {
        if (view->transpose || view->stride != static_cast<uint64_t>(matrix->n) || view->indices.size() != 2u ||
            view->source->shape.size() != 2u || !view->indices[0].same_as(matrix->axes[0]) || !view->indices[1].same_as(matrix->axes[1])) { return {}; }
        auto m = view->source->shape[0].as<tvm::IntImmNode>();
        auto n = view->source->shape[1].as<tvm::IntImmNode>();
        if (m == nullptr || n == nullptr || m->value != matrix->m || n->value != matrix->n) { return {}; }
    }
    return MatrixCarry{matrix->c->source, matrix->d.source, static_cast<uint64_t>(matrix->m), static_cast<uint64_t>(matrix->n)};
}

std::optional<MatrixLoopEmission::Output> metal_matrix_output(
    const tvm::tirx::For &loop, const MatrixCarry &carry, luisa::span<const tvm::tirx::ForNode *const> ancestors) {
    auto independent = loop->annotations.Get(independent_elements_annotation);
    auto rank = independent ? independent.value().as<tvm::IntImmNode>() : nullptr;
    if (rank == nullptr || rank->value != 2 || loop->annotations.size() != 1u) { return {}; }
    auto column = loop->body.as<tvm::tirx::ForNode>();
    if (matrix_extent(loop.get()) != static_cast<int64_t>(carry.rows) || matrix_extent(column) != static_cast<int64_t>(carry.columns) ||
        !column->annotations.empty()) { return {}; }
    auto body = column->body;
    tvm::PrimExpr valid = tvm::IntImm::Bool(true);
    while (auto guard = body.as<tvm::tirx::IfThenElseNode>()) {
        if (guard->else_case) { return {}; }
        valid = valid && guard->condition;
        body = guard->then_case;
    }
    auto store = body.as<tvm::tirx::BufferStoreNode>();
    if (store == nullptr || store->buffer.scope() != "global") { return {}; }
    auto load = store->value.as<tvm::tirx::BufferLoadNode>();
    if (load == nullptr || load->predicate || !load->buffer.same_as(carry.initial) || load->indices.size() != 2u ||
        !load->indices[0].same_as(loop->loop_var) || !load->indices[1].same_as(column->loop_var)) { return {}; }
    Axes axes{loop->loop_var, column->loop_var, tvm::tirx::PrimVar{"unused_k", tvm::PrimType::Int(64)}};
    auto view = matrix_projection(store->buffer, store->indices, axes, 0u, 1u, carry.rows, carry.columns, store->buffer);
    if (!view) { return {}; }
    if (store->predicate) { valid = valid && store->predicate.value(); }
    for (auto i = 0u; i < store->indices.size(); i++) {
        valid = valid && store->indices[i] >= tvm::IntImm::Int64(0) && store->indices[i] < store->buffer->shape[i];
    }
    luisa::vector<const tvm::tirx::ForNode *> domain{ancestors.begin(), ancestors.end()};
    domain.emplace_back(loop.get());
    domain.emplace_back(column);
    if (!prove_in_loop_domain(std::move(valid), domain)) { return {}; }
    return MatrixLoopEmission::Output{store->buffer, store->indices, loop->loop_var, column->loop_var, view->stride, view->transpose};
}

tvm::tirx::Stmt try_metal_matrix(
    const tvm::tirx::For &loop, const tvm::tirx::PrimVar &thread, uint64_t threads,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    const MatrixDistribution &distribution, MatrixLoopEmission *loop_emission, bool metal_mpp,
    luisa::span<const tvm::tirx::ForNode *const> ancestors) {
    if (threads < 32u || threads % 32u != 0u) { return {}; }
    auto matched = match_metal_matrix(loop, map_buffer, metal_mpp, ancestors);
    if (!matched) { return {}; }
    auto &[axes, a_view, b_view, d_view, c, initial, m, n, k, reduction_length] = *matched;
    auto a = &a_view;
    auto b = &b_view;
    auto d = &d_view;
    if (distribution.rectangular()) {
        MatrixWorkload workload{static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k)};
        workload.accumulator_iterations = loop_emission == nullptr ? 0u : 1u;
        workload.has_direct_output = loop_emission != nullptr && loop_emission->output.has_value();
        if (threads > std::numeric_limits<uint32_t>::max() ||
            !verify_matrix_distribution(workload, distribution, static_cast<uint32_t>(threads), 32u)) { return {}; }
        if (loop_emission != nullptr && (!distribution.persistent_accumulator || !metal_matrix_carry(loop, map_buffer, metal_mpp, ancestors))) { return {}; }
        if (distribution.direct_accumulator_store &&
            (loop_emission == nullptr || !loop_emission->output || loop_emission->initial.as<tvm::FloatImmNode>() == nullptr ||
             loop_emission->initial.ty() != tvm::PrimType::Float(32))) { return {}; }
        return metal_mpp ? mpp_matrix(*matched, distribution, thread, loop_emission) :
                           rectangular_matrix(*matched, distribution, thread, loop_emission);
    }

    if (metal_mpp) { throw std::runtime_error{"Metal MPP currently requires an exact rectangular subgroup plan"}; }

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
