#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include <tvm/arith/analyzer.h>
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

// Erase an access only when it has the exact same element owner. A pure call
// attribute is necessary but does not license pointer escape, another memory
// input, or dependence on logical row/column coordinates owned by MPP.
class MatrixElementReads final : public tvm::tirx::StmtExprMutator {
private:
    const MatrixCarry &_carry;
    const MatrixEpilogue &_epilogue;
    tvm::tirx::PrimVar _row, _column;

protected:
    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::VarNode *variable) final {
        // Uniform outer coordinates/parameters need a separate availability
        // proof. This first contract accepts closed scalar DAGs only.
        valid = false;
        return tvm::ffi::GetRef<tvm::tirx::Var>(variable);
    }
    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::ProducerLoadNode *load) final {
        valid = false;
        return tvm::ffi::GetRef<tvm::tirx::ProducerLoad>(load);
    }
    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::CallNode *call) final {
        static auto effects = tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
        auto op = call->op.as<tvm::Op>();
        valid &= op && effects.count(op.value()) &&
                 effects[op.value()] <= static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure) &&
                 !call->op.same_as(tvm::tirx::builtin::address_of());
        return StmtExprMutator::VisitExpr_(call);
    }
    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (load->predicate || load->indices.size() != 2u ||
            !load->indices[0].same_as(_row) || !load->indices[1].same_as(_column)) {
            valid = false;
        }
        if (load->buffer.same_as(_carry.initial)) { return _epilogue.input; }
        for (auto &binding : _epilogue.bindings) {
            if (load->buffer.same_as(binding.buffer)) { return binding.scalar; }
        }
        valid = false;
        return tvm::ffi::GetRef<tvm::tirx::BufferLoad>(load);
    }

public:
    bool valid{true};
    MatrixElementReads(const MatrixCarry &carry, const MatrixEpilogue &epilogue,
                       tvm::tirx::PrimVar row, tvm::tirx::PrimVar column)
        : _carry{carry}, _epilogue{epilogue}, _row{std::move(row)}, _column{std::move(column)} {}
    [[nodiscard]] tvm::PrimExpr value(const tvm::PrimExpr &expression) { return VisitPrimExpr(expression); }
};

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
    // Proved positive K prefix and, optionally, a zero-padded M/N prefix.
    // The latter is relative to the logical matrix, before subgroup slicing.
    tvm::PrimExpr reduction_length;
    tvm::PrimExpr outer_length;
};

// TIRx simplification need not canonicalize the order/association of Boolean
// conjunctions. A memory-axis permutation must not change their meaning.
// Match every nontrivial clause in both directions; this is a sufficient
// equivalence proof, not permission to discard an additional output mask.
[[nodiscard]] bool equivalent_conjunctions(
    const tvm::PrimExpr &a, const tvm::PrimExpr &b,
    luisa::span<const tvm::tirx::ForNode *const> domain) {
    if (prove_in_loop_domain(tvm::equal(a, b), domain)) { return true; }
    auto clauses = [&](const tvm::PrimExpr &expression) {
        luisa::vector<tvm::PrimExpr> result;
        auto visit = [&](auto &&self, const tvm::PrimExpr &term) -> void {
            if (auto conjunction = term.as<tvm::tirx::AndNode>()) {
                self(self, conjunction->a);
                self(self, conjunction->b);
            } else if (!prove_in_loop_domain(term, domain)) {
                result.emplace_back(term);
            }
        };
        visit(visit, expression);
        return result;
    };
    auto left = clauses(a);
    auto right = clauses(b);
    auto covered = [&](const auto &from, const auto &to) {
        return std::all_of(from.begin(), from.end(), [&](const auto &term) {
            return std::any_of(to.begin(), to.end(), [&](const auto &candidate) {
                return prove_in_loop_domain(tvm::equal(term, candidate), domain);
            });
        });
    };
    return covered(left, right) && covered(right, left);
}

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
    auto mn_capability = tvm::ffi::Function::GetGlobal("target.metal.mpp_bounded_mnk_contract_version");
    auto bounded_mn = mn_capability && (*mn_capability)().cast<int64_t>() == 1;
    auto load = conditional->args[1].as<tvm::tirx::BufferLoadNode>();
    if (load == nullptr || load->predicate || load->indices.size() != 2u || load->buffer->shape.size() != 2u) { return {}; }
    auto buffer = map_buffer(load->buffer);
    // Bounded memory inputs must be explicitly authorized immutable globals;
    // this is not permission to omit padding of shared/manual storage.
    if (!buffer.defined() || buffer.scope() != "global") { return {}; }
    tvm::PrimExpr bounds = tvm::IntImm::Bool(true);
    tvm::PrimExpr length;
    tvm::PrimExpr outer_length;
    auto outer = row_axis == 2u ? column_axis : row_axis;
    auto outer_extent = row_axis == 2u ? columns : rows;
    auto reduction_extent = row_axis == 2u ? rows : columns;
    auto outer_dimensions = 0u;
    auto transpose = false;
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
            if (!prove_in_loop_domain(index->base >= 0 && last <= buffer->shape[i], domain)) {
                if (!bounded_mn || !prove_in_loop_domain(index->base >= 0, domain)) { return {}; }
                outer_length = tvm::max(tvm::IntImm::Int64(0), tvm::min(buffer->shape[i] - index->base,
                                                                        tvm::IntImm::Int64(static_cast<int64_t>(outer_extent))));
            }
        } else {
            return {};
        }
        if (i == 0u) { transpose = index->strides[row_axis] == 0u; }
        bounds = bounds && load->indices[i] >= 0 && load->indices[i] < buffer->shape[i];
    }
    if (!length.defined() || outer_dimensions != 1u ||
        !prove_in_loop_domain(tvm::equal(conditional->args[0].as_or_throw<tvm::PrimExpr>(), bounds), domain)) { return {}; }
    // Bounds, not nominal padded extents, constrain the physical rectangle.
    // Unit projections identify orientation even if a physical dimension is
    // one and both logical strides happen to be equal.
    auto result = matrix_projection(buffer, load->indices, axes, row_axis, column_axis,
                                    row_axis == 2u || outer_length.defined() ? 1u : rows,
                                    column_axis == 2u || outer_length.defined() ? 1u : columns, load->buffer);
    if (result) {
        result->reduction_length = std::move(length);
        result->outer_length = std::move(outer_length);
        result->transpose = transpose;
        result->stride = static_cast<uint64_t>(buffer->shape[1].as<tvm::IntImmNode>()->value);
    }
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

[[nodiscard]] tvm::Expr matrix_address(const MatrixView &view, const Coordinates &coordinates, const tvm::PrimExpr &present = {}) {
    auto indices = tvm::tirx::Substitute(view.indices, coordinates);
    if (present.defined()) {
        // An empty subgroup rectangle still needs a valid pointer value.
        // Its zero-product realization never reads the absent operand.
        indices = indices.Map([&](const tvm::PrimExpr &index) { return tvm::if_then_else(present, index, tvm::IntImm::Int64(0)); });
    }
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

// Mean of a positive constant/affine extent, optionally capped by a constant,
// over a rectangular static domain. The common bounded-memory K prefix is
// min(cap, base + stride * ordinal). Sum it analytically, without unrolling
// kernels or sampling endpoints as a proxy for a nonlinear expression.
// More general piecewise/multi-axis cases retain the nominal cost bound.
[[nodiscard]] std::optional<double> mean_extent(
    const tvm::PrimExpr &expression, luisa::span<const tvm::tirx::ForNode *const> domain) {
    tvm::arith::Analyzer analyzer;
    tvm::ffi::Array<tvm::tirx::PrimVar> variables;
    for (auto loop : domain) {
        auto minimum = loop->min.as<tvm::IntImmNode>();
        auto extent = loop->extent.as<tvm::IntImmNode>();
        auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
        if (minimum == nullptr || extent == nullptr || extent->value <= 0 || loop->kind != tvm::tirx::ForKind::kSerial ||
            loop->thread_binding || (loop->step && (step == nullptr || step->value != 1))) { return {}; }
        if (minimum->value > std::numeric_limits<int64_t>::max() - (extent->value - 1)) { return {}; }
        analyzer->Bind(loop->loop_var, tvm::Range::FromMinExtent(loop->min, loop->extent));
        variables.push_back(loop->loop_var);
    }
    auto value = analyzer->Simplify(expression);
    if (auto constant = value.as<tvm::IntImmNode>()) {
        return constant->value > 0 ? std::optional{static_cast<double>(constant->value)} : std::nullopt;
    }
    auto cap = std::numeric_limits<int64_t>::max();
    // Preserve the original cap before simplifying its affine operand.
    // Canonical simplification may turn min(c - i*s, k) into c - max(i*s, c-k).
    auto minimum = expression.as<tvm::tirx::MinNode>();
    if (minimum == nullptr) { minimum = value.as<tvm::tirx::MinNode>(); }
    if (minimum != nullptr) {
        auto constant = minimum->a.as<tvm::IntImmNode>();
        auto linear = minimum->b;
        if (constant == nullptr) {
            constant = minimum->b.as<tvm::IntImmNode>();
            linear = minimum->a;
        }
        if (constant == nullptr || constant->value <= 0) { return {}; }
        cap = constant->value;
        value = analyzer->Simplify(linear);
    }
    static auto detect_linear = tvm::ffi::Function::GetGlobalRequired("arith.DetectLinearEquation");
    auto coefficients = detect_linear(value, variables).cast<tvm::ffi::Array<tvm::PrimExpr>>();
    if (coefficients.size() != variables.size() + 1u) { return {}; }
    const tvm::tirx::ForNode *varying = nullptr;
    auto stride = int64_t{0};
    for (auto i = 0u; i < variables.size(); i++) {
        auto coefficient = analyzer->Simplify(coefficients[i]);
        auto constant = coefficient.as<tvm::IntImmNode>();
        if (constant == nullptr) { return {}; }
        if (constant->value != 0) {
            if (varying != nullptr || constant->value == std::numeric_limits<int64_t>::min()) { return {}; }
            varying = domain[i];
            stride = std::abs(constant->value);
        }
    }
    if (varying == nullptr) { return {}; }
    auto count = varying->extent.as<tvm::IntImmNode>()->value;
    auto first = analyzer->Simplify(tvm::tirx::Substitute(value, Coordinates{{varying->loop_var, varying->min}}));
    auto last = analyzer->Simplify(tvm::tirx::Substitute(value, Coordinates{{varying->loop_var, varying->min + varying->extent - 1}}));
    auto a = first.as<tvm::IntImmNode>();
    auto b = last.as<tvm::IntImmNode>();
    if (a == nullptr || b == nullptr || a->value <= 0 || b->value <= 0) { return {}; }
    auto low = std::min(a->value, b->value);
    auto high = std::max(a->value, b->value);
    if (low >= cap) { return static_cast<double>(cap); }
    if (high <= cap) { return 0.5 * static_cast<double>(low) + 0.5 * static_cast<double>(high); }
    auto uncapped = (cap - low) / stride + 1;
    auto fraction = static_cast<double>(uncapped) / static_cast<double>(count);
    auto last_uncapped = low + (uncapped - 1) * stride;
    return fraction * (0.5 * static_cast<double>(low) + 0.5 * static_cast<double>(last_uncapped)) +
           (1.0 - fraction) * static_cast<double>(cap);
}

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
    if (reassociate == nullptr || reassociate->value != 1 || rank == nullptr || rank->value < 2 || rank->value > 16) { return {}; }
    if (rank->value != 2) {
        // Unit factors do not change the physical matrix projection. Keep
        // buffer rank/layout intact and substitute their unique coordinate;
        // only the execution nest is projected, never a tensor-name pattern.
        luisa::vector<const tvm::tirx::ForNode *> retained;
        Coordinates units;
        auto current = loop.get();
        tvm::tirx::Stmt point;
        auto remaining = rank->value;
        for (auto i = int64_t{0}; i < rank->value; i++) {
            if (!current || current->kind != tvm::tirx::ForKind::kSerial || current->thread_binding ||
                current->loop_var.ty() != tvm::PrimType::Int(64) || (i != 0 && !current->annotations.empty())) { return {}; }
            auto extent = current->extent.as<tvm::IntImmNode>();
            auto minimum = current->min.as<tvm::IntImmNode>();
            auto step = current->step ? current->step.value().as<tvm::IntImmNode>() : nullptr;
            if (!extent || extent->value <= 0 || !minimum || minimum->value != 0 ||
                (current->step && (!step || step->value != 1))) { return {}; }
            if (remaining > 2 && extent->value == 1) {
                units.Set(current->loop_var, current->min);
                remaining--;
            } else {
                retained.emplace_back(current);
            }
            point = current->body;
            current = point.as<tvm::tirx::ForNode>();
        }
        if (retained.size() != 2u) { return {}; }
        point = tvm::tirx::Substitute(point, units);
        for (auto i = retained.size(); i != 0u; i--) {
            auto axis = retained[i - 1u];
            tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations;
            if (i == 1u) {
                annotations.Set(independent_elements_annotation, tvm::IntImm::Int64(2));
                annotations.Set(mma_annotation, permission.value());
            }
            point = tvm::tirx::For{axis->loop_var, axis->min, axis->extent, tvm::tirx::ForKind::kSerial,
                                   std::move(point), std::nullopt, std::move(annotations)};
        }
        return match_metal_matrix(point.as_or_throw<tvm::tirx::For>(), map_buffer, bounded_k, ancestors);
    }
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
    auto remaining = [&](const tvm::PrimExpr &length, uint32_t axis, int64_t extent) -> tvm::PrimExpr {
        if (!length.defined()) { return tvm::IntImm::Int64(extent); }
        return tvm::max(tvm::IntImm::Int64(0), tvm::min(tvm::IntImm::Int64(extent),
                                                        length - coordinates.at(matrix.axes[axis]).as_or_throw<tvm::PrimExpr>()));
    };
    auto actual_m = remaining(matrix.a.outer_length, 0u, m);
    auto actual_n = remaining(matrix.b.outer_length, 1u, n);
    auto bounded_mn = matrix.a.outer_length.defined() || matrix.b.outer_length.defined();
    auto cf = tvm::tirx::decl_buffer({tvm::IntImm::Int64(m * n)}, tvm::PrimType::Float(32),
                                     matrix.axes[0]->name + "_mpp_c", "metal.cooperative_tensor");
    auto zero = tvm::IntImm::Int32(0);
    auto transfer = [&](const MatrixView &view, bool store, const tvm::PrimExpr &rows = {}, const tvm::PrimExpr &columns = {}) {
        static const auto load_op = tvm::Op::Get("tirx.cooperative_tensor_load");
        static const auto store_op = tvm::Op::Get("tirx.cooperative_tensor_store");
        auto present = rows.defined() ? rows > 0 && columns > 0 : tvm::PrimExpr{};
        tvm::ffi::Array<tvm::Expr> args{cf, zero, matrix_address(view, coordinates, present), tvm::IntImm::Int64(static_cast<int64_t>(view.stride)), tvm::IntImm::Int64(m), tvm::IntImm::Int64(n), tvm::IntImm::Bool(view.transpose), tvm::IntImm::Int64(m), tvm::IntImm::Int64(n), tvm::IntImm::Int64(k), tvm::IntImm::Int32(2)};
        if (rows.defined()) {
            args.push_back(rows);
            args.push_back(columns);
        }
        return tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), store ? store_op : load_op, std::move(args)}};
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
        matrix_address(matrix.a, coordinates, matrix.a.outer_length.defined() ? actual_m > 0 : tvm::PrimExpr{}), tvm::IntImm::Int64(static_cast<int64_t>(matrix.a.stride)),
        matrix_address(matrix.b, coordinates, matrix.b.outer_length.defined() ? actual_n > 0 : tvm::PrimExpr{}), tvm::IntImm::Int64(static_cast<int64_t>(matrix.b.stride))};
    if (!overwrite) {
        mma_args.push_back(cf);
        mma_args.push_back(zero);
    }
    mma_args.push_back(tvm::IntImm::Int64(m));
    mma_args.push_back(tvm::IntImm::Int64(n));
    mma_args.push_back(tvm::IntImm::Int64(k));
    mma_args.push_back(tvm::IntImm::Bool(matrix.a.transpose));
    mma_args.push_back(tvm::IntImm::Bool(matrix.b.transpose));
    if (bounded_mn) {
        mma_args.push_back(actual_m);
        mma_args.push_back(actual_n);
        mma_args.push_back(matrix.reduction_length.defined() ? matrix.reduction_length : tvm::IntImm::Int64(k));
    } else if (matrix.reduction_length.defined()) {
        mma_args.push_back(matrix.reduction_length);
    }
    auto multiply = tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), mma_op, std::move(mma_args)}};
    auto destination = loop_emission == nullptr ? matrix.d : *matrix.c;
    tvm::PrimExpr output_rows, output_columns;
    if (direct) {
        auto &output = *loop_emission->output;
        auto indices = tvm::tirx::Substitute(output.indices, Coordinates{{output.row, matrix.axes[0]}, {output.column, matrix.axes[1]}});
        destination = MatrixView{output.buffer, std::move(indices), output.stride, output.transpose, output.buffer};
        if (output.rows.defined()) {
            // The sink's domain, not the input padding, decides which values
            // are observable. Compose that domain with this subgroup's origin.
            output_rows = remaining(output.rows, 0u, m);
            output_columns = remaining(output.columns, 1u, n);
        }
    }
    auto final = tvm::tirx::Stmt{transfer(destination, true, output_rows, output_columns)};
    if (direct && loop_emission->output->epilogue) {
        auto &epilogue = *loop_emission->output->epilogue;
        auto index = tvm::tirx::PrimVar{"mpp_element_index", tvm::PrimType::Int(32)};
        auto capacity = tvm::Call{tvm::PrimType::Int(32), tvm::Op::Get("tirx.cooperative_tensor_capacity"), {cf, zero}};
        auto valid = tvm::Call{tvm::PrimType::Bool(), tvm::Op::Get("tirx.cooperative_tensor_is_valid_element"), {cf, zero, index}};
        auto value = tvm::Call{tvm::PrimType::Float(32), tvm::Op::Get("tirx.cooperative_tensor_element_load"), {cf, zero, index}};
        tvm::ffi::Array<tvm::tirx::Stmt> body{tvm::tirx::Bind{epilogue.input, value}};
        for (auto &binding : epilogue.bindings) { body.push_back(tvm::tirx::Bind{binding.scalar, binding.value}); }
        body.push_back(tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), tvm::Op::Get("tirx.cooperative_tensor_element_store"), {cf, zero, index, epilogue.value}}});
        auto update = tvm::tirx::For{index, zero, capacity, tvm::tirx::ForKind::kSerial,
                                     tvm::tirx::IfThenElse{valid, tvm::tirx::SeqStmt::Flatten(body)}};
        final = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{update, final});
    }
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
        MatrixWorkload result{static_cast<uint64_t>(matched->m), static_cast<uint64_t>(matched->n), static_cast<uint64_t>(matched->k)};
        result.overwrites_accumulator = !matched->c && is_positive_zero(matched->initial);
        if (matched->reduction_length.defined()) {
            if (auto mean = mean_extent(matched->reduction_length, ancestors)) { result.mean_contraction = *mean; }
        }
        return result;
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

bool metal_matrix_epilogue_binding(const tvm::tirx::For &loop, const MatrixCarry &carry, MatrixEpilogue &epilogue) {
    auto provenance = loop->annotations.Get(materialized_pure_tile_annotation);
    auto version = provenance ? provenance.value().as<tvm::IntImmNode>() : nullptr;
    auto independent = loop->annotations.Get(independent_elements_annotation);
    auto rank = independent ? independent.value().as<tvm::IntImmNode>() : nullptr;
    auto column = loop->body.as<tvm::tirx::ForNode>();
    if (!version || version->value != 1 || !rank || rank->value != 2 || loop->annotations.size() != 2u ||
        matrix_extent(loop.get()) != static_cast<int64_t>(carry.rows) || matrix_extent(column) != static_cast<int64_t>(carry.columns) ||
        !column->annotations.empty()) { return false; }
    auto store = column->body.as<tvm::tirx::BufferStoreNode>();
    if (!store || store->predicate || store->indices.size() != 2u ||
        !store->indices[0].same_as(loop->loop_var) || !store->indices[1].same_as(column->loop_var) ||
        store->value.ty() != store->buffer->dtype || store->buffer->dtype.lanes() != 1 || store->buffer->dtype.IsScalableVector()) { return false; }
    MatrixElementReads reads{carry, epilogue, loop->loop_var, column->loop_var};
    auto value = reads.value(store->value);
    if (!reads.valid) { return false; }
    epilogue.bindings.emplace_back(MatrixEpilogue::Binding{store->buffer,
                                                           tvm::tirx::PrimVar{store->buffer.name() + "_element", store->buffer->dtype}, value, loop.get()});
    return true;
}

std::optional<MatrixLoopEmission::Output> metal_matrix_output(
    const tvm::tirx::For &loop, const MatrixCarry &carry, luisa::span<const tvm::tirx::ForNode *const> ancestors,
    bool bounded, const MatrixEpilogue *epilogue) {
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
    auto empty_epilogue = MatrixEpilogue{};
    auto &graph = epilogue == nullptr ? empty_epilogue : *epilogue;
    MatrixElementReads reads{carry, graph, loop->loop_var, column->loop_var};
    auto value = reads.value(store->value);
    if (!reads.valid || value.ty() != tvm::PrimType::Float(32) || store->buffer->dtype != tvm::PrimType::Float(32)) { return {}; }
    std::optional<MatrixEpilogue> scalar_epilogue;
    if (!value.same_as(graph.input) || !graph.bindings.empty()) {
        auto capability = bounded && epilogue != nullptr ? tvm::ffi::Function::GetGlobal("target.metal.mpp_element_contract_version") : std::nullopt;
        if (!capability || (*capability)().cast<int64_t>() != 1) { return {}; }
        scalar_epilogue = graph;
        scalar_epilogue->value = std::move(value);
    }
    Axes axes{loop->loop_var, column->loop_var, tvm::tirx::PrimVar{"unused_k", tvm::PrimType::Int(64)}};
    auto view = matrix_projection(store->buffer, store->indices, axes, 0u, 1u, carry.rows, carry.columns, store->buffer);
    if (store->predicate) { valid = valid && store->predicate.value(); }
    tvm::PrimExpr bounds = tvm::IntImm::Bool(true);
    for (auto i = 0u; i < store->indices.size(); i++) {
        bounds = bounds && store->indices[i] >= tvm::IntImm::Int64(0) && store->indices[i] < store->buffer->shape[i];
    }
    luisa::vector<const tvm::tirx::ForNode *> domain{ancestors.begin(), ancestors.end()};
    domain.emplace_back(loop.get());
    domain.emplace_back(column);
    if (view && prove_in_loop_domain(valid && bounds, domain)) {
        return MatrixLoopEmission::Output{store->buffer, store->indices, loop->loop_var, column->loop_var, view->stride, view->transpose, {}, {}, std::move(scalar_epilogue)};
    }
    auto capability = bounded ? tvm::ffi::Function::GetGlobal("target.metal.mpp_bounded_store_contract_version") : std::nullopt;
    if (!capability || (*capability)().cast<int64_t>() != 1 || store->indices.size() != 2u || store->buffer->shape.size() != 2u ||
        !equivalent_conjunctions(valid, bounds, domain)) { return {}; }
    // A bounds guard is not an arbitrary mask. Prove a unit row/column
    // projection and nonnegative origin before deriving its valid prefix.
    // Negative origins and more general affine masks keep scalar stores.
    std::array<tvm::PrimExpr, 2u> lengths;
    auto transpose = false;
    for (auto i = 0u; i < 2u; i++) {
        auto index = affine_index(store->indices[i], axes);
        auto extent = store->buffer->shape[i].as<tvm::IntImmNode>();
        if (!index || extent == nullptr || extent->value <= 0 || extent->value > std::numeric_limits<int32_t>::max() ||
            !prove_in_loop_domain(index->base >= 0, domain)) { return {}; }
        auto axis = index->strides == std::array<uint64_t, 3u>{1u, 0u, 0u} ? 0u :
                    index->strides == std::array<uint64_t, 3u>{0u, 1u, 0u} ? 1u :
                                                                             2u;
        if (axis == 2u || lengths[axis].defined()) { return {}; }
        auto nominal = static_cast<int64_t>(axis == 0u ? carry.rows : carry.columns);
        lengths[axis] = tvm::max(tvm::IntImm::Int64(0), tvm::min(tvm::IntImm::Int64(nominal), store->buffer->shape[i] - index->base));
        if (i == 0u) { transpose = axis == 1u; }
    }
    // Only the actual rectangle must fit the physical leading stride. This
    // also handles a one-element physical dimension without guessing layout.
    view = matrix_projection(store->buffer, store->indices, axes, 0u, 1u, 1u, 1u, store->buffer);
    if (!view) { return {}; }
    return MatrixLoopEmission::Output{store->buffer, store->indices, loop->loop_var, column->loop_var,
                                      static_cast<uint64_t>(store->buffer->shape[1].as<tvm::IntImmNode>()->value), transpose, lengths[0], lengths[1], std::move(scalar_epilogue)};
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
