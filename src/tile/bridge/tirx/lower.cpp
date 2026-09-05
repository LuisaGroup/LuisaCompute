#include <cmath>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>

#include <tvm/ir/attrs.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/layout.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/verifier.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx {

namespace detail {

namespace {

struct WholeGemmContract {
    uint64_t m;
    uint64_t n;
    uint64_t k;
};

[[nodiscard]] std::optional<uint64_t> static_extent(
    const IndexSpace &space, size_t axis) noexcept {
    if (axis >= space.rank()) { return std::nullopt; }
    auto &&extent = space.axis(axis).extent;
    if (!extent.is_constant()) { return std::nullopt; }
    return extent.constant_value();
}

[[nodiscard]] bool has_shape(
    const Type &type, std::initializer_list<uint64_t> extents) noexcept {
    auto space = type.index_space();
    if (space == nullptr || space->rank() != extents.size()) { return false; }
    auto axis = size_t{0u};
    for (auto extent : extents) {
        if (static_extent(*space, axis++) != extent) { return false; }
    }
    return true;
}

[[nodiscard]] std::optional<uint64_t> unsigned_constant(
    const Value *value) noexcept {
    if (value == nullptr || value->origin() != Value::Origin::OPERATION_RESULT) {
        return std::nullopt;
    }
    auto operation = value->defining_operation();
    if (operation == nullptr || operation->kind() != OperationKind::CONSTANT ||
        operation->result_count() != 1u || operation->result(0u) != value) {
        return std::nullopt;
    }
    auto attribute = operation->attribute("value");
    if (attribute == nullptr) { return std::nullopt; }
    auto &&payload = attribute->value();
    if (auto item = luisa::get_if<uint64_t>(&payload)) { return *item; }
    if (auto item = luisa::get_if<int64_t>(&payload); item != nullptr && *item >= 0) {
        return static_cast<uint64_t>(*item);
    }
    return std::nullopt;
}

[[nodiscard]] bool is_zero_constant(const Value *value) noexcept {
    if (auto item = unsigned_constant(value)) { return *item == 0u; }
    if (value == nullptr || value->origin() != Value::Origin::OPERATION_RESULT) {
        return false;
    }
    auto operation = value->defining_operation();
    if (operation == nullptr || operation->kind() != OperationKind::CONSTANT) {
        return false;
    }
    auto attribute = operation->attribute("value");
    if (attribute == nullptr) { return false; }
    if (auto item = luisa::get_if<double>(&attribute->value())) { return *item == 0.0; }
    return false;
}

[[nodiscard]] bool is_infinity_constant(
    const Value *value, bool negative) noexcept {
    if (value == nullptr || value->origin() != Value::Origin::OPERATION_RESULT) {
        return false;
    }
    auto operation = value->defining_operation();
    if (operation == nullptr || operation->kind() != OperationKind::CONSTANT) {
        return false;
    }
    auto attribute = operation->attribute("value");
    if (attribute == nullptr) { return false; }
    auto item = luisa::get_if<double>(&attribute->value());
    return item != nullptr && std::isinf(*item) && std::signbit(*item) == negative;
}

[[nodiscard]] std::optional<int64_t> match_reduction_contract(
    const Operation &operation) noexcept {
    if (operation.kind() != OperationKind::REDUCE || !operation.domain() ||
        operation.domain()->rank() != 1u || operation.operand_count() != 1u ||
        operation.result_count() != 1u || operation.region_count() != 1u ||
        operation.operand(0u)->type().kind() != TypeKind::SCALAR ||
        operation.operand(0u)->type().scalar_type() != ScalarType::FLOAT32 ||
        operation.result(0u)->type() != operation.operand(0u)->type() ||
        operation.region(0u)->block_count() != 1u) { return std::nullopt; }
    auto body = operation.region(0u)->block(0u);
    if (body->argument_count() != 2u || body->operation_count() != 3u) {
        return std::nullopt;
    }
    auto extract = body->operation(0u);
    auto combine = body->operation(1u);
    auto yield = body->operation(2u);
    if (extract->kind() != OperationKind::TILE_EXTRACT ||
        extract->result_count() != 1u || extract->result(0u)->type().kind() != TypeKind::SCALAR ||
        extract->result(0u)->type().scalar_type() != ScalarType::FLOAT32 ||
        combine->kind() != OperationKind::ELEMENTWISE || combine->operand_count() != 2u ||
        combine->result_count() != 1u || combine->result(0u)->type() != operation.result(0u)->type() ||
        yield->kind() != OperationKind::YIELD || yield->operand_count() != 1u ||
        yield->operand(0u) != combine->result(0u)) { return std::nullopt; }
    auto carry = body->argument(1u);
    auto element = extract->result(0u);
    if (!((combine->operand(0u) == carry && combine->operand(1u) == element) ||
          (combine->operand(1u) == carry && combine->operand(0u) == element))) {
        return std::nullopt;
    }
    switch (combine->elementwise_op()) {
        case ElementwiseOp::ADD:
            return is_zero_constant(operation.operand(0u)) ?
                       std::optional<int64_t>{reduction_add_contract} :
                       std::nullopt;
        case ElementwiseOp::MAX:
            return is_infinity_constant(operation.operand(0u), true) ?
                       std::optional<int64_t>{reduction_max_contract} :
                       std::nullopt;
        case ElementwiseOp::MIN:
            return is_infinity_constant(operation.operand(0u), false) ?
                       std::optional<int64_t>{reduction_min_contract} :
                       std::nullopt;
        default: return std::nullopt;
    }
}

[[nodiscard]] bool scaled_index(
    const Value *value, const Value *index, uint64_t scale) noexcept {
    if (value == nullptr || index == nullptr ||
        value->origin() != Value::Origin::OPERATION_RESULT) { return false; }
    auto operation = value->defining_operation();
    if (operation == nullptr || operation->kind() != OperationKind::ELEMENTWISE ||
        operation->elementwise_op() != ElementwiseOp::MUL ||
        operation->operand_count() != 2u || operation->result_count() != 1u ||
        operation->result(0u) != value) { return false; }
    auto lhs = operation->operand(0u);
    auto rhs = operation->operand(1u);
    return (lhs == index && unsigned_constant(rhs) == scale) ||
           (rhs == index && unsigned_constant(lhs) == scale);
}

[[nodiscard]] bool same_dimension(
    const IndexSpace &lhs, size_t lhs_axis,
    const IndexSpace &rhs, size_t rhs_axis) noexcept {
    return lhs_axis < lhs.rank() && rhs_axis < rhs.rank() &&
           lhs.axis(lhs_axis).dimension == rhs.axis(rhs_axis).dimension;
}

[[nodiscard]] constexpr uint64_t ceil_div_positive(
    uint64_t value, uint64_t divisor) noexcept {
    return value / divisor + static_cast<uint64_t>(value % divisor != 0u);
}

[[nodiscard]] std::optional<WholeGemmContract> match_whole_gemm(
    const Function &function) noexcept {
    if (function.body().block_count() != 1u) { return std::nullopt; }
    auto root = function.body().block(0u);
    if (root->argument_count() != 3u || root->operation_count() != 1u) {
        return std::nullopt;
    }
    auto a_view = root->argument(0u);
    auto b_view = root->argument(1u);
    auto c_view = root->argument(2u);
    for (auto view : {a_view, b_view, c_view}) {
        if (!view->type().is_view() || view->type().scalar_type() != ScalarType::FLOAT32 ||
            view->type().index_space() == nullptr || view->type().index_space()->rank() != 2u) {
            return std::nullopt;
        }
    }
    auto a_space = a_view->type().index_space();
    auto b_space = b_view->type().index_space();
    auto c_space = c_view->type().index_space();
    auto m = static_extent(*a_space, 0u);
    auto k = static_extent(*a_space, 1u);
    auto b_k = static_extent(*b_space, 0u);
    auto n = static_extent(*b_space, 1u);
    if (!m || !n || !k || *m == 0u || *n == 0u || *k == 0u ||
        b_k != k || static_extent(*c_space, 0u) != m ||
        static_extent(*c_space, 1u) != n) { return std::nullopt; }

    auto parallel = root->operation(0u);
    if (parallel->kind() != OperationKind::PARALLEL ||
        parallel->operand_count() != 0u || parallel->result_count() != 0u ||
        parallel->region_count() != 1u || !parallel->domain() ||
        parallel->domain()->rank() != 2u ||
        parallel->execution_scope_constraint() ||
        parallel->resource_class_constraint() ||
        parallel->region(0u)->block_count() != 1u) { return std::nullopt; }
    auto body = parallel->region(0u)->block(0u);
    if (body->argument_count() != 2u || body->operation_count() != 8u) {
        return std::nullopt;
    }

    const Operation *pipeline = nullptr;
    const Operation *store = nullptr;
    const Operation *outer_yield = nullptr;
    size_t constants = 0u;
    size_t multiplies = 0u;
    for (auto operation : body->operations()) {
        switch (operation->kind()) {
            case OperationKind::CONSTANT: constants++; break;
            case OperationKind::ELEMENTWISE:
                if (operation->elementwise_op() != ElementwiseOp::MUL) { return std::nullopt; }
                multiplies++;
                break;
            case OperationKind::PIPELINE:
                if (pipeline != nullptr) { return std::nullopt; }
                pipeline = operation;
                break;
            case OperationKind::VIEW_STORE:
                if (store != nullptr) { return std::nullopt; }
                store = operation;
                break;
            case OperationKind::YIELD:
                if (outer_yield != nullptr) { return std::nullopt; }
                outer_yield = operation;
                break;
            default: return std::nullopt;
        }
    }
    if (constants != 3u || multiplies != 2u || pipeline == nullptr ||
        store == nullptr || outer_yield == nullptr ||
        outer_yield->operand_count() != 0u ||
        outer_yield != body->operation(body->operation_count() - 1u)) {
        return std::nullopt;
    }
    if (pipeline->operand_count() != 1u || pipeline->result_count() != 1u ||
        pipeline->region_count() != 1u || !pipeline->domain() ||
        pipeline->domain()->rank() != 1u ||
        pipeline->execution_scope_constraint() || pipeline->resource_class_constraint() ||
        pipeline->region(0u)->block_count() != 1u) { return std::nullopt; }
    auto accumulator = pipeline->operand(0u);
    auto result = pipeline->result(0u);
    if (!is_zero_constant(accumulator) ||
        accumulator->type().scalar_type() != ScalarType::FLOAT32 ||
        !(accumulator->type() == result->type()) ||
        !accumulator->type().is_tile()) { return std::nullopt; }
    auto result_space = result->type().index_space();
    if (result_space == nullptr || result_space->rank() != 2u) { return std::nullopt; }
    auto bm = static_extent(*result_space, 0u);
    auto bn = static_extent(*result_space, 1u);
    if (!bm || !bn || *bm == 0u || *bn == 0u ||
        static_extent(*parallel->domain(), 0u) != ceil_div_positive(*m, *bm) ||
        static_extent(*parallel->domain(), 1u) != ceil_div_positive(*n, *bn)) {
        return std::nullopt;
    }
    auto m0 = store->operand_count() == 4u ? store->operand(1u) : nullptr;
    auto n0 = store->operand_count() == 4u ? store->operand(2u) : nullptr;
    if (store->bounds_mode() != BoundsMode::ZERO || !store->domain() ||
        store->operand_count() != 4u || store->operand(0u) != c_view ||
        store->operand(3u) != result || *store->domain() != *result_space ||
        !scaled_index(m0, body->argument(0u), *bm) ||
        !scaled_index(n0, body->argument(1u), *bn)) { return std::nullopt; }

    auto pipeline_body = pipeline->region(0u)->block(0u);
    if (pipeline_body->argument_count() != 2u ||
        pipeline_body->argument(1u)->type() != accumulator->type()) {
        return std::nullopt;
    }
    const Operation *mma = nullptr;
    const Operation *inner_yield = nullptr;
    luisa::vector<const Operation *> loads;
    size_t inner_constants = 0u;
    size_t inner_multiplies = 0u;
    for (auto operation : pipeline_body->operations()) {
        switch (operation->kind()) {
            case OperationKind::CONSTANT: inner_constants++; break;
            case OperationKind::ELEMENTWISE:
                if (operation->elementwise_op() != ElementwiseOp::MUL) { return std::nullopt; }
                inner_multiplies++;
                break;
            case OperationKind::STAGE: break;
            case OperationKind::VIEW_LOAD: loads.emplace_back(operation); break;
            case OperationKind::MMA:
                if (mma != nullptr) { return std::nullopt; }
                mma = operation;
                break;
            case OperationKind::YIELD:
                if (inner_yield != nullptr) { return std::nullopt; }
                inner_yield = operation;
                break;
            default: return std::nullopt;
        }
    }
    if (inner_constants != 1u || inner_multiplies != 1u || loads.size() != 2u ||
        mma == nullptr || inner_yield == nullptr ||
        inner_yield != pipeline_body->operation(pipeline_body->operation_count() - 1u) ||
        mma->operand_count() != 3u || mma->result_count() != 1u ||
        !mma->mma_policy().allow_reassociation ||
        mma->operand(2u) != pipeline_body->argument(1u) ||
        inner_yield->operand_count() != 1u || inner_yield->operand(0u) != mma->result(0u)) {
        return std::nullopt;
    }
    auto a_load = mma->operand(0u)->defining_operation();
    auto b_load = mma->operand(1u)->defining_operation();
    if (a_load == nullptr || b_load == nullptr || a_load == b_load ||
        a_load->kind() != OperationKind::VIEW_LOAD ||
        b_load->kind() != OperationKind::VIEW_LOAD ||
        (a_load != loads[0u] && a_load != loads[1u]) ||
        (b_load != loads[0u] && b_load != loads[1u]) ||
        a_load->bounds_mode() != BoundsMode::ZERO ||
        b_load->bounds_mode() != BoundsMode::ZERO ||
        a_load->operand_count() != 3u || b_load->operand_count() != 3u ||
        a_load->operand(0u) != a_view || b_load->operand(0u) != b_view ||
        !a_load->domain() || !b_load->domain()) { return std::nullopt; }
    auto a_tile = a_load->result(0u)->type().index_space();
    auto b_tile = b_load->result(0u)->type().index_space();
    if (a_tile == nullptr || b_tile == nullptr || a_tile->rank() != 2u || b_tile->rank() != 2u ||
        static_extent(*a_tile, 0u) != bm || static_extent(*b_tile, 1u) != bn) {
        return std::nullopt;
    }
    auto bk = static_extent(*a_tile, 1u);
    if (!bk || *bk == 0u || static_extent(*b_tile, 0u) != bk ||
        !same_dimension(*a_tile, 0u, *result_space, 0u) ||
        !same_dimension(*a_tile, 1u, *b_tile, 0u) ||
        !same_dimension(*b_tile, 1u, *result_space, 1u) ||
        static_extent(*pipeline->domain(), 0u) != ceil_div_positive(*k, *bk)) {
        return std::nullopt;
    }
    auto k0 = a_load->operand(2u);
    if (a_load->operand(1u) != m0 || b_load->operand(1u) != k0 ||
        b_load->operand(2u) != n0 ||
        !scaled_index(k0, pipeline_body->argument(0u), *bk) ||
        !(mma->result(0u)->type() == result->type())) { return std::nullopt; }
    return WholeGemmContract{*m, *n, *k};
}

}// namespace

class FunctionLowerer final {

private:
    using Indices = tvm::ffi::Array<tvm::PrimExpr>;
    using Statements = tvm::ffi::Array<tvm::tirx::Stmt>;
    struct StageBoundary {
        size_t position;
        tvm::ffi::String name;
    };
    using TileExpression = std::function<tvm::PrimExpr(const Indices &)>;
    const Function &_function;
    LowerOptions _options;
    luisa::string _error;
    luisa::unordered_map<const Value *, tvm::PrimExpr> _expressions;
    luisa::unordered_map<const Value *, tvm::tirx::BufferVar> _views;
    luisa::unordered_map<const Value *, tvm::tirx::BufferVar> _memories;
    luisa::unordered_map<const Value *, TileExpression> _tiles;
    uint64_t _temporary_index{0u};

private:
    void _fail(luisa::string message) noexcept {
        if (_error.empty()) { _error = std::move(message); }
    }

    [[nodiscard]] static tvm::PrimType _primitive_type(ScalarType type) {
        switch (type) {
            case ScalarType::BOOL: return tvm::PrimType::Bool();
            case ScalarType::INT8: return tvm::PrimType::Int(8);
            case ScalarType::UINT8: return tvm::PrimType::UInt(8);
            case ScalarType::INT16: return tvm::PrimType::Int(16);
            case ScalarType::UINT16: return tvm::PrimType::UInt(16);
            case ScalarType::INT32: return tvm::PrimType::Int(32);
            case ScalarType::UINT32: return tvm::PrimType::UInt(32);
            case ScalarType::INT64: return tvm::PrimType::Int(64);
            case ScalarType::UINT64: return tvm::PrimType::UInt(64);
            case ScalarType::FLOAT8_E4M3:
                return tvm::PrimType{DLDataTypeCode::kDLFloat8_e4m3fn, 8};
            case ScalarType::FLOAT8_E5M2:
                return tvm::PrimType{DLDataTypeCode::kDLFloat8_e5m2, 8};
            case ScalarType::BFLOAT16: return tvm::PrimType::BFloat(16);
            case ScalarType::FLOAT16: return tvm::PrimType::Float(16);
            case ScalarType::FLOAT32: return tvm::PrimType::Float(32);
            case ScalarType::FLOAT64: return tvm::PrimType::Float(64);
            case ScalarType::INVALID: break;
        }
        return tvm::PrimType::Void();
    }

    [[nodiscard]] static tvm::PrimType _primitive_type(const Type &type) {
        if (type.kind() == TypeKind::INDEX) { return tvm::PrimType::Int(64); }
        if (type.kind() == TypeKind::SCALAR || type.is_tile()) { return _primitive_type(type.scalar_type()); }
        return tvm::PrimType::Void();
    }

    [[nodiscard]] static tvm::ffi::Array<tvm::PrimExpr> _shape(const IndexSpace &space) {
        tvm::ffi::Array<tvm::PrimExpr> result;
        result.reserve(space.rank());
        for (auto &&axis : space.axes()) {
            if (!axis.extent.is_constant() ||
                axis.extent.constant_value() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                return {};
            }
            result.push_back(tvm::IntImm::Int64(static_cast<int64_t>(axis.extent.constant_value())));
        }
        return result;
    }

    [[nodiscard]] tvm::PrimExpr _expression(const Value *value) noexcept {
        if (value == nullptr) {
            _fail("TIRx lowering encountered a null TileIR value");
            return {};
        }
        if (auto iter = _expressions.find(value); iter != _expressions.end()) { return iter->second; }
        _fail(luisa::format("TileIR value %{} has no TIRx expression", value->id()));
        return {};
    }

    [[nodiscard]] tvm::tirx::BufferVar _view(const Value *value) noexcept {
        if (value != nullptr) {
            if (auto iter = _views.find(value); iter != _views.end()) { return iter->second; }
        }
        _fail("TileIR view has no corresponding TIRx buffer parameter");
        return {};
    }

    [[nodiscard]] tvm::ffi::Array<tvm::PrimExpr> _indices(
        const Operation &operation,
        size_t begin,
        size_t count) noexcept {
        tvm::ffi::Array<tvm::PrimExpr> result;
        result.reserve(count);
        for (auto i = 0u; i < count; i++) {
            auto index = _expression(operation.operand(begin + i));
            if (!index.defined()) { return {}; }
            result.push_back(std::move(index));
        }
        return result;
    }

    [[nodiscard]] tvm::PrimExpr _constant(const Operation &operation) {
        auto attribute = operation.attribute("value");
        if (attribute == nullptr || operation.result_count() != 1u) {
            _fail("TileIR constant requires exactly one value attribute and result");
            return {};
        }
        auto type = _primitive_type(operation.result(0u)->type());
        if (type.code() == DLDataTypeCode::kDLUInt && type.bits() == 0) {
            _fail("TileIR constant has a type unsupported by the scalar TIRx bridge");
            return {};
        }
        auto &&value = attribute->value();
        if (auto item = luisa::get_if<bool>(&value)) {
            return tvm::IntImm{type, *item ? 1 : 0};
        }
        if (auto item = luisa::get_if<int64_t>(&value)) {
            return tvm::IntImm{type, *item};
        }
        if (auto item = luisa::get_if<uint64_t>(&value)) {
            return tvm::IntImm{type, static_cast<int64_t>(*item)};
        }
        if (auto item = luisa::get_if<double>(&value)) {
            return tvm::FloatImm{type, *item};
        }
        _fail("TileIR scalar constant has an incompatible attribute payload");
        return {};
    }

    [[nodiscard]] static tvm::PrimExpr _apply_elementwise(
        ElementwiseOp op, luisa::span<const tvm::PrimExpr> operands, tvm::PrimType result_type) {
        switch (op) {
            case ElementwiseOp::ADD: return tvm::add(operands[0u], operands[1u]);
            case ElementwiseOp::SUB: return tvm::sub(operands[0u], operands[1u]);
            case ElementwiseOp::MUL: return tvm::mul(operands[0u], operands[1u]);
            case ElementwiseOp::DIV: return tvm::div(operands[0u], operands[1u]);
            case ElementwiseOp::MOD: return tvm::truncmod(operands[0u], operands[1u]);
            case ElementwiseOp::NEG: return tvm::neg(operands[0u]);
            case ElementwiseOp::MIN: return tvm::min(operands[0u], operands[1u]);
            case ElementwiseOp::MAX: return tvm::max(operands[0u], operands[1u]);
            case ElementwiseOp::CAST:
                return tvm::cast(result_type, operands[0u]);
            case ElementwiseOp::SELECT:
                return tvm::if_then_else(operands[0u], operands[1u], operands[2u]);
            case ElementwiseOp::EQ: return tvm::equal(operands[0u], operands[1u]);
            case ElementwiseOp::NE: return tvm::not_equal(operands[0u], operands[1u]);
            case ElementwiseOp::LT: return tvm::less(operands[0u], operands[1u]);
            case ElementwiseOp::LE: return tvm::less_equal(operands[0u], operands[1u]);
            case ElementwiseOp::GT: return tvm::greater(operands[0u], operands[1u]);
            case ElementwiseOp::GE: return tvm::greater_equal(operands[0u], operands[1u]);
            case ElementwiseOp::LOGICAL_AND: return tvm::logical_and(operands[0u], operands[1u]);
            case ElementwiseOp::LOGICAL_OR: return tvm::logical_or(operands[0u], operands[1u]);
            case ElementwiseOp::LOGICAL_NOT: return tvm::logical_not(operands[0u]);
            case ElementwiseOp::EXP: return tvm::exp(operands[0u]);
            case ElementwiseOp::LOG: return tvm::log(operands[0u]);
            case ElementwiseOp::SQRT: return tvm::sqrt(operands[0u]);
            case ElementwiseOp::TANH: return tvm::tanh(operands[0u]);
            case ElementwiseOp::ABS: return tvm::abs(operands[0u]);
            case ElementwiseOp::INVALID: break;
        }
        return {};
    }

    [[nodiscard]] tvm::PrimExpr _elementwise(const Operation &operation) {
        luisa::vector<tvm::PrimExpr> operands;
        for (auto i = 0u; i < operation.operand_count(); i++) {
            auto operand = _expression(operation.operand(i));
            if (!operand.defined()) { return {}; }
            operands.emplace_back(std::move(operand));
        }
        return _apply_elementwise(operation.elementwise_op(), operands, _primitive_type(operation.result(0)->type()));
    }

    [[nodiscard]] TileExpression _tile(const Value *value) {
        if (auto iter = _tiles.find(value); iter != _tiles.end()) { return iter->second; }
        _fail("TileIR value has no native Tile expression");
        return {};
    }

    // Project the current logical coordinates onto one operand. Shared Dim
    // identities align; missing dimensions broadcast, independently of memory.
    [[nodiscard]] TileExpression _in_domain(const Value *value, const IndexSpace &domain) {
        if (!value->type().is_tile()) {
            auto expression = _expression(value);
            return [expression = std::move(expression)](const Indices &) { return expression; };
        }
        auto expression = _tile(value);
        luisa::vector<std::pair<size_t, bool>> projection;
        for (auto &&axis : value->type().index_space()->axes()) {
            auto index = domain.axis_index(axis.dimension);
            if (!index) {
                _fail("Tile operand dimensions are absent from the expression domain");
                return {};
            }
            auto broadcast = axis.extent.is_constant() && axis.extent.constant_value() == 1u;
            projection.emplace_back(*index, broadcast);
        }
        return [expression = std::move(expression), projection = std::move(projection)](const Indices &indices) {
            Indices projected;
            for (auto [index, broadcast] : projection) { projected.push_back(broadcast ? tvm::IntImm::Int64(0) : indices[index]); }
            return expression(projected);
        };
    }

    [[nodiscard]] tvm::tirx::Stmt _for_each(const IndexSpace &space,
                                            const std::function<tvm::tirx::Stmt(const Indices &)> &body,
                                            bool independent = true) {
        Indices indices;
        tvm::ffi::Array<tvm::tirx::PrimVar> variables;
        auto extents = _shape(space);
        if (extents.size() != space.rank()) {
            _fail("native Tile lowering requires JIT-specialized static extents");
            return {};
        }
        auto count = uint64_t{1u};
        for (auto &&extent : extents) {
            auto value = static_cast<uint64_t>(extent.as<tvm::IntImmNode>()->value);
            if (value != 0u && count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / value) {
                _fail("native Tile element domain exceeds int64 range");
                return {};
            }
            count *= value;
        }
        for (auto i = 0u; i < space.rank(); i++) {
            auto name = tvm::ffi::String{std::string{"tile_i_"} + std::to_string(_temporary_index++)};
            auto variable = tvm::tirx::PrimVar{std::move(name), tvm::PrimType::Int(64)};
            variables.push_back(variable);
            indices.push_back(std::move(variable));
        }
        auto statement = body(indices);
        if (!statement.defined()) { return {}; }
        if (variables.empty()) {
            // Even a scalar map needs a region so its entire body executes
            // once, with private temporaries, when distributed across a group.
            auto name = tvm::ffi::String{std::string{"tile_i_"} + std::to_string(_temporary_index++)};
            variables.push_back(tvm::tirx::PrimVar{std::move(name), tvm::PrimType::Int(64)});
            extents.push_back(tvm::IntImm::Int64(1));
        }
        // Preserve the rectangular loop nest for CPU optimization. Only a
        // cooperative target binding should flatten it into worker chunks.
        for (auto i = variables.size(); i != 0u; i--) {
            tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations;
            if (independent && i == 1u) {
                annotations.Set(independent_elements_annotation, tvm::IntImm::Int64(static_cast<int64_t>(variables.size())));
            }
            statement = tvm::tirx::For{variables[i - 1u], tvm::IntImm::Int64(0), extents[i - 1u],
                                       tvm::tirx::ForKind::kSerial, std::move(statement), std::nullopt, std::move(annotations)};
        }
        return statement;
    }

    [[nodiscard]] tvm::tirx::BufferVar _new_storage(const Type &type, Statements &statements) {
        auto name = tvm::ffi::String{std::string{"tile_storage_"} + std::to_string(_temporary_index++)};
        auto shape = type.is_tile() ? _shape(*type.index_space()) : Indices{tvm::IntImm::Int64(1)};
        auto buffer = tvm::tirx::decl_buffer(std::move(shape), _primitive_type(type), std::move(name), "local");
        statements.push_back(tvm::tirx::AllocBuffer{buffer});
        return buffer;
    }

    void _bind_storage(const Value *value, tvm::tirx::BufferVar buffer) {
        if (value->type().is_tile()) {
            _tiles.insert_or_assign(value, [buffer = std::move(buffer)](const Indices &indices) {
                return tvm::tirx::BufferLoad{buffer, indices};
            });
        } else {
            _bind_expression(value, tvm::tirx::BufferLoad{std::move(buffer), {tvm::IntImm::Int64(0)}});
        }
    }

    [[nodiscard]] tvm::tirx::Stmt _copy_to_storage(const Value *value, const tvm::tirx::BufferVar &buffer) {
        if (!value->type().is_tile()) {
            return tvm::tirx::BufferStore{buffer, _expression(value), {tvm::IntImm::Int64(0)}};
        }
        auto element = _tile(value);
        if (!element) { return {}; }
        return _for_each(*value->type().index_space(), [&](const Indices &indices) {
            return tvm::tirx::BufferStore{buffer, element(indices), indices};
        });
    }

    [[nodiscard]] bool _preserve_shared_tile(ElementwiseOp op) const noexcept {
        if (_options.shared_tiles == SharedTileMaterialization::PRESERVE) {
            return true;
        }
        switch (op) {
            case ElementwiseOp::EXP:
            case ElementwiseOp::LOG:
            case ElementwiseOp::SQRT:
            case ElementwiseOp::TANH: return true;
            default: return false;
        }
    }

    void _lower_tile_elementwise(const Operation &operation, Statements &statements) {
        auto result = operation.result(0);
        luisa::vector<TileExpression> inputs;
        for (auto i = 0u; i < operation.operand_count(); i++) {
            inputs.emplace_back(_in_domain(operation.operand(i), *result->type().index_space()));
        }
        auto op = operation.elementwise_op();
        auto type = _primitive_type(result->type());
        TileExpression expression = [inputs = std::move(inputs), op, type](const Indices &indices) {
            luisa::vector<tvm::PrimExpr> elements;
            for (auto &&input : inputs) { elements.emplace_back(input(indices)); }
            return _apply_elementwise(op, elements, type);
        };
        // A multi-use Tile is one logical SSA definition. Preserve that
        // sharing in structural TIRx instead of irreversibly cloning its
        // producer into every consumer. Target planners may compact this
        // logical storage or deliberately inline/recompute it later.
        if (result->use_count() > 1u && _preserve_shared_tile(op)) {
            auto buffer = _new_storage(result->type(), statements);
            auto materialization = _for_each(*result->type().index_space(), [&](const Indices &indices) {
                return tvm::tirx::BufferStore{buffer, expression(indices), indices};
            });
            if (auto loop = materialization.as<tvm::tirx::For>()) {
                loop.value().CopyOnWrite()->annotations.Set(
                    materialized_pure_tile_annotation,
                    tvm::IntImm::Int32(1));
                materialization = loop.value();
            }
            statements.push_back(std::move(materialization));
            _bind_storage(result, std::move(buffer));
        } else {
            _tiles.insert_or_assign(result, std::move(expression));
        }
    }

    void _lower_tile_load(const Operation &operation, Statements &statements) {
        auto result = operation.result(0);
        auto view = _view(operation.operand(0));
        auto &&space = *operation.domain();
        auto origin = _indices(operation, 1u, space.rank());
        auto buffer = _new_storage(result->type(), statements);
        auto fallback = operation.operand_count() == space.rank() + 2u ?
                            _expression(operation.operand(space.rank() + 1u)) :
                            tvm::cast(_primitive_type(result->type()), tvm::IntImm::Int64(0));
        auto view_shape = _shape(*operation.operand(0)->type().index_space());
        statements.push_back(_for_each(space, [&](const Indices &indices) {
            Indices address;
            tvm::PrimExpr valid = tvm::IntImm{tvm::PrimType::Bool(), 1};
            for (auto i = 0u; i < space.rank(); i++) {
                auto index = origin[i] + indices[i];
                address.push_back(index);
                valid = valid && (index >= 0) && (index < view_shape[i]);
            }
            tvm::PrimExpr value = tvm::tirx::BufferLoad{view, address};
            if (operation.bounds_mode() == BoundsMode::ZERO) {
                value = tvm::if_then_else(std::move(valid), std::move(value), fallback);
            }
            return tvm::tirx::BufferStore{buffer, std::move(value), indices};
        }));
        _bind_storage(result, std::move(buffer));
    }

    void _lower_tile_store(const Operation &operation, Statements &statements) {
        auto view = _view(operation.operand(0));
        auto &&space = *operation.domain();
        auto origin = _indices(operation, 1u, space.rank());
        auto element = _tile(operation.operand(space.rank() + 1u));
        auto view_shape = _shape(*operation.operand(0)->type().index_space());
        statements.push_back(_for_each(space, [&](const Indices &indices) -> tvm::tirx::Stmt {
            Indices address;
            tvm::PrimExpr valid = tvm::IntImm{tvm::PrimType::Bool(), 1};
            for (auto i = 0u; i < space.rank(); i++) {
                auto index = origin[i] + indices[i];
                address.push_back(index);
                valid = valid && (index >= 0) && (index < view_shape[i]);
            }
            tvm::tirx::Stmt store = tvm::tirx::BufferStore{view, element(indices), address};
            return operation.bounds_mode() == BoundsMode::ZERO ?
                       tvm::tirx::IfThenElse{std::move(valid), std::move(store)} :
                       store;
        }));
    }

    void _lower_tile_map(const Operation &operation, Statements &statements) {
        auto result = operation.result(0);
        auto buffer = _new_storage(result->type(), statements);
        auto body = operation.region(0)->block(0);
        statements.push_back(_for_each(*operation.domain(), [&](const Indices &indices) {
            for (auto i = 0u; i < indices.size(); i++) { _bind_expression(body->argument(i), indices[i]); }
            return _lower_block(*body, {buffer}, true, &indices);
        }));
        _bind_storage(result, std::move(buffer));
    }

    void _lower_memory_alloc(const Operation &operation, Statements &statements) {
        auto memory = operation.result(0u);
        auto &&type = memory->type();
        auto &&layout = operation.memory_layout();
        if (layout) {
            auto proof = layout->prove();
            if (!proof.is_storage_safe()) {
                _fail("native Memory layout requires a proved total, in-bounds, injective map; unresolved maps use a finite proof budget of 1048576 logical points");
                return;
            }
        }
        auto &&storage_space = layout ? layout->codomain() : *type.index_space();
        auto shape = _shape(storage_space);
        if (shape.size() != storage_space.rank()) {
            _fail("native Memory allocation requires JIT-specialized static extents");
            return;
        }
        auto name = tvm::ffi::String{std::string{"tile_memory_"} + std::to_string(memory->id())};
        auto buffer = tvm::tirx::decl_buffer(std::move(shape), _primitive_type(type.scalar_type()), std::move(name), "local");
        auto volume = storage_space.static_volume();
        auto element_bytes = static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
        if (!volume || element_bytes == 0u || *volume > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / element_bytes) {
            _fail("native Memory allocation exceeds signed 64-bit byte addressing");
            return;
        }
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations;
        if (auto resource = operation.resource_class_constraint()) {
            annotations.Set(memory_resource_annotation, tvm::ffi::String{std::string{*resource}});
        }
        annotations.Set(manual_memory_annotation, tvm::IntImm::Bool(true));
        statements.push_back(tvm::tirx::AllocBuffer{buffer, std::move(annotations)});
        _memories.emplace(memory, std::move(buffer));
    }

    void _lower_memory_access(const Operation &operation, Statements &statements) {
        auto memory = _memories.find(operation.operand(0u));
        if (memory == _memories.end()) {
            _fail("native Memory access requires a lexically visible allocation");
            return;
        }
        auto volume = operation.operand(0u)->type().index_space()->static_volume();
        if (volume && *volume == 0u) {
            // The map is total vacuously: there are no address events. Keep
            // allocation/resource validation, but do not evaluate the map's
            // unreachable arithmetic while constructing a zero-trip loop.
            if (operation.kind() == OperationKind::MEMORY_LOAD) {
                auto result = operation.result(0u);
                _bind_storage(result, _new_storage(result->type(), statements));
            }
            return;
        }
        auto &&layout = operation.operand(0u)->defining_operation()->memory_layout();
        auto physical_indices = [&](const Indices &indices) {
            return layout ? lower_index_map(*layout, indices) : NativeIndices{indices, {}};
        };
        if (operation.kind() == OperationKind::MEMORY_STORE) {
            auto tile = operation.operand(2u);
            auto element = _tile(tile);
            if (!element) { return; }
            auto statement = _for_each(*tile->type().index_space(), [&](const Indices &indices) -> tvm::tirx::Stmt {
                auto mapped = physical_indices(indices);
                if (!mapped) {
                    _fail(std::move(mapped.error));
                    return {};
                }
                return tvm::tirx::BufferStore{memory->second, element(indices), std::move(mapped.value)};
            });
            if (statement.defined()) { statements.push_back(std::move(statement)); }
        } else {
            auto result = operation.result(0u);
            auto snapshot = _new_storage(result->type(), statements);
            auto statement = _for_each(*result->type().index_space(), [&](const Indices &indices) -> tvm::tirx::Stmt {
                auto mapped = physical_indices(indices);
                if (!mapped) {
                    _fail(std::move(mapped.error));
                    return {};
                }
                return tvm::tirx::BufferStore{snapshot, tvm::tirx::BufferLoad{memory->second, std::move(mapped.value)}, indices};
            });
            if (statement.defined()) { statements.push_back(std::move(statement)); }
            _bind_storage(result, std::move(snapshot));
        }
        // MemoryState is an ordering token, not a runtime payload. The
        // verifier has established the reaching state; serial TIRx effects
        // plus target-generated synchronization implement that dependency.
    }

    void _lower_mma(const Operation &operation, Statements &statements) {
        auto result = operation.result(0);
        auto &&space = *result->type().index_space();
        auto contraction = IndexSpace{};
        auto domain = space;
        for (auto &&axis : operation.operand(0)->type().index_space()->axes()) {
            if (!space.contains(axis.dimension)) {
                static_cast<void>(contraction.add(axis.dimension, axis.extent));
                static_cast<void>(domain.add(axis.dimension, axis.extent));
            }
        }
        auto a = _in_domain(operation.operand(0), domain);
        auto b = _in_domain(operation.operand(1), domain);
        auto initial = _tile(operation.operand(2));
        auto buffer = _new_storage(result->type(), statements);
        auto type = _primitive_type(result->type());
        auto statement = _for_each(space, [&](const Indices &indices) {
            Statements body{tvm::tirx::BufferStore{buffer, initial(indices), indices}};
            body.push_back(_for_each(contraction, [&](const Indices &contracted) {
                auto coordinates = indices;
                for (auto &&index : contracted) { coordinates.push_back(index); }
                auto product = tvm::cast(type, a(coordinates)) * tvm::cast(type, b(coordinates));
                auto sum = tvm::tirx::BufferLoad{buffer, indices} + product;
                return tvm::tirx::BufferStore{buffer, std::move(sum), indices}; }, false));
            return tvm::tirx::SeqStmt::Flatten(body);
        });
        if (auto loop = statement.as<tvm::tirx::For>()) {
            loop.value().CopyOnWrite()->annotations.Set(mma_annotation, tvm::IntImm::Int32(operation.mma_policy().allow_reassociation));
            statement = loop.value();
        }
        statements.push_back(std::move(statement));
        _bind_storage(result, std::move(buffer));
    }

    void _bind_expression(const Value *value, tvm::PrimExpr expression) {
        if (value == nullptr || !expression.defined()) {
            _fail("cannot bind an undefined TIRx expression");
            return;
        }
        _expressions.insert_or_assign(value, std::move(expression));
    }

    [[nodiscard]] tvm::PrimExpr _materialize_expression(
        tvm::PrimExpr expression,
        tvm::ffi::Array<tvm::tirx::Stmt> &statements) {
        if (!expression.defined()) { return {}; }
        auto name = tvm::ffi::String{std::string{"tile_value_"} + std::to_string(_temporary_index++)};
        auto buffer = tvm::tirx::decl_buffer(
            {tvm::IntImm::Int64(1)}, expression.ty(), std::move(name), "local");
        statements.push_back(tvm::tirx::AllocBuffer{buffer});
        statements.push_back(tvm::tirx::BufferStore{
            buffer, std::move(expression), {tvm::IntImm::Int64(0)}});
        return tvm::tirx::BufferLoad{buffer, {tvm::IntImm::Int64(0)}};
    }

    void _materialize(
        const Value *value,
        tvm::PrimExpr expression,
        tvm::ffi::Array<tvm::tirx::Stmt> &statements) {
        _bind_expression(value, _materialize_expression(std::move(expression), statements));
    }

    [[nodiscard]] tvm::tirx::Stmt _lower_structured(const Operation &operation) {
        auto &&domain = *operation.domain();
        auto body = operation.region(0u)->block(0u);
        auto is_parallel = operation.kind() == OperationKind::PARALLEL;
        auto is_pipeline = operation.kind() == OperationKind::PIPELINE;
        auto flatten_domain = is_parallel || is_pipeline;
        tvm::ffi::Array<tvm::tirx::PrimVar> loop_variables;
        tvm::ffi::Array<tvm::PrimExpr> loop_extents;
        luisa::vector<uint64_t> constant_extents;
        loop_variables.reserve(domain.rank());
        loop_extents.reserve(domain.rank());
        constant_extents.reserve(domain.rank());
        for (auto i = 0u; i < domain.rank(); i++) {
            auto &&axis = domain.axis(i);
            if (!axis.extent.is_constant() ||
                axis.extent.constant_value() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                _fail("the initial native TIRx bridge requires JIT-specialized static execution extents");
                return {};
            }
            auto axis_name = _function.parent_module()->dimensions().name(axis.dimension);
            auto name = axis_name.empty() ? std::string{"axis"} : std::string{axis_name};
            name += "_" + std::to_string(operation.id()) + "_" + std::to_string(i);
            tvm::tirx::PrimVar variable{tvm::ffi::String{name}, tvm::PrimType::Int(64)};
            loop_variables.push_back(variable);
            loop_extents.push_back(tvm::IntImm::Int64(static_cast<int64_t>(axis.extent.constant_value())));
            constant_extents.emplace_back(axis.extent.constant_value());
            if (!flatten_domain) { _bind_expression(body->argument(i), variable); }
        }

        tvm::tirx::PrimVar parallel_variable;
        uint64_t parallel_extent = 1u;
        if (flatten_domain) {
            for (auto extent : constant_extents) {
                if (extent != 0u && parallel_extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / extent) {
                    _fail("the flattened TIRx execution domain exceeds int64 range");
                    return {};
                }
                parallel_extent *= extent;
            }
            auto name = tvm::ffi::String{
                std::string{is_parallel ? "parallel_" : "pipeline_"} + std::to_string(operation.id())};
            parallel_variable = tvm::tirx::PrimVar{std::move(name), tvm::PrimType::Int(64)};
            auto trailing_extent = parallel_extent;
            for (auto i = 0u; i < domain.rank(); i++) {
                auto extent = constant_extents[i];
                tvm::PrimExpr coordinate;
                if (parallel_extent == 0u || extent == 1u) {
                    coordinate = tvm::IntImm::Int64(0);
                } else {
                    trailing_extent /= extent;
                    coordinate = parallel_variable;
                    if (trailing_extent != 1u) {
                        coordinate = tvm::floordiv(
                            std::move(coordinate),
                            tvm::IntImm::Int64(static_cast<int64_t>(trailing_extent)));
                    }
                    if (extent != 1u) {
                        coordinate = tvm::floormod(
                            std::move(coordinate),
                            tvm::IntImm::Int64(static_cast<int64_t>(extent)));
                    }
                }
                _bind_expression(body->argument(i), std::move(coordinate));
            }
        }

        luisa::vector<tvm::tirx::BufferVar> carries;
        tvm::ffi::Array<tvm::tirx::Stmt> prefix;
        carries.reserve(operation.result_count());
        for (auto i = 0u; i < operation.result_count(); i++) {
            if (operation.result(i)->type().kind() == TypeKind::MEMORY_STATE) {
                carries.emplace_back();
                continue;
            }
            auto buffer = _new_storage(operation.result(i)->type(), prefix);
            carries.push_back(buffer);
            prefix.push_back(_copy_to_storage(operation.operand(i), buffer));
            _bind_storage(body->argument(domain.rank() + i), buffer);
        }

        auto loop_body = _lower_block(*body, carries, true);
        if (!loop_body.defined()) { return {}; }
        if (flatten_domain) {
            tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations;
            if (is_parallel) {
                annotations.Set(logical_parallel_annotation, tvm::IntImm::Int64(static_cast<int64_t>(operation.id())));
                if (auto &&scope = operation.execution_scope_constraint()) {
                    annotations.Set(execution_scope_annotation, tvm::ffi::String{std::string{*scope}});
                }
            } else {
                annotations.Set(logical_pipeline_annotation, tvm::IntImm::Int64(static_cast<int64_t>(operation.id())));
                auto unsigned_attribute = [&](luisa::string_view name, uint64_t fallback) {
                    auto attribute = operation.attribute(name);
                    auto value = attribute == nullptr ? nullptr : luisa::get_if<uint64_t>(&attribute->value());
                    return tvm::IntImm::Int64(static_cast<int64_t>(value == nullptr ? fallback : *value));
                };
                annotations.Set(pipeline_window_annotation, unsigned_attribute("stages", 0u));
                annotations.Set(pipeline_interval_annotation, unsigned_attribute("initiation_interval", 1u));
            }
            loop_body = tvm::tirx::For{
                parallel_variable,
                tvm::IntImm::Int64(0),
                tvm::IntImm::Int64(static_cast<int64_t>(parallel_extent)),
                tvm::tirx::ForKind::kSerial,
                std::move(loop_body),
                std::nullopt,
                std::move(annotations)};
        } else {
            for (auto i = domain.rank(); i != 0u; i--) {
                loop_body = tvm::tirx::For{
                    loop_variables[i - 1u],
                    tvm::IntImm::Int64(0),
                    loop_extents[i - 1u],
                    tvm::tirx::ForKind::kSerial,
                    std::move(loop_body)};
            }
        }
        if (auto contract = match_reduction_contract(operation)) {
            if (auto loop = loop_body.as<tvm::tirx::For>()) {
                loop.value().CopyOnWrite()->annotations.Set(
                    reduction_contract_annotation,
                    tvm::IntImm::Int32(*contract));
                loop_body = loop.value();
            }
        }
        prefix.push_back(std::move(loop_body));
        for (auto i = 0u; i < operation.result_count(); i++) {
            if (carries[i].defined()) { _bind_storage(operation.result(i), carries[i]); }
        }
        return tvm::tirx::SeqStmt::Flatten(prefix);
    }

    void _lower_operation(
        const Operation &operation,
        tvm::ffi::Array<tvm::tirx::Stmt> &statements) {
        if (!_error.empty()) { return; }
        switch (operation.kind()) {
            case OperationKind::CONSTANT: {
                auto constant = _constant(operation);
                if (operation.result(0)->type().is_tile()) {
                    _tiles.insert_or_assign(operation.result(0), [constant = std::move(constant)](const Indices &) { return constant; });
                } else {
                    _bind_expression(operation.result(0u), std::move(constant));
                }
                break;
            }
            case OperationKind::ELEMENTWISE:
                if (operation.result(0)->type().is_tile()) {
                    _lower_tile_elementwise(operation, statements);
                } else {
                    _bind_expression(operation.result(0u), _elementwise(operation));
                }
                break;
            case OperationKind::VIEW_LOAD: {
                if (operation.domain()) {
                    _lower_tile_load(operation, statements);
                    break;
                }
                auto view = _view(operation.operand(0u));
                if (!view.defined()) { return; }
                auto rank = operation.operand(0u)->type().index_space()->rank();
                auto indices = _indices(operation, 1u, rank);
                if (!_error.empty()) { return; }
                // A TileIR load is an SSA value at this program point. Keep it
                // stable across later writes by materializing it; TIRx passes
                // can scalarize/eliminate the one-element buffer afterwards.
                tvm::PrimExpr value = tvm::tirx::BufferLoad{view, indices};
                if (operation.operand_count() == rank + 3u) {
                    auto predicate = _expression(operation.operand(rank + 1u));
                    auto fallback = _expression(operation.operand(rank + 2u));
                    if (!predicate.defined() || !fallback.defined()) { return; }
                    value = tvm::if_then_else(
                        std::move(predicate), std::move(value), std::move(fallback));
                }
                _materialize(operation.result(0u), std::move(value), statements);
                break;
            }
            case OperationKind::VIEW_STORE: {
                if (operation.domain()) {
                    _lower_tile_store(operation, statements);
                    break;
                }
                auto view = _view(operation.operand(0u));
                if (!view.defined()) { return; }
                auto rank = operation.operand(0u)->type().index_space()->rank();
                auto indices = _indices(operation, 1u, rank);
                auto value = _expression(operation.operand(rank + 1u));
                if (!_error.empty()) { return; }
                statements.push_back(tvm::tirx::BufferStore{view, std::move(value), std::move(indices)});
                break;
            }
            case OperationKind::PARALLEL:
            case OperationKind::SERIAL:
            case OperationKind::PIPELINE:
            case OperationKind::REDUCE: {
                auto statement = _lower_structured(operation);
                if (statement.defined()) { statements.push_back(std::move(statement)); }
                break;
            }
            case OperationKind::STAGE:
                // The containing block consumes cuts into stage segments.
                break;
            case OperationKind::YIELD:
                _fail("yield must be consumed by its enclosing structured operation");
                break;
            case OperationKind::TILE_MAP: _lower_tile_map(operation, statements); break;
            case OperationKind::TILE_EXTRACT: {
                auto tile = _tile(operation.operand(0));
                if (!tile) { return; }
                auto indices = _indices(operation, 1u, operation.operand_count() - 1u);
                _bind_expression(operation.result(0), tile(indices));
                break;
            }
            case OperationKind::MMA: _lower_mma(operation, statements); break;
            case OperationKind::MEMORY_ALLOC: _lower_memory_alloc(operation, statements); break;
            case OperationKind::MEMORY_LOAD:
            case OperationKind::MEMORY_STORE: _lower_memory_access(operation, statements); break;
            case OperationKind::CUSTOM:
                _fail(luisa::format("TileIR operation '{}' is not supported by the native TIRx bridge", operation.name()));
                break;
        }
    }

    [[nodiscard]] tvm::tirx::Stmt _lower_block(
        const Block &block,
        const luisa::vector<tvm::tirx::BufferVar> &carries,
        bool allow_yield,
        const Indices *element_indices = nullptr) {
        tvm::ffi::Array<tvm::tirx::Stmt> statements;
        luisa::vector<StageBoundary> stages;
        bool saw_yield = false;
        for (auto operation_ptr : block.operations()) {
            auto &&operation = *operation_ptr;
            if (operation.kind() == OperationKind::STAGE) {
                auto attribute = operation.attribute("name");
                auto name = attribute == nullptr ? nullptr : luisa::get_if<luisa::string>(&attribute->value());
                // Pure prelude expressions may precede the first cursor cut.
                // The first cut begins stage zero, not a second empty stage.
                stages.push_back({stages.empty() ? 0u : statements.size(),
                                  name == nullptr ? tvm::ffi::String{} : tvm::ffi::String{std::string{*name}}});
                continue;
            }
            if (operation.kind() != OperationKind::YIELD) {
                _lower_operation(operation, statements);
                if (!_error.empty()) { return {}; }
                continue;
            }
            if (!allow_yield || saw_yield || operation.operand_count() != carries.size()) {
                _fail("structured TileIR yield does not match its TIRx loop-carried state");
                return {};
            }
            saw_yield = true;
            if (element_indices != nullptr) {
                statements.push_back(tvm::tirx::BufferStore{
                    carries[0], _expression(operation.operand(0)), *element_indices});
                continue;
            }
            Statements updates;
            for (auto i = 0u; i < carries.size(); i++) {
                auto value = operation.operand(i);
                auto &&type = value->type();
                if (type.kind() == TypeKind::MEMORY_STATE) { continue; }
                auto direct = _copy_to_storage(value, carries[i]);
                auto reads_carry = false;
                tvm::tirx::PostOrderVisit(direct, [&](const tvm::ffi::ObjectRef &node) {
                    if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
                        for (auto &&carry : carries) {
                            reads_carry |= carry.defined() && load->buffer.same_as(carry);
                        }
                    }
                });
                // Carries are distinct, compiler-owned allocations. A value
                // reading none of them is already stable across the entire
                // parallel assignment, including a previously materialized
                // tile. Do not copy it into another snapshot first.
                if (!reads_carry) {
                    updates.push_back(std::move(direct));
                    continue;
                }
                auto buffer = _new_storage(value->type(), statements);
                statements.push_back(_copy_to_storage(value, buffer));
                if (type.is_tile()) {
                    updates.push_back(_for_each(*type.index_space(), [&](const Indices &indices) {
                        return tvm::tirx::BufferStore{carries[i], tvm::tirx::BufferLoad{buffer, indices}, indices};
                    }));
                } else {
                    Indices indices{tvm::IntImm::Int64(0)};
                    updates.push_back(tvm::tirx::BufferStore{
                        carries[i], tvm::tirx::BufferLoad{buffer, indices}, indices});
                }
            }
            // Even direct updates must wait until ALL dependent expressions
            // have been snapshotted. Mixed stable/dependent yields and swaps
            // still have simultaneous SSA semantics, not sequential stores.
            for (auto &&update : updates) { statements.push_back(update); }
        }
        if (!carries.empty() && !saw_yield) {
            _fail("loop-carried TileIR region is missing a yield");
            return {};
        }
        if (stages.empty()) { return tvm::tirx::SeqStmt::Flatten(statements); }
        Statements allocations;
        Statements segments;
        // A cut partitions execution, not lexical storage visibility. Keep
        // immediate allocations in the iteration scope so an SSA load in one
        // stage remains visible to later stages. Never lift through a child
        // loop, conditional, or other execution region.
        std::function<void(const tvm::tirx::Stmt &, Statements &)> partition =
            [&](const tvm::tirx::Stmt &statement, Statements &body) {
                if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
                    for (auto &&child : sequence->seq) { partition(child, body); }
                } else if (statement.as<tvm::tirx::AllocBufferNode>() != nullptr) {
                    allocations.push_back(statement);
                } else {
                    body.push_back(statement);
                }
            };
        for (auto i = 0u; i < stages.size(); i++) {
            Statements body;
            auto end = i + 1u == stages.size() ? statements.size() : stages[i + 1u].position;
            for (auto j = stages[i].position; j < end; j++) { partition(statements[j], body); }
            segments.push_back(tvm::tirx::AttrStmt{
                stages[i].name, pipeline_stage_annotation, tvm::IntImm::Int64(static_cast<int64_t>(i)),
                tvm::tirx::SeqStmt::Flatten(body)});
        }
        for (auto &&segment : segments) { allocations.push_back(segment); }
        return tvm::tirx::SeqStmt::Flatten(allocations);
    }

public:
    explicit FunctionLowerer(const Function &function,
                             const LowerOptions &options) noexcept
        : _function{function}, _options{options} {}

    [[nodiscard]] NativeFunction run() {
        NativeFunction result;
        auto module = _function.parent_module();
        if (module == nullptr) {
            result.error = "TileIR function is detached from its module";
            return result;
        }
        auto verified = verify(*module);
        if (!verified) {
            result.error = verified.diagnostics().empty() ?
                               luisa::string{"TileIR verification failed before TIRx lowering"} :
                               luisa::format("TileIR verification failed: {}", verified.diagnostics().front().message);
            return result;
        }
        if (_function.body().block_count() != 1u) {
            result.error = "the initial TIRx bridge requires exactly one TileIR entry block";
            return result;
        }
        auto root = _function.body().block(0u);
        tvm::ffi::Array<tvm::tirx::Var> parameters;
        parameters.reserve(root->argument_count());
        for (auto &&argument_ptr : root->arguments()) {
            auto argument = argument_ptr.get();
            if (!argument->type().is_view()) {
                result.error = "the initial TIRx bridge only supports View kernel parameters";
                return result;
            }
            auto space = argument->type().index_space();
            auto native_shape = _shape(*space);
            if (native_shape.size() != space->rank()) {
                result.error = "the initial TIRx bridge requires JIT-specialized static View shapes";
                return result;
            }
            auto name = argument->name().empty() ?
                            std::string{"view_"} + std::to_string(argument->id()) :
                            std::string{argument->name()};
            auto view = tvm::tirx::decl_buffer(
                std::move(native_shape),
                _primitive_type(argument->type().scalar_type()),
                tvm::ffi::String{std::move(name)});
            parameters.push_back(view.var());
            _views.emplace(argument, std::move(view));
        }
        auto body = _lower_block(*root, {}, false);
        if (!_error.empty() || !body.defined()) {
            result.error = _error.empty() ? luisa::string{"TIRx lowering produced no body"} : std::move(_error);
            return result;
        }
        result.value = tvm::tirx::PrimFunc{std::move(parameters), std::move(body)};
        if (auto contract = match_whole_gemm(_function)) {
            result.value = tvm::WithAttr(std::move(result.value), whole_gemm_contract_annotation, int64_t{1});
            result.value = tvm::WithAttr(std::move(result.value), whole_gemm_m_annotation, static_cast<int64_t>(contract->m));
            result.value = tvm::WithAttr(std::move(result.value), whole_gemm_n_annotation, static_cast<int64_t>(contract->n));
            result.value = tvm::WithAttr(std::move(result.value), whole_gemm_k_annotation, static_cast<int64_t>(contract->k));
        }
        return result;
    }
};

}// namespace detail

NativeFunction lower(const Function &function,
                     const LowerOptions &options) noexcept {
    try {
        switch (options.shared_tiles) {
            case SharedTileMaterialization::PRESERVE:
            case SharedTileMaterialization::EXPENSIVE_ONLY: break;
            default:
                return NativeFunction{
                    {}, "unknown shared-Tile materialization policy"};
        }
        return detail::FunctionLowerer{function, options}.run();
    } catch (const tvm::ffi::Error &error) {
        return NativeFunction{{}, luisa::string{error.what()}};
    } catch (const std::exception &error) {
        return NativeFunction{{}, luisa::string{error.what()}};
    } catch (...) {
        return NativeFunction{{}, "unknown failure while lowering TileIR to native TIRx"};
    }
}

}// namespace luisa::compute::tile::bridge::tirx
