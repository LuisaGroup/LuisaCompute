#include <exception>
#include <limits>
#include <string>

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/verifier.h>

namespace luisa::compute::tile::bridge::tirx {

namespace detail {

constexpr auto logical_parallel_annotation = "luisa.tile.logical_parallel";

class FunctionLowerer final {

private:
    const Function &_function;
    luisa::string _error;
    luisa::unordered_map<const Value *, tvm::PrimExpr> _expressions;
    luisa::unordered_map<const Value *, tvm::tirx::BufferVar> _views;
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
        if (type.kind() == TypeKind::SCALAR) { return _primitive_type(type.scalar_type()); }
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

    [[nodiscard]] tvm::PrimExpr _constant(const Operation &operation) noexcept {
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

    [[nodiscard]] tvm::PrimExpr _elementwise(const Operation &operation) noexcept {
        luisa::vector<tvm::PrimExpr> operands;
        operands.reserve(operation.operand_count());
        for (auto i = 0u; i < operation.operand_count(); i++) {
            auto operand = _expression(operation.operand(i));
            if (!operand.defined()) { return {}; }
            operands.emplace_back(std::move(operand));
        }
        switch (operation.elementwise_op()) {
            case ElementwiseOp::ADD: return tvm::add(operands[0u], operands[1u]);
            case ElementwiseOp::SUB: return tvm::sub(operands[0u], operands[1u]);
            case ElementwiseOp::MUL: return tvm::mul(operands[0u], operands[1u]);
            case ElementwiseOp::DIV: return tvm::div(operands[0u], operands[1u]);
            case ElementwiseOp::MOD: return tvm::truncmod(operands[0u], operands[1u]);
            case ElementwiseOp::NEG: return tvm::neg(operands[0u]);
            case ElementwiseOp::MIN: return tvm::min(operands[0u], operands[1u]);
            case ElementwiseOp::MAX: return tvm::max(operands[0u], operands[1u]);
            case ElementwiseOp::CAST:
                return tvm::cast(_primitive_type(operation.result(0u)->type()), operands[0u]);
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
        _fail("unsupported elementwise TileIR opcode");
        return {};
    }

    void _bind_expression(const Value *value, tvm::PrimExpr expression) noexcept {
        if (value == nullptr || !expression.defined()) {
            _fail("cannot bind an undefined TIRx expression");
            return;
        }
        _expressions.insert_or_assign(value, std::move(expression));
    }

    [[nodiscard]] tvm::PrimExpr _materialize_expression(
        tvm::PrimExpr expression,
        tvm::ffi::Array<tvm::tirx::Stmt> &statements) noexcept {
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
        tvm::ffi::Array<tvm::tirx::Stmt> &statements) noexcept {
        _bind_expression(value, _materialize_expression(std::move(expression), statements));
    }

    [[nodiscard]] tvm::tirx::Stmt _lower_structured(const Operation &operation) noexcept {
        auto &&domain = *operation.domain();
        auto body = operation.region(0u)->block(0u);
        auto is_parallel = operation.kind() == OperationKind::PARALLEL;
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
            if (!is_parallel) { _bind_expression(body->argument(i), variable); }
        }

        tvm::tirx::PrimVar parallel_variable;
        uint64_t parallel_extent = 1u;
        if (is_parallel) {
            for (auto extent : constant_extents) {
                if (extent != 0u && parallel_extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / extent) {
                    _fail("the flattened TIRx parallel domain exceeds int64 range");
                    return {};
                }
                parallel_extent *= extent;
            }
            auto name = tvm::ffi::String{
                std::string{"parallel_"} + std::to_string(operation.id())};
            parallel_variable = tvm::tirx::PrimVar{std::move(name), tvm::PrimType::Int(64)};
            auto trailing_extent = parallel_extent;
            for (auto i = 0u; i < domain.rank(); i++) {
                auto extent = constant_extents[i];
                tvm::PrimExpr coordinate;
                if (parallel_extent == 0u) {
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

        tvm::ffi::Array<tvm::tirx::BufferVar> carries;
        tvm::ffi::Array<tvm::tirx::Stmt> prefix;
        carries.reserve(operation.result_count());
        for (auto i = 0u; i < operation.result_count(); i++) {
            auto type = _primitive_type(operation.result(i)->type());
            if (type.code() == DLDataTypeCode::kDLUInt && type.bits() == 0) {
                _fail("TIRx loop-carried state must have scalar or index type");
                return {};
            }
            auto name = tvm::ffi::String{
                std::string{"tile_carry_"} + std::to_string(operation.id()) + "_" + std::to_string(i)};
            auto buffer = tvm::tirx::decl_buffer(
                {tvm::IntImm::Int64(1)}, type, std::move(name), "local");
            carries.push_back(buffer);
            prefix.push_back(tvm::tirx::AllocBuffer{buffer});
            auto initial = _expression(operation.operand(i));
            if (!initial.defined()) { return {}; }
            prefix.push_back(tvm::tirx::BufferStore{
                buffer, std::move(initial), {tvm::IntImm::Int64(0)}});
            _bind_expression(
                body->argument(domain.rank() + i),
                tvm::tirx::BufferLoad{buffer, {tvm::IntImm::Int64(0)}});
        }

        auto loop_body = _lower_block(*body, carries, true);
        if (!loop_body.defined()) { return {}; }
        if (is_parallel) {
            tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations{
                {logical_parallel_annotation, tvm::IntImm::Int64(static_cast<int64_t>(operation.id()))}};
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
        prefix.push_back(std::move(loop_body));
        for (auto i = 0u; i < operation.result_count(); i++) {
            _bind_expression(
                operation.result(i),
                tvm::tirx::BufferLoad{carries[i], {tvm::IntImm::Int64(0)}});
        }
        return tvm::tirx::SeqStmt::Flatten(prefix);
    }

    void _lower_operation(
        const Operation &operation,
        tvm::ffi::Array<tvm::tirx::Stmt> &statements) noexcept {
        if (!_error.empty()) { return; }
        switch (operation.kind()) {
            case OperationKind::CONSTANT:
                _bind_expression(operation.result(0u), _constant(operation));
                break;
            case OperationKind::ELEMENTWISE:
                _bind_expression(operation.result(0u), _elementwise(operation));
                break;
            case OperationKind::VIEW_LOAD: {
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
                // Stage markers are schedule boundaries, not runtime effects.
                break;
            case OperationKind::YIELD:
                _fail("yield must be consumed by its enclosing structured operation");
                break;
            case OperationKind::TILE_MAP:
            case OperationKind::MMA:
            case OperationKind::MEMORY_ALLOC:
            case OperationKind::MEMORY_LOAD:
            case OperationKind::MEMORY_STORE:
            case OperationKind::CUSTOM:
                _fail(luisa::format("TileIR operation '{}' is not in the scalar TIRx lowering subset", operation.name()));
                break;
        }
    }

    [[nodiscard]] tvm::tirx::Stmt _lower_block(
        const Block &block,
        const tvm::ffi::Array<tvm::tirx::BufferVar> &carries,
        bool allow_yield) noexcept {
        tvm::ffi::Array<tvm::tirx::Stmt> statements;
        bool saw_yield = false;
        for (auto operation_ptr : block.operations()) {
            auto &&operation = *operation_ptr;
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
            luisa::vector<tvm::PrimExpr> snapshots;
            snapshots.reserve(carries.size());
            for (auto i = 0u; i < carries.size(); i++) {
                auto value = _expression(operation.operand(i));
                if (!value.defined()) { return {}; }
                snapshots.emplace_back(
                    _materialize_expression(std::move(value), statements));
            }
            // A structured yield updates every carried value simultaneously.
            // Snapshot all SSA expressions before writing any carry buffer so
            // one update cannot change the expression of a later update.
            for (auto i = 0u; i < carries.size(); i++) {
                statements.push_back(tvm::tirx::BufferStore{
                    carries[i], std::move(snapshots[i]), {tvm::IntImm::Int64(0)}});
            }
        }
        if (!carries.empty() && !saw_yield) {
            _fail("loop-carried TileIR region is missing a yield");
            return {};
        }
        return tvm::tirx::SeqStmt::Flatten(statements);
    }

public:
    explicit FunctionLowerer(const Function &function) noexcept
        : _function{function} {}

    [[nodiscard]] NativeFunction run() noexcept {
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
        return result;
    }
};

}// namespace detail

NativeFunction lower(const Function &function) noexcept {
    try {
        return detail::FunctionLowerer{function}.run();
    } catch (const tvm::ffi::Error &error) {
        return NativeFunction{{}, luisa::string{error.what()}};
    } catch (const std::exception &error) {
        return NativeFunction{{}, luisa::string{error.what()}};
    } catch (...) {
        return NativeFunction{{}, "unknown failure while lowering TileIR to native TIRx"};
    }
}

}// namespace luisa::compute::tile::bridge::tirx
