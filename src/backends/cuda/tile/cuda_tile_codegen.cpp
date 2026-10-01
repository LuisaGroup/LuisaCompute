#include "cuda_tile_codegen.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/verifier.h>

namespace luisa::compute::cuda::native_tile {
namespace {
using namespace tile;

class Emitter {
private:
    const Function &_function;
    Artifact _artifact;
    luisa::unordered_map<const Value *, luisa::string> _values;
    luisa::unordered_map<const Value *, size_t> _buffers;
    uint32_t _indent{1u};
    bool _inside_parallel{false};
    uint32_t _parallel_count{0u};
    const IndexSpace *_map_space{nullptr};
    const Block *_map_body{nullptr};
    luisa::unordered_set<const Value *> _mapped_values;

    void _fail(const Operation *op, luisa::string_view message) noexcept {
        if (_artifact.error.empty()) {
            _artifact.error = op == nullptr ? luisa::format("CUDA Tile IR: {}", message) :
                                              luisa::format("CUDA Tile IR: operation #{} ({}): {}", op->id(), to_string(op->kind()), message);
        }
    }
    void _line(luisa::string_view text) noexcept {
        _artifact.source.append(_indent * 4u, ' ');
        _artifact.source.append(text);
        _artifact.source += '\n';
    }
    [[nodiscard]] static luisa::string _name(const Value *value) noexcept { return luisa::format("v{}", value->id()); }
    [[nodiscard]] luisa::string _value(const Value *value) noexcept {
        if (auto it = _values.find(value); it != _values.end()) { return it->second; }
        _fail(nullptr, "value has no dominating source binding");
        return "invalid_value";
    }
    [[nodiscard]] static luisa::string_view _scalar(ScalarType type) noexcept {
        switch (type) {
            case ScalarType::BOOL: return "bool";
            case ScalarType::INT32: return "int";
            case ScalarType::UINT32: return "unsigned";
            case ScalarType::INT64: return "long long";
            case ScalarType::UINT64: return "unsigned long long";
            case ScalarType::FLOAT16: return "__half";
            case ScalarType::BFLOAT16: return "__nv_bfloat16";
            case ScalarType::FLOAT32: return "float";
            case ScalarType::FLOAT64: return "double";
            default: return {};
        }
    }
    [[nodiscard]] bool _space(const IndexSpace &space, bool tile, const Operation *op) noexcept {
        if ((!tile && space.rank() == 0u) || space.rank() > 3u) {
            _fail(op, "only rank 1..3 buffers/loops and rank 0..3 Tiles are supported");
            return false;
        }
        uint64_t volume = 1u;
        for (auto &&axis : space.axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u ||
                axis.extent.constant_value() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / volume) {
                _fail(op, "extents must be positive static integers with an int64-representable volume");
                return false;
            }
            auto n = axis.extent.constant_value();
            if (tile && !std::has_single_bit(n)) {
                _fail(op, "native Tile extents must be powers of two; implicit padding changes logical semantics");
                return false;
            }
            volume *= n;
        }
        return true;
    }
    [[nodiscard]] luisa::string _shape(const IndexSpace &space) noexcept {
        luisa::string result = "ct::shape<";
        for (auto i = 0u; i < space.rank(); i++) {
            if (i != 0u) { result += ", "; }
            result += luisa::format("{}", space.axis(i).extent.constant_value());
        }
        return result + ">";
    }
    [[nodiscard]] bool _type(const Type &type, const Operation *op) noexcept {
        if (type.kind() == TypeKind::INDEX) { return true; }
        if (type.scalar_type() == ScalarType::FLOAT16 || type.scalar_type() == ScalarType::BFLOAT16 ||
            type.scalar_type() == ScalarType::FLOAT64) {
            _fail(op, "this native runtime slice supports only FP32 floating-point values");
            return false;
        }
        if ((type.kind() != TypeKind::SCALAR && !type.is_tile()) || _scalar(type.scalar_type()).empty()) {
            _fail(op, "unsupported value type (only index, bool, i32/u32/i64/u64, and f32 scalars/Tiles)");
            return false;
        }
        return !type.is_tile() || _space(*type.index_space(), true, op);
    }
    [[nodiscard]] luisa::string_view _element(const Type &type) noexcept {
        return type.kind() == TypeKind::INDEX ? "long long" : _scalar(type.scalar_type());
    }
    void _bind(const Value *value, luisa::string expression) noexcept {
        auto name = _name(value);
        _line(luisa::format("auto {} = {};", name, expression));
        _values.emplace(value, std::move(name));
    }
    [[nodiscard]] static luisa::string _shape(luisa::span<const uint64_t> extents) noexcept {
        luisa::string result = "ct::shape<";
        for (auto i = 0u; i < extents.size(); i++) {
            if (i != 0u) { result += ", "; }
            result += luisa::format("{}", extents[i]);
        }
        return result + ">";
    }
    [[nodiscard]] static luisa::string _permute(luisa::string expression, luisa::span<const uint32_t> order) noexcept {
        auto identity = true;
        for (auto i = 0u; i < order.size(); i++) { identity &= order[i] == i; }
        if (identity) { return expression; }
        luisa::string mapping = "ct::dimension_map<";
        for (auto i = 0u; i < order.size(); i++) {
            if (i != 0u) { mapping += ", "; }
            mapping += luisa::format("{}", order[i]);
        }
        return luisa::format("ct::permute({}, {}>{{}})", expression, mapping);
    }
    // Put source axes into destination order before adding singleton axes.
    // CUDA's positional broadcast can then realize the IR's named-axis rule.
    [[nodiscard]] luisa::string _align(luisa::string expression, const IndexSpace &source,
                                       const IndexSpace &destination, const Operation &op) noexcept {
        luisa::vector<uint32_t> order;
        luisa::vector<uint64_t> extents;
        for (auto &&axis : destination.axes()) {
            auto index = source.axis_index(axis.dimension);
            if (index) {
                auto extent = source.axis(*index).extent.constant_value();
                if (extent != 1u && extent != axis.extent.constant_value()) {
                    _fail(&op, "named-axis broadcast has incompatible extents");
                    return "invalid_broadcast";
                }
                order.emplace_back(static_cast<uint32_t>(*index));
                extents.emplace_back(extent);
            } else {
                extents.emplace_back(1u);
            }
        }
        if (order.size() != source.rank()) {
            _fail(&op, "named-axis broadcast would discard a source dimension");
            return "invalid_broadcast";
        }
        expression = _permute(std::move(expression), order);
        return luisa::format("ct::broadcast(ct::reshape({}, {}{{}}), {}{{}})",
                             expression, _shape(extents), _shape(destination));
    }
    [[nodiscard]] luisa::string _elementwise_value(const Value *value, const Type &result, const Operation &op) noexcept {
        auto expression = _value(value);
        if (result.is_tile() && value->type().is_tile()) {
            return _align(std::move(expression), *value->type().index_space(), *result.index_space(), op);
        }
        return expression;
    }
    [[nodiscard]] luisa::string _map_broadcast(luisa::string expression) noexcept {
        return _map_space == nullptr ? expression :
               luisa::format("ct::broadcast({}, {}{{}})", expression, _shape(*_map_space));
    }
    void _constant(const Operation &op) noexcept {
        if (op.result_count() != 1u) {
            _fail(&op, "constants require one result");
            return;
        }
        auto attr = op.attribute("value");
        auto &&type = op.result(0u)->type();
        if (attr == nullptr) {
            _fail(&op, "constant has no value");
            return;
        }
        luisa::string scalar;
        auto &&value = attr->value();
        if (auto x = luisa::get_if<bool>(&value)) {
            scalar = *x ? "true" : "false";
        } else if (auto x = luisa::get_if<int64_t>(&value)) {
            scalar = luisa::format("ct::element_bitcast<long long>(0x{:016x}ull)", std::bit_cast<uint64_t>(*x));
        } else if (auto x = luisa::get_if<uint64_t>(&value)) {
            scalar = luisa::format("0x{:016x}ull", *x);
        } else if (auto x = luisa::get_if<double>(&value)) {
            // Preserve the attribute bits before the one explicit target cast.
            // A host float intermediate can double-round f16/bf16 constants.
            scalar = luisa::format("ct::element_bitcast<double>(0x{:016x}ull)", std::bit_cast<uint64_t>(*x));
        } else {
            _fail(&op, "unsupported constant attribute");
            return;
        }
        scalar = luisa::format("ct::element_cast<{}>({})", _element(type), scalar);
        _bind(op.result(0u), type.is_tile() ? luisa::format("ct::full<ct::tile<{}, {}>>({})", _element(type), _shape(*type.index_space()), scalar) : scalar);
    }
    void _elementwise(const Operation &op) noexcept {
        auto &&result_type = op.result(0u)->type();
        auto a = _elementwise_value(op.operand(0u), result_type, op);
        auto b = op.operand_count() > 1u ? _elementwise_value(op.operand(1u), result_type, op) : luisa::string{};
        auto element = op.result(0u)->type().scalar_type();
        if ((element == ScalarType::FLOAT16 || element == ScalarType::BFLOAT16) &&
            op.elementwise_op() != ElementwiseOp::CAST && op.elementwise_op() != ElementwiseOp::SELECT) {
            _fail(&op, "narrow elementwise arithmetic rounding is not implemented (load/cast/select/MMA are supported)");
            return;
        }
        luisa::string expression;
        luisa::string_view binary;
        switch (op.elementwise_op()) {
            case ElementwiseOp::ADD: binary = "+"; break;
            case ElementwiseOp::SUB: binary = "-"; break;
            case ElementwiseOp::MUL: binary = "*"; break;
            case ElementwiseOp::DIV: binary = "/"; break;
            case ElementwiseOp::MOD: binary = "%"; break;
            case ElementwiseOp::EQ: binary = "=="; break;
            case ElementwiseOp::NE: binary = "!="; break;
            case ElementwiseOp::LT: binary = "<"; break;
            case ElementwiseOp::LE: binary = "<="; break;
            case ElementwiseOp::GT: binary = ">"; break;
            case ElementwiseOp::GE: binary = ">="; break;
            case ElementwiseOp::LOGICAL_AND: binary = "&&"; break;
            case ElementwiseOp::LOGICAL_OR: binary = "||"; break;
            case ElementwiseOp::NEG: expression = luisa::format("(-{})", a); break;
            case ElementwiseOp::LOGICAL_NOT: expression = luisa::format("(!{})", a); break;
            case ElementwiseOp::CAST: expression = luisa::format("ct::element_cast<{}>({})", _element(op.result(0u)->type()), a); break;
            case ElementwiseOp::MIN:
            case ElementwiseOp::MAX: {
                auto name = op.elementwise_op() == ElementwiseOp::MIN ? "min" : "max";
                expression = element == ScalarType::FLOAT32 ?
                    luisa::format("ct::{}({}, {}, ct::suppress_nan_t{{}}, ct::preserve_subnormals_t{{}})", name, a, b) :
                    luisa::format("ct::{}({}, {})", name, a, b);
                break;
            }
            case ElementwiseOp::EXP: expression = luisa::format("ct::exp({}, ct::round_full_t{{}})", a); break;
            case ElementwiseOp::LOG: expression = luisa::format("ct::log({})", a); break;
            case ElementwiseOp::SQRT: expression = luisa::format("ct::sqrt({}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", a); break;
            case ElementwiseOp::TANH: expression = luisa::format("ct::tanh({}, ct::round_full_t{{}})", a); break;
            case ElementwiseOp::ABS: expression = luisa::format("ct::abs({})", a); break;
            case ElementwiseOp::SELECT: {
                auto branch = [&](const Value *value) noexcept {
                    auto expression = _elementwise_value(value, result_type, op);
                    if (result_type.is_tile() && !value->type().is_tile()) {
                        expression = luisa::format("ct::full<ct::tile<{}, {}>>({})", _element(result_type), _shape(*result_type.index_space()), expression);
                    }
                    return _map_broadcast(std::move(expression));
                };
                // Unlike arithmetic operators, ct::select deduces one exact
                // type for both branches. Broadcast scalar branches explicitly.
                expression = luisa::format("ct::select({}, {}, {})", a, branch(op.operand(1u)), branch(op.operand(2u)));
                break;
            }
            default: _fail(&op, "elementwise opcode has no validated native numerical mapping yet"); return;
        }
        if (!binary.empty()) {
            luisa::string_view precise;
            if (element == ScalarType::FLOAT32 || element == ScalarType::FLOAT64) {
                switch (op.elementwise_op()) {
                    case ElementwiseOp::ADD: precise = "add"; break;
                    case ElementwiseOp::SUB: precise = "sub"; break;
                    case ElementwiseOp::MUL: precise = "mul"; break;
                    case ElementwiseOp::DIV: precise = "div"; break;
                    default: break;
                }
            }
            expression = precise.empty() ? luisa::format("({} {} {})", a, binary, b) :
                                           luisa::format("ct::{}({}, {}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", precise, a, b);
        }
        _bind(op.result(0u), luisa::format("ct::element_cast<{}>({})", _element(op.result(0u)->type()), expression));
        for (auto i = 0u; i < op.operand_count(); i++) {
            if (_mapped_values.contains(op.operand(i))) { _mapped_values.emplace(op.result(0u)); }
        }
    }
    void _view(const Operation &op) noexcept {
        auto view = op.operand(0u);
        auto found = _buffers.find(view);
        if (found == _buffers.end() || !op.domain()) {
            _fail(&op, "only direct buffer parameters and domain-bearing Tile loads/stores are supported");
            return;
        }
        auto &&space = *op.domain();
        if (!_space(space, true, &op)) { return; }
        auto &&view_space = *view->type().index_space();
        auto read = op.kind() == OperationKind::VIEW_LOAD;
        auto &argument = _artifact.arguments[found->second];
        argument.read |= read;
        argument.written |= !read;
        auto prefix = luisa::format("mem{}", op.id());
        // Flattened iota is shaped identically to the logical load/store Tile.
        // Each coordinate is derived from the original named-axis order.
        _line(luisa::format("auto {}_lane = ct::iota<ct::tile<long long, {}>>();", prefix, _shape(space)));
        if (op.bounds_mode() == BoundsMode::ZERO) {
            _line(luisa::format("auto {}_zero = ct::full<ct::tile<long long, {}>>(0ll);", prefix, _shape(space)));
        }
        luisa::string offset = "0ll", mask = "true";
        uint64_t trailing = *space.static_volume();
        for (auto i = 0u; i < space.rank(); i++) {
            auto n = space.axis(i).extent.constant_value();
            trailing /= n;
            auto coord = luisa::format("{}_c{}", prefix, i);
            _line(luisa::format("auto {} = {} + ({}_lane / {}ll) % {}ll;", coord, _value(op.operand(1u + i)), prefix, trailing, n));
            auto extent = view_space.axis(i).extent.constant_value();
            mask = luisa::format("({} && ({} >= 0ll) && ({} < {}ll))", mask, coord, coord, extent);
            auto safe_coord = op.bounds_mode() == BoundsMode::ZERO ?
                                  luisa::format("ct::select(({} >= 0ll) && ({} < {}ll), {}, {}_zero)", coord, coord, extent, coord, prefix) :
                                  coord;
            offset = luisa::format("(({}) * {}ll + {})", offset, extent, safe_coord);
        }
        // Masked-out lanes use a valid in-buffer pointer, so arbitrary negative
        // origins do not form out-of-object addresses before the masked access.
        if (op.bounds_mode() == BoundsMode::ZERO) {
            _line(luisa::format("auto {}_mask = {};", prefix, mask));
            offset = luisa::format("ct::select({}_mask, {}, {}_zero)", prefix, offset, prefix);
        }
        _line(luisa::format("auto {}_ptr = {} + {};", prefix, _value(view), offset));
        if (read) {
            auto expression = luisa::format("ct::load({}_ptr)", prefix);
            if (op.bounds_mode() == BoundsMode::ZERO) {
                auto fallback = op.operand_count() == space.rank() + 2u ? _value(op.operand(space.rank() + 1u)) :
                                                                          luisa::format("ct::element_cast<{}>(0)", _scalar(argument.element));
                expression = luisa::format("ct::load_masked({}_ptr, {}_mask, {})", prefix, prefix, fallback);
            }
            _bind(op.result(0u), std::move(expression));
        } else {
            auto value = _value(op.operand(space.rank() + 1u));
            _line(op.bounds_mode() == BoundsMode::ZERO ?
                      luisa::format("ct::store_masked({}_ptr, {}, {}_mask);", prefix, value, prefix) :
                      luisa::format("ct::store({}_ptr, {});", prefix, value));
        }
    }
    void _mma(const Operation &op) noexcept {
        auto &&a_type = op.operand(0u)->type();
        auto &&b_type = op.operand(1u)->type();
        auto &&c_type = op.operand(2u)->type();
        auto &&a = *a_type.index_space();
        auto &&b = *b_type.index_space();
        auto &&c = *c_type.index_space();
        if (a.rank() != 2u || b.rank() != 2u || c.rank() != 2u || a_type.scalar_type() != b_type.scalar_type() ||
            c_type.scalar_type() != ScalarType::FLOAT32) {
            _fail(&op, "MMA requires rank-2 matrices, equal input element types, and an FP32 accumulator");
            return;
        }
        auto m = c.axis(0u).dimension, n = c.axis(1u).dimension;
        auto ai = a.axis_index(m), bi = b.axis_index(n);
        if (!ai || !bi || a.contains(n) || b.contains(m)) {
            _fail(&op, "MMA requires independent output M/N axes and exactly one shared contraction K axis");
            return;
        }
        auto ak = 1u - *ai, bk = 1u - *bi;
        if (a.axis(ak) != b.axis(bk)) {
            _fail(&op, "MMA contraction axes differ");
            return;
        }
        if (a.axis(ak).extent.constant_value() > std::numeric_limits<uint32_t>::max()) {
            _fail(&op, "MMA contraction extent exceeds the initial native loop index range");
            return;
        }
        auto lhs = _value(op.operand(0u)), rhs = _value(op.operand(1u));
        if (*ai != 0u) { lhs = luisa::format("ct::transpose({})", lhs); }
        if (*bi != 1u) { rhs = luisa::format("ct::transpose({})", rhs); }
        auto input = a_type.scalar_type();
        if (input != ScalarType::FLOAT32 && input != ScalarType::FLOAT16 && input != ScalarType::BFLOAT16) {
            _fail(&op, "MMA input precision is not implemented; no implicit TF32/narrowing conversion is permitted");
            return;
        }
        if (input != ScalarType::FLOAT32 && op.mma_policy().allow_reassociation) {
            _bind(op.result(0u), luisa::format("ct::mma({}, {}, {})", lhs, rhs, _value(op.operand(2u))));
            return;
        }
        auto prefix = luisa::format("mma{}", op.id());
        _line(luisa::format("auto {}_a = ct::element_cast<float>({});", prefix, lhs));
        _line(luisa::format("auto {}_b = ct::element_cast<float>({});", prefix, rhs));
        _bind(op.result(0u), _value(op.operand(2u)));
        auto acc = _value(op.result(0u));
        _line(luisa::format("for (unsigned {}_k = 0u; {}_k < {}ull; ++{}_k) {{", prefix, prefix, a.axis(ak).extent.constant_value(), prefix));
        _indent++;
        _line(luisa::format("{} = ct::fma(ct::extract({}_a, ct::shape<{}, 1>{{}}, 0u, {}_k), ct::extract({}_b, ct::shape<1, {}>{{}}, {}_k, 0u), {}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}});",
                            acc, prefix, c.axis(0u).extent.constant_value(), prefix, prefix, c.axis(1u).extent.constant_value(), prefix, acc));
        _indent--;
        _line("}");
    }
    void _tile_extract(const Operation &op) noexcept {
        auto &&source = *op.operand(0u)->type().index_space();
        luisa::vector<uint64_t> extents(source.rank(), 1u);
        luisa::vector<int32_t> source_axes(_map_space ? _map_space->rank() : 0u, -1);
        luisa::string indices;
        for (auto i = 0u; i < source.rank(); i++) {
            auto coordinate = op.operand(1u + i);
            auto mapped = false;
            if (_map_body != nullptr) {
                for (auto j = 0u; j < _map_body->argument_count(); j++) {
                    if (coordinate == _map_body->argument(j)) {
                        if (source_axes[j] >= 0 || source.axis(i).extent != _map_space->axis(j).extent) {
                            _fail(&op, "map extraction requires independent matching coordinate extents");
                            return;
                        }
                        source_axes[j] = static_cast<int32_t>(i);
                        extents[i] = source.axis(i).extent.constant_value();
                        indices += ", 0ull";
                        mapped = true;
                        break;
                    }
                }
            }
            if (!mapped) {
                if (_mapped_values.contains(coordinate)) {
                    _fail(&op, "lane-dependent gather indices require a native gather realization");
                    return;
                }
                indices += luisa::format(", static_cast<unsigned long long>({})", _value(coordinate));
            }
        }
        auto expression = luisa::format("ct::extract({}, {}{{}}{})", _value(op.operand(0u)), _shape(extents), indices);
        if (_map_space != nullptr) {
            luisa::vector<uint32_t> order;
            luisa::vector<uint64_t> mapped_extents;
            for (auto index : source_axes) {
                mapped_extents.emplace_back(index < 0 ? 1u : extents[static_cast<size_t>(index)]);
                if (index >= 0) { order.emplace_back(static_cast<uint32_t>(index)); }
            }
            for (auto i = 0u; i < source.rank(); i++) {
                if (std::find(order.begin(), order.end(), i) == order.end()) { order.emplace_back(i); }
            }
            expression = _permute(std::move(expression), order);
            expression = luisa::format("ct::broadcast(ct::reshape({}, {}{{}}), {}{{}})",
                                       expression, _shape(mapped_extents), _shape(*_map_space));
            _mapped_values.emplace(op.result(0u));
        } else {
            expression = luisa::format("static_cast<{}>(ct::reshape({}, ct::shape<>{{}}))", _element(op.result(0u)->type()), expression);
        }
        _bind(op.result(0u), std::move(expression));
    }
    void _tile_map(const Operation &op) noexcept {
        if (_map_space != nullptr) {
            _fail(&op, "nested Tile maps require explicit coordinate composition");
            return;
        }
        auto &&space = *op.domain();
        auto body = op.region(0u)->block(0u);
        _map_space = &space;
        _map_body = body;
        if (space.rank() != 0u) {
            auto lane = luisa::format("map{}_lane", op.id());
            _line(luisa::format("auto {} = ct::iota<ct::tile<long long, {}>>();", lane, _shape(space)));
            auto trailing = *space.static_volume();
            for (auto i = 0u; i < space.rank(); i++) {
                auto extent = space.axis(i).extent.constant_value();
                trailing /= extent;
                _bind(body->argument(i), luisa::format("({} / {}ll) % {}ll", lane, trailing, extent));
                _mapped_values.emplace(body->argument(i));
            }
        }
        _bind(op.result(0u), luisa::format("ct::full<ct::tile<{}, {}>>(0)", _element(op.result(0u)->type()), _shape(space)));
        luisa::string result[]{_value(op.result(0u))};
        _block(*body, result);
        _map_space = nullptr;
        _map_body = nullptr;
    }
    // Prove a closed one-state reducer over an already materialized Tile.
    // Only an unordered tree with its declared identity uses a native tree;
    // every other region retains the ordered scalar contribution sequence.
    [[nodiscard]] bool _reduction_tree(const Operation &op) noexcept {
        if (_map_space == nullptr || op.reduction_policy() != ReductionPolicy::UNORDERED_TREE ||
            op.operand_count() != 1u || op.result(0u)->type() != Type::scalar(ScalarType::FLOAT32)) { return false; }
        auto body = op.region(0u)->block(0u);
        luisa::vector<const Operation *> operations;
        for (auto operation : body->operations()) { operations.emplace_back(operation); }
        if (operations.size() != 3u || operations[0u]->kind() != OperationKind::TILE_EXTRACT ||
            operations[1u]->kind() != OperationKind::ELEMENTWISE || operations[2u]->kind() != OperationKind::YIELD) { return false; }
        auto extract = operations[0u], merge = operations[1u], yield = operations[2u];
        for (auto operation : operations) {
            if (operation->execution_scope_constraint() || operation->resource_class_constraint() || operation->memory_layout()) { return false; }
        }
        auto carry = body->argument(op.domain()->rank());
        if (merge->operand_count() != 2u || yield->operand_count() != 1u || yield->operand(0u) != merge->result(0u) ||
            !((merge->operand(0u) == carry && merge->operand(1u) == extract->result(0u)) ||
              (merge->operand(1u) == carry && merge->operand(0u) == extract->result(0u)))) { return false; }
        auto seed = op.operand(0u)->defining_operation();
        auto attribute = seed && seed->kind() == OperationKind::CONSTANT ? seed->attribute("value") : nullptr;
        auto identity = attribute ? luisa::get_if<double>(&attribute->value()) : nullptr;
        if (identity == nullptr) { return false; }
        auto opcode = merge->elementwise_op();
        if (opcode == ElementwiseOp::ADD) {
            if (*identity != 0.0 || std::signbit(*identity)) { return false; }
        } else if (opcode == ElementwiseOp::MIN || opcode == ElementwiseOp::MAX) {
            if (!std::isinf(*identity) || std::signbit(*identity) != (opcode == ElementwiseOp::MAX)) { return false; }
        } else { return false; }
        auto &&source = *extract->operand(0u)->type().index_space();
        luisa::vector<uint32_t> axes;
        IndexSpace remaining;
        for (auto i = 0u; i < source.rank(); i++) {
            auto index = extract->operand(i + 1u);
            auto reduced = op.domain()->axis_index(source.axis(i).dimension);
            if (reduced) {
                if (index != body->argument(*reduced) || source.axis(i).extent != op.domain()->axis(*reduced).extent) { return false; }
                axes.emplace_back(i);
            } else {
                auto mapped = _map_space->axis_index(source.axis(i).dimension);
                if (!mapped || index != _map_body->argument(*mapped) || source.axis(i).extent != _map_space->axis(*mapped).extent) { return false; }
                static_cast<void>(remaining.add(source.axis(i).dimension, source.axis(i).extent));
            }
        }
        if (axes.size() != op.domain()->rank()) { return false; }
        auto expression = _value(extract->operand(0u));
        for (auto axis : axes) {
            expression = opcode == ElementwiseOp::ADD ?
                luisa::format("ct::sum({}, ct::integral_constant<{}>{{}}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", expression, axis) :
                luisa::format("ct::reduce_{}({}, ct::integral_constant<{}>{{}}, ct::suppress_nan_t{{}}, ct::preserve_subnormals_t{{}})", opcode == ElementwiseOp::MIN ? "min" : "max", expression, axis);
        }
        expression = luisa::format("ct::reshape({}, {}{{}})", expression, _shape(remaining));
        expression = _align(std::move(expression), remaining, *_map_space, op);
        auto name = opcode == ElementwiseOp::ADD ? "add" : opcode == ElementwiseOp::MIN ? "min" : "max";
        auto policy = opcode == ElementwiseOp::ADD ? "ct::round_ties_to_even_t{}" : "ct::suppress_nan_t{}";
        _bind(op.result(0u), luisa::format("ct::{}({}, {}, {}, ct::preserve_subnormals_t{{}})", name, _value(op.operand(0u)), expression, policy));
        _mapped_values.emplace(op.result(0u));
        return true;
    }
    void _structured(const Operation &op) noexcept {
        auto parallel = op.kind() == OperationKind::PARALLEL;
        auto &&domain = *op.domain();
        if (!_space(domain, false, &op) || op.region(0u)->block_count() != 1u) {
            _fail(&op, "structured regions require one body block");
            return;
        }
        auto body = op.region(0u)->block(0u);
        if (op.kind() == OperationKind::REDUCE && _reduction_tree(op)) { return; }
        if (parallel) {
            if (_inside_parallel || ++_parallel_count != 1u || op.parent_block() != _function.body().block(0u)) {
                _fail(&op, "exactly one root PARALLEL is supported; nested/multiple grids need separate launches");
                return;
            }
            static constexpr luisa::string_view axes[] = {"x", "y", "z"};
            for (auto i = 0u; i < domain.rank(); i++) {
                auto n = domain.axis(i).extent.constant_value();
                if (n > (i == 0u ? 0x7fffffffull : 65535ull)) {
                    _fail(&op, "parallel extent exceeds native CUDA grid dimension limit");
                    return;
                }
                _artifact.grid[i] = static_cast<uint32_t>(n);
                _bind(body->argument(i), luisa::format("ct::element_cast<long long>(ct::bid().{})", axes[i]));
            }
            _inside_parallel = true;
            _block(*body, {});
            _inside_parallel = false;
            return;
        }
        if (!_inside_parallel && _map_space == nullptr) {
            _fail(&op, "ordered loops outside the root PARALLEL are unsupported");
            return;
        }
        luisa::vector<luisa::string> carries;
        for (auto i = 0u; i < op.result_count(); i++) {
            _bind(op.result(i), _map_broadcast(_value(op.operand(i))));
            carries.emplace_back(_value(op.result(i)));
            _values.emplace(body->argument(domain.rank() + i), carries.back());
            if (_map_space != nullptr) {
                _mapped_values.emplace(op.result(i));
                _mapped_values.emplace(body->argument(domain.rank() + i));
            }
        }
        for (auto i = 0u; i < domain.rank(); i++) {
            auto name = _name(body->argument(i));
            _values.emplace(body->argument(i), name);
            if (op.kind() == OperationKind::REDUCE && op.reduction_policy() == ReductionPolicy::FOLD_RIGHT) {
                _line(luisa::format("for (long long {} = {}ll; {}-- > 0ll;) {{", name, domain.axis(i).extent.constant_value(), name));
            } else {
                _line(luisa::format("for (long long {} = 0ll; {} < {}ll; ++{}) {{", name, name, domain.axis(i).extent.constant_value(), name));
            }
            _indent++;
        }
        _block(*body, carries);
        for (auto i = 0u; i < domain.rank(); i++) {
            _indent--;
            _line("}");
        }
    }
    void _block(const Block &block, luisa::span<const luisa::string> carries) noexcept {
        for (auto op : block.operations()) {
            if (!_artifact.error.empty()) { return; }
            if (op->execution_scope_constraint() || op->resource_class_constraint() || op->memory_layout()) {
                _fail(op, "explicit execution/resource/layout constraints are not implemented");
                return;
            }
            for (auto i = 0u; i < op->result_count(); i++) {
                if (!_type(op->result(i)->type(), op)) { return; }
            }
            if (!_inside_parallel && op->kind() != OperationKind::PARALLEL && op->kind() != OperationKind::CONSTANT &&
                op->kind() != OperationKind::ELEMENTWISE && op->kind() != OperationKind::TILE_MAP &&
                op->kind() != OperationKind::YIELD && _map_space == nullptr) {
                _fail(op, "effects outside the root PARALLEL cannot be replicated per grid program");
                return;
            }
            switch (op->kind()) {
                case OperationKind::CONSTANT: _constant(*op); break;
                case OperationKind::ELEMENTWISE: _elementwise(*op); break;
                case OperationKind::TILE_MAP: _tile_map(*op); break;
                case OperationKind::TILE_EXTRACT: _tile_extract(*op); break;
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE: _view(*op); break;
                case OperationKind::MMA: _mma(*op); break;
                case OperationKind::PARALLEL:
                case OperationKind::SERIAL:
                case OperationKind::REDUCE:
                case OperationKind::PIPELINE: _structured(*op); break;
                case OperationKind::STAGE: _line("// Stage boundary; ordered execution is a valid non-overlapped schedule."); break;
                case OperationKind::YIELD:
                    if (op->operand_count() != carries.size()) {
                        _fail(op, "yield arity does not match loop carries");
                        return;
                    }
                    for (auto i = 0u; i < carries.size(); i++) {
                        _line(luisa::format("auto yield{}_{} = {};", op->id(), i, _map_broadcast(_value(op->operand(i)))));
                    }
                    for (auto i = 0u; i < carries.size(); i++) {
                        _line(luisa::format("{} = yield{}_{};", carries[i], op->id(), i));
                    }
                    break;
                default: _fail(op, "unsupported operation; no hidden fallback is available"); return;
            }
        }
    }

public:
    explicit Emitter(const Function &function) noexcept : _function{function} {}
    [[nodiscard]] Artifact run() noexcept {
        auto module = _function.parent_module();
        if (module == nullptr) {
            _fail(nullptr, "function has no module");
            return std::move(_artifact);
        }
        auto verified = tile::verify(*module);
        if (!verified) {
            _fail(nullptr, verified.diagnostics().front().message);
            return std::move(_artifact);
        }
        if (_function.form() != IRForm::CANDIDATE || _function.body().block_count() != 1u) {
            _fail(nullptr, "only a one-block candidate function is supported");
            return std::move(_artifact);
        }
        auto body = _function.body().block(0u);
        _artifact.source = "#include <cuda_tile.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\nnamespace ct = cuda::tiles;\nextern \"C\" __tile_global__ void luisa_tile_main(";
        if (body->argument_count() > 31u) { _fail(nullptr, "buffer argument count exceeds existing CUDAShaderTile ABI"); }
        for (auto i = 0u; i < body->argument_count() && _artifact.error.empty(); i++) {
            auto arg = body->argument(i);
            auto &&type = arg->type();
            auto element = type.scalar_type();
            auto supported_element = element == ScalarType::BOOL || element == ScalarType::INT32 || element == ScalarType::UINT32 ||
                                     element == ScalarType::INT64 || element == ScalarType::UINT64 || element == ScalarType::FLOAT32;
            if (!type.is_view() || !supported_element || !_space(*type.index_space(), false, nullptr)) {
                _fail(nullptr, "kernel parameters must be statically sized contiguous bool/integer/FP32 buffer views");
                break;
            }
            auto volume = *type.index_space()->static_volume();
            auto element_size = scalar_type_size(type.scalar_type());
            if (volume > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / element_size) {
                _fail(nullptr, "buffer byte size exceeds int64");
                break;
            }
            if (i != 0u) { _artifact.source += ", "; }
            auto name = luisa::format("buffer{}", i);
            _artifact.source += luisa::format("{} *{}", _scalar(type.scalar_type()), name);
            _values.emplace(arg, name);
            _buffers.emplace(arg, i);
            _artifact.arguments.emplace_back(BufferArgument{type.scalar_type(), volume * element_size});
        }
        _artifact.source += ") {\n";
        if (_artifact.error.empty()) { _block(*body, {}); }
        if (_parallel_count != 1u) { _fail(nullptr, "exactly one root PARALLEL is required"); }
        _artifact.source += "}\n";
        if (!_artifact.error.empty()) { _artifact.source.clear(); }
        return std::move(_artifact);
    }
};
}// namespace

Artifact generate(const tile::Function &function) noexcept { return Emitter{function}.run(); }
}// namespace luisa::compute::cuda::native_tile
