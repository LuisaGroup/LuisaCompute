#include <algorithm>
#include <stdexcept>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/verifier.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/verifier.h>

namespace luisa::compute::tile::bridge::xir {
namespace {

namespace x = compute::xir;
using XType = compute::Type;
using Elements = luisa::vector<x::Value *>;
using Coordinates = luisa::vector<uint64_t>;
using A = x::ArithmeticOp;

// Portable checked signed arithmetic, with the same overflow policy as the
// layout proof engine. Do not use floating-point bounds or compiler-only builtins.
[[nodiscard]] luisa::optional<int64_t> checked_add(int64_t a, int64_t b) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) { return {}; }
    return a + b;
}
[[nodiscard]] luisa::optional<int64_t> checked_subtract(int64_t a, int64_t b) {
    if ((b > 0 && a < INT64_MIN + b) || (b < 0 && a > INT64_MAX + b)) { return {}; }
    return a - b;
}
[[nodiscard]] luisa::optional<int64_t> checked_multiply(int64_t a, int64_t b) {
    if (a == 0 || b == 0) { return 0; }
    if ((a == -1 && b == INT64_MIN) || (b == -1 && a == INT64_MIN)) { return {}; }
    if (a > 0) {
        if ((b > 0 && a > INT64_MAX / b) || (b < 0 && b < INT64_MIN / a)) { return {}; }
    } else if ((b > 0 && a < INT64_MIN / b) || (b < 0 && a < INT64_MAX / b)) {
        return {};
    }
    return a * b;
}

class Lowerer final {
private:
    const Function &_input;
    LowerOptions _options;
    NativeFunction _output;
    x::XIRBuilder _builder;
    x::BasicBlock *_block{nullptr};
    luisa::unordered_map<const Value *, Elements> _values;
    luisa::unordered_map<const Value *, uint32_t> _arguments;
    struct IndexRange {
        int64_t lo, hi;
    };
    luisa::unordered_map<const Value *, IndexRange> _coordinate_ranges;
    uint64_t _expanded_values{0u};
    bool _inside_parallel{false};
    bool _saw_parallel{false};

    [[noreturn]] static void _fail(luisa::string_view message) {
        throw std::runtime_error{std::string{message}};
    }
    void _charge(uint64_t count = 1u) {
        if (count > _options.max_expanded_values || _expanded_values > _options.max_expanded_values - count) {
            _fail("XIR realization exceeds its static SSA expansion budget; choose smaller Tiles");
        }
        _expanded_values += count;
    }
    [[nodiscard]] static const XType *_type(const Type &type) {
        if (type.kind() == TypeKind::INDEX) { return XType::of<int64_t>(); }
        switch (type.scalar_type()) {
            case ScalarType::BOOL: return XType::of<bool>();
            case ScalarType::INT32: return XType::of<int32_t>();
            case ScalarType::UINT32: return XType::of<uint32_t>();
            case ScalarType::INT64: return XType::of<int64_t>();
            case ScalarType::UINT64: return XType::of<uint64_t>();
            case ScalarType::FLOAT32: return XType::of<float>();
            case ScalarType::FLOAT64: return XType::of<double>();
            default: _fail("unsupported scalar type in Tile to XIR bridge");
        }
    }
    [[nodiscard]] static uint64_t _extent(const IndexSpace &space, size_t axis) {
        auto &extent = space.axis(axis).extent;
        if (!extent.is_constant() || extent.constant_value() > UINT32_MAX) { _fail("XIR realization requires static uint32-addressable extents"); }
        return extent.constant_value();
    }
    [[nodiscard]] static uint64_t _volume(const IndexSpace &space) {
        uint64_t count = 1u;
        for (size_t axis = 0u; axis < space.rank(); axis++) {
            auto extent = _extent(space, axis);
            if (extent != 0u && count > UINT32_MAX / extent) { _fail("XIR realization domain exceeds uint32 range"); }
            count *= extent;
        }
        return count;
    }
    [[nodiscard]] static Coordinates _coordinates(const IndexSpace &space, uint64_t flat) {
        Coordinates result(space.rank());
        for (auto axis = space.rank(); axis != 0u; axis--) {
            auto extent = _extent(space, axis - 1u);
            if (extent == 0u) { _fail("cannot index an empty Tile"); }
            result[axis - 1u] = flat % extent;
            flat /= extent;
        }
        return result;
    }
    template<typename T>
    [[nodiscard]] x::Value *_constant(T value) { return _output.module->create_constant(XType::of<T>(), &value); }
    [[nodiscard]] x::Value *_index(uint64_t value) { return _constant(static_cast<int64_t>(value)); }
    [[nodiscard]] x::Value *_alu(const XType *type, A op, std::initializer_list<x::Value *> operands) {
        _charge();
        return _builder.call(type, op, operands);
    }
    [[nodiscard]] x::Value *_binary(A op, x::Value *a, x::Value *b) { return _alu(a->type(), op, {a, b}); }
    [[nodiscard]] x::Value *_compare(A op, x::Value *a, x::Value *b) { return _alu(XType::of<bool>(), op, {a, b}); }
    void _at(x::BasicBlock *block) {
        _block = block;
        _builder.set_insertion_point(block);
    }
    [[nodiscard]] const Elements &_get(const Value *value) const {
        auto found = _values.find(value);
        if (found == _values.end()) { _fail("TileIR value has no dominating XIR definition"); }
        return found->second;
    }
    [[nodiscard]] x::Value *_scalar(const Value *value) const {
        auto &elements = _get(value);
        if (value->type().is_tile() || elements.size() != 1u) { _fail("expected scalar TileIR operand"); }
        return elements.front();
    }
    // An integer proof, independent of the planner's floating-point slope
    // heuristic. Unknown values, narrow arithmetic and any possible signed
    // overflow retain the original bounds checks.
    [[nodiscard]] luisa::optional<IndexRange> _range(const Value *value, uint32_t depth = 0u) const {
        if (depth > 32u) { return {}; }
        if (auto found = _coordinate_ranges.find(value); found != _coordinate_ranges.end()) { return found->second; }
        if (value->type().kind() != TypeKind::INDEX && value->type().scalar_type() != ScalarType::INT64) { return {}; }
        auto op = value->defining_operation();
        if (!op) { return {}; }
        if (op->kind() == OperationKind::CONSTANT) {
            if (auto attribute = op->attribute("value")) {
                if (auto v = luisa::get_if<int64_t>(&attribute->value())) { return IndexRange{*v, *v}; }
                if (auto v = luisa::get_if<uint64_t>(&attribute->value()); v && *v <= INT64_MAX) { return IndexRange{static_cast<int64_t>(*v), static_cast<int64_t>(*v)}; }
            }
            return {};
        }
        if (op->kind() != OperationKind::ELEMENTWISE || op->operand_count() != 2u) { return {}; }
        auto a = _range(op->operand(0u), depth + 1u), b = _range(op->operand(1u), depth + 1u);
        if (!a || !b) { return {}; }
        IndexRange result{};
        switch (op->elementwise_op()) {
            case ElementwiseOp::ADD: {
                auto lo = checked_add(a->lo, b->lo), hi = checked_add(a->hi, b->hi);
                if (lo && hi) { return IndexRange{*lo, *hi}; }
                return {};
            }
            case ElementwiseOp::SUB: {
                auto lo = checked_subtract(a->lo, b->hi), hi = checked_subtract(a->hi, b->lo);
                if (lo && hi) { return IndexRange{*lo, *hi}; }
                return {};
            }
            case ElementwiseOp::MUL:
                result = {INT64_MAX, INT64_MIN};
                for (auto x : {a->lo, a->hi}) {
                    for (auto y : {b->lo, b->hi}) {
                        auto product = checked_multiply(x, y);
                        if (!product) { return {}; }
                        result.lo = std::min(result.lo, *product);
                        result.hi = std::max(result.hi, *product);
                    }
                }
                return result;
            case ElementwiseOp::DIV:
                if (a->lo >= 0 && b->lo > 0 && b->lo == b->hi) { return IndexRange{a->lo / b->lo, a->hi / b->lo}; }
                break;
            case ElementwiseOp::MOD:
                if (a->lo >= 0 && b->lo > 0 && b->lo == b->hi) { return IndexRange{0, b->lo - 1}; }
                break;
            default: break;
        }
        return {};
    }
    [[nodiscard]] x::Value *_project(const Value *value, const IndexSpace &domain, const Coordinates &coordinates) const {
        if (!value->type().is_tile()) { return _scalar(value); }
        auto &space = *value->type().index_space();
        uint64_t flat = 0u;
        for (size_t i = 0u; i < space.rank(); i++) {
            auto axis = domain.axis_index(space.axis(i).dimension);
            if (!axis) { _fail("Tile operand dimension is absent from its XIR expression domain"); }
            auto extent = _extent(space, i);
            auto coordinate = extent == 1u ? 0u : coordinates[*axis];
            if (coordinate >= extent) { _fail("Tile projection is out of bounds"); }
            flat = flat * extent + coordinate;
        }
        return _get(value).at(flat);
    }
    [[nodiscard]] x::Value *_elementwise(ElementwiseOp op, const XType *type, const Elements &v) {
        switch (op) {
            case ElementwiseOp::ADD: return _alu(type, A::BINARY_ADD, {v[0], v[1]});
            case ElementwiseOp::SUB: return _alu(type, A::BINARY_SUB, {v[0], v[1]});
            case ElementwiseOp::MUL: return _alu(type, A::BINARY_MUL, {v[0], v[1]});
            case ElementwiseOp::DIV: return _alu(type, A::BINARY_DIV, {v[0], v[1]});
            case ElementwiseOp::MOD: return _alu(type, A::BINARY_MOD, {v[0], v[1]});
            case ElementwiseOp::NEG: return _alu(type, A::UNARY_MINUS, {v[0]});
            case ElementwiseOp::MIN: return _alu(type, A::MIN, {v[0], v[1]});
            case ElementwiseOp::MAX: return _alu(type, A::MAX, {v[0], v[1]});
            case ElementwiseOp::CAST: return _builder.static_cast_if_necessary(type, v[0]);
            // Tile ite(condition, true, false); XIR/Luisa select(false, true, condition).
            case ElementwiseOp::SELECT: return _alu(type, A::SELECT, {v[2], v[1], v[0]});
            case ElementwiseOp::EQ: return _compare(A::BINARY_EQUAL, v[0], v[1]);
            case ElementwiseOp::NE: return _compare(A::BINARY_NOT_EQUAL, v[0], v[1]);
            case ElementwiseOp::LT: return _compare(A::BINARY_LESS, v[0], v[1]);
            case ElementwiseOp::LE: return _compare(A::BINARY_LESS_EQUAL, v[0], v[1]);
            case ElementwiseOp::GT: return _compare(A::BINARY_GREATER, v[0], v[1]);
            case ElementwiseOp::GE: return _compare(A::BINARY_GREATER_EQUAL, v[0], v[1]);
            case ElementwiseOp::LOGICAL_AND: return _alu(type, A::BINARY_BIT_AND, {v[0], v[1]});
            case ElementwiseOp::LOGICAL_OR: return _alu(type, A::BINARY_BIT_OR, {v[0], v[1]});
            case ElementwiseOp::LOGICAL_NOT: return _compare(A::BINARY_EQUAL, v[0], _constant(false));
            case ElementwiseOp::EXP: return _alu(type, A::EXP, {v[0]});
            case ElementwiseOp::LOG: return _alu(type, A::LOG, {v[0]});
            case ElementwiseOp::SQRT: return _alu(type, A::SQRT, {v[0]});
            case ElementwiseOp::TANH: return _alu(type, A::TANH, {v[0]});
            case ElementwiseOp::ABS: return _alu(type, A::ABS, {v[0]});
            default: _fail("unsupported Tile elementwise opcode");
        }
    }
    [[nodiscard]] x::Value *_literal(const Operation &op) {
        auto attribute = op.attribute("value");
        if (attribute == nullptr) { _fail("Tile constant is missing its value"); }
        x::Value *value = nullptr;
        auto &payload = attribute->value();
        if (auto item = luisa::get_if<bool>(&payload)) { value = _constant(*item); }
        if (auto item = luisa::get_if<int64_t>(&payload)) { value = _constant(*item); }
        if (auto item = luisa::get_if<uint64_t>(&payload)) { value = _constant(*item); }
        if (auto item = luisa::get_if<double>(&payload)) { value = _constant(*item); }
        if (value == nullptr) { _fail("invalid Tile constant payload"); }
        return _builder.static_cast_if_necessary(_type(op.result(0)->type()), value);
    }
    [[nodiscard]] x::Value *_guarded_load(x::Value *condition, x::Value *buffer, x::Value *address, x::Value *fallback) {
        auto header = _block;
        auto read = _output.function->create_basic_block();
        auto merge = _output.function->create_basic_block();
        _builder.cond_br(condition, read, merge);
        _at(read);
        auto value = _builder.call(fallback->type(), x::ResourceReadOp::BUFFER_READ, {buffer, address});
        _builder.br(merge);
        _at(merge);
        return _builder.phi(fallback->type(), {{fallback, header}, {value, read}});
    }
    void _guarded_store(x::Value *condition, x::Value *buffer, x::Value *address, x::Value *value) {
        auto write = _output.function->create_basic_block();
        auto merge = _output.function->create_basic_block();
        _builder.cond_br(condition, write, merge);
        _at(write);
        _builder.call(x::ResourceWriteOp::BUFFER_WRITE, {buffer, address, value});
        _builder.br(merge);
        _at(merge);
    }
    void _view_access(const Operation &op) {
        auto view = op.operand(0u);
        auto found = _arguments.find(view);
        if (found == _arguments.end()) { _fail("XIR view access requires a direct buffer argument"); }
        auto slot = found->second;
        auto load = op.kind() == OperationKind::VIEW_LOAD;
        _output.argument_usages[slot] = static_cast<Usage>(static_cast<uint32_t>(_output.argument_usages[slot]) | static_cast<uint32_t>(load ? Usage::READ : Usage::WRITE));
        auto buffer = _get(view).front();
        auto &space = *view->type().index_space();
        auto count = op.domain() ? _volume(*op.domain()) : 1u;
        _charge(count);
        Elements result;
        for (uint64_t item = 0u; item < count; item++) {
            auto indices = op.domain() ? _coordinates(*op.domain(), item) : Coordinates(space.rank(), 0u);
            x::Value *address = _index(0u);
            x::Value *valid = _constant(true);
            auto needs_guard = false;
            for (size_t i = 0u; i < space.rank(); i++) {
                auto coordinate = _scalar(op.operand(i + 1u));
                if (op.domain()) { coordinate = _binary(A::BINARY_ADD, coordinate, _index(indices[i])); }
                address = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, address, _index(_extent(space, i))), coordinate);
                if (op.domain() && op.bounds_mode() == BoundsMode::ZERO) {
                    if (auto range = _range(op.operand(i + 1u))) {
                        auto offset = static_cast<int64_t>(indices[i]);
                        auto lo = checked_add(range->lo, offset), hi = checked_add(range->hi, offset);
                        if (lo && hi && *lo >= 0 && *hi < static_cast<int64_t>(_extent(space, i))) { continue; }
                    }
                    needs_guard = true;
                    valid = _binary(A::BINARY_BIT_AND, valid, _compare(A::BINARY_GREATER_EQUAL, coordinate, _index(0u)));
                    valid = _binary(A::BINARY_BIT_AND, valid, _compare(A::BINARY_LESS, coordinate, _index(_extent(space, i))));
                }
            }
            address = _builder.static_cast_if_necessary(XType::of<uint64_t>(), address);
            if (load) {
                auto type = _type(op.result(0u)->type());
                auto fallback = _output.module->create_constant_zero(type);
                x::Value *fill = fallback;
                auto guarded = needs_guard;
                if (op.domain() && op.operand_count() == space.rank() + 2u) { fill = _scalar(op.operand(space.rank() + 1u)); }
                if (!op.domain() && op.operand_count() == space.rank() + 3u) {
                    guarded = true;
                    valid = _scalar(op.operand(space.rank() + 1u));
                    fill = _scalar(op.operand(space.rank() + 2u));
                }
                result.emplace_back(guarded ? _guarded_load(valid, buffer, address, fill) : _builder.call(type, x::ResourceReadOp::BUFFER_READ, {buffer, address}));
            } else {
                auto value = _get(op.operand(space.rank() + 1u)).at(item);
                if (needs_guard) {
                    _guarded_store(valid, buffer, address, value);
                } else {
                    _builder.call(x::ResourceWriteOp::BUFFER_WRITE, {buffer, address, value});
                }
            }
        }
        if (load) { _values.insert_or_assign(op.result(0u), std::move(result)); }
    }
    void _bind_coordinates(const Block &body, const IndexSpace &domain, x::Value *flat, luisa::span<const uint32_t> order = {}) {
        auto trailing = _volume(domain);
        for (size_t position = 0u; position < domain.rank(); position++) {
            auto i = order.empty() ? position : order[position];
            auto extent = _extent(domain, i);
            auto coordinate = _index(0u);
            if (trailing != 0u && extent != 0u) {
                trailing /= extent;
                coordinate = _binary(A::BINARY_MOD, _binary(A::BINARY_DIV, flat, _index(trailing)), _index(extent));
            }
            _values.insert_or_assign(body.argument(i), Elements{coordinate});
            // The body executes only for valid coordinates; zero-trip loop
            // bodies are unreachable. Never infer ranges for carried values.
            if (extent != 0u) { _coordinate_ranges.insert_or_assign(body.argument(i), IndexRange{0, static_cast<int64_t>(extent - 1u)}); }
        }
    }
    [[nodiscard]] luisa::vector<Elements> _region(const Block &body) {
        for (auto op : body.operations()) {
            if (op->kind() == OperationKind::YIELD) {
                luisa::vector<Elements> yielded;
                for (size_t i = 0u; i < op->operand_count(); i++) { yielded.emplace_back(_get(op->operand(i))); }
                return yielded;
            }
            _operation(*op);
        }
        return {};
    }
    void _loop(const Operation &op) {
        auto &domain = *op.domain();
        auto body = op.region(0u)->block(0u);
        if (auto scope = op.execution_scope_constraint(); scope && *scope != "worker" && *scope != "auto") {
            _fail("XIR worker realization cannot honor this explicit execution binding");
        }
        if (op.kind() == OperationKind::PARALLEL && !_inside_parallel) {
            if (_saw_parallel || op.result_count() != 0u) { _fail("XIR bridge requires one independent root parallel with no escaping results"); }
            _output.dispatch_size = static_cast<uint32_t>(_volume(domain));
            if (_output.dispatch_size == 0u) { _fail("empty root parallel has no executable launch"); }
            _saw_parallel = true;
            _inside_parallel = true;
            auto dispatch = _alu(XType::of<uint32_t>(), A::EXTRACT, {_output.module->create_dispatch_id(), _constant(uint32_t{0})});
            auto &order = _options.root_axis_order;
            if (!order.empty()) {
                if (order.size() != domain.rank()) { _fail("XIR execution order must be a complete permutation"); }
                luisa::vector<bool> seen(domain.rank(), false);
                for (auto axis : order) {
                    if (axis >= domain.rank() || seen[axis]) { _fail("XIR execution order must be a complete permutation"); }
                    seen[axis] = true;
                }
            }
            _bind_coordinates(*body, domain, _builder.static_cast_(XType::of<int64_t>(), dispatch), order);
            if (!_region(*body).empty()) { _fail("root parallel yielded state"); }
            _inside_parallel = false;
            return;
        }
        if (!_inside_parallel) { _fail("serial work outside the root parallel requires a multi-launch program"); }
        auto preheader = _block;
        auto header = _output.function->create_basic_block();
        auto loop_body = _output.function->create_basic_block();
        auto exit = _output.function->create_basic_block();
        _builder.br(header);
        _at(header);
        auto induction = _builder.phi(XType::of<int64_t>(), {{_index(0u), preheader}});
        luisa::vector<luisa::vector<x::PhiInst *>> carries;
        for (size_t i = 0u; i < op.result_count(); i++) {
            luisa::vector<x::PhiInst *> phis;
            Elements elements;
            for (auto value : _get(op.operand(i))) {
                auto phi = _builder.phi(value->type(), {{value, preheader}});
                phis.emplace_back(phi);
                elements.emplace_back(phi);
            }
            _values.insert_or_assign(body->argument(domain.rank() + i), elements);
            _values.insert_or_assign(op.result(i), std::move(elements));
            carries.emplace_back(std::move(phis));
        }
        _builder.cond_br(_compare(A::BINARY_LESS, induction, _index(_volume(domain))), loop_body, exit);
        _at(loop_body);
        _bind_coordinates(*body, domain, induction);
        auto yielded = _region(*body);
        if (yielded.size() != carries.size()) { _fail("XIR loop yield does not match its carried state"); }
        // Every incoming uses the old iteration's SSA definitions. No ordered
        // stores, including for swaps and interdependent carried Tiles.
        for (size_t i = 0u; i < carries.size(); i++) {
            if (yielded[i].size() != carries[i].size()) { _fail("XIR loop carry shape mismatch"); }
            for (size_t j = 0u; j < carries[i].size(); j++) { carries[i][j]->add_incoming(yielded[i][j], _block); }
        }
        induction->add_incoming(_binary(A::BINARY_ADD, induction, _index(1u)), _block);
        _builder.br(header);
        _at(exit);
    }
    void _mma(const Operation &op) {
        auto result = op.result(0u);
        auto &space = *result->type().index_space();
        auto contraction = IndexSpace{};
        auto domain = space;
        for (auto &axis : op.operand(0u)->type().index_space()->axes()) {
            if (!space.contains(axis.dimension)) {
                static_cast<void>(contraction.add(axis.dimension, axis.extent));
                static_cast<void>(domain.add(axis.dimension, axis.extent));
            }
        }
        Elements elements;
        auto type = _type(result->type());
        for (uint64_t i = 0u; i < _volume(space); i++) {
            auto coordinates = _coordinates(space, i);
            auto sum = _get(op.operand(2u)).at(i);
            for (uint64_t k = 0u; k < _volume(contraction); k++) {
                auto full = coordinates;
                for (auto coordinate : _coordinates(contraction, k)) { full.emplace_back(coordinate); }
                auto a = _builder.static_cast_if_necessary(type, _project(op.operand(0u), domain, full));
                auto b = _builder.static_cast_if_necessary(type, _project(op.operand(1u), domain, full));
                sum = _binary(A::BINARY_ADD, sum, _binary(A::BINARY_MUL, a, b));
            }
            elements.emplace_back(sum);
        }
        _values.insert_or_assign(result, std::move(elements));
    }
    void _operation(const Operation &op) {
        switch (op.kind()) {
            case OperationKind::CONSTANT: {
                auto result = op.result(0u);
                auto count = result->type().is_tile() ? _volume(*result->type().index_space()) : 1u;
                _charge(count);
                _values.insert_or_assign(result, Elements(count, _literal(op)));
                break;
            }
            case OperationKind::ELEMENTWISE: {
                auto result = op.result(0u);
                auto domain = result->type().is_tile() ? *result->type().index_space() : IndexSpace{};
                Elements elements;
                for (uint64_t i = 0u; i < _volume(domain); i++) {
                    auto coordinates = _coordinates(domain, i);
                    Elements inputs;
                    for (size_t j = 0u; j < op.operand_count(); j++) { inputs.emplace_back(_project(op.operand(j), domain, coordinates)); }
                    elements.emplace_back(_elementwise(op.elementwise_op(), _type(result->type()), inputs));
                }
                _values.insert_or_assign(result, std::move(elements));
                break;
            }
            case OperationKind::VIEW_LOAD:
            case OperationKind::VIEW_STORE: _view_access(op); break;
            case OperationKind::PARALLEL:
            case OperationKind::SERIAL:
            case OperationKind::REDUCE:
            case OperationKind::PIPELINE: _loop(op); break;
            case OperationKind::STAGE: break;// Ordered CPU realization retains source phase order.
            case OperationKind::MMA: _mma(op); break;
            case OperationKind::TILE_MAP: {
                Elements values;
                auto body = op.region(0u)->block(0u);
                for (uint64_t i = 0u; i < _volume(*op.domain()); i++) {
                    auto coordinates = _coordinates(*op.domain(), i);
                    for (size_t j = 0u; j < coordinates.size(); j++) { _values.insert_or_assign(body->argument(j), Elements{_index(coordinates[j])}); }
                    auto yielded = _region(*body);
                    if (yielded.size() != 1u || yielded[0].size() != 1u) { _fail("Tile map must yield exactly one scalar"); }
                    values.emplace_back(yielded[0][0]);
                }
                _values.insert_or_assign(op.result(0u), std::move(values));
                break;
            }
            case OperationKind::TILE_EXTRACT: {
                auto tile = op.operand(0u);
                auto &space = *tile->type().index_space();
                x::Value *flat = _index(0u);
                for (size_t i = 0u; i < space.rank(); i++) { flat = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, flat, _index(_extent(space, i))), _scalar(op.operand(i + 1u))); }
                auto type = _type(op.result(0u)->type());
                x::Value *value = _output.module->create_constant_zero(type);
                auto &elements = _get(tile);
                for (size_t i = 0u; i < elements.size(); i++) { value = _alu(type, A::SELECT, {value, elements[i], _compare(A::BINARY_EQUAL, flat, _index(i))}); }
                _values.insert_or_assign(op.result(0u), Elements{value});
                break;
            }
            default: _fail("unsupported TileIR operation in XIR worker realization; no fallback or effect erasure");
        }
    }

public:
    Lowerer(const Function &input, LowerOptions options) : _input{input}, _options{options} {}
    [[nodiscard]] NativeFunction run() {
        if (_input.parent_module() == nullptr || !verify(*_input.parent_module())) { _fail("TileIR verification failed before XIR lowering"); }
        if (_input.body().block_count() != 1u || !x::KernelFunction::is_valid_block_size(luisa::make_uint3(_options.block_size, 1u, 1u)) || _options.max_expanded_values == 0u) { _fail("invalid XIR realization options or entry region"); }
        _output.module = luisa::make_unique<x::Module>();
        _output.function = _output.module->create_kernel();
        _output.function->set_name(_input.name());
        _output.function->set_block_size(luisa::make_uint3(_options.block_size, 1u, 1u));
        _at(_output.function->create_body_block());
        auto root = _input.body().block(0u);
        for (auto &argument : root->arguments()) {
            auto value = argument.get();
            if (!value->type().is_view()) { _fail("XIR Tile kernels currently require buffer View arguments"); }
            auto type = _type(value->type());
            auto count = _volume(*value->type().index_space());
            if (count == 0u || count > SIZE_MAX / type->size()) { _fail("invalid XIR buffer footprint"); }
            auto buffer = _output.function->create_resource_argument(XType::buffer(type));
            buffer->set_name(value->name());
            _arguments.emplace(value, static_cast<uint32_t>(_output.argument_usages.size()));
            _values.emplace(value, Elements{buffer});
            _output.argument_usages.emplace_back(Usage::NONE);
            _output.argument_sizes_bytes.emplace_back(count * type->size());
        }
        for (auto op : root->operations()) {
            if (op->kind() != OperationKind::CONSTANT && op->kind() != OperationKind::ELEMENTWISE && op->kind() != OperationKind::PARALLEL) { _fail("root effects require one explicit parallel execution domain"); }
            _operation(*op);
        }
        if (!_saw_parallel) { _fail("XIR realization requires a root parallel domain"); }
        _builder.return_void();
        auto verified = x::xir_verify_module(_output.module.get(), {.require_reachable_blocks = true});
        if (!verified.succeeded()) { _fail(verified.errors.front().message); }
        return std::move(_output);
    }
};

}// namespace

NativeFunction lower(const Function &function, const LowerOptions &options) noexcept {
    try {
        return Lowerer{function, options}.run();
    } catch (const std::exception &error) {
        NativeFunction result;
        result.error = error.what();
        return result;
    } catch (...) {
        NativeFunction result;
        result.error = "unknown error lowering TileIR to XIR";
        return result;
    }
}

}// namespace luisa::compute::tile::bridge::xir
