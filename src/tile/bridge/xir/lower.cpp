#include <algorithm>
#include <stdexcept>

#include <luisa/core/stl/format.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/verifier.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/verifier.h>
#include "representation.h"

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
    struct Representation {
        const Type *type{nullptr};
        Elements elements;
        x::Value *storage{nullptr};
        const Operation *expression{nullptr};
        luisa::vector<const Representation *> inputs;
        bool splat{false};
        bool pending_load{false};
    };
    luisa::vector<luisa::unique_ptr<Representation>> _definitions;
    luisa::unordered_map<const Value *, const Representation *> _values;
    luisa::unordered_map<const Value *, uint32_t> _arguments;
    struct IndexRange {
        int64_t lo, hi;
    };
    struct ViewAccess {
        const Operation *operation;
        x::Value *buffer;
        Elements origins;
        luisa::vector<luisa::optional<IndexRange>> ranges;
        x::Value *fill{nullptr};
        x::Value *mask{nullptr};
    };
    struct FusedLoad {
        ViewAccess access;
        Representation *result;
    };
    luisa::unordered_map<const Operation *, luisa::vector<FusedLoad>> _pending_loads;
    luisa::unordered_map<const Representation *, x::Value *> _fused_elements;
    luisa::unordered_map<const Value *, IndexRange> _coordinate_ranges;
    uint64_t _expanded_values{0u};
    uint64_t _local_bytes{0u};
    bool _inside_parallel{false};
    bool _saw_parallel{false};
    x::Value *_lane{nullptr};
    x::Value *_local_slot{nullptr};

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
    [[nodiscard]] bool _bounded(uint64_t count) const noexcept {
        return _options.max_unrolled_tile_elements != 0u && count > _options.max_unrolled_tile_elements;
    }
    [[nodiscard]] bool _distributed(uint64_t count) const noexcept { return _options.local_lanes > 1u && count >= _options.local_lanes; }
    [[nodiscard]] uint64_t _storage_count(const Type &type) const {
        auto count = _volume(*type.index_space());
        return _distributed(count) ? ceil_div(count, static_cast<uint64_t>(_options.local_lanes)) : count;
    }
    [[nodiscard]] x::Value *_storage_index(const Type &type, x::Value *flat) {
        if (!_distributed(_volume(*type.index_space()))) { return flat; }
        // Admission establishes that every nonunit projection preserves the
        // current common-axis owner. Keep the split pair (slot, lane) instead
        // of reconstructing slot = (slot * W + lane) / W in varying i64 IR.
        // This is an exact realization fact, not an LLVM reassociation hint.
        if (!_local_slot) { _fail("distributed Tile access has no active owner coordinate"); }
        return _local_slot;
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
    [[nodiscard]] const Representation *_get(const Value *value) const {
        auto found = _values.find(value);
        if (found == _values.end()) { _fail("TileIR value has no dominating XIR definition"); }
        return found->second;
    }
    [[nodiscard]] x::Value *_scalar(const Value *value) const {
        auto &elements = _get(value)->elements;
        if (value->type().is_tile() || elements.size() != 1u) { _fail("expected scalar TileIR operand"); }
        return elements.front();
    }
    [[nodiscard]] Representation *_representation(const Value *value) {
        auto data = luisa::make_unique<Representation>();
        data->type = &value->type();
        auto result = data.get();
        _definitions.emplace_back(std::move(data));
        _values.insert_or_assign(value, result);
        return result;
    }
    [[nodiscard]] x::Value *_allocate(const Type &type) {
        auto element = _type(type);
        auto count = _storage_count(type);
        auto bytes = count * element->size();
        if (bytes > _options.max_local_bytes || _local_bytes > _options.max_local_bytes - bytes) {
            _fail("XIR realization exceeds its local snapshot storage budget");
        }
        _local_bytes += bytes;
        _charge();
        auto storage = _builder.alloca_local(XType::array(element, count));
        storage->set_name("tile_snapshot");
        return storage;
    }
    template<typename F>
    void _serial_for(uint64_t count, F &&emit, bool force_loop = false) {
        if (!_bounded(count) && !force_loop) {
            for (uint64_t i = 0u; i < count; i++) { emit(_index(i)); }
            return;
        }
        auto preheader = _block;
        auto header = _output.function->create_basic_block();
        auto body = _output.function->create_basic_block();
        auto exit = _output.function->create_basic_block();
        _builder.br(header);
        _at(header);
        auto index = _builder.phi(XType::of<int64_t>(), {{_index(0u), preheader}});
        _builder.cond_br(_compare(A::BINARY_LESS, index, _index(count)), body, exit);
        _at(body);
        emit(index);
        index->add_incoming(_binary(A::BINARY_ADD, index, _index(1u)), _block);
        _builder.br(header);
        _at(exit);
    }
    template<typename F>
    void _for_each(uint64_t count, F &&emit) {
        if (!_distributed(count)) {
            _serial_for(count, emit);
            return;
        }
        auto lanes = _options.local_lanes;
        auto full = count / lanes;
        auto element = [&](x::Value *chunk) {
            auto previous = _local_slot;
            _local_slot = chunk;
            emit(_binary(A::BINARY_ADD, _binary(A::BINARY_MUL, chunk, _index(lanes)), _lane));
            _local_slot = previous;
        };
        _serial_for(full, element, _bounded(count));
        if (count % lanes != 0u) {
            auto tail = _output.function->create_basic_block();
            auto exit = _output.function->create_basic_block();
            _builder.cond_br(_compare(A::BINARY_LESS, _lane, _index(count % lanes)), tail, exit);
            _at(tail);
            element(_index(full));
            _builder.br(exit);
            _at(exit);
        }
    }
    template<typename F>
    [[nodiscard]] x::Value *_fold(uint64_t count, x::Value *initial, F &&emit) {
        if (!_bounded(count)) {
            for (uint64_t i = 0u; i < count; i++) { initial = emit(_index(i), initial); }
            return initial;
        }
        auto preheader = _block;
        auto header = _output.function->create_basic_block();
        auto body = _output.function->create_basic_block();
        auto exit = _output.function->create_basic_block();
        _builder.br(header);
        _at(header);
        auto index = _builder.phi(XType::of<int64_t>(), {{_index(0u), preheader}});
        auto sum = _builder.phi(initial->type(), {{initial, preheader}});
        _builder.cond_br(_compare(A::BINARY_LESS, index, _index(count)), body, exit);
        _at(body);
        auto next = emit(index, sum);
        sum->add_incoming(next, _block);
        index->add_incoming(_binary(A::BINARY_ADD, index, _index(1u)), _block);
        _builder.br(header);
        _at(exit);
        return sum;
    }
    [[nodiscard]] Elements _coordinates(const IndexSpace &space, x::Value *flat) {
        Elements result(space.rank());
        uint64_t constant = 0u;
        if (x::try_decode_constant_nonnegative_integer(flat, constant)) {
            auto coordinates = _coordinates(space, constant);
            for (size_t i = 0u; i < coordinates.size(); i++) { result[i] = _index(coordinates[i]); }
        } else {
            for (auto axis = space.rank(); axis != 0u; axis--) {
                auto extent = _extent(space, axis - 1u);
                if (!extent) { _fail("cannot index an empty Tile"); }
                auto major = true;
                for (size_t j = 0u; j + 1u < axis; j++) { major &= _extent(space, j) == 1u; }
                result[axis - 1u] = extent == 1u ? _index(0u) :
                                    major        ? flat :
                                                   _binary(A::BINARY_MOD, flat, _index(extent));
                flat = _binary(A::BINARY_DIV, flat, _index(extent));
            }
        }
        return result;
    }
    void _store_local(const Type &type, x::Value *storage, x::Value *flat, x::Value *element) {
        _charge(2u);
        _builder.store(_builder.gep(_type(type), storage, {_storage_index(type, flat)}), element);
    }
    template<typename F>
    void _emit_tile(const Value *value, F &&emit) {
        auto count = _volume(*value->type().index_space());
        if (_bounded(count) || _distributed(count)) {
            auto storage = _allocate(value->type());
            _for_each(count, [&](x::Value *flat) { _store_local(value->type(), storage, flat, emit(flat)); });
            _representation(value)->storage = storage;
        } else {
            Elements elements;
            _for_each(count, [&](x::Value *flat) { elements.emplace_back(emit(flat)); });
            _define(value, std::move(elements));
        }
    }
    void _define(const Value *value, Elements elements) {
        auto data = _representation(value);
        if (elements.size() > 1u && detail::needs_indexable_snapshot(value, _options.max_unrolled_tile_elements)) {
            auto storage = _allocate(value->type());
            for (size_t i = 0u; i < elements.size(); i++) {
                _store_local(value->type(), storage, _index(i), elements[i]);
            }
            // Stores occur at this SSA definition, not at the first extract:
            // external writes cannot change a previously loaded Tile, and a
            // reduction does not re-materialize its entire input per iteration.
            data->storage = storage;
        }
        data->elements = std::move(elements);
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
    [[nodiscard]] x::Value *_read(const Representation *data, x::Value *flat) {
        if (auto found = _fused_elements.find(data); found != _fused_elements.end()) { return found->second; }
        if (data->splat) { return data->elements.front(); }
        if (data->expression) {
            auto &domain = *data->type->index_space();
            auto coordinates = _coordinates(domain, flat);
            Elements inputs;
            for (auto input : data->inputs) { inputs.emplace_back(_project(input, domain, coordinates)); }
            return _elementwise(data->expression->elementwise_op(), _type(*data->type), inputs);
        }
        uint64_t constant = 0u;
        if (!data->elements.empty() && x::try_decode_constant_nonnegative_integer(flat, constant)) {
            return data->elements.at(constant);
        }
        if (data->storage) {
            _charge(2u);
            return _builder.load(_type(*data->type), _builder.gep(_type(*data->type), data->storage, {_storage_index(*data->type, flat)}));
        }
        if (data->pending_load) { _fail("elided load snapshot has a consumer outside its fused traversal"); }
        auto type = _type(*data->type);
        x::Value *value = _output.module->create_constant_zero(type);
        for (size_t i = 0u; i < data->elements.size(); i++) {
            value = _alu(type, A::SELECT, {value, data->elements[i], _compare(A::BINARY_EQUAL, flat, _index(i))});
        }
        return value;
    }
    [[nodiscard]] x::Value *_project(const Representation *data, const IndexSpace &domain, const Elements &coordinates) {
        if (!data->type->is_tile()) { return data->elements.front(); }
        auto &space = *data->type->index_space();
        x::Value *flat = _index(0u);
        luisa::optional<uint64_t> constant_flat{0u};
        for (size_t i = 0u; i < space.rank(); i++) {
            auto axis = domain.axis_index(space.axis(i).dimension);
            if (!axis) { _fail("Tile operand dimension is absent from its XIR expression domain"); }
            auto extent = _extent(space, i);
            auto coordinate = extent == 1u ? _index(0u) : coordinates[*axis];
            uint64_t constant = 0u;
            if (constant_flat && x::try_decode_constant_nonnegative_integer(coordinate, constant)) {
                if (constant >= extent) { _fail("Tile projection is out of bounds"); }
                *constant_flat = *constant_flat * extent + constant;
                flat = _index(*constant_flat);
            } else {
                constant_flat.reset();
                flat = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, flat, _index(extent)), coordinate);
            }
        }
        return _read(data, flat);
    }
    void _copy(const Representation *source, x::Value *destination) {
        _for_each(_volume(*source->type->index_space()), [&](x::Value *flat) {
            _store_local(*source->type, destination, flat, _read(source, flat));
        });
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
    [[nodiscard]] ViewAccess _capture_view_access(const Operation &op) {
        auto view = op.operand(0u);
        auto found = _arguments.find(view);
        if (found == _arguments.end()) { _fail("XIR view access requires a direct buffer argument"); }
        auto slot = found->second;
        auto load = op.kind() == OperationKind::VIEW_LOAD;
        _output.argument_usages[slot] = static_cast<Usage>(static_cast<uint32_t>(_output.argument_usages[slot]) | static_cast<uint32_t>(load ? Usage::READ : Usage::WRITE));
        auto &space = *view->type().index_space();
        ViewAccess access{&op, _get(view)->elements.front(), {}, {}};
        for (size_t i = 0u; i < space.rank(); i++) {
            access.origins.emplace_back(_scalar(op.operand(i + 1u)));
            access.ranges.emplace_back(_range(op.operand(i + 1u)));
        }
        if (load) {
            access.fill = _output.module->create_constant_zero(_type(op.result(0u)->type()));
            if (op.domain() && op.operand_count() == space.rank() + 2u) { access.fill = _scalar(op.operand(space.rank() + 1u)); }
            if (!op.domain() && op.operand_count() == space.rank() + 3u) {
                access.mask = _scalar(op.operand(space.rank() + 1u));
                access.fill = _scalar(op.operand(space.rank() + 2u));
            }
        }
        return access;
    }
    [[nodiscard]] x::Value *_view_element(const ViewAccess &access, x::Value *flat) {
        auto &op = *access.operation;
        auto &space = *op.operand(0u)->type().index_space();
        _charge();
        auto indices = op.domain() ? _coordinates(*op.domain(), flat) : Elements(space.rank(), _index(0u));
        x::Value *address = _index(0u);
        x::Value *valid = _constant(true);
        auto needs_guard = false;
        for (size_t i = 0u; i < space.rank(); i++) {
            auto coordinate = access.origins[i];
            if (op.domain()) { coordinate = _binary(A::BINARY_ADD, coordinate, indices[i]); }
            address = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, address, _index(_extent(space, i))), coordinate);
            if (op.domain() && op.bounds_mode() == BoundsMode::ZERO) {
                if (auto range = access.ranges[i]) {
                    uint64_t offset = 0u;
                    auto fixed = x::try_decode_constant_nonnegative_integer(indices[i], offset);
                    auto lo = checked_add(range->lo, fixed ? static_cast<int64_t>(offset) : 0);
                    auto hi = checked_add(range->hi, fixed ? static_cast<int64_t>(offset) : static_cast<int64_t>(_extent(*op.domain(), i)) - 1);
                    if (lo && hi && *lo >= 0 && *hi < static_cast<int64_t>(_extent(space, i))) { continue; }
                }
                needs_guard = true;
                valid = _binary(A::BINARY_BIT_AND, valid, _compare(A::BINARY_GREATER_EQUAL, coordinate, _index(0u)));
                valid = _binary(A::BINARY_BIT_AND, valid, _compare(A::BINARY_LESS, coordinate, _index(_extent(space, i))));
            }
        }
        address = _builder.static_cast_if_necessary(XType::of<uint64_t>(), address);
        if (op.kind() == OperationKind::VIEW_LOAD) {
            auto type = _type(op.result(0u)->type());
            if (access.mask) { valid = access.mask; }
            return needs_guard || access.mask ? _guarded_load(valid, access.buffer, address, access.fill) :
                                                _builder.call(type, x::ResourceReadOp::BUFFER_READ, {access.buffer, address});
        } else {
            auto value = _read(_get(op.operand(space.rank() + 1u)), flat);
            if (needs_guard) {
                _guarded_store(valid, access.buffer, address, value);
            } else {
                _builder.call(x::ResourceWriteOp::BUFFER_WRITE, {access.buffer, address, value});
            }
        }
        return nullptr;
    }
    void _view_access(const Operation &op) {
        auto captured = _capture_view_access(op);
        auto count = op.domain() ? _volume(*op.domain()) : 1u;
        auto access = [&](x::Value *flat) { return _view_element(captured, flat); };
        if (op.kind() == OperationKind::VIEW_LOAD) {
            if (_options.enable_load_reduction_fusion) {
                if (auto fusion = detail::load_reduction_fusion(op.result(0u), _options.max_unrolled_tile_elements,
                                                                _options.local_lanes, _options.reduction_partitions)) {
                    auto result = _representation(op.result(0u));
                    result->pending_load = true;
                    if (fusion->retain_snapshot) { result->storage = _allocate(*result->type); }
                    _pending_loads[fusion->reduction].emplace_back(FusedLoad{std::move(captured), result});
                    _output.fused_reduction_loads++;
                    _output.elided_load_snapshots += !fusion->retain_snapshot;
                    return;
                }
            }
            if (op.result(0u)->type().is_tile()) {
                _emit_tile(op.result(0u), access);
            } else {
                _define(op.result(0u), Elements{access(_index(0u))});
            }
        } else {
            if (_options.local_lanes > 1u && count == 1u) {
                auto leader = _output.function->create_basic_block();
                auto exit = _output.function->create_basic_block();
                _builder.cond_br(_compare(A::BINARY_EQUAL, _lane, _index(0u)), leader, exit);
                _at(leader);
                access(_index(0u));
                _builder.br(exit);
                _at(exit);
            } else {
                _for_each(count, access);
            }
        }
    }
    void _bind_coordinates(const Block &body, const IndexSpace &domain, x::Value *flat, luisa::span<const uint32_t> order = {}) {
        auto trailing = _volume(domain);
        auto major = true;
        for (size_t position = 0u; position < domain.rank(); position++) {
            auto i = order.empty() ? position : order[position];
            auto extent = _extent(domain, i);
            auto coordinate = _index(0u);
            if (trailing != 0u && extent != 0u) {
                trailing /= extent;
                coordinate = extent == 1u ? _index(0u) : _binary(A::BINARY_DIV, flat, _index(trailing));
                if (extent != 1u && !major) { coordinate = _binary(A::BINARY_MOD, coordinate, _index(extent)); }
            }
            major &= extent == 1u;
            _define(body.argument(i), Elements{coordinate});
            // The body executes only for valid coordinates; zero-trip loop
            // bodies are unreachable. Never infer ranges for carried values.
            if (extent != 0u) { _coordinate_ranges.insert_or_assign(body.argument(i), IndexRange{0, static_cast<int64_t>(extent - 1u)}); }
        }
    }
    [[nodiscard]] luisa::vector<const Representation *> _region(const Block &body) {
        for (auto op : body.operations()) {
            if (op->kind() == OperationKind::YIELD) {
                luisa::vector<const Representation *> yielded;
                for (size_t i = 0u; i < op->operand_count(); i++) { yielded.emplace_back(_get(op->operand(i))); }
                return yielded;
            }
            _operation(*op);
        }
        return {};
    }
    [[nodiscard]] bool _partial_reduction(const Operation &op) {
        auto closed = detail::closed_reduction(op);
        if (!closed) { return false; }
        auto total = _volume(*op.domain());
        auto distributed = _distributed(total);
        if (!distributed && (!_bounded(total) || _options.reduction_partitions <= 1u)) { return false; }
        auto body = op.region(0u)->block(0u);
        auto update = closed->update;
        auto yield = closed->yield;
        auto left = closed->carry_left;
        auto kind = update->elementwise_op();
        auto contribution = closed->contribution;
        auto lanes = distributed ? _options.local_lanes : 1u;
        auto count = total / lanes;
        auto partitions = std::min<uint64_t>(_options.reduction_partitions, count);
        auto type = _type(op.result(0u)->type());
        auto initial = _scalar(op.operand(0u));
        // Consume, rather than retain, this host-side plan. The same source
        // operation can be lowered again by an enclosing expanded loop/map.
        luisa::vector<FusedLoad> loads;
        if (auto found = _pending_loads.find(&op); found != _pending_loads.end()) {
            loads = std::move(found->second);
            _pending_loads.erase(found);
        }
        auto evaluate = [&](x::Value *ordinal) {
            auto previous = _local_slot;
            if (distributed) { _local_slot = ordinal; }
            if (distributed) { ordinal = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, ordinal, _index(lanes)), _lane); }
            _bind_coordinates(*body, *op.domain(), ordinal);
            for (auto &load : loads) {
                auto &space = *load.result->type->index_space();
                auto flat = _index(0u);
                for (size_t i = 0u; i < space.rank(); i++) {
                    auto axis = op.domain()->axis_index(space.axis(i).dimension);
                    auto coordinate = _extent(space, i) == 1u ? _index(0u) : _scalar(body->argument(*axis));
                    flat = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, flat, _index(_extent(space, i))), coordinate);
                }
                auto element = _view_element(load.access, flat);
                if (load.result->storage) { _store_local(*load.result->type, load.result->storage, flat, element); }
                _fused_elements.emplace(load.result, element);
            }
            for (auto operation : body->operations()) {
                if (operation != update && operation != yield) { _operation(*operation); }
            }
            auto result = _scalar(contribution);
            for (auto &load : loads) { _fused_elements.erase(load.result); }
            _local_slot = previous;
            return result;
        };
        auto combine = [&](x::Value *a, x::Value *b) {
            return _elementwise(kind, type, left ? Elements{a, b} : Elements{b, a});
        };
        Elements seeds;
        for (uint64_t p = 0u; p < partitions; p++) { seeds.emplace_back(evaluate(_index(p))); }
        // Seed each nonempty partition from its first actual contribution.
        // Do not invent zero/one identities (notably for signed zero/NaN),
        // and include the user's initial accumulator exactly once at the end.
        auto preheader = _block;
        auto header = _output.function->create_basic_block();
        auto loop_body = _output.function->create_basic_block();
        auto exit = _output.function->create_basic_block();
        _builder.br(header);
        _at(header);
        auto induction = _builder.phi(XType::of<int64_t>(), {{_index(partitions), preheader}});
        luisa::vector<x::PhiInst *> partials;
        for (auto seed : seeds) { partials.emplace_back(_builder.phi(type, {{seed, preheader}})); }
        auto bulk = count - count % partitions;
        _builder.cond_br(_compare(A::BINARY_LESS, induction, _index(bulk)), loop_body, exit);
        _at(loop_body);
        Elements next;
        for (uint64_t p = 0u; p < partitions; p++) {
            next.emplace_back(combine(partials[p], evaluate(_binary(A::BINARY_ADD, induction, _index(p)))));
        }
        for (size_t p = 0u; p < partials.size(); p++) { partials[p]->add_incoming(next[p], _block); }
        induction->add_incoming(_binary(A::BINARY_ADD, induction, _index(partitions)), _block);
        _builder.br(header);
        _at(exit);
        Elements results{partials.begin(), partials.end()};
        for (uint64_t p = 0u; p < count % partitions; p++) { results[p] = combine(results[p], evaluate(_index(bulk + p))); }
        if (distributed) {
            // A final partial chunk updates only the owning active lanes. All
            // lanes reconverge before shuffles; no empty-lane identity or
            // duplicated initial accumulator is introduced.
            if (total % lanes != 0u) {
                auto before = _block;
                auto tail = _output.function->create_basic_block();
                auto exit = _output.function->create_basic_block();
                _builder.cond_br(_compare(A::BINARY_LESS, _lane, _index(total % lanes)), tail, exit);
                _at(tail);
                auto value = combine(results[0u], evaluate(_index(count)));
                auto after = _block;
                _builder.br(exit);
                _at(exit);
                results[0u] = _builder.phi(type, {{results[0u], before}, {value, after}});
            }
            auto value = results[0u];
            for (size_t p = 1u; p < results.size(); p++) { value = combine(value, results[p]); }
            auto lane = _builder.static_cast_if_necessary(XType::of<uint32_t>(), _lane);
            for (uint32_t distance = 1u; distance < lanes; distance *= 2u) {
                auto peer = _binary(A::BINARY_BIT_XOR, lane, _constant(distance));
                auto other = _builder.call(type, x::ThreadGroupOp::WARP_READ_LANE, {value, peer});
                value = combine(value, other);
            }
            // Every lane consumes the exact same tree root, even when its own
            // butterfly operand ordering would produce a different FP value.
            auto root = _builder.call(type, x::ThreadGroupOp::WARP_READ_LANE, {value, _constant(uint32_t{0u})});
            initial = combine(initial, root);
        } else {
            for (auto partial : results) { initial = combine(initial, partial); }
        }
        _define(op.result(0u), Elements{initial});
        return true;
    }
    void _loop(const Operation &op) {
        auto &domain = *op.domain();
        auto body = op.region(0u)->block(0u);
        if (auto scope = op.execution_scope_constraint(); scope && *scope != "worker" && *scope != "auto") {
            _fail("XIR worker realization cannot honor this explicit execution binding");
        }
        if (op.kind() == OperationKind::PARALLEL && !_inside_parallel) {
            if (_saw_parallel || op.result_count() != 0u) { _fail("XIR bridge requires one independent root parallel with no escaping results"); }
            auto count = _volume(domain);
            if (count > UINT32_MAX / _options.local_lanes) { _fail("packet-local XIR dispatch exceeds uint32 range"); }
            _output.dispatch_size = static_cast<uint32_t>(count * _options.local_lanes);
            if (_output.dispatch_size == 0u) { _fail("empty root parallel has no executable launch"); }
            _saw_parallel = true;
            _inside_parallel = true;
            auto dispatch = _alu(XType::of<uint32_t>(), A::EXTRACT, {_output.module->create_dispatch_id(), _constant(uint32_t{0})});
            if (_options.local_lanes > 1u) {
                auto lane = _output.module->create_warp_lane_id();
                _lane = _builder.static_cast_(XType::of<int64_t>(), lane);
                // Subtract the physical lane first: the packet's program index
                // is uniform, unlike the element coordinate distributed below.
                dispatch = _binary(A::BINARY_DIV, _binary(A::BINARY_SUB, dispatch, lane), _constant(_options.local_lanes));
            }
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
        if (_partial_reduction(op)) { return; }
        struct Carry {
            luisa::vector<x::PhiInst *> phis;
            x::Value *current{nullptr};
            x::Value *next{nullptr};
        };
        luisa::vector<Carry> carries(op.result_count());
        for (size_t i = 0u; i < carries.size(); i++) {
            auto &type = op.result(i)->type();
            if (type.is_tile() && _bounded(_volume(*type.index_space()))) {
                carries[i].current = _allocate(type);
                carries[i].next = _allocate(type);
                _copy(_get(op.operand(i)), carries[i].current);
            }
        }
        auto preheader = _block;
        auto header = _output.function->create_basic_block();
        auto loop_body = _output.function->create_basic_block();
        auto exit = _output.function->create_basic_block();
        _builder.br(header);
        _at(header);
        auto induction = _builder.phi(XType::of<int64_t>(), {{_index(0u), preheader}});
        for (size_t i = 0u; i < op.result_count(); i++) {
            if (carries[i].current) { continue; }
            for (auto value : _get(op.operand(i))->elements) {
                auto phi = _builder.phi(value->type(), {{value, preheader}});
                carries[i].phis.emplace_back(phi);
            }
        }
        _builder.cond_br(_compare(A::BINARY_LESS, induction, _index(_volume(domain))), loop_body, exit);
        _at(loop_body);
        // All PHIs must precede non-PHI instructions. Materialize carried
        // snapshots only on an executed iteration, after simultaneous updates.
        for (size_t i = 0u; i < carries.size(); i++) {
            auto argument = body->argument(domain.rank() + i);
            if (carries[i].current) {
                _representation(argument)->storage = carries[i].current;
            } else {
                auto &phis = carries[i].phis;
                _define(argument, Elements{phis.begin(), phis.end()});
            }
        }
        x::Value *ordinal = induction;
        if (op.kind() == OperationKind::REDUCE && op.reduction_policy() == reduction::fold_right) {
            // Reverse the logical sequence, not the accumulator operands or
            // the root worker permutation. This expression is unreachable for
            // an empty domain, so there is no unsigned host-side underflow.
            ordinal = _binary(A::BINARY_SUB, _index(_volume(domain)),
                              _binary(A::BINARY_ADD, induction, _index(1u)));
        }
        _bind_coordinates(*body, domain, ordinal);
        auto yielded = _region(*body);
        if (yielded.size() != carries.size()) { _fail("XIR loop yield does not match its carried state"); }
        // Stage every large incoming before overwriting any current carry.
        // This is a parallel copy, including swaps and interdependent Tiles.
        for (size_t i = 0u; i < carries.size(); i++) {
            if (carries[i].next) { _copy(yielded[i], carries[i].next); }
        }
        for (size_t i = 0u; i < carries.size(); i++) {
            if (carries[i].current) {
                Representation staged;
                staged.type = &op.result(i)->type();
                staged.storage = carries[i].next;
                _copy(&staged, carries[i].current);
            }
        }
        for (size_t i = 0u; i < carries.size(); i++) {
            if (carries[i].current) { continue; }
            auto &phis = carries[i].phis;
            if (yielded[i]->elements.size() != phis.size()) { _fail("XIR loop carry shape mismatch"); }
            for (size_t j = 0u; j < phis.size(); j++) { phis[j]->add_incoming(yielded[i]->elements[j], _block); }
        }
        induction->add_incoming(_binary(A::BINARY_ADD, induction, _index(1u)), _block);
        _builder.br(header);
        _at(exit);
        for (size_t i = 0u; i < carries.size(); i++) {
            if (carries[i].current) {
                _representation(op.result(i))->storage = carries[i].current;
            } else {
                auto &phis = carries[i].phis;
                _define(op.result(i), Elements{phis.begin(), phis.end()});
            }
        }
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
        auto type = _type(result->type());
        _emit_tile(result, [&](x::Value *flat) {
            auto coordinates = _coordinates(space, flat);
            auto initial = _read(_get(op.operand(2u)), flat);
            return _fold(_volume(contraction), initial, [&](x::Value *k, x::Value *sum) {
                auto full = coordinates;
                for (auto coordinate : _coordinates(contraction, k)) { full.emplace_back(coordinate); }
                auto a = _builder.static_cast_if_necessary(type, _project(_get(op.operand(0u)), domain, full));
                auto b = _builder.static_cast_if_necessary(type, _project(_get(op.operand(1u)), domain, full));
                return _binary(A::BINARY_ADD, sum, _binary(A::BINARY_MUL, a, b));
            });
        });
    }
    void _operation(const Operation &op) {
        switch (op.kind()) {
            case OperationKind::CONSTANT: {
                auto result = op.result(0u);
                auto count = result->type().is_tile() ? _volume(*result->type().index_space()) : 1u;
                if (_bounded(count) || _distributed(count)) {
                    _charge();
                    auto data = _representation(result);
                    data->splat = true;
                    data->elements.emplace_back(_literal(op));
                } else {
                    _charge(count);
                    _define(result, Elements(count, _literal(op)));
                }
                break;
            }
            case OperationKind::ELEMENTWISE: {
                auto result = op.result(0u);
                auto domain = result->type().is_tile() ? *result->type().index_space() : IndexSpace{};
                if (detail::deferred_elementwise(result, _options.max_unrolled_tile_elements, _options.local_lanes)) {
                    // Capture immutable physical operands now, not mutable
                    // TileIR-to-XIR bindings that another map/carry may replace.
                    // Only pure single-use arithmetic is deferred. Loads and
                    // multi-consumer values stay materialized at their definition.
                    _charge();
                    auto data = _representation(result);
                    data->expression = &op;
                    for (size_t j = 0u; j < op.operand_count(); j++) { data->inputs.emplace_back(_get(op.operand(j))); }
                    break;
                }
                auto evaluate = [&](x::Value *flat) {
                    auto coordinates = _coordinates(domain, flat);
                    Elements inputs;
                    for (size_t j = 0u; j < op.operand_count(); j++) { inputs.emplace_back(_project(_get(op.operand(j)), domain, coordinates)); }
                    return _elementwise(op.elementwise_op(), _type(result->type()), inputs);
                };
                if (result->type().is_tile()) {
                    _emit_tile(result, evaluate);
                } else {
                    _define(result, Elements{evaluate(_index(0u))});
                }
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
                auto body = op.region(0u)->block(0u);
                _emit_tile(op.result(0u), [&](x::Value *flat) {
                    auto coordinates = _coordinates(*op.domain(), flat);
                    for (size_t j = 0u; j < coordinates.size(); j++) {
                        _define(body->argument(j), Elements{coordinates[j]});
                        uint64_t c = 0u;
                        auto fixed = x::try_decode_constant_nonnegative_integer(coordinates[j], c);
                        _coordinate_ranges.insert_or_assign(body->argument(j), fixed ? IndexRange{static_cast<int64_t>(c), static_cast<int64_t>(c)} : IndexRange{0, static_cast<int64_t>(_extent(*op.domain(), j)) - 1});
                    }
                    auto yielded = _region(*body);
                    if (yielded.size() != 1u || yielded[0]->elements.size() != 1u) { _fail("Tile map must yield exactly one scalar"); }
                    return yielded[0]->elements[0];
                });
                break;
            }
            case OperationKind::TILE_EXTRACT: {
                auto tile = op.operand(0u);
                auto &space = *tile->type().index_space();
                auto data = _get(tile);
                auto count = _volume(space);
                auto type = _type(op.result(0u)->type());
                x::Value *value = _output.module->create_constant_zero(type);
                // Constant map coordinates project SSA directly. Use checked
                // integer arithmetic; never a floating-point planner estimate.
                luisa::optional<int64_t> constant_flat{0};
                for (size_t i = 0u; i < space.rank() && constant_flat; i++) {
                    auto range = _range(op.operand(i + 1u));
                    auto product = checked_multiply(*constant_flat, static_cast<int64_t>(_extent(space, i)));
                    constant_flat = range && range->lo == range->hi && product ? checked_add(*product, range->lo) : luisa::nullopt;
                }
                if (constant_flat) {
                    if (*constant_flat >= 0 && static_cast<uint64_t>(*constant_flat) < count) { value = _read(data, _index(*constant_flat)); }
                    _define(op.result(0u), Elements{value});
                    break;
                }
                x::Value *flat = _index(0u);
                auto in_bounds = count != 0u;
                for (size_t i = 0u; i < space.rank(); i++) {
                    auto range = _range(op.operand(i + 1u));
                    in_bounds &= range && range->lo >= 0 && range->hi < static_cast<int64_t>(_extent(space, i));
                    flat = _binary(A::BINARY_ADD, _binary(A::BINARY_MUL, flat, _index(_extent(space, i))), _scalar(op.operand(i + 1u)));
                }
                if (in_bounds) {
                    // Declared region coordinates plus checked integer ranges
                    // establish valid projection independently of parallel's
                    // conflict contract. In particular, a proven lane-local
                    // extract must not introduce a divergent control region.
                    _define(op.result(0u), Elements{_read(data, flat)});
                    break;
                }
                if (count != 0u) {
                    auto valid = _binary(A::BINARY_BIT_AND, _compare(A::BINARY_GREATER_EQUAL, flat, _index(0u)),
                                         _compare(A::BINARY_LESS, flat, _index(count)));
                    // Preserve the existing flat-index zero fallback, including
                    // negative indices. Never issue an out-of-bounds local load.
                    auto header = _block;
                    auto read = _output.function->create_basic_block();
                    auto merge = _output.function->create_basic_block();
                    _builder.cond_br(valid, read, merge);
                    _at(read);
                    auto loaded = _read(data, flat);
                    auto read_exit = _block;
                    _builder.br(merge);
                    _at(merge);
                    value = _builder.phi(type, {{value, header}, {loaded, read_exit}});
                }
                _define(op.result(0u), Elements{value});
                break;
            }
            default: _fail("unsupported TileIR operation in XIR worker realization; no fallback or effect erasure");
        }
    }

public:
    Lowerer(const Function &input, LowerOptions options) : _input{input}, _options{options} {}
    [[nodiscard]] NativeFunction run() {
        if (_input.parent_module() == nullptr || !verify(*_input.parent_module())) { _fail("TileIR verification failed before XIR lowering"); }
        if (_input.body().block_count() != 1u || !x::KernelFunction::is_valid_block_size(luisa::make_uint3(_options.block_size, 1u, 1u)) || _options.max_expanded_values == 0u ||
            _options.reduction_partitions == 0u || _options.reduction_partitions > 16u ||
            !_options.local_lanes || _options.local_lanes > 16u || (_options.local_lanes & (_options.local_lanes - 1u)) ||
            _options.block_size % _options.local_lanes) { _fail("invalid XIR realization options or entry region"); }
        if (_options.local_lanes > 1u && !detail::packet_local_program(_input, _options.local_lanes)) {
            _fail("XIR packet-local realization requires a common pointwise axis and closed unordered reductions with owner-preserving extracts");
        }
        _output.module = luisa::make_unique<x::Module>();
        _output.required_packet_width = _options.local_lanes > 1u ? _options.local_lanes : 0u;
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
            _define(value, Elements{buffer});
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
