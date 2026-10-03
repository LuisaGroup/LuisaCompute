#include "cuda_tile_codegen.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/verifier.h>
#include <luisa/tile/collective_plan.h>

namespace luisa::compute::cuda::native_tile {
namespace {
using namespace tile;

class Emitter {
private:
    const tile::Function &_function;
    const bool _enable_fast_math;
    const bool _enable_aligned16;
    const uint32_t _worker_warps;
    const uint32_t _target_sm;
    const uint32_t _scan_chunk_extent;
    const uint32_t _independent_axis_extent;
    luisa::unordered_set<uint64_t> _chunked_scan_operations;
    luisa::unordered_map<uint64_t, tile::CollectiveKind> _partition_collectives;
    uint32_t _aligned16_seen{0u};
    uint32_t _aligned16_rejected{0u};
    struct AlignedViewLoad {
        size_t offset;
        size_t length;
        uint32_t argument_index;
        luisa::string replacement;
    };
    luisa::vector<AlignedViewLoad> _aligned_view_loads;
    Artifact _artifact;
    luisa::unordered_map<const Value *, luisa::string> _values;
    luisa::unordered_map<const Value *, size_t> _buffers;
    uint32_t _indent{1u};
    bool _inside_parallel{false};
    uint32_t _parallel_count{0u};
    const IndexSpace *_map_space{nullptr};
    const Block *_map_body{nullptr};
    luisa::unordered_set<const Value *> _mapped_values;

    // A conservative proof for scalar INDEX/int64 values only. The DSL uses
    // scalar int64 results for arithmetic on loop INDEX arguments. All values
    // are nonnegative and fit int64. Unknown/overflowing expressions keep
    // masked pointer accesses; this analysis never changes their arithmetic.
    struct IndexFacts {
        uint64_t minimum;
        uint64_t maximum;
    };
    luisa::unordered_map<const Value *, IndexFacts> _index_facts;

    void _record_index_facts(const Operation &op) noexcept {
        if (_map_space != nullptr || op.result_count() != 1u) { return; }
        auto &&type = op.result(0u)->type();
        if (type.kind() != TypeKind::INDEX &&
            !(type.kind() == TypeKind::SCALAR && type.scalar_type() == ScalarType::INT64)) { return; }
        constexpr auto kMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        auto result = op.result(0u);
        if (op.kind() == OperationKind::CONSTANT) {
            if (auto attr = op.attribute("value")) {
                if (auto x = luisa::get_if<int64_t>(&attr->value()); x != nullptr && *x >= 0) {
                    auto value = static_cast<uint64_t>(*x);
                    _index_facts.emplace(result, IndexFacts{value, value});
                } else if (auto x = luisa::get_if<uint64_t>(&attr->value()); x != nullptr && *x <= kMax) {
                    _index_facts.emplace(result, IndexFacts{*x, *x});
                }
            }
            return;
        }
        if (op.kind() != OperationKind::ELEMENTWISE || op.operand_count() == 0u) { return; }
        auto lhs = _index_facts.find(op.operand(0u));
        if (lhs == _index_facts.end()) { return; }
        if (op.elementwise_op() == ElementwiseOp::CAST && op.operand_count() == 1u) {
            // Only INDEX/int64 values enter this table, so this cast cannot
            // narrow an integer or reinterpret a floating-point operand.
            _index_facts.emplace(result, lhs->second);
            return;
        }
        if (op.operand_count() != 2u) { return; }
        auto rhs = _index_facts.find(op.operand(1u));
        if (rhs == _index_facts.end()) { return; }
        auto a = lhs->second, b = rhs->second;
        if (op.elementwise_op() == ElementwiseOp::ADD) {
            if (a.maximum > kMax - b.maximum) { return; }
            _index_facts.emplace(result, IndexFacts{a.minimum + b.minimum, a.maximum + b.maximum});
        } else if (op.elementwise_op() == ElementwiseOp::MUL &&
                   (a.minimum == a.maximum || b.minimum == b.maximum)) {
            if (a.maximum != 0u && b.maximum > kMax / a.maximum) { return; }
            _index_facts.emplace(result, IndexFacts{a.minimum * b.minimum, a.maximum * b.maximum});
        } else if (op.elementwise_op() == ElementwiseOp::DIV &&
                   b.minimum == b.maximum && b.minimum != 0u) {
            // Nonnegative signed division is monotone truncation. No zero
            // denominator or INT64_MIN / -1 can enter this proof domain.
            _index_facts.emplace(result, IndexFacts{a.minimum / b.minimum, a.maximum / b.minimum});
        }
    }

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
        if (!tile && space.rank() == 0u) {
            _fail(op, "buffer views and loop domains must have at least one dimension");
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
            if (tile && volume > (1ull << 24u)) {
                _fail(op, "native CUDA Tile shapes cannot exceed 2^24 elements");
                return false;
            }
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
    [[nodiscard]] bool _type(const tile::Type &type, const Operation *op) noexcept {
        if (type.kind() == TypeKind::INDEX) { return true; }
        if (type.scalar_type() == ScalarType::FLOAT64) {
            _fail(op, "this native runtime slice supports FP32 arithmetic and explicit FP16/BF16 storage/conversion/MMA");
            return false;
        }
        if ((type.kind() != TypeKind::SCALAR && !type.is_tile()) || _scalar(type.scalar_type()).empty()) {
            _fail(op, "unsupported value type (index, bool, i32/u32/i64/u64, f16/bf16/f32 scalars/Tiles)");
            return false;
        }
        return !type.is_tile() || _space(*type.index_space(), true, op);
    }
    [[nodiscard]] luisa::string_view _element(const tile::Type &type) noexcept {
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
    [[nodiscard]] luisa::string _elementwise_value(const Value *value, const tile::Type &result, const Operation &op) noexcept {
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
    [[nodiscard]] luisa::string _fast_div_sqrt(const Operation &op, luisa::string_view numerator) noexcept {
        auto denominator = op.operand(1u);
        auto root = denominator->defining_operation();
        if (op.operand(0u)->type().scalar_type() != ScalarType::FLOAT32 ||
            root == nullptr || root->kind() != OperationKind::ELEMENTWISE ||
            root->elementwise_op() != ElementwiseOp::SQRT || root->operand_count() != 1u ||
            root->result_count() != 1u || root->result(0u) != denominator ||
            denominator->type().scalar_type() != ScalarType::FLOAT32 ||
            root->operand(0u)->type().scalar_type() != ScalarType::FLOAT32) { return {}; }
        // Keep both named-axis alignments from the original SQRT then DIV.
        // Computing the reciprocal in the producer's space allows a row
        // scalar to remain a row scalar before the consumer broadcasts it.
        auto argument = _elementwise_value(root->operand(0u), denominator->type(), *root);
        auto reciprocal = luisa::format("ct::rsqrt({}, ct::round_subnormals_to_zero_t{{}})", argument);
        auto &&result = op.result(0u)->type();
        if (result.is_tile() && denominator->type().is_tile()) {
            reciprocal = _align(std::move(reciprocal), *denominator->type().index_space(), *result.index_space(), op);
        }
        // This is an explicit fast FP32 policy, not a general reciprocal
        // transform. SQRT emission is retained for any other users; an unused
        // pure producer may be removed by the native compiler. Scalar values
        // in map bodies retain the ordinary _mapped_values propagation below.
        return luisa::format("ct::mul({}, {}, ct::round_ties_to_even_t{{}}, ct::round_subnormals_to_zero_t{{}})", numerator, reciprocal);
    }
    void _elementwise(const Operation &op) noexcept {
        auto &&result_type = op.result(0u)->type();
        auto a = _elementwise_value(op.operand(0u), result_type, op);
        auto b = op.operand_count() > 1u ? _elementwise_value(op.operand(1u), result_type, op) : luisa::string{};
        auto element = op.result(0u)->type().scalar_type();
        auto narrow = element == ScalarType::FLOAT16 || element == ScalarType::BFLOAT16;
        for (auto i = 0u; i < op.operand_count(); i++) {
            auto type = op.operand(i)->type().scalar_type();
            narrow |= type == ScalarType::FLOAT16 || type == ScalarType::BFLOAT16;
        }
        if (narrow &&
            op.elementwise_op() != ElementwiseOp::CAST && op.elementwise_op() != ElementwiseOp::SELECT) {
            _fail(&op, "narrow elementwise arithmetic rounding is not implemented (load/cast/select/MMA are supported)");
            return;
        }
        auto fast_fp32 = _enable_fast_math && element == ScalarType::FLOAT32;
        luisa::string expression;
        luisa::string_view binary;
        switch (op.elementwise_op()) {
            case ElementwiseOp::ADD: binary = "+"; break;
            case ElementwiseOp::SUB: binary = "-"; break;
            case ElementwiseOp::MUL: binary = "*"; break;
            case ElementwiseOp::DIV:
                if (fast_fp32) { expression = _fast_div_sqrt(op, a); }
                if (expression.empty()) { binary = "/"; }
                break;
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
            case ElementwiseOp::BITCAST: expression = luisa::format("ct::element_bitcast<{}>({})", _element(op.result(0u)->type()), a); break;
            case ElementwiseOp::MIN:
            case ElementwiseOp::MAX: {
                auto name = op.elementwise_op() == ElementwiseOp::MIN ? "min" : "max";
                expression = element == ScalarType::FLOAT32 ?
                                 luisa::format("ct::{}({}, {}, ct::suppress_nan_t{{}}, ct::preserve_subnormals_t{{}})", name, a, b) :
                                 luisa::format("ct::{}({}, {})", name, a, b);
                break;
            }
            case ElementwiseOp::EXP:
                expression = fast_fp32 ? luisa::format("ct::exp({}, ct::round_approximate_t{{}})", a) :
                                         luisa::format("ct::exp({}, ct::round_full_t{{}})", a);
                break;
            case ElementwiseOp::LOG: expression = luisa::format("ct::log({})", a); break;
            case ElementwiseOp::SQRT:
                expression = fast_fp32 ? luisa::format("ct::sqrt({}, ct::round_approximate_t{{}}, ct::round_subnormals_to_zero_t{{}})", a) :
                                         luisa::format("ct::sqrt({}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", a);
                break;
            // The SDK approximate tanh exceeds the elementwise error budget
            // for ordinary finite inputs. Keep full precision in both modes.
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
            if (op.elementwise_op() == ElementwiseOp::ADD &&
                (element == ScalarType::INT32 || element == ScalarType::INT64)) {
                // CUDA Tile signed addition permits overflow UB. Keep Luisa's
                // fixed-width addition in the unsigned domain, including the
                // scalar fallback used by ordered reductions and scans.
                auto unsigned_element = element == ScalarType::INT32 ? "unsigned" : "unsigned long long";
                expression = luisa::format("ct::element_bitcast<{}>(ct::element_bitcast<{}>({}) + ct::element_bitcast<{}>({}))",
                                           _element(result_type), unsigned_element, a, unsigned_element, b);
            } else if (fast_fp32 && !precise.empty()) {
                auto rounding = op.elementwise_op() == ElementwiseOp::DIV ? "ct::round_approximate_t{}" : "ct::round_ties_to_even_t{}";
                expression = luisa::format("ct::{}({}, {}, {}, ct::round_subnormals_to_zero_t{{}})", precise, a, b, rounding);
            } else {
                expression = precise.empty() ? luisa::format("({} {} {})", a, binary, b) :
                                               luisa::format("ct::{}({}, {}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", precise, a, b);
            }
        }
        _bind(op.result(0u), luisa::format("ct::element_cast<{}>({})", _element(op.result(0u)->type()), expression));
        for (auto i = 0u; i < op.operand_count(); i++) {
            if (_mapped_values.contains(op.operand(i))) { _mapped_values.emplace(op.result(0u)); }
        }
    }
    [[nodiscard]] bool _view_fully_in_bounds(const Operation &op, const IndexSpace &space,
                                             const IndexSpace &view_space) const noexcept {
        if (_map_space != nullptr || space.rank() == 0u || space.rank() != view_space.rank()) { return false; }
        for (auto i = 0u; i < space.rank(); i++) {
            if (_mapped_values.contains(op.operand(1u + i))) { return false; }
            auto fact = _index_facts.find(op.operand(1u + i));
            if (fact == _index_facts.end()) { return false; }
            auto extent = view_space.axis(i).extent.constant_value();
            auto tile = space.axis(i).extent.constant_value();
            // Known facts are nonnegative and int64-representable. Avoid an
            // origin + tile addition, and prove every lane of every invocation.
            // Pointer Tiles need no origin divisibility or stronger alignment.
            if (tile > extent || fact->second.maximum > extent - tile) { return false; }
        }
        return true;
    }

    // partition_view indices name whole Tile chunks, not element origins.
    // This bounded proof uses the existing nonnegative/no-overflow facts.
    [[nodiscard]] bool _origin_multiple_of(const Value *value, uint64_t divisor,
                                            uint32_t depth = 0u) const noexcept {
        if (divisor == 0u || depth >= 32u) { return false; }
        auto fact = _index_facts.find(value);
        if (fact == _index_facts.end() || _mapped_values.contains(value)) { return false; }
        if (divisor == 1u) { return true; }
        if (fact->second.minimum == fact->second.maximum) {
            return fact->second.minimum % divisor == 0u;
        }
        if (auto product = _index_operation(value, ElementwiseOp::MUL)) {
            for (auto i = 0u; i < 2u; i++) {
                auto constant = _index_facts.find(product->operand(i));
                if (constant != _index_facts.end() && constant->second.minimum == constant->second.maximum &&
                    constant->second.minimum % divisor == 0u) { return true; }
            }
        }
        if (auto sum = _index_operation(value, ElementwiseOp::ADD)) {
            return _origin_multiple_of(sum->operand(0u), divisor, depth + 1u) &&
                   _origin_multiple_of(sum->operand(1u), divisor, depth + 1u);
        }
        return false;
    }
    [[nodiscard]] luisa::string _aligned_partition_load(const Operation &op, const IndexSpace &space,
                                                        const IndexSpace &view_space, luisa::string_view prefix) noexcept {
        if (!_enable_aligned16 || !_view_fully_in_bounds(op, space, view_space)) { return {}; }
        luisa::string extents{"ct::extents<long long"}, chunks;
        for (auto i = 0u; i < space.rank(); i++) {
            auto extent = space.axis(i).extent.constant_value();
            if (!_origin_multiple_of(op.operand(1u + i), extent)) { return {}; }
            extents += luisa::format(", {}", view_space.axis(i).extent.constant_value());
            if (i != 0u) { chunks += ", "; }
            chunks += luisa::format("({}) / {}ll", _value(op.operand(1u + i)), extent);
        }
        extents += ">{}";
        auto indent = luisa::string(_indent * 4u, ' ');
        return luisa::format("{}auto {}_span = ct::tensor_span{{{}, {}}};\n"
                             "{}auto {}_partition = ct::partition_view{{{}_span, {}{{}}}};\n"
                             "{}auto {} = {}_partition.load({});\n",
                             indent, prefix, _value(op.operand(0u)), extents,
                             indent, prefix, prefix, _shape(space),
                             indent, _name(op.result(0u)), prefix, chunks);
    }
    void _record_aligned16_view(const Operation &op, size_t argument_index,
                                const IndexSpace &space, const IndexSpace &view_space) noexcept {
        if (!_enable_aligned16) { return; }
        auto bit = uint32_t{1u} << argument_index;
        _aligned16_seen |= bit;
        auto element = _artifact.arguments[argument_index].element;
        // This initial specialization is deliberately limited to narrow,
        // row-major accesses with complete static bounds. A row's
        // stride and every contiguous eight-element group start are 16-byte
        // multiples relative to the root. A single unproved access excludes
        // that root, while other independently proved roots remain eligible.
        auto eligible = (element == ScalarType::FLOAT16 || element == ScalarType::BFLOAT16) &&
                        _view_fully_in_bounds(op, space, view_space);
        if (eligible) {
            auto last = space.rank() - 1u;
            eligible = space.axis(last).extent.constant_value() % 8u == 0u &&
                       view_space.axis(last).extent.constant_value() % 8u == 0u &&
                       _origin_multiple_of(op.operand(1u + last), 8u);
        }
        if (!eligible) { _aligned16_rejected |= bit; }
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
        _record_aligned16_view(op, found->second, space, view_space);
        auto prefix = luisa::format("mem{}", op.id());
        // Keep the existing pointer Tile shape/layout. Elide masks only when
        // the scalar index facts prove every lane in bounds; all partial tails,
        // negative/unproved origins and custom out-of-bounds fills stay masked.
        auto masked = op.bounds_mode() == BoundsMode::ZERO && !_view_fully_in_bounds(op, space, view_space);
        // Flattened iota is shaped identically to the logical load/store Tile.
        // Each coordinate is derived from the original named-axis order.
        _line(luisa::format("auto {}_lane = ct::iota<ct::tile<long long, {}>>();", prefix, _shape(space)));
        if (masked) {
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
            auto safe_coord = masked ?
                                  luisa::format("ct::select(({} >= 0ll) && ({} < {}ll), {}, {}_zero)", coord, coord, extent, coord, prefix) :
                                  coord;
            offset = luisa::format("(({}) * {}ll + {})", offset, extent, safe_coord);
        }
        // Masked-out lanes use a valid in-buffer pointer, so arbitrary negative
        // origins do not form out-of-object addresses before the masked access.
        if (masked) {
            _line(luisa::format("auto {}_mask = {};", prefix, mask));
            offset = luisa::format("ct::select({}_mask, {}, {}_zero)", prefix, offset, prefix);
        }
        _line(luisa::format("auto {}_ptr = {} + {};", prefix, _value(view), offset));
        if (read) {
            auto expression = luisa::format("ct::load({}_ptr)", prefix);
            if (masked) {
                auto fallback = op.operand_count() == space.rank() + 2u ? _value(op.operand(space.rank() + 1u)) :
                                                                          luisa::format("ct::element_cast<{}>(0)", _scalar(argument.element));
                expression = luisa::format("ct::load_masked({}_ptr, {}_mask, {})", prefix, prefix, fallback);
            }
            auto binding_offset = _artifact.source.size();
            _bind(op.result(0u), std::move(expression));
            if (auto replacement = _aligned_partition_load(op, space, view_space, prefix); !replacement.empty()) {
                _aligned_view_loads.emplace_back(AlignedViewLoad{
                    binding_offset, _artifact.source.size() - binding_offset,
                    static_cast<uint32_t>(found->second), std::move(replacement)});
            }
        } else {
            auto value = _value(op.operand(space.rank() + 1u));
            _line(masked ?
                      luisa::format("ct::store_masked({}_ptr, {}, {}_mask);", prefix, value, prefix) :
                      luisa::format("ct::store({}_ptr, {});", prefix, value));
        }
    }
    void _mma(const Operation &op) noexcept {
        auto &&a_type = op.operand(0u)->type();
        auto &&b_type = op.operand(1u)->type();
        auto &&c_type = op.operand(2u)->type();
        auto &&a_full = *a_type.index_space();
        auto &&b_full = *b_type.index_space();
        auto &&c_full = *c_type.index_space();
        auto squeeze = [&](const IndexSpace &source) noexcept {
            IndexSpace result;
            for (auto &&axis : source.axes()) {
                if (a_full.contains(axis.dimension) && b_full.contains(axis.dimension) && c_full.contains(axis.dimension)) {
                    if (axis.extent.constant_value() != 1u) { _fail(&op, "MMA only supports singleton shared batch dimensions"); }
                } else {
                    static_cast<void>(result.add(axis.dimension, axis.extent));
                }
            }
            return result;
        };
        auto a = squeeze(a_full), b = squeeze(b_full), c = squeeze(c_full);
        if (!_artifact.error.empty()) { return; }
        if (a.rank() != 2u || b.rank() != 2u || c.rank() != 2u || a_type.scalar_type() != b_type.scalar_type() ||
            c_type.scalar_type() != ScalarType::FLOAT32) {
            _fail(&op, "MMA requires rank-2 matrices after removing singleton batches, equal input element types, and an FP32 accumulator");
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
        auto squeezed = c.rank() != c_full.rank();
        auto initial = _value(op.operand(2u));
        if (squeezed) {
            lhs = luisa::format("ct::reshape({}, {}{{}})", lhs, _shape(a));
            rhs = luisa::format("ct::reshape({}, {}{{}})", rhs, _shape(b));
            initial = luisa::format("ct::reshape({}, {}{{}})", initial, _shape(c));
        }
        if (*ai != 0u) { lhs = luisa::format("ct::transpose({})", lhs); }
        if (*bi != 1u) { rhs = luisa::format("ct::transpose({})", rhs); }
        auto input = a_type.scalar_type();
        if (input != ScalarType::FLOAT32 && input != ScalarType::FLOAT16 && input != ScalarType::BFLOAT16) {
            _fail(&op, "MMA input precision is not implemented; no implicit TF32/narrowing conversion is permitted");
            return;
        }
        if (op.mma_policy().allow_reassociation) {
            auto expression = luisa::format("ct::mma({}, {}, {})", lhs, rhs, initial);
            if (squeezed) { expression = luisa::format("ct::reshape({}, {}{{}})", expression, _shape(c_full)); }
            _bind(op.result(0u), std::move(expression));
            return;
        }
        auto prefix = luisa::format("mma{}", op.id());
        _line(luisa::format("auto {}_a = ct::element_cast<float>({});", prefix, lhs));
        _line(luisa::format("auto {}_b = ct::element_cast<float>({});", prefix, rhs));
        auto acc = squeezed ? luisa::format("{}_acc", prefix) : _name(op.result(0u));
        if (squeezed) {
            _line(luisa::format("auto {} = {};", acc, initial));
        } else {
            _bind(op.result(0u), std::move(initial));
        }
        auto emit_fma = [&](luisa::string_view index) noexcept {
            _line(luisa::format("{} = ct::fma(ct::extract({}_a, ct::shape<{}, 1>{{}}, 0u, {}), ct::extract({}_b, ct::shape<1, {}>{{}}, {}, 0u), {}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}});",
                                acc, prefix, c.axis(0u).extent.constant_value(), index, prefix, c.axis(1u).extent.constant_value(), index, acc));
        };
        auto extent = a.axis(ak).extent.constant_value();
        // Constant slice indices expose the small contraction to Tile IR before
        // physical layout selection. Preserve the exact ascending-K FMA chain.
        constexpr auto kMaxUnrolledContraction = 32u;
        constexpr auto kMaxExtendedContraction = 128u;
        constexpr auto kMaxExtendedOutputElements = 4096u;
        constexpr auto kMaxExtendedFmaOperations = 65536u;
        // Keep the existing <=32 behavior. Bound both the accumulator and
        // total expanded work; divide the budget to avoid product overflow.
        auto output_elements = *c.static_volume();
        auto extend = input == ScalarType::FLOAT32 && extent <= kMaxExtendedContraction &&
                      output_elements != 0u && output_elements <= kMaxExtendedOutputElements &&
                      extent <= kMaxExtendedFmaOperations / output_elements;
        if (extent <= kMaxUnrolledContraction || extend) {
            for (auto k = uint32_t{0u}; k < extent; k++) {
                emit_fma(luisa::format("{}u", k));
            }
        } else {
            _line(luisa::format("for (unsigned {}_k = 0u; {}_k < {}ull; ++{}_k) {{", prefix, prefix, extent, prefix));
            _indent++;
            emit_fma(luisa::format("{}_k", prefix));
            _indent--;
            _line("}");
        }
        if (squeezed) { _bind(op.result(0u), luisa::format("ct::reshape({}, {}{{}})", acc, _shape(c_full))); }
    }
    [[nodiscard]] static const Value *_index_value(const Value *value) noexcept {
        for (;;) {
            auto operation = value->defining_operation();
            if (operation == nullptr || operation->kind() != OperationKind::ELEMENTWISE ||
                operation->elementwise_op() != ElementwiseOp::CAST) { return value; }
            auto integer64 = [](const tile::Type &type) noexcept {
                return type.kind() == TypeKind::INDEX || type == tile::Type::scalar(ScalarType::INT64);
            };
            if (!integer64(value->type()) || !integer64(operation->operand(0u)->type())) { return value; }
            value = operation->operand(0u);
        }
    }
    [[nodiscard]] static const Operation *_index_operation(const Value *value, ElementwiseOp opcode) noexcept {
        value = _index_value(value);
        auto operation = value->defining_operation();
        if (value->type().kind() != TypeKind::INDEX && value->type() != tile::Type::scalar(ScalarType::INT64)) { return nullptr; }
        return operation != nullptr && operation->kind() == OperationKind::ELEMENTWISE &&
                       operation->elementwise_op() == opcode && operation->operand_count() == 2u ?
                   operation :
                   nullptr;
    }
    [[nodiscard]] static luisa::optional<int64_t> _index_constant(const Value *value) noexcept {
        value = _index_value(value);
        auto operation = value->defining_operation();
        auto attribute = operation != nullptr && operation->kind() == OperationKind::CONSTANT ? operation->attribute("value") : nullptr;
        if (attribute != nullptr) {
            if (auto integer = luisa::get_if<int64_t>(&attribute->value())) { return *integer; }
            if (auto integer = luisa::get_if<uint64_t>(&attribute->value()); integer != nullptr && *integer <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                return static_cast<int64_t>(*integer);
            }
        }
        return {};
    }
    // Prove i/(2*s)*(2*s) + (i+s)%(2*s) == i xor s over the
    // nonnegative map coordinate range. No data-dependent index is accepted.
    [[nodiscard]] static luisa::optional<uint64_t> _xor_stride(const Value *value, const Value *coordinate, uint64_t extent) noexcept {
        auto add = _index_operation(value, ElementwiseOp::ADD);
        if (add == nullptr) { return {}; }
        for (auto side = 0u; side < 2u; side++) {
            auto product = _index_operation(add->operand(side), ElementwiseOp::MUL);
            auto remainder = _index_operation(add->operand(1u - side), ElementwiseOp::MOD);
            if (product == nullptr || remainder == nullptr) { continue; }
            auto group = _index_constant(remainder->operand(1u));
            auto shifted = _index_operation(remainder->operand(0u), ElementwiseOp::ADD);
            if (!group || *group < 2 || static_cast<uint64_t>(*group) > extent ||
                !std::has_single_bit(static_cast<uint64_t>(*group)) || shifted == nullptr) { continue; }
            auto stride = *group / 2;
            auto matching_shift = false;
            for (auto term = 0u; term < 2u; term++) {
                matching_shift |= _index_value(shifted->operand(term)) == coordinate &&
                                  _index_constant(shifted->operand(1u - term)) == stride;
            }
            if (!matching_shift) { continue; }
            for (auto term = 0u; term < 2u; term++) {
                auto quotient = _index_operation(product->operand(term), ElementwiseOp::DIV);
                if (quotient != nullptr && _index_value(quotient->operand(0u)) == coordinate &&
                    _index_constant(quotient->operand(1u)) == *group && _index_constant(product->operand(1u - term)) == *group) {
                    return static_cast<uint64_t>(stride);
                }
            }
        }
        return {};
    }
    [[nodiscard]] luisa::string _xor_permute(luisa::string expression, const IndexSpace &space,
                                             uint32_t axis, uint64_t stride, const Operation &op) noexcept {
        uint64_t prefix = 1u, suffix = 1u;
        for (auto i = 0u; i < axis; i++) { prefix *= space.axis(i).extent.constant_value(); }
        for (auto i = axis + 1u; i < space.rank(); i++) { suffix *= space.axis(i).extent.constant_value(); }
        auto groups = prefix * (space.axis(axis).extent.constant_value() / (2u * stride));
        auto inner = stride * suffix;
        auto name = luisa::format("permute{}_{}", op.id(), axis);
        _line(luisa::format("auto {} = ct::reshape({}, ct::shape<{}, 2, {}>{{}});", name, expression, groups, inner));
        auto high = luisa::format("ct::extract({}, ct::shape<{}, 1, {}>{{}}, 0u, 1u, 0u)", name, groups, inner);
        auto low = luisa::format("ct::extract({}, ct::shape<{}, 1, {}>{{}}, 0u, 0u, 0u)", name, groups, inner);
        return luisa::format("ct::reshape(ct::cat({}, {}, ct::integral_constant<1>{{}}), {}{{}})", high, low, _shape(space));
    }
    void _tile_extract(const Operation &op) noexcept {
        auto &&source = *op.operand(0u)->type().index_space();
        auto source_expression = _value(op.operand(0u));
        luisa::vector<uint64_t> extents(source.rank(), 1u);
        luisa::vector<int32_t> source_axes(_map_space ? _map_space->rank() : 0u, -1);
        luisa::string indices;
        for (auto i = 0u; i < source.rank(); i++) {
            auto coordinate = op.operand(1u + i);
            auto mapped = false;
            if (_map_body != nullptr) {
                for (auto j = 0u; j < _map_body->argument_count(); j++) {
                    auto source_extent = source.axis(i).extent.constant_value();
                    auto target_extent = _map_space->axis(j).extent.constant_value();
                    auto direct = _index_value(coordinate) == _map_body->argument(j);
                    auto stride = source_extent == target_extent ? _xor_stride(coordinate, _map_body->argument(j), source_extent) : luisa::optional<uint64_t>{};
                    if (direct || stride) {
                        if (source_axes[j] >= 0 || source_extent < target_extent) {
                            _fail(&op, "map extraction requires independent coordinates within source extents");
                            return;
                        }
                        if (stride) { source_expression = _xor_permute(std::move(source_expression), source, i, *stride, op); }
                        source_axes[j] = static_cast<int32_t>(i);
                        extents[i] = target_extent;
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
        auto expression = luisa::format("ct::extract({}, {}{{}}{})", source_expression, _shape(extents), indices);
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
    [[nodiscard]] static bool _tree_scalar(const tile::Type &type) noexcept {
        if (type.kind() != TypeKind::SCALAR) { return false; }
        auto element = type.scalar_type();
        return element == ScalarType::FLOAT32 || element == ScalarType::INT32 || element == ScalarType::UINT32 ||
               element == ScalarType::INT64 || element == ScalarType::UINT64;
    }
    [[nodiscard]] static bool _tree_identity(const Value *value, ElementwiseOp opcode) noexcept {
        if (!_tree_scalar(value->type())) { return false; }
        auto operation = value->defining_operation();
        auto attribute = operation != nullptr && operation->kind() == OperationKind::CONSTANT ? operation->attribute("value") : nullptr;
        if (attribute == nullptr) { return false; }
        auto element = value->type().scalar_type();
        if (element == ScalarType::FLOAT32) {
            auto number = luisa::get_if<double>(&attribute->value());
            if (number == nullptr) { return false; }
            if (opcode == ElementwiseOp::ADD) { return *number == 0.0 && !std::signbit(*number); }
            return (opcode == ElementwiseOp::MIN || opcode == ElementwiseOp::MAX) &&
                   std::isinf(*number) && std::signbit(*number) == (opcode == ElementwiseOp::MAX);
        }
        if (element == ScalarType::INT32 || element == ScalarType::INT64) {
            auto number = luisa::get_if<int64_t>(&attribute->value());
            if (number == nullptr) { return false; }
            if (opcode == ElementwiseOp::ADD) { return *number == 0; }
            if (opcode == ElementwiseOp::MIN) {
                return *number == (element == ScalarType::INT32 ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int64_t>::max());
            }
            return opcode == ElementwiseOp::MAX &&
                   *number == (element == ScalarType::INT32 ? std::numeric_limits<int32_t>::lowest() : std::numeric_limits<int64_t>::lowest());
        }
        auto number = luisa::get_if<uint64_t>(&attribute->value());
        if (number == nullptr) { return false; }
        if (opcode == ElementwiseOp::ADD || opcode == ElementwiseOp::MAX) { return *number == 0u; }
        return opcode == ElementwiseOp::MIN &&
               *number == (element == ScalarType::UINT32 ? std::numeric_limits<uint32_t>::max() : std::numeric_limits<uint64_t>::max());
    }
    // Only matched unordered FP32 prefix operations enter this opt-in path.
    // The original immutable source remains materialized. No global access,
    // memory ordering, alias assumption or storage conversion is introduced.
    [[nodiscard]] luisa::string _chunked_scan(luisa::string input, const IndexSpace &space,
                                              uint32_t axis, const Operation &op) noexcept {
        auto extent = space.axis(axis).extent.constant_value();
        auto chunks = extent / _scan_chunk_extent;
        luisa::vector<uint64_t> shape, carry_shape;
        for (auto i = 0u; i < space.rank(); i++) {
            auto dimension = space.axis(i).extent.constant_value();
            shape.emplace_back(i == axis ? _scan_chunk_extent : dimension);
            carry_shape.emplace_back(i == axis ? 1u : dimension);
        }
        auto chunk_shape = _shape(shape);
        auto last_shape = _shape(carry_shape);
        auto prefix = luisa::format("scan{}_", op.id());
        luisa::vector<luisa::string> scanned;
        luisa::string carry;
        for (auto chunk = uint64_t{0u}; chunk < chunks; chunk++) {
            luisa::string indices, last_indices;
            for (auto i = 0u; i < space.rank(); i++) {
                indices += luisa::format(", {}ull", i == axis ? chunk : 0u);
                last_indices += luisa::format(", {}ull", i == axis ? _scan_chunk_extent - 1u : 0u);
            }
            auto extracted = luisa::format("{}input{}", prefix, chunk);
            auto local = luisa::format("{}local{}", prefix, chunk);
            auto result = luisa::format("{}prefix{}", prefix, chunk);
            _line(luisa::format("auto {} = ct::extract({}, {}{{}}{});", extracted, input, chunk_shape, indices));
            _line(luisa::format("auto {} = ct::partial_sum({}, ct::integral_constant<{}>{{}}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}}, ct::scan_forward_t{{}});", local, extracted, axis));
            if (chunk == 0u) {
                result = local;
            } else {
                _line(luisa::format("auto {} = ct::add({}, {}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}});", result, carry, local));
            }
            scanned.emplace_back(result);
            if (chunk + 1u != chunks) {
                carry = luisa::format("{}carry{}", prefix, chunk);
                _line(luisa::format("auto {} = ct::extract({}, {}{{}}{});", carry, result, last_shape, last_indices));
            }
        }
        // SDK cat requires two equal-shaped operand types. A balanced tree
        // doubles one extent at each level and remains power-of-two throughout.
        for (auto level = 0u; scanned.size() > 1u; level++) {
            luisa::vector<luisa::string> next;
            for (auto i = size_t{0u}; i < scanned.size(); i += 2u) {
                auto result = luisa::format("{}join{}_{}", prefix, level, i / 2u);
                _line(luisa::format("auto {} = ct::cat({}, {}, ct::integral_constant<{}>{{}});", result, scanned[i], scanned[i + 1u], axis));
                next.emplace_back(std::move(result));
            }
            scanned = std::move(next);
        }
        _artifact.chunked_scan_operations++;
        return scanned.front();
    }

    // Partition immutable values only, never programs or memory operations.
    // Independent axes have no carries between partitions. The whole source
    // and result snapshots remain live according to the compiler's allocation.
    [[nodiscard]] luisa::optional<luisa::string> _partition_collective(
        const luisa::string &input, const IndexSpace &space,
        luisa::span<const uint32_t> collective_axes, const Operation &op,
        tile::CollectiveKind kind) noexcept {
        auto admitted = _partition_collectives.find(op.id());
        if (_independent_axis_extent == 0u || admitted == _partition_collectives.end() ||
            admitted->second != kind || op.result(0u)->type().scalar_type() != ScalarType::FLOAT32) { return {}; }
        // A bounded diagnostic code-size budget, not a profitability estimate.
        constexpr auto kMaxPartitions = uint64_t{16u};
        luisa::optional<uint32_t> selected;
        auto selected_extent = uint64_t{0u};
        for (auto i = 0u; i < space.rank(); i++) {
            if (std::find(collective_axes.begin(), collective_axes.end(), i) != collective_axes.end()) { continue; }
            auto &&extent = space.axis(i).extent;
            if (!extent.is_constant()) { continue; }
            auto size = extent.constant_value();
            if (!std::has_single_bit(size) || size <= _independent_axis_extent ||
                size % _independent_axis_extent != 0u || size / _independent_axis_extent > kMaxPartitions) { continue; }
            if (size > selected_extent) {
                selected = i;
                selected_extent = size;
            }
        }
        if (!selected) { return {}; }
        auto axis = *selected;
        auto partitions = selected_extent / _independent_axis_extent;
        luisa::vector<uint64_t> extents;
        for (auto i = 0u; i < space.rank(); i++) {
            extents.emplace_back(i == axis ? _independent_axis_extent : space.axis(i).extent.constant_value());
        }
        auto shape = _shape(extents);
        auto prefix = luisa::format("independent{}_", op.id());
        luisa::vector<luisa::string> results;
        for (auto partition = uint64_t{0u}; partition < partitions; partition++) {
            luisa::string indices;
            for (auto i = 0u; i < space.rank(); i++) {
                // extract takes partition coordinates, not element offsets.
                indices += luisa::format(", {}ull", i == axis ? partition : 0u);
            }
            auto extracted = luisa::format("{}input{}", prefix, partition);
            _line(luisa::format("auto {} = ct::extract({}, {}{{}}{});", extracted, input, shape, indices));
            auto expression = extracted;
            for (auto collective_axis : collective_axes) {
                if (kind == tile::CollectiveKind::INCLUSIVE_SUM) {
                    expression = luisa::format("ct::partial_sum({}, ct::integral_constant<{}>{{}}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}}, ct::scan_forward_t{{}})", expression, collective_axis);
                } else if (kind == tile::CollectiveKind::SUM) {
                    expression = luisa::format("ct::sum({}, ct::integral_constant<{}>{{}}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", expression, collective_axis);
                } else {
                    expression = luisa::format("ct::reduce_{}({}, ct::integral_constant<{}>{{}}, ct::suppress_nan_t{{}}, ct::preserve_subnormals_t{{}})", kind == tile::CollectiveKind::MINIMUM ? "min" : "max", expression, collective_axis);
                }
            }
            auto result = luisa::format("{}result{}", prefix, partition);
            _line(luisa::format("auto {} = {};", result, expression));
            results.emplace_back(std::move(result));
        }
        // cat requires identical shapes. Pairwise concatenation doubles only
        // the selected axis; reduced axes retain their singleton extents.
        for (auto level = 0u; results.size() > 1u; level++) {
            luisa::vector<luisa::string> next;
            for (auto i = size_t{0u}; i < results.size(); i += 2u) {
                auto result = luisa::format("{}join{}_{}", prefix, level, i / 2u);
                _line(luisa::format("auto {} = ct::cat({}, {}, ct::integral_constant<{}>{{}});", result, results[i], results[i + 1u], axis));
                next.emplace_back(std::move(result));
            }
            results = std::move(next);
        }
        _artifact.partitioned_collective_operations++;
        return std::move(results.front());
    }

    // A prefix scan is a closed sum whose contribution at k is selected by
    // k <= output_coordinate. Tree permission is required; fold policies keep
    // their scalar sequence even when their current inputs happen to agree.
    [[nodiscard]] bool _scan_tree(const Operation &op) noexcept {
        if (_map_space == nullptr || op.reduction_policy() != ReductionPolicy::UNORDERED_TREE ||
            op.domain()->rank() != 1u || op.operand_count() != 1u ||
            !_tree_scalar(op.result(0u)->type()) || !_tree_identity(op.operand(0u), ElementwiseOp::ADD)) { return false; }
        auto body = op.region(0u)->block(0u);
        if (body->operations().empty() || body->operations().back()->kind() != OperationKind::YIELD) { return false; }
        auto yield = body->operations().back();
        auto merge = yield->operand_count() == 1u ? yield->operand(0u)->defining_operation() : nullptr;
        if (merge == nullptr || merge->kind() != OperationKind::ELEMENTWISE || merge->elementwise_op() != ElementwiseOp::ADD) { return false; }
        auto carry = body->argument(1u);
        auto term = merge->operand(0u) == carry ? merge->operand(1u) : merge->operand(1u) == carry ? merge->operand(0u) :
                                                                                                     nullptr;
        auto select = term != nullptr ? term->defining_operation() : nullptr;
        if (select == nullptr || select->kind() != OperationKind::ELEMENTWISE || select->elementwise_op() != ElementwiseOp::SELECT ||
            !_tree_identity(select->operand(2u), ElementwiseOp::ADD)) { return false; }
        auto predicate = select->operand(0u)->defining_operation();
        auto extract = select->operand(1u)->defining_operation();
        if (predicate == nullptr || predicate->kind() != OperationKind::ELEMENTWISE || predicate->elementwise_op() != ElementwiseOp::LE ||
            predicate->operand(0u) != body->argument(0u) || extract == nullptr || extract->kind() != OperationKind::TILE_EXTRACT) { return false; }
        auto &&source = *extract->operand(0u)->type().index_space();
        if (source.rank() != _map_space->rank()) { return false; }
        luisa::optional<uint32_t> scan_axis;
        for (auto i = 0u; i < source.rank(); i++) {
            auto mapped = _map_space->axis_index(source.axis(i).dimension);
            if (!mapped || source.axis(i).extent != _map_space->axis(*mapped).extent) { return false; }
            auto coordinate = extract->operand(i + 1u);
            if (coordinate == body->argument(0u)) {
                if (scan_axis || source.axis(i).extent != op.domain()->axis(0u).extent ||
                    predicate->operand(1u) != _map_body->argument(*mapped)) { return false; }
                scan_axis = i;
            } else if (coordinate != _map_body->argument(*mapped)) {
                return false;
            }
        }
        if (!scan_axis) { return false; }
        auto padding = select->operand(2u)->defining_operation();
        for (auto operation : body->operations()) {
            if (operation != yield && operation != merge && operation != select && operation != predicate && operation != extract && operation != padding) { return false; }
            if (operation->execution_scope_constraint() || operation->resource_class_constraint() || operation->memory_layout()) { return false; }
        }
        auto element = op.result(0u)->type().scalar_type();
        auto signed_integer = element == ScalarType::INT32 || element == ScalarType::INT64;
        auto expression = _value(extract->operand(0u));
        if (signed_integer) {
            expression = luisa::format("ct::element_bitcast<{}>({})", element == ScalarType::INT32 ? "unsigned" : "unsigned long long", expression);
        }
        uint32_t collective_axes[]{*scan_axis};
        if (element == ScalarType::FLOAT32 && _chunked_scan_operations.contains(op.id())) {
            expression = _chunked_scan(std::move(expression), source, *scan_axis, op);
        } else if (auto partitioned = _partition_collective(expression, source, collective_axes, op, tile::CollectiveKind::INCLUSIVE_SUM)) {
            expression = std::move(*partitioned);
        } else {
            expression = element == ScalarType::FLOAT32 ?
                             luisa::format("ct::partial_sum({}, ct::integral_constant<{}>{{}}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}}, ct::scan_forward_t{{}})", expression, *scan_axis) :
                             luisa::format("ct::partial_sum({}, ct::integral_constant<{}>{{}}, ct::scan_forward_t{{}})", expression, *scan_axis);
        }
        if (signed_integer) { expression = luisa::format("ct::element_bitcast<{}>({})", _element(op.result(0u)->type()), expression); }
        expression = _align(std::move(expression), source, *_map_space, op);
        // Float +0 has a signed-zero effect; the matched integer zero is exact.
        if (element == ScalarType::FLOAT32) {
            expression = luisa::format("ct::add({}, {}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", _value(op.operand(0u)), expression);
        }
        _bind(op.result(0u), expression);
        _mapped_values.emplace(op.result(0u));
        return true;
    }
    // Prove a closed one-state reducer over an already materialized Tile.
    // Only an unordered tree with its declared identity uses a native tree;
    // every other region retains the ordered scalar contribution sequence.
    [[nodiscard]] bool _reduction_tree(const Operation &op) noexcept {
        if (_map_space == nullptr || op.reduction_policy() != ReductionPolicy::UNORDERED_TREE ||
            op.operand_count() != 1u || !_tree_scalar(op.result(0u)->type())) { return false; }
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
        auto opcode = merge->elementwise_op();
        if (!_tree_identity(op.operand(0u), opcode)) { return false; }
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
        auto element = op.result(0u)->type().scalar_type();
        auto signed_sum = opcode == ElementwiseOp::ADD && (element == ScalarType::INT32 || element == ScalarType::INT64);
        auto expression = _value(extract->operand(0u));
        if (signed_sum) {
            expression = luisa::format("ct::element_bitcast<{}>({})", element == ScalarType::INT32 ? "unsigned" : "unsigned long long", expression);
        }
        auto kind = opcode == ElementwiseOp::ADD ? tile::CollectiveKind::SUM :
                    opcode == ElementwiseOp::MIN ? tile::CollectiveKind::MINIMUM :
                                                   tile::CollectiveKind::MAXIMUM;
        if (auto partitioned = _partition_collective(expression, source, axes, op, kind)) {
            expression = std::move(*partitioned);
        } else {
            for (auto axis : axes) {
                if (element == ScalarType::FLOAT32) {
                    expression = opcode == ElementwiseOp::ADD ?
                                     luisa::format("ct::sum({}, ct::integral_constant<{}>{{}}, ct::round_ties_to_even_t{{}}, ct::preserve_subnormals_t{{}})", expression, axis) :
                                     luisa::format("ct::reduce_{}({}, ct::integral_constant<{}>{{}}, ct::suppress_nan_t{{}}, ct::preserve_subnormals_t{{}})", opcode == ElementwiseOp::MIN ? "min" : "max", expression, axis);
                } else {
                    auto name = opcode == ElementwiseOp::ADD ? "sum" : opcode == ElementwiseOp::MIN ? "reduce_min" :
                                                                                                      "reduce_max";
                    expression = luisa::format("ct::{}({}, ct::integral_constant<{}>{{}})", name, expression, axis);
                }
            }
        }
        if (signed_sum) { expression = luisa::format("ct::element_bitcast<{}>({})", _element(op.result(0u)->type()), expression); }
        expression = luisa::format("ct::reshape({}, {}{{}})", expression, _shape(remaining));
        expression = _align(std::move(expression), remaining, *_map_space, op);
        if (element == ScalarType::FLOAT32) {
            auto name = opcode == ElementwiseOp::ADD ? "add" : opcode == ElementwiseOp::MIN ? "min" :
                                                                                              "max";
            auto policy = opcode == ElementwiseOp::ADD ? "ct::round_ties_to_even_t{}" : "ct::suppress_nan_t{}";
            expression = luisa::format("ct::{}({}, {}, {}, ct::preserve_subnormals_t{{}})", name, _value(op.operand(0u)), expression, policy);
        }
        _bind(op.result(0u), expression);
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
        if (op.kind() == OperationKind::REDUCE && (_scan_tree(op) || _reduction_tree(op))) { return; }
        if (parallel) {
            if (domain.rank() > 3u) {
                _fail(&op, "native CUDA launch grids support only rank 1..3");
                return;
            }
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
                _index_facts.emplace(body->argument(i), IndexFacts{0u, n - 1u});
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
            if (_map_space == nullptr) {
                auto n = domain.axis(i).extent.constant_value();
                _index_facts.emplace(body->argument(i), IndexFacts{0u, n - 1u});
            }
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
            _record_index_facts(*op);
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
    explicit Emitter(const tile::Function &function, bool enable_fast_math, bool enable_aligned16,
                     uint32_t worker_warps, uint32_t target_sm, uint32_t scan_chunk_extent, uint32_t independent_axis_extent) noexcept
        : _function{function}, _enable_fast_math{enable_fast_math}, _enable_aligned16{enable_aligned16},
          _worker_warps{worker_warps}, _target_sm{target_sm},
          _scan_chunk_extent{scan_chunk_extent}, _independent_axis_extent{independent_axis_extent} {}
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
        if (_scan_chunk_extent != 0u && _independent_axis_extent != 0u) {
            _fail(nullptr, "experimental collective realizations must be measured separately");
            return std::move(_artifact);
        }
        if (_scan_chunk_extent != 0u || _independent_axis_extent != 0u) {
            if (_scan_chunk_extent != 0u && _scan_chunk_extent != 1024u && _scan_chunk_extent != 2048u) {
                _fail(nullptr, "experimental scan chunks require 1024 or 2048 elements");
                return std::move(_artifact);
            }
            if (_independent_axis_extent != 0u && !std::has_single_bit(_independent_axis_extent)) {
                _fail(nullptr, "experimental independent-axis extent must be a power of two");
                return std::move(_artifact);
            }
            // Both candidates consume the same target-independent admission.
            // Analyze once; the zero/default path neither analyzes nor changes source.
            auto analysis = tile::analyze_collective_work(_function);
            if (!analysis.ok()) {
                _fail(nullptr, luisa::format("experimental collective planning rejected: {}", analysis.error));
                return std::move(_artifact);
            }
            for (auto &&work : analysis.collectives) {
                if (work.element != ScalarType::FLOAT32) { continue; }
                if (_independent_axis_extent != 0u) {
                    _partition_collectives.emplace(work.operation_id, work.kind);
                }
                // Bound generated code, independently of a future profitability model.
                constexpr auto kMaxScanChunks = uint64_t{16u};
                if (_scan_chunk_extent != 0u && work.kind == tile::CollectiveKind::INCLUSIVE_SUM &&
                    work.contribution_extent > _scan_chunk_extent &&
                    work.contribution_extent % _scan_chunk_extent == 0u &&
                    work.contribution_extent / _scan_chunk_extent <= kMaxScanChunks &&
                    std::has_single_bit(work.contribution_extent / _scan_chunk_extent)) {
                    _chunked_scan_operations.emplace(work.operation_id);
                }
            }
            if (_scan_chunk_extent != 0u && _chunked_scan_operations.empty()) {
                _fail(nullptr, "experimental scan planning found no divisible wider FP32 prefix within the code-size budget");
                return std::move(_artifact);
            }
            _artifact.scan_chunk_extent = _scan_chunk_extent;
            _artifact.independent_axis_extent = _independent_axis_extent;
        }
        if (_worker_warps != 0u &&
            ((_worker_warps != 4u && _worker_warps != 8u) || _target_sm != 89u)) {
            _fail(nullptr, "worker-warp calibration requires SM89 and 4 or 8 worker warps");
            return std::move(_artifact);
        }
        _artifact.source = "#include <cuda_tile.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\nnamespace ct = cuda::tiles;\n";
        _artifact.source += _worker_warps == 0u ? "extern \"C\" __tile_global__ void luisa_tile_main(" :
                                                  luisa::format("extern \"C\" {{\n[[cutile::hint({}, num_worker_warps_per_cta = {})]]\n__tile_global__ void luisa_tile_main(", _target_sm * 10u, _worker_warps);
        if (body->argument_count() > 31u) { _fail(nullptr, "buffer argument count exceeds existing CUDAShaderTile ABI"); }
        for (auto i = 0u; i < body->argument_count() && _artifact.error.empty(); i++) {
            auto arg = body->argument(i);
            auto &&type = arg->type();
            auto element = type.scalar_type();
            auto supported_element = element == ScalarType::BOOL || element == ScalarType::INT32 || element == ScalarType::UINT32 ||
                                     element == ScalarType::INT64 || element == ScalarType::UINT64 || element == ScalarType::FLOAT32 ||
                                     element == ScalarType::FLOAT16 || element == ScalarType::BFLOAT16;
            if (!type.is_view() || !supported_element || !_space(*type.index_space(), false, nullptr)) {
                _fail(nullptr, "kernel parameters must be statically sized contiguous bool/integer/FP32/FP16/BF16 buffer views");
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
        if (_artifact.error.empty() && _scan_chunk_extent != 0u &&
            _artifact.chunked_scan_operations != _chunked_scan_operations.size()) {
            _fail(nullptr, "experimental scan plan was not fully realized");
        }
        if (_artifact.error.empty() && _independent_axis_extent != 0u &&
            _artifact.partitioned_collective_operations == 0u) {
            _fail(nullptr, "experimental independent-axis planning found no supported non-collective partition");
        }
        _artifact.source += "}\n";
        if (_worker_warps != 0u) { _artifact.source += "}\n"; }
        if (!_artifact.error.empty()) { _artifact.source.clear(); }
        auto mask = _aligned16_seen & ~_aligned16_rejected;
        if (_artifact.ok() && mask != 0u) {
            _artifact.aligned16_entry = "luisa_tile_aligned16";
            _artifact.aligned16_buffer_mask = mask;
            // Keep the entire original entry byte-for-byte. Same-entry
            // guarded assumptions can affect the compiler's fallback branch,
            // so the host chooses between two separate entry functions.
            auto begin = _artifact.source.find("extern \"C\" ");
            auto aligned = _artifact.source.substr(begin);
            // Apply only to the separate aligned entry after all accesses have
            // contributed to the root eligibility mask. Keep load positions,
            // immutable SSA snapshots and every store/arithmetic line unchanged.
            // Reverse order keeps the recorded original source offsets stable.
            for (auto i = _aligned_view_loads.size(); i != 0u; i--) {
                auto &&load = _aligned_view_loads[i - 1u];
                if ((mask & (uint32_t{1u} << load.argument_index)) != 0u) {
                    aligned.replace(load.offset - begin, load.length, load.replacement);
                    _artifact.aligned16_partition_loads++;
                }
            }
            auto entry = aligned.find(_artifact.entry);
            aligned.replace(entry, _artifact.entry.size(), _artifact.aligned16_entry);
            auto body = aligned.find(") {\n") + 4u;
            luisa::string assumptions;
            for (auto i = 0u; i < _artifact.arguments.size(); i++) {
                if ((mask & (uint32_t{1u} << i)) != 0u) {
                    assumptions += luisa::format("    buffer{} = ct::assume_aligned<16>(buffer{});\n", i, i);
                }
            }
            aligned.insert(body, assumptions);
            _artifact.source += '\n';
            _artifact.source += aligned;
        }
        return std::move(_artifact);
    }
};
}// namespace

Artifact generate(const tile::Function &function, bool enable_fast_math, bool enable_aligned16,
                  uint32_t worker_warps, uint32_t target_sm, uint32_t scan_chunk_extent, uint32_t independent_axis_extent) noexcept {
    return Emitter{function, enable_fast_math, enable_aligned16, worker_warps, target_sm, scan_chunk_extent, independent_axis_extent}.run();
}
}// namespace luisa::compute::cuda::native_tile
