#include "cuda_tile_codegen.h"

#include <bit>
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
        if (space.rank() == 0u || space.rank() > 3u) {
            _fail(op, "only rank 1..3 spaces are supported");
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
    // Native trailing-axis broadcasting is not the Tile IR's named-axis rule.
    // Restrict Tile operands to exactly the result space; scalar broadcasting
    // is supported. MMA below has its own explicit named-axis permutation.
    [[nodiscard]] bool _elementwise_shape(const Operation &op) noexcept {
        auto &&result = op.result(0u)->type();
        for (auto i = 0u; i < op.operand_count(); i++) {
            auto &&operand = op.operand(i)->type();
            if (operand.is_tile() && (!result.is_tile() || *operand.index_space() != *result.index_space())) {
                _fail(&op, "elementwise named-axis broadcasting/permutation is not implemented; use equal spaces or scalar operands");
                return false;
            }
        }
        return true;
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
        if (!_elementwise_shape(op)) { return; }
        auto a = _value(op.operand(0u));
        auto b = op.operand_count() > 1u ? _value(op.operand(1u)) : luisa::string{};
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
            case ElementwiseOp::SELECT: {
                auto &&result_type = op.result(0u)->type();
                auto branch = [&](const Value *value) noexcept {
                    auto expression = _value(value);
                    if (result_type.is_tile() && !value->type().is_tile()) {
                        expression = luisa::format("ct::full<ct::tile<{}, {}>>({})", _element(result_type), _shape(*result_type.index_space()), expression);
                    }
                    return expression;
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
    void _structured(const Operation &op) noexcept {
        auto parallel = op.kind() == OperationKind::PARALLEL;
        auto &&domain = *op.domain();
        if (!_space(domain, false, &op) || op.region(0u)->block_count() != 1u) {
            _fail(&op, "structured regions require one body block");
            return;
        }
        auto body = op.region(0u)->block(0u);
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
        if (!_inside_parallel) {
            _fail(&op, "ordered loops outside the root PARALLEL are unsupported");
            return;
        }
        luisa::vector<luisa::string> carries;
        for (auto i = 0u; i < op.result_count(); i++) {
            _bind(op.result(i), _value(op.operand(i)));
            carries.emplace_back(_value(op.result(i)));
            _values.emplace(body->argument(domain.rank() + i), carries.back());
        }
        for (auto i = 0u; i < domain.rank(); i++) {
            auto name = _name(body->argument(i));
            _values.emplace(body->argument(i), name);
            _line(luisa::format("for (long long {} = 0ll; {} < {}ll; ++{}) {{", name, name, domain.axis(i).extent.constant_value(), name));
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
                op->kind() != OperationKind::ELEMENTWISE && op->kind() != OperationKind::YIELD) {
                _fail(op, "effects outside the root PARALLEL cannot be replicated per grid program");
                return;
            }
            switch (op->kind()) {
                case OperationKind::CONSTANT: _constant(*op); break;
                case OperationKind::ELEMENTWISE: _elementwise(*op); break;
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE: _view(*op); break;
                case OperationKind::MMA: _mma(*op); break;
                case OperationKind::PARALLEL:
                case OperationKind::SERIAL:
                case OperationKind::PIPELINE: _structured(*op); break;
                case OperationKind::STAGE: _line("// Stage boundary; ordered execution is a valid non-overlapped schedule."); break;
                case OperationKind::YIELD:
                    if (op->operand_count() != carries.size()) {
                        _fail(op, "yield arity does not match loop carries");
                        return;
                    }
                    for (auto i = 0u; i < carries.size(); i++) {
                        _line(luisa::format("auto yield{}_{} = {};", op->id(), i, _value(op->operand(i))));
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
            if (!type.is_view() || type.scalar_type() != ScalarType::FLOAT32 || !_space(*type.index_space(), false, nullptr)) {
                _fail(nullptr, "kernel parameters must be statically sized contiguous FP32 buffer views");
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
