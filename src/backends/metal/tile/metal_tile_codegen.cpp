#include "metal_tile_codegen.h"
#include <luisa/tile/verifier.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <algorithm>
#include <limits>

namespace luisa::compute::metal {
namespace {

using tile::Operation;
using tile::OperationKind;
using tile::Value;

struct IndexValue {
    luisa::string expression;
    int64_t minimum{0};
    int64_t maximum{0};
};

// A target realization, not a name-based kernel substitution. Each record
// retains its exact semantic store/MMA/load identities until source emission.
struct MppAtom {
    const Operation *store;
    const Operation *mma;
    const Operation *a;
    const Operation *b;
    uint32_t participants;
    uint64_t rows;
    uint64_t columns;
    uint64_t reduction;
    bool transpose_a;
    bool transpose_b;
};

class Lowering {
private:
    const tile::Function &_function;
    tile::CompileOptions _options;
    uint32_t _max_threads;
    MetalTileCode _result;
    luisa::unordered_map<const Value *, IndexValue> _indices;
    luisa::vector<MppAtom> _atoms;
    luisa::string _body;
    uint64_t _groups{0u};
    uint32_t _subgroups{0u};
    size_t _root_count{0u};

    void _error(const Operation *op, luisa::string_view message) noexcept {
        if (_result.metadata.error.empty()) {
            _result.metadata.error = op == nullptr ? luisa::string{message} :
                                                     luisa::format("Tile operation {} ({}): {}", op->id(), op->name(), message);
        }
    }

    [[nodiscard]] static uint64_t _extent(const tile::IndexSpace &space, size_t axis) noexcept {
        return space.axis(axis).extent.constant_value();
    }

    [[nodiscard]] static bool _matrix_type(const tile::Type &type) noexcept {
        auto space = type.index_space();
        if (type.scalar_type() != tile::ScalarType::FLOAT32 || space == nullptr || space->rank() != 2u) { return false; }
        for (auto &axis : space->axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u ||
                axis.extent.constant_value() > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) { return false; }
        }
        return true;
    }

    [[nodiscard]] static bool _zero(const Value *value) noexcept {
        auto op = value->defining_operation();
        if (op == nullptr || op->kind() != OperationKind::CONSTANT) { return false; }
        auto attribute = op->attribute("value");
        return attribute != nullptr && luisa::holds_alternative<double>(attribute->value()) &&
               luisa::get<double>(attribute->value()) == 0.0;
    }

    [[nodiscard]] static bool _index_type(const tile::Type &type) noexcept {
        return type.kind() == tile::TypeKind::INDEX ||
               (type.kind() == tile::TypeKind::SCALAR && type.scalar_type() == tile::ScalarType::INT64);
    }

    [[nodiscard]] const IndexValue *_index(const Value *value, const Operation *user) noexcept {
        auto it = _indices.find(value);
        if (it == _indices.end()) {
            _error(user, "expected a supported, dominating integer coordinate");
            return nullptr;
        }
        return &it->second;
    }

    [[nodiscard]] bool _zero_index(const Value *value) const noexcept {
        auto it = _indices.find(value);
        return it != _indices.end() && it->second.minimum == 0 && it->second.maximum == 0;
    }

    void _constant(const Operation *op) noexcept {
        auto value = op->result(0u);
        if (value->type().is_tile()) {
            if (!_matrix_type(value->type()) || !_zero(value)) { _error(op, "only zero FP32 matrix accumulators are supported"); }
            return;
        }
        auto attribute = op->attribute("value");
        if (!_index_type(value->type()) || attribute == nullptr || !luisa::holds_alternative<int64_t>(attribute->value())) {
            _error(op, "only signed 64-bit coordinate constants are supported");
            return;
        }
        auto x = luisa::get<int64_t>(attribute->value());
        _indices.emplace(value, IndexValue{luisa::format("{}L", x), x, x});
    }

    void _arithmetic(const Operation *op) noexcept {
        if (op->operand_count() != 2u || !_index_type(op->result(0u)->type())) {
            _error(op, "only binary signed 64-bit coordinate arithmetic is supported");
            return;
        }
        auto a = _index(op->operand(0u), op);
        auto b = _index(op->operand(1u), op);
        if (a == nullptr || b == nullptr) { return; }
        int64_t lo{}, hi{};
        const char *symbol = nullptr;
        auto overflow = false;
        switch (op->elementwise_op()) {
            case tile::ElementwiseOp::ADD:
                symbol = "+";
                overflow = __builtin_add_overflow(a->minimum, b->minimum, &lo) || __builtin_add_overflow(a->maximum, b->maximum, &hi);
                break;
            case tile::ElementwiseOp::SUB:
                symbol = "-";
                overflow = __builtin_sub_overflow(a->minimum, b->maximum, &lo) || __builtin_sub_overflow(a->maximum, b->minimum, &hi);
                break;
            case tile::ElementwiseOp::MUL: {
                symbol = "*";
                int64_t products[4]{};
                overflow = __builtin_mul_overflow(a->minimum, b->minimum, &products[0]) ||
                           __builtin_mul_overflow(a->minimum, b->maximum, &products[1]) ||
                           __builtin_mul_overflow(a->maximum, b->minimum, &products[2]) ||
                           __builtin_mul_overflow(a->maximum, b->maximum, &products[3]);
                lo = *std::min_element(std::begin(products), std::end(products));
                hi = *std::max_element(std::begin(products), std::end(products));
                break;
            }
            default: _error(op, "unsupported coordinate operation"); return;
        }
        if (overflow) {
            _error(op, "coordinate range may overflow signed 64-bit arithmetic");
            return;
        }
        _indices.emplace(op->result(0u), IndexValue{luisa::format("({} {} {})", a->expression, symbol, b->expression), lo, hi});
    }

    [[nodiscard]] bool _argument(const Value *value) const noexcept {
        return value->origin() == Value::Origin::BLOCK_ARGUMENT && value->argument_block() == _function.body().block(0u);
    }

    void _load(const Operation *op) noexcept {
        if (op->operand_count() != 3u || !op->domain() || !_argument(op->operand(0u)) ||
            !_matrix_type(op->result(0u)->type()) || op->result(0u)->use_count() != 1u) {
            _error(op, "MPP operands must be single-use rank-two FP32 argument Tile loads without custom padding");
            return;
        }
        auto user = op->result(0u)->use_list().begin();
        if ((*user)->user()->kind() != OperationKind::MMA) {
            _error(op, "only MMA consumes forwarded Tile loads in this realization");
            return;
        }
        auto &arg = _result.metadata.arguments[op->operand(0u)->index()];
        arg.usage = static_cast<Usage>(to_underlying(arg.usage) | to_underlying(Usage::READ));
    }

    void _store(const Operation *op, uint32_t participants) noexcept {
        if (participants == 0u || op->operand_count() != 4u || !op->domain() || !_argument(op->operand(0u))) {
            _error(op, "expected a rank-two argument Tile store in a group or subgroup");
            return;
        }
        auto mma = op->operand(3u)->defining_operation();
        if (mma == nullptr || mma->kind() != OperationKind::MMA || mma->result(0u)->use_count() != 1u ||
            mma->parent_block() != op->parent_block() || !_matrix_type(mma->result(0u)->type()) ||
            !mma->mma_policy().allow_reassociation || !_zero(mma->operand(2u))) {
            _error(op, "expected a sole-use, same-scope FP32 MMA with zero initial value and permitted reassociation");
            return;
        }
        auto a = mma->operand(0u)->defining_operation();
        auto b = mma->operand(1u)->defining_operation();
        if (a == nullptr || b == nullptr || a->kind() != OperationKind::VIEW_LOAD || b->kind() != OperationKind::VIEW_LOAD ||
            a->parent_block() != mma->parent_block() || b->parent_block() != mma->parent_block()) {
            _error(mma, "MPP input snapshots must be direct loads in the MMA scope");
            return;
        }
        auto &out = *mma->result(0u)->type().index_space();
        auto &as = *a->result(0u)->type().index_space();
        auto &bs = *b->result(0u)->type().index_space();
        auto am = as.axis_index(out.axis(0u).dimension);
        auto bn = bs.axis_index(out.axis(1u).dimension);
        if (!am || !bn || *op->domain() != out) {
            _error(mma, "MMA/store dimensions require explicit reindexing");
            return;
        }
        auto ak = 1u - *am;
        auto bk = 1u - *bn;
        auto &av = *a->operand(0u)->type().index_space();
        auto &bv = *b->operand(0u)->type().index_space();
        auto &cv = *op->operand(0u)->type().index_space();
        auto rows = _extent(out, 0u), columns = _extent(out, 1u), reduction = _extent(as, ak);
        if (rows % 8u != 0u || columns % 8u != 0u || (rows % 16u != 0u && columns % 16u != 0u)) {
            _error(mma, "MPP tile M/N must be multiples of 8, with at least one a multiple of 16");
            return;
        }
        if (as.axis(ak).dimension != bs.axis(bk).dimension || _extent(bs, bk) != reduction ||
            _extent(av, ak) != reduction || _extent(bv, bk) != reduction ||
            !_zero_index(a->operand(ak + 1u)) || !_zero_index(b->operand(bk + 1u))) {
            _error(mma, "this MPP realization requires a complete K dimension at origin zero; K pipelines are not lowered yet");
            return;
        }
        auto m0 = _index(op->operand(1u), op), n0 = _index(op->operand(2u), op);
        auto a0 = _index(a->operand(*am + 1u), a), b0 = _index(b->operand(*bn + 1u), b);
        if (!m0 || !n0 || !a0 || !b0) { return; }
        if (m0->expression != a0->expression || n0->expression != b0->expression ||
            _extent(av, *am) != _extent(cv, 0u) || _extent(bv, *bn) != _extent(cv, 1u) ||
            m0->minimum < 0 || n0->minimum < 0 || m0->maximum > INT32_MAX || n0->maximum > INT32_MAX) {
            _error(op, "operand and output free-axis origins/bounds must agree and fit nonnegative int32");
            return;
        }
        _atoms.emplace_back(MppAtom{op, mma, a, b, participants, rows, columns, reduction, *am == 1u, *bn == 0u});
        auto &arg = _result.metadata.arguments[op->operand(0u)->index()];
        arg.usage = static_cast<Usage>(to_underlying(arg.usage) | to_underlying(Usage::WRITE));
        _body += luisa::format("  tile_atom_{}(args, int({}), int({}));\n", mma->id(), m0->expression, n0->expression);
    }

    void _parallel(const Operation *op, uint32_t depth) noexcept {
        if (!op->domain() || op->region_count() != 1u || op->region(0u)->block_count() != 1u ||
            op->operand_count() != 0u || op->result_count() != 0u) {
            _error(op, "only independent parallel nests without carried results are supported");
            return;
        }
        auto volume = op->domain()->static_volume();
        if (!volume || *volume == 0u || *volume > UINT32_MAX) {
            _error(op, "execution domain must have a positive uint32 static volume");
            return;
        }
        auto constraint = op->execution_scope_constraint();
        auto scope = constraint ? luisa::string_view{*constraint} : luisa::string_view{};
        const char *coordinate = nullptr;
        uint32_t participants{};
        if (depth == 0u && (scope.empty() || scope == "group")) {
            if (++_root_count != 1u) {
                _error(op, "multiple root launches require a later multi-dispatch planner");
                return;
            }
            _groups = *volume;
            coordinate = "long(group.x)";
            // Resolve the complete group only after visiting its child nests.
            participants = UINT32_MAX;
        } else if (depth == 1u && scope == "subgroup") {
            if (*volume > _max_threads / 32u || (_subgroups != 0u && _subgroups != *volume)) {
                _error(op, "subgroup cohort exceeds the target or conflicts with another cohort");
                return;
            }
            _subgroups = static_cast<uint32_t>(*volume);
            coordinate = "long(subgroup)";
            participants = 1u;
        } else {
            _error(op, "supported mapping is root group with optional explicit subgroup child");
            return;
        }
        auto block = op->region(0u)->block(0u);
        if (block->argument_count() != op->domain()->rank()) {
            _error(op, "unexpected parallel argument schema");
            return;
        }
        auto stride = *volume;
        for (auto i = size_t{0}; i < op->domain()->rank(); i++) {
            auto extent = _extent(*op->domain(), i);
            stride /= extent;
            _indices.emplace(block->argument(i), IndexValue{luisa::format("(({} / {}L) % {}L)", coordinate, stride, extent), 0, static_cast<int64_t>(extent - 1u)});
        }
        _block(block, depth + 1u, participants);
    }

    void _block(const tile::Block *block, uint32_t depth, uint32_t participants) noexcept {
        for (auto op : block->operations()) {
            if (!_result.ok()) { return; }
            switch (op->kind()) {
                case OperationKind::CONSTANT: _constant(op); break;
                case OperationKind::ELEMENTWISE: _arithmetic(op); break;
                case OperationKind::PARALLEL: _parallel(op, depth); break;
                case OperationKind::VIEW_LOAD: _load(op); break;
                case OperationKind::VIEW_STORE: _store(op, participants); break;
                case OperationKind::MMA:
                    if (op->result(0u)->use_count() != 1u) { _error(op, "MMA must have one explicit output store"); }
                    break;
                case OperationKind::YIELD:
                    if (op->operand_count() != 0u) { _error(op, "carried/yielded values need a later realization"); }
                    break;
                default: _error(op, "unsupported by native MPP lowering; no implicit fallback or effect erasure"); break;
            }
        }
    }

    [[nodiscard]] luisa::string _tensor(const Value *argument, luisa::string_view name) const noexcept {
        auto &space = *argument->type().index_space();
        return luisa::format("  tensor<device float, dextents<int, 2>, tensor_inline> {}(args.b{}.data, dextents<int, 2>({}, {}), array<int, 2>{{1, {}}});\n",
                             name, argument->index(), _extent(space, 1u), _extent(space, 0u), _extent(space, 1u));
    }

    void _emit() noexcept {
        auto &src = _result.metadata.source;
        src = "#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp::tensor_ops;\nstruct alignas(16) TileBuffer { device float *data; ulong size; };\nstruct Arguments {\n";
        for (auto i = size_t{0}; i < _result.metadata.arguments.size(); i++) { src += luisa::format("  TileBuffer b{};\n", i); }
        src += "};\n";
        for (auto &atom : _atoms) {
            auto id = atom.mma->id();
            auto &space = *atom.store->operand(0u)->type().index_space();
            auto m = _extent(space, 0u), n = _extent(space, 1u);
            src += luisa::format("template<typename A, typename B, typename C>\nvoid tile_mma_{}(thread A &a, thread B &b, thread C &c) {{\n", id);
            src += luisa::format("  constexpr auto desc = matmul2d_descriptor({}, {}, dynamic_length_v<int>, {}, {}, false);\n  matmul2d<desc, execution_simdgroups<{}>> op;\n", atom.rows, atom.columns, atom.transpose_a, atom.transpose_b, atom.participants);
            src += "  auto result = op.get_destination_cooperative_tensor<A, B, float>();\n  op.run(a, b, result);\n  result.store(c);\n}\n";
            src += luisa::format("void tile_atom_{}(constant Arguments &args, int m0, int n0) {{\n", id);
            src += luisa::format("  if (m0 >= {} || n0 >= {}) {{ return; }}\n", m, n);
            src += _tensor(atom.a->operand(0u), "a") + _tensor(atom.b->operand(0u), "b") + _tensor(atom.store->operand(0u), "c");
            auto ax = atom.transpose_a ? "m0" : "0", ay = atom.transpose_a ? "0" : "m0";
            auto bx = atom.transpose_b ? "0" : "n0", by = atom.transpose_b ? "n0" : "0";
            src += luisa::format("  if (m0 <= {}L - {}L && n0 <= {}L - {}L) {{\n", m, atom.rows, n, atom.columns);
            src += luisa::format("    auto aa = a.slice<{}, {}>({}, {});\n", atom.transpose_a ? luisa::format("{}", atom.rows) : "dynamic_extent", atom.transpose_a ? "dynamic_extent" : luisa::format("{}", atom.rows), ax, ay);
            src += luisa::format("    auto bb = b.slice<{}, {}>({}, {});\n", atom.transpose_b ? "dynamic_extent" : luisa::format("{}", atom.columns), atom.transpose_b ? luisa::format("{}", atom.columns) : "dynamic_extent", bx, by);
            src += luisa::format("    auto cc = c.slice<{}, {}>(n0, m0);\n    tile_mma_{}(aa, bb, cc);\n  }} else {{\n", atom.columns, atom.rows, id);
            src += luisa::format("    auto aa = a.slice({}, {});\n    auto bb = b.slice({}, {});\n    auto cc = c.slice(n0, m0);\n    tile_mma_{}(aa, bb, cc);\n  }}\n}}\n", ax, ay, bx, by, id);
        }
        src += "void tile_main(constant Arguments &args, uint3 group, uint subgroup) {\n" + _body + "}\n";
        src += "kernel void kernel_main(constant Arguments &args [[buffer(0)]], constant uint3 &ds [[buffer(1)]], uint3 group [[threadgroup_position_in_grid]], uint subgroup [[simdgroup_index_in_threadgroup]]) { tile_main(args, group, subgroup); }\n";
        src += "kernel void kernel_main_indirect(constant Arguments &args [[buffer(0)]], device uint4 &ds [[buffer(1)]], uint3 group [[threadgroup_position_in_grid]], uint subgroup [[simdgroup_index_in_threadgroup]]) { tile_main(args, group, subgroup); }\n";
    }

public:
    Lowering(const tile::Function &function, tile::CompileOptions options, uint32_t max_threads) noexcept
        : _function{function}, _options{options}, _max_threads{max_threads} {}

    [[nodiscard]] MetalTileCode run() noexcept {
        auto parent = _function.parent_module();
        auto attached = false;
        if (parent != nullptr) {
            for (auto function : parent->functions()) {
                if (function == &_function) {
                    attached = true;
                    break;
                }
            }
        }
        if (!attached) {
            _error(nullptr, "Tile function must belong to its owning module before verification/lowering");
            return std::move(_result);
        }
        auto verified = tile::verify(*parent);
        if (!verified) {
            _error(nullptr, verified.diagnostics().front().message);
            return std::move(_result);
        }
        if (_function.form() != tile::IRForm::CANDIDATE || _function.body().block_count() != 1u ||
            (_options.threads_per_group != 0u && (_options.threads_per_group % 32u != 0u || _options.threads_per_group > _max_threads))) {
            _error(nullptr, "expected Candidate TileIR and a legal whole-subgroup thread constraint");
            return std::move(_result);
        }
        auto entry = _function.body().block(0u);
        // MetalShader uses a 64 KiB, 16-byte-aligned root argument block.
        if (entry->argument_count() > 4096u) {
            _error(nullptr, "Metal root argument block exceeds 64 KiB");
            return std::move(_result);
        }
        for (auto &arg : entry->arguments()) {
            if (!arg->type().is_view() || !_matrix_type(arg->type())) {
                _error(nullptr, "native MPP arguments currently require dense rank-two FP32 TensorViews");
                return std::move(_result);
            }
            auto volume = arg->type().index_space()->static_volume();
            if (!volume || *volume > INT32_MAX || *volume > SIZE_MAX / sizeof(float)) {
                _error(nullptr, "dense MPP argument exceeds signed int32 element addressing");
                return std::move(_result);
            }
            _result.metadata.arguments.emplace_back(tile::KernelArgument{tile::ScalarType::FLOAT32, *volume * sizeof(float), Usage::NONE});
        }
        _block(entry, 0u, 0u);
        auto threads = _subgroups != 0u ? _subgroups * 32u : (_options.threads_per_group == 0u ? 128u : _options.threads_per_group);
        if (_subgroups != 0u && _options.threads_per_group != 0u && threads != _options.threads_per_group) { _error(nullptr, "thread constraint conflicts with explicit subgroup domain"); }
        if (_atoms.empty() || _groups == 0u || threads > _max_threads || _groups > UINT32_MAX / threads) { _error(nullptr, "empty/unsupported MPP program or dispatch capacity exceeded"); }
        for (auto &atom : _atoms) {
            if (atom.participants == UINT32_MAX) { atom.participants = threads / 32u; }
            if (atom.participants != 1u && atom.participants * 32u != threads) { _error(atom.mma, "MPP collective must use one subgroup or the complete threadgroup"); }
        }
        for (auto &arg : _result.metadata.arguments) {
            if (arg.usage == Usage::READ_WRITE) { _error(nullptr, "argument is both read and written; snapshot-safe forwarding is not proved"); }
        }
        if (_result.ok()) {
            _result.block_size = make_uint3(threads, 1u, 1u);
            _result.metadata.dispatch_size = make_uint3(static_cast<uint32_t>(_groups * threads), 1u, 1u);
            _result.metadata.disjoint_writes = true;
            _result.metadata.realization = luisa::format("Metal MPP: {} groups, {} threads/group, {} explicit subgroups, {} typed MMA atoms; inline dense views; no TVM", _groups, threads, _subgroups, _atoms.size());
            _emit();
        }
        return std::move(_result);
    }
};

}// namespace

MetalTileCode lower_tile_to_mpp(const tile::Function &function, const tile::CompileOptions &options, uint32_t max_threads) noexcept {
    return Lowering{function, options, max_threads}.run();
}

}// namespace luisa::compute::metal
