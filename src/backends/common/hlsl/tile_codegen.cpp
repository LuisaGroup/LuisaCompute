#include "tile_codegen.h"

#include <luisa/ast/external_function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/verifier.h>

#include <algorithm>
#include <limits>

namespace lc::hlsl {
namespace {

using tile::Operation;
using tile::OperationKind;
using tile::Value;

struct IndexValue {
    luisa::string expression;
    int64_t minimum{0};
    int64_t maximum{0};
};

struct MmaAtom {
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
    bool shared{false};
    bool full{false};
};

[[nodiscard]] bool add_overflow(int64_t a, int64_t b, int64_t &result) noexcept {
    constexpr auto lo = std::numeric_limits<int64_t>::min();
    constexpr auto hi = std::numeric_limits<int64_t>::max();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) { return true; }
    result = a + b;
    return false;
}

[[nodiscard]] bool sub_overflow(int64_t a, int64_t b, int64_t &result) noexcept {
    constexpr auto lo = std::numeric_limits<int64_t>::min();
    constexpr auto hi = std::numeric_limits<int64_t>::max();
    if ((b > 0 && a < lo + b) || (b < 0 && a > hi + b)) { return true; }
    result = a - b;
    return false;
}

[[nodiscard]] bool mul_overflow(int64_t a, int64_t b, int64_t &result) noexcept {
    constexpr auto lo = std::numeric_limits<int64_t>::min();
    constexpr auto hi = std::numeric_limits<int64_t>::max();
    if (a == 0 || b == 0) {
        result = 0;
        return false;
    }
    if ((a == -1 && b == lo) || (b == -1 && a == lo)) { return true; }
    if (a > 0) {
        if ((b > 0 && a > hi / b) || (b < 0 && b < lo / a)) { return true; }
    } else {
        if ((b > 0 && a < lo / b) || (b < 0 && a < hi / b)) { return true; }
    }
    result = a * b;
    return false;
}

class Lowering {
private:
    const tile::Function &_function;
    tile::CompileOptions _options;
    HlslTileTargetConfig _target;
    HlslTileCode _result;
    luisa::unordered_map<const Value *, IndexValue> _indices;
    luisa::vector<MmaAtom> _atoms;
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
            _error(op, "only signed coordinate constants are supported");
            return;
        }
        auto x = luisa::get<int64_t>(attribute->value());
        if (x < std::numeric_limits<int32_t>::min() || x > std::numeric_limits<int32_t>::max()) {
            _error(op, "coordinate constant exceeds signed 32-bit HLSL arithmetic");
            return;
        }
        _indices.emplace(value, IndexValue{luisa::format("{}", x), x, x});
    }

    void _arithmetic(const Operation *op) noexcept {
        if (op->operand_count() != 2u || !_index_type(op->result(0u)->type())) {
            _error(op, "only binary signed coordinate arithmetic is supported");
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
                overflow = add_overflow(a->minimum, b->minimum, lo) || add_overflow(a->maximum, b->maximum, hi);
                break;
            case tile::ElementwiseOp::SUB:
                symbol = "-";
                overflow = sub_overflow(a->minimum, b->maximum, lo) || sub_overflow(a->maximum, b->minimum, hi);
                break;
            case tile::ElementwiseOp::MUL: {
                symbol = "*";
                int64_t products[4]{};
                overflow = mul_overflow(a->minimum, b->minimum, products[0]) ||
                           mul_overflow(a->minimum, b->maximum, products[1]) ||
                           mul_overflow(a->maximum, b->minimum, products[2]) ||
                           mul_overflow(a->maximum, b->maximum, products[3]);
                lo = *std::min_element(std::begin(products), std::end(products));
                hi = *std::max_element(std::begin(products), std::end(products));
                break;
            }
            default: _error(op, "unsupported coordinate operation"); return;
        }
        if (overflow || lo < std::numeric_limits<int32_t>::min() || hi > std::numeric_limits<int32_t>::max()) {
            _error(op, "coordinate range exceeds signed 32-bit HLSL arithmetic");
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
            _error(op, "native HLSL operands must be single-use rank-two FP32 argument Tile loads without custom padding");
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
            !_zero(mma->operand(2u))) {
            _error(op, "expected a sole-use, same-scope FP32 MMA with zero initial value");
            return;
        }
        auto a = mma->operand(0u)->defining_operation();
        auto b = mma->operand(1u)->defining_operation();
        if (a == nullptr || b == nullptr || a->kind() != OperationKind::VIEW_LOAD || b->kind() != OperationKind::VIEW_LOAD ||
            a->parent_block() != mma->parent_block() || b->parent_block() != mma->parent_block()) {
            _error(mma, "native HLSL input snapshots must be direct loads in the MMA scope");
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
        auto rows = _extent(out, 0u), columns = _extent(out, 1u);
        if (rows % 8u != 0u || columns % 8u != 0u || (rows % 16u != 0u && columns % 16u != 0u)) {
            _error(mma, "native HLSL tile M/N must be multiples of 8, with at least one a multiple of 16");
            return;
        }
        // From this point the input belongs to the bounded native family. A
        // backend must report subsequent policy/schedule errors rather than
        // silently compiling a different realization through the XIR fallback.
        _result.native_candidate = true;
        if (!mma->mma_policy().allow_reassociation) {
            _error(mma, "native HLSL scalar MMA realization requires permitted reassociation");
            return;
        }
        auto ak = 1u - *am;
        auto bk = 1u - *bn;
        auto &av = *a->operand(0u)->type().index_space();
        auto &bv = *b->operand(0u)->type().index_space();
        auto &cv = *op->operand(0u)->type().index_space();
        auto reduction = _extent(as, ak);
        if (as.axis(ak).dimension != bs.axis(bk).dimension || _extent(bs, bk) != reduction ||
            _extent(av, ak) != reduction || _extent(bv, bk) != reduction ||
            !_zero_index(a->operand(ak + 1u)) || !_zero_index(b->operand(bk + 1u))) {
            _error(mma, "this native HLSL realization requires a complete K dimension at origin zero; K pipelines are not lowered yet");
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
        _atoms.emplace_back(MmaAtom{op, mma, a, b, participants, rows, columns, reduction, *am == 1u, *bn == 0u});
        _atoms.back().full = m0->maximum <= static_cast<int64_t>(_extent(cv, 0u)) - static_cast<int64_t>(rows) &&
                             n0->maximum <= static_cast<int64_t>(_extent(cv, 1u)) - static_cast<int64_t>(columns);
        auto &arg = _result.metadata.arguments[op->operand(0u)->index()];
        arg.usage = static_cast<Usage>(to_underlying(arg.usage) | to_underlying(Usage::WRITE));
        auto worker = participants == 1u ? "lane" : "local";
        auto workers = participants == 1u ? luisa::format("{}u", _target.warp_size) : luisa::string{"group_workers"};
        _body += luisa::format("  _lc_tile_mma_{}(b{}, b{}, b{}, int({}), int({}), {}, {});\n",
                               mma->id(), a->operand(0u)->index(), b->operand(0u)->index(), op->operand(0u)->index(),
                               m0->expression, n0->expression, worker, workers);
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
            coordinate = "group";
            participants = UINT32_MAX;
        } else if (depth == 1u && scope == "subgroup") {
            if (*volume > _target.max_block_size / _target.warp_size || (_subgroups != 0u && _subgroups != *volume)) {
                _error(op, "subgroup cohort exceeds the target or conflicts with another cohort");
                return;
            }
            _subgroups = static_cast<uint32_t>(*volume);
            coordinate = "subgroup";
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
            if (extent == 0u || extent > static_cast<uint64_t>(INT32_MAX)) {
                _error(op, "parallel extents must fit positive signed 32-bit coordinates");
                return;
            }
            stride /= extent;
            luisa::string expression;
            if (extent == 1u) {
                expression = "0";
            } else {
                auto linear = stride == 1u ? luisa::format("(uint){}", coordinate) :
                                             luisa::format("((uint){} / {}u)", coordinate, stride);
                if (stride * extent != *volume) { linear = luisa::format("({} % {}u)", linear, extent); }
                expression = luisa::format("(int){}", linear);
            }
            _indices.emplace(block->argument(i), IndexValue{std::move(expression), 0, static_cast<int64_t>(extent - 1u)});
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
                default: _error(op, "unsupported by native HLSL lowering; the general XIR fallback remains available"); break;
            }
        }
    }

    void _emit_atom(const MmaAtom &atom) noexcept {
        auto &src = _result.metadata.source;
        auto id = atom.mma->id();
        auto &space = *atom.store->operand(0u)->type().index_space();
        auto m = _extent(space, 0u), n = _extent(space, 1u);
        if (atom.shared) {
            src += luisa::format("groupshared float _lc_tile_a_{}[{}u];\n", id, atom.rows * atom.reduction);
            src += luisa::format("groupshared float _lc_tile_b_{}[{}u];\n\n", id, atom.reduction * atom.columns);
        }
        src += luisa::format(
            "void _lc_tile_mma_{}(StructuredBuffer<float> a, StructuredBuffer<float> b, RWStructuredBuffer<float> c, int m0, int n0, uint worker, uint workers) {{\n",
            id);
        if (atom.shared) {
            // Cache canonical A[row, k] and B[k, column] tiles. Iterating in
            // source order keeps global loads coalesced for transposed views.
            src += luisa::format("  for (uint index = worker; index < {}u; index += workers) {{\n", atom.rows * atom.reduction);
            src += atom.transpose_a ?
                       luisa::format("    const uint row = index % {}u;\n    const uint k = index / {}u;\n", atom.rows, atom.rows) :
                       luisa::format("    const uint row = index / {}u;\n    const uint k = index % {}u;\n", atom.reduction, atom.reduction);
            src += luisa::format("    const int global_m = min(m0 + (int)row, {});\n", static_cast<int64_t>(m) - 1);
            auto a_index = atom.transpose_a ?
                               luisa::format("k * {}u + (uint)global_m", m) :
                               luisa::format("(uint)global_m * {}u + k", atom.reduction);
            src += luisa::format("    _lc_tile_a_{}[row * {}u + k] = a[{}];\n  }}\n", id, atom.reduction, a_index);
        }
        if (atom.shared) {
            src += luisa::format("  for (uint index = worker; index < {}u; index += workers) {{\n", atom.reduction * atom.columns);
            src += atom.transpose_b ?
                       luisa::format("    const uint column = index / {}u;\n    const uint k = index % {}u;\n", atom.reduction, atom.reduction) :
                       luisa::format("    const uint column = index % {}u;\n    const uint k = index / {}u;\n", atom.columns, atom.columns);
            src += luisa::format("    const int global_n = min(n0 + (int)column, {});\n", static_cast<int64_t>(n) - 1);
            auto b_index = atom.transpose_b ?
                               luisa::format("(uint)global_n * {}u + k", atom.reduction) :
                               luisa::format("k * {}u + (uint)global_n", n);
            src += luisa::format("    _lc_tile_b_{}[k * {}u + column] = b[{}];\n  }}\n", id, atom.columns, b_index);
            src += "  GroupMemoryBarrierWithGroupSync();\n";
        } else if (!atom.full) {
            src += luisa::format("  if (m0 >= {} || n0 >= {}) {{ return; }}\n", m, n);
        }
        auto workers = atom.participants == 1u ? _target.warp_size : atom.participants * _target.warp_size;
        auto elements = atom.rows * atom.columns;
        constexpr auto register_rows = uint32_t{4u};
        constexpr auto register_columns = uint32_t{2u};
        constexpr auto register_outputs = register_rows * register_columns;
        if (!atom.shared && atom.participants != 1u && elements == workers * register_outputs) {
            // A 4x2 register tile needs six global loads per eight FMAs,
            // compared with sixteen for eight independent outputs.
            auto column_workers = atom.columns / register_columns;
            src += luisa::format("  const uint row = (worker / {0}u) * {1}u;\n", column_workers, register_rows);
            src += luisa::format("  const uint column = (worker % {0}u) * {1}u;\n", column_workers, register_columns);
            for (auto i = uint32_t{0u}; i < register_outputs; i++) {
                src += luisa::format("  float sum{} = 0.0f;\n", i);
            }
            for (auto r = uint32_t{0u}; r < register_rows; r++) {
                src += atom.full ?
                           luisa::format("  const int a_row{} = m0 + (int)(row + {}u);\n", r, r) :
                           luisa::format("  const int a_row{} = min(m0 + (int)(row + {}u), {});\n", r, r, static_cast<int64_t>(m) - 1);
            }
            for (auto col = uint32_t{0u}; col < register_columns; col++) {
                src += atom.full ?
                           luisa::format("  const int b_column{} = n0 + (int)(column + {}u);\n", col, col) :
                           luisa::format("  const int b_column{} = min(n0 + (int)(column + {}u), {});\n", col, col, static_cast<int64_t>(n) - 1);
            }
            src += luisa::format("  for (uint k = 0u; k < {}u; ++k) {{\n", atom.reduction);
            for (auto r = uint32_t{0u}; r < register_rows; r++) {
                auto a_index = atom.transpose_a ?
                                   luisa::format("k * {}u + (uint)a_row{}", m, r) :
                                   luisa::format("(uint)a_row{} * {}u + k", r, atom.reduction);
                src += luisa::format("    const float a_value{} = a[{}];\n", r, a_index);
            }
            for (auto col = uint32_t{0u}; col < register_columns; col++) {
                auto b_index = atom.transpose_b ?
                                   luisa::format("(uint)b_column{} * {}u + k", col, atom.reduction) :
                                   luisa::format("k * {}u + (uint)b_column{}", n, col);
                src += luisa::format("    const float b_value{} = b[{}];\n", col, b_index);
            }
            for (auto r = uint32_t{0u}; r < register_rows; r++) {
                for (auto col = uint32_t{0u}; col < register_columns; col++) {
                    auto i = r * register_columns + col;
                    src += luisa::format("    sum{} += a_value{} * b_value{};\n", i, r, col);
                }
            }
            src += "  }\n";
            for (auto col = uint32_t{0u}; col < register_columns; col++) {
                src += col == 0u ?
                           luisa::string{"  const int global_n0 = n0 + (int)column;\n"} :
                           luisa::format("  const int global_n{} = global_n0 + {};\n", col, col);
            }
            for (auto r = uint32_t{0u}; r < register_rows; r++) {
                src += luisa::format("  const int global_m{} = m0 + (int)(row + {}u);\n", r, r);
                if (!atom.full) { src += luisa::format("  if (global_m{} < {}) {{\n", r, m); }
                for (auto col = uint32_t{0u}; col < register_columns; col++) {
                    auto i = r * register_columns + col;
                    if (atom.full) {
                        src += luisa::format("  c[(uint)global_m{} * {}u + (uint)global_n{}] = sum{};\n", r, n, col, i);
                    } else {
                        src += luisa::format("    if (global_n{0} < {1}) {{ c[(uint)global_m{2} * {1}u + (uint)global_n{0}] = sum{3}; }}\n",
                                             col, n, r, i);
                    }
                }
                if (!atom.full) { src += "  }\n"; }
            }
            src += "}\n";
            return;
        }
        if (atom.shared && workers <= elements && elements % workers == 0u &&
            workers % atom.columns == 0u && elements / workers <= 8u) {
            // Keep a small output tile in registers. With a whole-number row
            // stride, all accumulators share one B value at each K step.
            auto outputs = elements / workers;
            auto row_stride = workers / atom.columns;
            src += luisa::format("  const uint row = worker / {}u;\n  const uint column = worker % {}u;\n", atom.columns, atom.columns);
            src += "  const int global_n = n0 + (int)column;\n";
            for (auto i = uint64_t{0u}; i < outputs; i++) {
                src += luisa::format("  float sum{} = 0.0f;\n", i);
            }
            src += luisa::format("  for (uint k = 0u; k < {}u; ++k) {{\n", atom.reduction);
            src += luisa::format("    const float b_value = _lc_tile_b_{}[k * {}u + column];\n", id, atom.columns);
            for (auto i = uint64_t{0u}; i < outputs; i++) {
                auto output_row = i == 0u ? luisa::string{"row"} : luisa::format("(row + {}u)", i * row_stride);
                src += luisa::format("    sum{} += _lc_tile_a_{}[{} * {}u + k] * b_value;\n",
                                     i, id, output_row, atom.reduction);
            }
            src += "  }\n";
            src += luisa::format("  if (global_n < {}) {{\n", n);
            for (auto i = uint64_t{0u}; i < outputs; i++) {
                auto output_row = i == 0u ? luisa::string{"row"} : luisa::format("(row + {}u)", i * row_stride);
                src += luisa::format("    const int global_m{} = m0 + (int){};\n", i, output_row);
                src += luisa::format("    if (global_m{0} < {1}) {{ c[(uint)global_m{0} * {2}u + (uint)global_n] = sum{0}; }}\n",
                                     i, m, n);
            }
            src += "  }\n}\n";
            return;
        }
        src += luisa::format("  for (uint element = worker; element < {}u; element += workers) {{\n", elements);
        src += luisa::format("    const uint row = element / {}u;\n    const uint column = element % {}u;\n", atom.columns, atom.columns);
        src += "    const int global_m = m0 + (int)row;\n    const int global_n = n0 + (int)column;\n";
        src += luisa::format("    if (global_m < {} && global_n < {}) {{\n", m, n);
        src += "      float sum = 0.0f;\n";
        src += luisa::format("      for (uint k = 0u; k < {}u; ++k) {{\n", atom.reduction);
        if (atom.shared) {
            src += luisa::format("        sum += _lc_tile_a_{}[row * {}u + k] * _lc_tile_b_{}[k * {}u + column];\n",
                                 id, atom.reduction, id, atom.columns);
        } else {
            auto a_index = atom.transpose_a ?
                               luisa::format("k * {}u + (uint)global_m", m) :
                               luisa::format("(uint)global_m * {}u + k", atom.reduction);
            auto b_index = atom.transpose_b ?
                               luisa::format("(uint)global_n * {}u + k", atom.reduction) :
                               luisa::format("k * {}u + (uint)global_n", n);
            src += luisa::format("        sum += a[{}] * b[{}];\n", a_index, b_index);
        }
        src += "      }\n";
        src += luisa::format("      c[(uint)global_m * {}u + (uint)global_n] = sum;\n", n);
        src += "    }\n  }\n}\n";
    }

    void _emit(uint32_t threads) noexcept {
        for (auto &atom : _atoms) { _emit_atom(atom); }
        auto &src = _result.metadata.source;
        src += "void _lc_tile_native(";
        for (auto i = size_t{0}; i < _result.metadata.arguments.size(); i++) {
            auto writable = (to_underlying(_result.metadata.arguments[i].usage) & to_underlying(Usage::WRITE)) != 0u;
            src += luisa::format("{} b{}, ", writable ? "RWStructuredBuffer<float>" : "StructuredBuffer<float>", i);
        }
        src += "uint3 lc_group_id, uint3 lc_thread_id) {\n";
        src += "  const int group = (int)lc_group_id.x;\n";
        src += "  const uint local = lc_thread_id.x;\n";
        src += luisa::format("  const int subgroup = (int)(local / {}u);\n", _target.warp_size);
        src += luisa::format("  const uint lane = local % {}u;\n", _target.warp_size);
        src += luisa::format("  const uint group_workers = {}u;\n", threads);
        src += _body;
        src += "}\n";
    }

public:
    Lowering(const tile::Function &function, tile::CompileOptions options, HlslTileTargetConfig target) noexcept
        : _function{function}, _options{options}, _target{target} {}

    [[nodiscard]] HlslTileCode run() noexcept {
        if (_options.lowering != tile::Lowering::NATIVE || _options.tirx != nullptr || _options.xir != nullptr) {
            _error(nullptr, "Invalid Tile lowering choice or backend-specific options passed to native HLSL lowering");
            return std::move(_result);
        }
        if (_target.warp_size == 0u || _target.max_block_size < _target.warp_size) {
            _error(nullptr, "native HLSL target requires a nonzero warp size and a compatible block-size limit");
            return std::move(_result);
        }
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
            (_options.threads_per_group != 0u &&
             (_options.threads_per_group % _target.warp_size != 0u || _options.threads_per_group > _target.max_block_size))) {
            _error(nullptr, "expected Candidate TileIR and a legal whole-subgroup thread constraint");
            return std::move(_result);
        }
        auto entry = _function.body().block(0u);
        // DX uses two root-signature units per buffer in addition to its fixed
        // dispatch constants. Keep the common HLSL wrapper within that limit.
        if (entry->argument_count() >= 30u) {
            _error(nullptr, "native HLSL root argument count exceeds the DirectX root-signature budget");
            return std::move(_result);
        }
        for (auto &arg : entry->arguments()) {
            if (!arg->type().is_view() || !_matrix_type(arg->type())) {
                _error(nullptr, "native HLSL arguments currently require dense rank-two FP32 TensorViews");
                return std::move(_result);
            }
            auto volume = arg->type().index_space()->static_volume();
            if (!volume || *volume > INT32_MAX || *volume > SIZE_MAX / sizeof(float)) {
                _error(nullptr, "dense native HLSL argument exceeds signed int32 element addressing");
                return std::move(_result);
            }
            _result.metadata.arguments.emplace_back(tile::KernelArgument{tile::ScalarType::FLOAT32, *volume * sizeof(float), Usage::NONE});
        }
        _block(entry, 0u, 0u);
        auto threads = _subgroups != 0u ? _subgroups * _target.warp_size :
                                          (_options.threads_per_group == 0u ? 4u * _target.warp_size : _options.threads_per_group);
        if (_subgroups != 0u && _options.threads_per_group != 0u && threads != _options.threads_per_group) {
            _error(nullptr, "thread constraint conflicts with explicit subgroup domain");
        }
        if (_atoms.empty() || _groups == 0u || threads > _target.max_block_size || _groups > UINT32_MAX / threads) {
            _error(nullptr, "empty/unsupported native HLSL program or dispatch capacity exceeded");
        }
        auto shared_bytes = size_t{0u};
        auto shared_atoms = size_t{0u};
        for (auto &atom : _atoms) {
            if (atom.participants == UINT32_MAX) { atom.participants = threads / _target.warp_size; }
            if (atom.participants != 1u && atom.participants * _target.warp_size != threads) {
                _error(atom.mma, "native HLSL collective must use one subgroup or the complete threadgroup");
            }
            auto elements = atom.rows * atom.columns;
            auto register_blocked = atom.participants != 1u && threads <= elements &&
                                    elements % threads == 0u && threads % atom.columns == 0u &&
                                    elements / threads <= 8u;
            if (register_blocked && elements != threads * 8u) {
                auto bytes = static_cast<size_t>((atom.rows + atom.columns) * atom.reduction * sizeof(float));
                if (shared_bytes <= _target.max_shared_memory && bytes <= _target.max_shared_memory - shared_bytes) {
                    atom.shared = true;
                    shared_bytes += bytes;
                    shared_atoms++;
                }
            }
        }
        for (auto &arg : _result.metadata.arguments) {
            if (arg.usage == Usage::READ_WRITE) { _error(nullptr, "argument is both read and written; snapshot-safe forwarding is not proved"); }
        }
        if (_result.ok()) {
            _result.block_size = make_uint3(threads, 1u, 1u);
            _result.metadata.dispatch_size = make_uint3(static_cast<uint32_t>(_groups * threads), 1u, 1u);
            _result.metadata.disjoint_writes = true;
            _result.metadata.realization = luisa::format(
                "{} native Tile: {} groups, {} threads/group, {} explicit subgroups, {} scalar FP32 MMA atoms ({} group-shared); dense view forwarding; no XIR/TVM",
                _target.backend_label, _groups, threads, _subgroups, _atoms.size(), shared_atoms);
            _emit(threads);
        }
        return std::move(_result);
    }
};

}// namespace

HlslTileCode lower_tile_to_hlsl(const tile::Function &function,
                                const tile::CompileOptions &options,
                                const HlslTileTargetConfig &target) noexcept {
    return Lowering{function, options, target}.run();
}

ShaderCreationInfo create_tile_kernel_via_hlsl(
    DeviceInterface *device, const ShaderOption &option,
    const tile::Function &kernel, const tile::CompileOptions &tile_options,
    tile::KernelMetadata &metadata, const HlslTileTargetConfig &target,
    bool &native_selected) noexcept {
    metadata = {};
    native_selected = false;
    auto fail = [&metadata](luisa::string message) noexcept {
        metadata.error = std::move(message);
        return ShaderCreationInfo::make_invalid();
    };
    if (option.compile_only) {
        native_selected = true;
        return fail("Tile compile-only archives are not supported on this backend");
    }
    if (tile_options.lowering == tile::Lowering::TIRX || tile_options.tirx != nullptr) {
        native_selected = true;
        return fail("TIRx lowering is unavailable on this backend; native HLSL and the XIR fallback require native lowering");
    }
    // An explicit XIR planner is an explicit request for the general fallback.
    if (tile_options.xir != nullptr) { return ShaderCreationInfo::make_invalid(); }

    auto code = lower_tile_to_hlsl(kernel, tile_options, target);
    if (!code.ok()) {
        native_selected = code.native_candidate;
        if (native_selected) { metadata = std::move(code.metadata); }
        return ShaderCreationInfo::make_invalid();
    }
    native_selected = true;
    if (!option.native_include.empty()) {
        return fail("ShaderOption::native_include cannot be combined with native Tile HLSL generation");
    }

    auto block_size = code.block_size;
    metadata = std::move(code.metadata);
    luisa::vector<const Type *> argument_types;
    luisa::vector<Usage> argument_usages;
    argument_types.reserve(metadata.arguments.size() + 2u);
    argument_usages.reserve(metadata.arguments.size() + 2u);
    for (auto &arg : metadata.arguments) {
        argument_types.emplace_back(Type::of<Buffer<float>>());
        argument_usages.emplace_back(arg.usage);
    }
    argument_types.emplace_back(Type::of<uint3>());
    argument_types.emplace_back(Type::of<uint3>());
    argument_usages.emplace_back(Usage::READ);
    argument_usages.emplace_back(Usage::READ);
    auto external = luisa::make_shared<ExternalFunction>(
        "_lc_tile_native", nullptr, std::move(argument_types), std::move(argument_usages));
    auto argument_count = metadata.arguments.size();
    auto builder = luisa::compute::detail::FunctionBuilder::define_kernel([&] {
        auto &current = *luisa::compute::detail::FunctionBuilder::current();
        current.set_block_size(block_size);
        current.set_name(kernel.name());
        luisa::vector<const Expression *> arguments;
        arguments.reserve(argument_count + 2u);
        for (auto i = size_t{0}; i < argument_count; i++) {
            arguments.emplace_back(current.buffer(Type::of<Buffer<float>>()));
        }
        arguments.emplace_back(current.block_id());
        arguments.emplace_back(current.thread_id());
        current.call(external, arguments);
    });
    auto adjusted_option = option;
    adjusted_option.native_include = metadata.source;
    auto info = device->create_shader(adjusted_option, builder->function());
    if (!info.valid()) {
        if (metadata.error.empty()) { metadata.error = "Native Tile HLSL kernel failed backend shader compilation"; }
        return ShaderCreationInfo::make_invalid();
    }
    return info;
}

}// namespace lc::hlsl
