#include <luisa/tile/collective_plan.h>
#include <luisa/tile/verifier.h>
#include <luisa/core/stl/unordered_map.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace luisa::compute::tile {
namespace {
namespace collective_plan_detail {

class CollectiveAnalyzer {
private:
    const tile::Function &_function;
    CollectiveWorkAnalysis _result;
    const Block *_program{nullptr};

    [[nodiscard]] bool _fail(luisa::string_view message) noexcept {
        _result.error.assign(message.data(), message.size());
        return false;
    }
    [[nodiscard]] static bool _add(uint64_t &destination, uint64_t amount) noexcept {
        if (amount > std::numeric_limits<uint64_t>::max() - destination) { return false; }
        destination += amount;
        return true;
    }
    [[nodiscard]] static bool _multiply(uint64_t lhs, uint64_t rhs, uint64_t &result) noexcept {
        if (rhs != 0u && lhs > std::numeric_limits<uint64_t>::max() / rhs) { return false; }
        result = lhs * rhs;
        return true;
    }
    [[nodiscard]] static uint64_t _volume(const IndexSpace *space) noexcept {
        if (space == nullptr) { return 0u; }
        auto volume = space->static_volume();
        return volume ? *volume : 0u;
    }
    [[nodiscard]] static bool _scalar_f32(const Value *value) noexcept {
        return value != nullptr && value->type() == tile::Type::scalar(ScalarType::FLOAT32);
    }
    [[nodiscard]] static bool _identity(const Value *value, ElementwiseOp opcode) noexcept {
        if (!_scalar_f32(value)) { return false; }
        auto op = value->defining_operation();
        if (op == nullptr || op->kind() != OperationKind::CONSTANT || op->operand_count() != 0u || op->result_count() != 1u || op->region_count() != 0u || op->domain()) { return false; }
        auto attribute = op->attribute("value");
        auto number = attribute == nullptr ? nullptr : luisa::get_if<double>(&attribute->value());
        if (number == nullptr) { return false; }
        if (opcode == ElementwiseOp::ADD) { return *number == 0.0 && !std::signbit(*number); }
        return (opcode == ElementwiseOp::MIN || opcode == ElementwiseOp::MAX) && std::isinf(*number) && std::signbit(*number) == (opcode == ElementwiseOp::MAX);
    }
    [[nodiscard]] bool _constraints(const Block &block) noexcept {
        for (auto op : block.operations()) {
            if (op->execution_scope_constraint() || op->resource_class_constraint() || op->memory_layout()) { return _fail("explicit execution, resource or layout constraint"); }
            for (auto &&attribute : op->attributes()) {
                if (op->kind() != OperationKind::CONSTANT || attribute.name != "value") { return _fail("unrecognized operation attribute"); }
            }
            for (auto &&region : op->regions()) {
                for (auto child : region->blocks()) {
                    if (!_constraints(*child)) { return false; }
                }
            }
        }
        return true;
    }
    [[nodiscard]] bool _collective(const Operation &op, const Operation &map) noexcept {
        if (op.reduction_policy() != ReductionPolicy::UNORDERED_TREE || !op.domain() || op.domain()->rank() != 1u ||
            op.operand_count() != 1u || op.result_count() != 1u || !_scalar_f32(op.result(0u)) ||
            op.region_count() != 1u || op.region(0u)->block_count() != 1u) { return _fail("requires a closed single-axis unordered FP32 collective"); }
        auto body = op.region(0u)->block(0u);
        auto map_body = map.region(0u)->block(0u);
        if (body->argument_count() != 2u || body->operations().empty()) { return _fail("invalid collective body"); }
        auto yield = body->operations().back();
        if (yield->kind() != OperationKind::YIELD || yield->operand_count() != 1u) { return _fail("collective yield is not closed"); }
        auto merge = yield->operand(0u)->defining_operation();
        if (merge == nullptr || merge->kind() != OperationKind::ELEMENTWISE || merge->operand_count() != 2u || merge->result_count() != 1u ||
            merge->result(0u)->use_count() != 1u || body->argument(1u)->use_count() != 1u) { return _fail("collective carry has an extra use"); }
        auto opcode = merge->elementwise_op();
        if (!_identity(op.operand(0u), opcode)) { return _fail("collective seed/merge law not recognized"); }
        auto carry = body->argument(1u);
        auto term = merge->operand(0u) == carry ? merge->operand(1u) : merge->operand(1u) == carry ? merge->operand(0u) :
                                                                                                     nullptr;
        auto term_op = term == nullptr ? nullptr : term->defining_operation();
        auto extract = term_op;
        const Operation *select = nullptr, *predicate = nullptr, *padding = nullptr;
        auto prefix = term_op != nullptr && term_op->kind() == OperationKind::ELEMENTWISE && term_op->elementwise_op() == ElementwiseOp::SELECT;
        if (prefix) {
            select = term_op;
            if (opcode != ElementwiseOp::ADD || select->operand_count() != 3u || !_identity(select->operand(2u), opcode)) { return _fail("prefix contribution is not a zero-padded sum"); }
            predicate = select->operand(0u)->defining_operation();
            extract = select->operand(1u)->defining_operation();
            padding = select->operand(2u)->defining_operation();
            if (predicate == nullptr || predicate->kind() != OperationKind::ELEMENTWISE || predicate->elementwise_op() != ElementwiseOp::LE ||
                predicate->operand_count() != 2u || predicate->operand(0u) != body->argument(0u)) { return _fail("prefix predicate is not an inclusive coordinate bound"); }
        }
        if (extract == nullptr || extract->kind() != OperationKind::TILE_EXTRACT || extract->result_count() != 1u ||
            extract->result(0u)->use_count() != 1u || !extract->operand(0u)->type().is_tile()) { return _fail("collective contribution is not a closed Tile extraction"); }
        auto source = extract->operand(0u)->type().index_space();
        if (_volume(source) == 0u || extract->operand_count() != source->rank() + 1u) { return _fail("collective input shape is not static"); }
        auto &&map_space = *map.domain();
        uint32_t reduced_axes = 0u;
        uint64_t contribution_extent = _volume(&*op.domain());
        for (auto i = size_t{0u}; i < source->rank(); i++) {
            auto &&axis = source->axis(i);
            auto coordinate = extract->operand(i + 1u);
            auto mapped = map_space.axis_index(axis.dimension);
            if (coordinate == body->argument(0u)) {
                reduced_axes++;
                if (axis.extent != op.domain()->axis(0u).extent) { return _fail("collective contribution extent differs from its source"); }
                if (prefix) {
                    if (!mapped || axis.extent != map_space.axis(*mapped).extent || predicate->operand(1u) != map_body->argument(*mapped)) { return _fail("prefix bound differs from its output coordinate"); }
                } else if (mapped || axis.dimension != op.domain()->axis(0u).dimension) {
                    return _fail("reduction axis is not removed from its result");
                }
            } else if (!mapped || axis.extent != map_space.axis(*mapped).extent || coordinate != map_body->argument(*mapped)) {
                return _fail("collective independent coordinates are not projections");
            }
        }
        if (reduced_axes != 1u || contribution_extent == 0u || (prefix ? source->rank() != map_space.rank() : source->rank() != map_space.rank() + 1u)) { return _fail("collective has unmatched axes"); }
        for (auto child : body->operations()) {
            if (child != yield && child != merge && child != extract && child != select && child != predicate && child != padding) { return _fail("collective body contains unmatched work or effects"); }
        }
        CollectiveWork work;
        work.operation_id = op.id();
        work.kind = prefix ? CollectiveKind::INCLUSIVE_SUM : opcode == ElementwiseOp::ADD ? CollectiveKind::SUM :
                                                         opcode == ElementwiseOp::MIN     ? CollectiveKind::MINIMUM :
                                                                                            CollectiveKind::MAXIMUM;
        work.element = ScalarType::FLOAT32;
        work.contribution_extent = contribution_extent;
        work.input_elements = _volume(source);
        work.independent_elements = work.input_elements / contribution_extent;
        _result.collectives.emplace_back(work);
        return true;
    }
    [[nodiscard]] static const Value *_index_value(const Value *value) noexcept {
        auto integer64 = [](const tile::Type &type) noexcept {
            return type.kind() == TypeKind::INDEX || type == tile::Type::scalar(ScalarType::INT64);
        };
        for (;;) {
            auto op = value->defining_operation();
            if (op == nullptr || op->kind() != OperationKind::ELEMENTWISE ||
                op->elementwise_op() != ElementwiseOp::CAST || op->operand_count() != 1u ||
                !integer64(value->type()) || !integer64(op->operand(0u)->type())) { return value; }
            value = op->operand(0u);
        }
    }
    [[nodiscard]] bool _mapped_extract(const Operation &op, const Operation *map) noexcept {
        if (map == nullptr || op.operand_count() == 0u || op.result_count() != 1u ||
            op.region_count() != 0u || !op.operand(0u)->type().is_tile()) { return _fail("extraction is not a pure mapped Tile projection"); }
        auto input = op.operand(0u);
        auto definition = input->defining_operation();
        auto source = input->type().index_space();
        if (definition == nullptr || definition->parent_block() != _program ||
            _volume(source) == 0u || op.operand_count() != source->rank() + 1u) { return _fail("mapped projection source is not a direct immutable Tile"); }
        auto &&destination = *map->domain();
        auto body = map->region(0u)->block(0u);
        for (auto i = size_t{0u}; i < source->rank(); i++) {
            auto &&axis = source->axis(i);
            auto mapped = destination.axis_index(axis.dimension);
            if (!mapped) { return _fail("mapped projection changes source dimension identity"); }
            auto coordinate = _index_value(op.operand(i + 1u));
            if (axis.extent == destination.axis(*mapped).extent && coordinate == body->argument(*mapped)) { continue; }
            auto constant = coordinate->defining_operation();
            auto attribute = constant != nullptr && constant->kind() == OperationKind::CONSTANT ? constant->attribute("value") : nullptr;
            auto zero = attribute == nullptr ? nullptr : luisa::get_if<int64_t>(&attribute->value());
            if (!axis.extent.is_constant() || axis.extent.constant_value() != 1u || zero == nullptr || *zero != 0) {
                return _fail("mapped extraction is not an exact coordinate or singleton-zero projection");
            }
        }
        return true;
    }

    [[nodiscard]] bool _visit(const Block &block, const Operation *map = nullptr) noexcept {
        auto repetitions = map == nullptr ? uint64_t{1u} : _volume(&*map->domain());
        for (auto op : block.operations()) {
            for (auto i = size_t{0u}; map != nullptr && i < op->result_count(); i++) {
                if (op->result(i)->type().is_tile()) { return _fail("materialized Tile nested inside a map"); }
            }
            switch (op->kind()) {
                case OperationKind::CONSTANT:
                case OperationKind::YIELD: break;
                case OperationKind::ELEMENTWISE: {
                    if (op->result_count() != 1u || op->region_count() != 0u) { return _fail("unsupported elementwise structure"); }
                    auto &&type = op->result(0u)->type();
                    auto volume = type.is_tile() ? _volume(type.index_space()) : repetitions;
                    if (volume == 0u || !_add(_result.elementwise_elements_per_program, volume)) { return _fail("elementwise work overflows"); }
                    break;
                }
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE: {
                    if (map != nullptr || !op->domain() || op->region_count() != 0u || !op->operand(0u)->type().is_view()) { return _fail("view access is not a materialized direct-body Tile"); }
                    uint64_t bytes;
                    if (!_multiply(_volume(&*op->domain()), scalar_type_size(op->operand(0u)->type().scalar_type()), bytes) || bytes == 0u ||
                        !_add(op->kind() == OperationKind::VIEW_LOAD ? _result.global_read_bytes_per_program : _result.global_write_bytes_per_program, bytes)) { return _fail("view byte demand overflows or is dynamic"); }
                    break;
                }
                case OperationKind::TILE_EXTRACT:
                    if (!_mapped_extract(*op, map)) { return false; }
                    break;
                case OperationKind::TILE_MAP: {
                    if (map != nullptr || !op->domain() || _volume(&*op->domain()) == 0u || op->region_count() != 1u || op->region(0u)->block_count() != 1u) { return _fail("nested or dynamic Tile map"); }
                    if (!_visit(*op->region(0u)->block(0u), op)) { return false; }
                    break;
                }
                case OperationKind::REDUCE:
                    if (map == nullptr) { return _fail("collective outside a Tile map"); }
                    if (!_collective(*op, *map)) { return false; }
                    break;
                default: return _fail("structure is outside straight-line collective analysis");
            }
        }
        return true;
    }
    [[nodiscard]] const Operation *_program_user(const Operation *user) const noexcept {
        for (;;) {
            if (user == nullptr) { return nullptr; }
            auto block = user->parent_block();
            if (block == _program) { return user; }
            if (block == nullptr || block->parent_region() == nullptr) { return nullptr; }
            user = block->parent_region()->parent_operation();
        }
    }
    [[nodiscard]] bool _live_tiles() noexcept {
        luisa::vector<const Operation *> operations;
        luisa::unordered_map<const Operation *, size_t> positions;
        for (auto op : _program->operations()) {
            positions.emplace(op, operations.size());
            operations.emplace_back(op);
        }
        luisa::vector<uint64_t> starts(operations.size(), 0u), ends(operations.size(), 0u);
        for (auto i = size_t{0u}; i < operations.size(); i++) {
            auto op = operations[i];
            for (auto j = size_t{0u}; j < op->result_count(); j++) {
                auto value = op->result(j);
                if (!value->type().is_tile()) { continue; }
                auto volume = _volume(value->type().index_space());
                uint64_t bytes;
                if (volume == 0u || !_multiply(volume, scalar_type_size(value->type().scalar_type()), bytes) || bytes == 0u || !_add(_result.materialized_tile_total_bytes, bytes)) { return _fail("materialized Tile footprint overflows or is dynamic"); }
                _result.largest_materialized_tile_elements = std::max(_result.largest_materialized_tile_elements, volume);
                auto last = i;
                for (auto use : value->use_list()) {
                    auto user = _program_user(use->user());
                    auto position = positions.find(user);
                    if (position == positions.end() || position->second < i) { return _fail("Tile value escapes its program body"); }
                    last = std::max(last, position->second);
                }
                if (!_add(starts[i], bytes) || !_add(ends[last], bytes)) { return _fail("materialized Tile liveness overflows"); }
            }
        }
        uint64_t live = 0u;
        for (auto i = size_t{0u}; i < operations.size(); i++) {
            if (!_add(live, starts[i]) || ends[i] > live) { return _fail("materialized Tile liveness overflows"); }
            _result.materialized_tile_peak_bytes = std::max(_result.materialized_tile_peak_bytes, live);
            live -= ends[i];
        }
        return true;
    }

public:
    explicit CollectiveAnalyzer(const tile::Function &function) noexcept : _function{function} {}
    [[nodiscard]] CollectiveWorkAnalysis run() noexcept {
        auto module = _function.parent_module();
        bool attached = false;
        if (module != nullptr) {
            for (auto function : module->functions()) { attached |= function == &_function; }
        }
        if (!attached || !verify(*module).ok() || _function.form() != IRForm::CANDIDATE || _function.body().block_count() != 1u) {
            static_cast<void>(_fail("requires a verified attached candidate function"));
            return std::move(_result);
        }
        auto root = _function.body().block(0u);
        const Operation *parallel = nullptr;
        for (auto op : root->operations()) {
            if (op->kind() == OperationKind::CONSTANT && op->result_count() == 1u && !op->result(0u)->type().is_tile()) { continue; }
            if (op->kind() != OperationKind::PARALLEL || parallel != nullptr) {
                static_cast<void>(_fail("requires one root parallel with only scalar constant captures"));
                return std::move(_result);
            }
            parallel = op;
        }
        if (parallel == nullptr || !parallel->domain() || (_result.programs = _volume(&*parallel->domain())) == 0u || parallel->operand_count() != 0u || parallel->result_count() != 0u ||
            parallel->region_count() != 1u || parallel->region(0u)->block_count() != 1u || !_constraints(*root)) {
            if (_result.error.empty()) { static_cast<void>(_fail("parallel domain is not static and closed")); }
            return std::move(_result);
        }
        _program = parallel->region(0u)->block(0u);
        if (!_visit(*_program) || !_live_tiles()) { return std::move(_result); }
        if (_result.collectives.empty()) { static_cast<void>(_fail("no supported collective")); }
        return std::move(_result);
    }
};

}  // namespace collective_plan_detail
}// namespace

CollectiveWorkAnalysis analyze_collective_work(const tile::Function &function) noexcept {
    return collective_plan_detail::CollectiveAnalyzer{function}.run();
}

}// namespace luisa::compute::tile
