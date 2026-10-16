#include <luisa/tile/collective_partition.h>
#include <cmath>
#include <limits>

namespace luisa::compute::tile {
namespace {
namespace collective_partition_detail {

[[nodiscard]] bool same_space(const IndexSpace &a, const IndexSpace &b) noexcept {
    if (a.rank() != b.rank()) { return false; }
    for (auto i = size_t{0u}; i < a.rank(); i++) {
        if (a.axis(i).dimension != b.axis(i).dimension || a.axis(i).extent != b.axis(i).extent) { return false; }
    }
    return true;
}

[[nodiscard]] bool integer64(const Type &type) noexcept {
    return type.kind() == TypeKind::INDEX || type == Type::scalar(ScalarType::INT64);
}

[[nodiscard]] const Value *index_value(const Value *value) noexcept {
    for (auto depth = 0u; depth < 32u && value != nullptr; depth++) {
        auto op = value->defining_operation();
        if (op == nullptr || op->kind() != OperationKind::ELEMENTWISE || op->elementwise_op() != ElementwiseOp::CAST ||
            op->operand_count() != 1u || !integer64(value->type()) || !integer64(op->operand(0u)->type())) { return value; }
        value = op->operand(0u);
    }
    return nullptr;
}

[[nodiscard]] bool integer_constant(const Value *value, uint64_t expected) noexcept {
    value = index_value(value);
    if (value == nullptr || !integer64(value->type())) { return false; }
    auto op = value->defining_operation();
    auto attribute = op != nullptr && op->kind() == OperationKind::CONSTANT ? op->attribute("value") : nullptr;
    auto number = attribute == nullptr ? nullptr : luisa::get_if<int64_t>(&attribute->value());
    return number != nullptr && *number >= 0 && static_cast<uint64_t>(*number) == expected;
}

[[nodiscard]] bool origin(const Value *value, const Value *program, uint64_t extent) noexcept {
    value = index_value(value);
    if (value == nullptr) { return false; }
    if (extent == 1u && value == program) { return true; }
    auto op = value->defining_operation();
    return op != nullptr && op->kind() == OperationKind::ELEMENTWISE && op->elementwise_op() == ElementwiseOp::MUL &&
           op->operand_count() == 2u && integer64(value->type()) &&
           ((index_value(op->operand(0u)) == program && integer_constant(op->operand(1u), extent)) ||
            (index_value(op->operand(1u)) == program && integer_constant(op->operand(0u), extent)));
}

[[nodiscard]] bool storage(ScalarType type) noexcept {
    return type == ScalarType::FLOAT32 || type == ScalarType::FLOAT16 || type == ScalarType::BFLOAT16;
}

[[nodiscard]] bool identity(const Value *value, CollectiveKind kind) noexcept {
    if (value->type() != Type::scalar(ScalarType::FLOAT32)) { return false; }
    auto op = value->defining_operation();
    auto attribute = op != nullptr && op->kind() == OperationKind::CONSTANT ? op->attribute("value") : nullptr;
    auto number = attribute == nullptr ? nullptr : luisa::get_if<double>(&attribute->value());
    if (number == nullptr) { return false; }
    return kind == CollectiveKind::SUM ? *number == 0.0 && !std::signbit(*number) :
                                         std::isinf(*number) && std::signbit(*number);
}

[[nodiscard]] bool multiply(uint64_t a, uint64_t b, uint64_t &result) noexcept {
    if (b != 0u && a > std::numeric_limits<uint64_t>::max() / b) { return false; }
    result = a * b;
    return true;
}

[[nodiscard]] const Operation *find(const Block &block, uint64_t id) noexcept {
    for (auto op : block.operations()) {
        if (op->id() == id) { return op; }
        for (auto &&region : op->regions()) {
            for (auto child : region->blocks()) {
                if (auto found = find(*child, id)) { return found; }
            }
        }
    }
    return nullptr;
}

[[nodiscard]] const Value *tile_cast(const Value *value, ScalarType from, ScalarType to, const Block *program) noexcept {
    auto op = value->defining_operation();
    if (!value->type().is_tile() || value->type().scalar_type() != to || op == nullptr || op->parent_block() != program ||
        op->kind() != OperationKind::ELEMENTWISE || op->elementwise_op() != ElementwiseOp::CAST || op->operand_count() != 1u ||
        !op->operand(0u)->type().is_tile() || op->operand(0u)->type().scalar_type() != from ||
        !same_space(*value->type().index_space(), *op->operand(0u)->type().index_space())) { return nullptr; }
    return op->operand(0u);
}

[[nodiscard]] bool coordinate_tile(const Value *value, Dim dimension, uint64_t extent, const Block *program) noexcept {
    auto map = value->defining_operation();
    if (!value->type().is_tile() || value->type().scalar_type() != ScalarType::INT64 || map == nullptr ||
        map->parent_block() != program || map->kind() != OperationKind::TILE_MAP || !map->domain() ||
        map->domain()->rank() != 1u || map->domain()->axis(0u).dimension != dimension ||
        !map->domain()->axis(0u).extent.is_constant() || map->domain()->axis(0u).extent.constant_value() != extent) { return false; }
    auto body = map->region(0u)->block(0u);
    auto yield = body->operations().back();
    return yield->kind() == OperationKind::YIELD && yield->operand_count() == 1u &&
           index_value(yield->operand(0u)) == body->argument(0u);
}

[[nodiscard]] bool identity_mask(const Value *predicate, Dim dimension, uint64_t extent, uint64_t logical_extent,
                                 const Block *program) noexcept {
    auto op = predicate->defining_operation();
    return predicate->type().is_tile() && predicate->type().scalar_type() == ScalarType::BOOL &&
           predicate->use_count() == 1u && op != nullptr && op->parent_block() == program &&
           op->kind() == OperationKind::ELEMENTWISE && op->elementwise_op() == ElementwiseOp::LT && op->operand_count() == 2u &&
           op->operand(0u)->use_count() == 1u && coordinate_tile(op->operand(0u), dimension, extent, program) &&
           integer_constant(op->operand(1u), logical_extent);
}

// A store can broadcast a reduced rank-one Tile onto a singleton output axis.
// The extraction must use the actual remaining dimension's coordinate.
[[nodiscard]] const Value *projection(const Value *value, Dim independent, uint64_t extent, const Block *program) noexcept {
    auto map = value->defining_operation();
    if (map == nullptr || map->parent_block() != program || map->kind() != OperationKind::TILE_MAP || !map->domain()) { return nullptr; }
    auto body = map->region(0u)->block(0u);
    auto yield = body->operations().back();
    if (yield->kind() != OperationKind::YIELD || yield->operand_count() != 1u) { return nullptr; }
    auto extract = yield->operand(0u)->defining_operation();
    auto mapped = map->domain()->axis_index(independent);
    if (extract == nullptr || extract->kind() != OperationKind::TILE_EXTRACT || extract->operand_count() != 2u ||
        extract->parent_block() != body || extract->result(0u)->use_count() != 1u || !mapped) { return nullptr; }
    auto source = extract->operand(0u);
    if (!source->type().is_tile() || source->type().index_space()->rank() != 1u ||
        source->type().index_space()->axis(0u).dimension != independent ||
        source->type().index_space()->axis(0u).extent != map->domain()->axis(*mapped).extent ||
        source->type().index_space()->axis(0u).extent.constant_value() != extent ||
        (index_value(extract->operand(1u)) != body->argument(*mapped) &&
         !(extent == 1u && integer_constant(extract->operand(1u), 0u)))) { return nullptr; }
    for (auto i = size_t{0u}; i < map->domain()->rank(); i++) {
        auto &&axis = map->domain()->axis(i);
        if (i != *mapped && (!axis.extent.is_constant() || axis.extent.constant_value() != 1u)) { return nullptr; }
    }
    return source;
}

}  // namespace collective_partition_detail
}// namespace

IndependentCollectivePlan plan_independent_collective(const Function &function, IndependentPartitionRequest request) noexcept {
    IndependentCollectivePlan plan;
    auto fail = [&](luisa::string_view why) noexcept {
        plan.error.assign(why.data(), why.size());
        return std::move(plan);
    };
    auto analysis = analyze_collective_work(function);
    if (!analysis.ok() || analysis.collectives.size() != 1u) { return fail("requires one verified closed collective"); }
    auto &&work = analysis.collectives.front();
    if ((work.kind != CollectiveKind::SUM && work.kind != CollectiveKind::MAXIMUM) || work.element != ScalarType::FLOAT32 ||
        (request.collective_operation_id != ~uint64_t{0u} && request.collective_operation_id != work.operation_id)) {
        return fail("requires the requested unordered FP32 SUM or MAXIMUM");
    }
    auto root = function.body().block(0u);
    const Operation *parallel = nullptr;
    for (auto op : root->operations()) {
        if (op->kind() == OperationKind::PARALLEL) { parallel = op; }
    }
    if (parallel == nullptr || parallel->domain()->rank() != 1u) { return fail("requires one independent program axis"); }
    auto program = parallel->region(0u)->block(0u);
    const Operation *load = nullptr, *store = nullptr;
    for (auto op : program->operations()) {
        if (op->kind() == OperationKind::VIEW_LOAD) {
            if (load != nullptr) { return fail("extra load extends the snapshot closure"); }
            load = op;
        } else if (op->kind() == OperationKind::VIEW_STORE) {
            if (store != nullptr) { return fail("extra store extends the effect closure"); }
            store = op;
        }
    }
    if (load == nullptr || store == nullptr || !load->domain() || !store->domain() || load->domain()->rank() != 2u ||
        (store->domain()->rank() != 1u && store->domain()->rank() != 2u) || load->bounds_mode() != BoundsMode::ZERO ||
        store->bounds_mode() != BoundsMode::ZERO || load->operand_count() != 3u ||
        store->operand_count() != store->domain()->rank() + 2u) { return fail("requires one zero-padded rank-two snapshot and rank-one or rank-two store"); }
    auto reduction = collective_partition_detail::find(*root, work.operation_id);
    auto map = reduction == nullptr ? nullptr : reduction->parent_block()->parent_region()->parent_operation();
    if (map == nullptr || map->kind() != OperationKind::TILE_MAP || map->parent_block() != program ||
        !map->domain() || map->domain()->rank() != 1u || map->result_count() != 1u) { return fail("reduction result is not one direct independent Tile map"); }
    auto map_body = map->region(0u)->block(0u);
    auto yield = map_body->operations().back();
    if (yield->kind() != OperationKind::YIELD || yield->operand_count() != 1u || yield->operand(0u) != reduction->result(0u) ||
        reduction->result(0u)->use_count() != 1u) { return fail("reduction has another use or scalar epilogue"); }
    auto independent = map->domain()->axis(0u).dimension;
    auto contribution = reduction->domain()->axis(0u).dimension;
    auto ia = load->domain()->axis_index(independent), ca = load->domain()->axis_index(contribution);
    auto oa = store->domain()->axis_index(independent);
    if (!ia || !ca || !oa || *ia == *ca || (request.independent_axis && request.independent_axis != independent)) {
        return fail("requested independent and contribution dimensions do not match the memory maps");
    }
    auto input = load->operand(0u), output = store->operand(0u);
    if (input->argument_block() != root || output->argument_block() != root || input == output ||
        !input->type().is_view() || !output->type().is_view() || input->type().index_space()->rank() != 2u ||
        output->type().index_space()->rank() != store->domain()->rank() ||
        input->index() > std::numeric_limits<uint32_t>::max() || output->index() > std::numeric_limits<uint32_t>::max()) {
        return fail("requires distinct direct root views with matched positional ranks");
    }
    if (!collective_partition_detail::storage(input->type().scalar_type()) || !collective_partition_detail::storage(output->type().scalar_type())) { return fail("requires FP32, FP16 or BF16 view storage"); }
    auto in_space = input->type().index_space(), out_space = output->type().index_space();
    for (auto &&axis : in_space->axes()) {
        if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u || axis.extent.constant_value() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return fail("input view extents are not positive signed-index constants");
        }
    }
    for (auto i = size_t{0u}; i < out_space->rank(); i++) {
        auto extent = out_space->axis(i).extent;
        if (!extent.is_constant() || extent.constant_value() == 0u ||
            (i == *oa ? extent != in_space->axis(*ia).extent : extent.constant_value() != 1u) ||
            (i != *oa && store->domain()->axis(i).extent.constant_value() != 1u)) { return fail("output does not cover the independent axis exactly once"); }
    }
    auto independent_extent = load->domain()->axis(*ia).extent.constant_value();
    auto contribution_extent = load->domain()->axis(*ca).extent.constant_value();
    auto logical_independent = in_space->axis(*ia).extent.constant_value();
    auto logical_contribution = in_space->axis(*ca).extent.constant_value();
    auto target = request.target_extent_per_program;
    if (independent_extent != work.independent_elements || contribution_extent != work.contribution_extent ||
        store->domain()->axis(*oa).extent != load->domain()->axis(*ia).extent ||
        contribution_extent < logical_contribution || target == 0u || target >= independent_extent || independent_extent % target != 0u) {
        return fail("target must properly divide the complete independent extent without changing contributions");
    }
    auto geometry = [logical_independent](uint64_t extent) noexcept {
        auto full = logical_independent / extent, tail = logical_independent % extent;
        return CollectiveCandidateGeometry{full + static_cast<uint64_t>(tail != 0u), extent, full, tail};
    };
    plan.original = geometry(independent_extent);
    plan.candidate = geometry(target);
    uint64_t padded_independent;
    if (analysis.programs != plan.original.programs || !collective_partition_detail::multiply(plan.original.programs, independent_extent, padded_independent) ||
        padded_independent - 1u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        contribution_extent - 1u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return fail("program grid does not give a safe complete independent cover"); }
    for (auto i = size_t{0u}; i < 2u; i++) {
        if (i == *ia ? !collective_partition_detail::origin(load->operand(i + 1u), program->argument(0u), independent_extent) : !collective_partition_detail::integer_constant(load->operand(i + 1u), 0u)) {
            return fail("input origin is not the proved independent ownership map");
        }
    }
    for (auto i = size_t{0u}; i < out_space->rank(); i++) {
        if (i == *oa ? !collective_partition_detail::origin(store->operand(i + 1u), program->argument(0u), independent_extent) : !collective_partition_detail::integer_constant(store->operand(i + 1u), 0u)) {
            return fail("output origin is not an injective independent ownership map");
        }
    }
    const Operation *extract = nullptr;
    for (auto op : reduction->region(0u)->block(0u)->operations()) {
        if (op->kind() == OperationKind::TILE_EXTRACT) { extract = op; }
    }
    if (extract == nullptr) { return fail("missing admitted contribution extraction"); }
    auto source = extract->operand(0u);
    if (!collective_partition_detail::same_space(*source->type().index_space(), *load->domain()) || source->use_count() != 1u) {
        return fail("contribution snapshot changes shape or has another use");
    }
    auto source_op = source->defining_operation();
    if (source_op != nullptr && source_op->parent_block() == program && source_op->kind() == OperationKind::ELEMENTWISE &&
        source_op->elementwise_op() == ElementwiseOp::SELECT) {
        if (source_op->operand_count() != 3u || !collective_partition_detail::identity(source_op->operand(2u), work.kind) ||
            !collective_partition_detail::identity_mask(source_op->operand(0u), contribution, contribution_extent, logical_contribution, program)) {
            return fail("contribution mask is not its actual coordinate bound and reducer identity");
        }
        source = source_op->operand(1u);
        plan.contribution_identity_mask = true;
    }
    auto loaded = load->result(0u);
    if (source->use_count() != 1u || loaded->use_count() != 1u ||
        (source != loaded && collective_partition_detail::tile_cast(source, input->type().scalar_type(), ScalarType::FLOAT32, program) != loaded)) {
        return fail("contribution is not the sole original snapshot or its FP32 conversion");
    }
    auto stored = store->operand(store->operand_count() - 1u);
    auto result = map->result(0u);
    auto saw_cast = false, saw_projection = false;
    for (auto step = 0u; step < 3u && stored != result; step++) {
        if (stored->use_count() != 1u) { return fail("store chain has an extra snapshot use"); }
        if (!saw_cast) {
            if (auto previous = collective_partition_detail::tile_cast(stored, ScalarType::FLOAT32, output->type().scalar_type(), program)) {
                saw_cast = true;
                stored = previous;
                continue;
            }
        }
        if (!saw_projection) {
            if (auto previous = collective_partition_detail::projection(stored, independent, independent_extent, program)) {
                saw_projection = true;
                stored = previous;
                continue;
            }
        }
        return fail("store chain is not a sole conversion and singleton axis projection");
    }
    if (stored != result || result->use_count() != 1u) { return fail("reduction Tile has another use or unmatched epilogue"); }
    uint64_t elements, input_bytes, output_bytes;
    if (!collective_partition_detail::multiply(logical_independent, logical_contribution, elements) ||
        !collective_partition_detail::multiply(elements, scalar_type_size(input->type().scalar_type()), input_bytes) ||
        !collective_partition_detail::multiply(logical_independent, scalar_type_size(output->type().scalar_type()), output_bytes)) {
        return fail("root view byte interval overflows");
    }
    plan.parallel_operation_id = parallel->id();
    plan.collective_operation_id = reduction->id();
    plan.load_operation_id = load->id();
    plan.store_operation_id = store->id();
    plan.kind = work.kind;
    plan.input_storage = input->type().scalar_type();
    plan.output_storage = output->type().scalar_type();
    plan.independent_axis = independent;
    plan.contribution_axis = contribution;
    plan.input_independent_axis = static_cast<uint32_t>(*ia);
    plan.input_contribution_axis = static_cast<uint32_t>(*ca);
    plan.output_independent_axis = static_cast<uint32_t>(*oa);
    plan.output_rank = static_cast<uint32_t>(out_space->rank());
    plan.logical_independent_extent = logical_independent;
    plan.logical_contribution_extent = logical_contribution;
    plan.tile_contribution_extent = contribution_extent;
    plan.disjoint = {{static_cast<uint32_t>(input->index()), 0u, input_bytes}, {static_cast<uint32_t>(output->index()), 0u, output_bytes}};
    plan.function = &function;
    return plan;
}

IndependentCollectiveWorkFacts analyze_independent_collective_candidate(
    const IndependentCollectivePlan &plan, IndependentCollectiveGeometryKind geometry_kind) noexcept {
    IndependentCollectiveWorkFacts facts;
    auto fail = [&](luisa::string_view reason) noexcept {
        facts.error.assign(reason.data(), reason.size());
        return std::move(facts);
    };
    if (!plan.ok() || (plan.kind != CollectiveKind::SUM && plan.kind != CollectiveKind::MAXIMUM) ||
        !collective_partition_detail::storage(plan.input_storage) || !collective_partition_detail::storage(plan.output_storage) ||
        !plan.independent_axis || !plan.contribution_axis || plan.independent_axis == plan.contribution_axis) {
        return fail("requires a successful independent collective plan");
    }
    if (geometry_kind != IndependentCollectiveGeometryKind::ORIGINAL &&
        geometry_kind != IndependentCollectiveGeometryKind::PARTITIONED) {
        return fail("unknown independent collective geometry kind");
    }
    auto rows = plan.logical_independent_extent;
    auto columns = plan.logical_contribution_extent;
    auto width = plan.tile_contribution_extent;
    auto original_extent = plan.original.independent_extent_per_program;
    auto candidate_extent = plan.candidate.independent_extent_per_program;
    if (rows == 0u || columns == 0u || width < columns || original_extent == 0u ||
        candidate_extent == 0u || candidate_extent >= original_extent || original_extent % candidate_extent != 0u) {
        return fail("inconsistent independent collective extents");
    }
    auto geometry_matches = [rows](const CollectiveCandidateGeometry &geometry) noexcept {
        auto extent = geometry.independent_extent_per_program;
        auto full = rows / extent;
        auto tail = rows % extent;
        return geometry.full_programs == full && geometry.tail_valid_extent == tail &&
               geometry.programs == full + static_cast<uint64_t>(tail != 0u);
    };
    if (!geometry_matches(plan.original) || !geometry_matches(plan.candidate)) {
        return fail("inconsistent independent collective program cover");
    }
    auto geometry = geometry_kind == IndependentCollectiveGeometryKind::ORIGINAL ? plan.original : plan.candidate;
    auto extent = geometry.independent_extent_per_program;
    auto input_bytes = static_cast<uint64_t>(scalar_type_size(plan.input_storage));
    auto output_bytes = static_cast<uint64_t>(scalar_type_size(plan.output_storage));
    if (!collective_partition_detail::multiply(geometry.programs, extent, facts.padded_independent_elements) ||
        !collective_partition_detail::multiply(extent, width, facts.collective_input_elements_per_program) ||
        !collective_partition_detail::multiply(geometry.programs, facts.collective_input_elements_per_program, facts.collective_input_elements_total) ||
        !collective_partition_detail::multiply(rows, columns, facts.valid_input_elements) ||
        !collective_partition_detail::multiply(facts.valid_input_elements, input_bytes, facts.valid_input_bytes) ||
        !collective_partition_detail::multiply(rows, output_bytes, facts.valid_output_bytes) ||
        !collective_partition_detail::multiply(facts.collective_input_elements_per_program, input_bytes, facts.input_snapshot_bytes_per_program) ||
        !collective_partition_detail::multiply(facts.collective_input_elements_per_program, uint64_t{4u}, facts.fp32_source_bytes_per_program) ||
        !collective_partition_detail::multiply(extent, uint64_t{4u}, facts.fp32_result_bytes_per_program) ||
        !collective_partition_detail::multiply(extent, output_bytes, facts.output_value_bytes_per_program)) {
        return fail("independent collective work arithmetic overflows");
    }
    if (plan.disjoint.input.argument_index == plan.disjoint.output.argument_index ||
        plan.disjoint.input.byte_count != facts.valid_input_bytes || plan.disjoint.output.byte_count != facts.valid_output_bytes ||
        plan.disjoint.input.byte_offset > std::numeric_limits<uint64_t>::max() - facts.valid_input_bytes ||
        plan.disjoint.output.byte_offset > std::numeric_limits<uint64_t>::max() - facts.valid_output_bytes) {
        return fail("independent collective root intervals disagree with work facts");
    }
    facts.geometry_kind = geometry_kind;
    facts.kind = plan.kind;
    facts.input_storage = plan.input_storage;
    facts.output_storage = plan.output_storage;
    facts.independent_axis = plan.independent_axis;
    facts.contribution_axis = plan.contribution_axis;
    facts.collective_operation_id = plan.collective_operation_id;
    facts.geometry = geometry;
    facts.logical_independent_extent = rows;
    facts.logical_contribution_extent = columns;
    facts.tile_contribution_extent = width;
    facts.valid_output_elements = rows;
    facts.independent_bounds_elidable = geometry.tail_valid_extent == 0u;
    facts.contribution_bounds_elidable = columns == width;
    facts.contribution_identity_mask = plan.contribution_identity_mask;
    return facts;
}

}// namespace luisa::compute::tile
