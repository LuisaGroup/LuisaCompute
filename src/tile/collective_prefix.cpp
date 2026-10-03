// Read-only memory/cast/origin closure around an existing admitted prefix.
#include <luisa/tile/collective_prefix.h>
#include <limits>

namespace luisa::compute::tile {
namespace {
namespace closed_prefix_detail {
[[nodiscard]] bool multiply(uint64_t a, uint64_t b, uint64_t &result) noexcept {
    if (b != 0u && a > std::numeric_limits<uint64_t>::max() / b) { return false; }
    result = a * b;
    return true;
}
[[nodiscard]] bool add(uint64_t a, uint64_t b, uint64_t &result) noexcept {
    if (a > std::numeric_limits<uint64_t>::max() - b) { return false; }
    result = a + b;
    return true;
}
[[nodiscard]] bool same_space(const IndexSpace &a, const IndexSpace &b) noexcept {
    if (a.rank() != b.rank()) { return false; }
    for (auto i = 0u; i < a.rank(); i++) {
        if (a.axis(i).dimension != b.axis(i).dimension || a.axis(i).extent != b.axis(i).extent) { return false; }
    }
    return true;
}
[[nodiscard]] bool integer64(const tile::Type &type) noexcept {
    return type.kind() == TypeKind::INDEX || type == tile::Type::scalar(ScalarType::INT64);
}
[[nodiscard]] const Value *strip_index_casts(const Value *value) noexcept {
    // Verifier and attached SSA admission precede all walks; the depth bound is
    // still conservative against unexpectedly large cast chains.
    for (auto depth = 0u; depth < 32u; depth++) {
        auto op = value->defining_operation();
        if (op == nullptr || op->kind() != OperationKind::ELEMENTWISE ||
            op->elementwise_op() != ElementwiseOp::CAST || op->operand_count() != 1u ||
            !integer64(value->type()) || !integer64(op->operand(0u)->type())) { return value; }
        value = op->operand(0u);
    }
    return nullptr;
}
[[nodiscard]] bool integer_constant(const Value *value, int64_t expected) noexcept {
    value = strip_index_casts(value);
    if (value == nullptr || !integer64(value->type())) { return false; }
    auto op = value->defining_operation();
    auto attribute = op != nullptr && op->kind() == OperationKind::CONSTANT ? op->attribute("value") : nullptr;
    auto number = attribute == nullptr ? nullptr : luisa::get_if<int64_t>(&attribute->value());
    return number != nullptr && *number == expected;
}
[[nodiscard]] bool row_origin(const Value *value, const Value *program_index, uint64_t rows) noexcept {
    value = strip_index_casts(value);
    if (value == nullptr) { return false; }
    if (rows == 1u && value == program_index) { return true; }
    auto op = value->defining_operation();
    if (op == nullptr || op->kind() != OperationKind::ELEMENTWISE ||
        op->elementwise_op() != ElementwiseOp::MUL || op->operand_count() != 2u || !integer64(value->type())) { return false; }
    return (strip_index_casts(op->operand(0u)) == program_index && integer_constant(op->operand(1u), static_cast<int64_t>(rows))) ||
           (strip_index_casts(op->operand(1u)) == program_index && integer_constant(op->operand(0u), static_cast<int64_t>(rows)));
}
[[nodiscard]] const Value *one_tile_cast(const Value *value, ScalarType from, ScalarType to,
                                         const Block *program) noexcept {
    if (value->type().scalar_type() != to || !value->type().is_tile()) { return nullptr; }
    auto op = value->defining_operation();
    if (op == nullptr || op->parent_block() != program || op->kind() != OperationKind::ELEMENTWISE ||
        op->elementwise_op() != ElementwiseOp::CAST || op->operand_count() != 1u || op->result_count() != 1u ||
        !op->operand(0u)->type().is_tile() || op->operand(0u)->type().scalar_type() != from ||
        !same_space(*value->type().index_space(), *op->operand(0u)->type().index_space())) { return nullptr; }
    return op->operand(0u);
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

[[nodiscard]] ClosedPrefixAnalysis analyze(const Function &function) noexcept {
    ClosedPrefixAnalysis plan;
    auto fail = [&](luisa::string_view why) noexcept {
        plan.error.assign(why.data(), why.size());
        return std::move(plan);
    };
    auto admission = tile::analyze_collective_work(function);
    if (!admission.ok() || admission.collectives.size() != 1u ||
        admission.collectives.front().kind != CollectiveKind::INCLUSIVE_SUM ||
        admission.collectives.front().element != ScalarType::FLOAT32) { return fail("requires exactly one admitted closed unordered FP32 prefix"); }
    auto root = function.body().block(0u);
    const Operation *parallel = nullptr;
    for (auto op : root->operations()) {
        if (op->kind() == OperationKind::PARALLEL) { parallel = op; }
    }
    if (parallel == nullptr || parallel->domain()->rank() != 1u || root->argument_count() > std::numeric_limits<uint32_t>::max()) {
        return fail("requires one rank-one program grid with representable root argument indices");
    }
    auto program = parallel->region(0u)->block(0u);
    const Operation *load = nullptr, *store = nullptr;
    for (auto op : program->operations()) {
        if (op->kind() == OperationKind::VIEW_LOAD) {
            if (load != nullptr) { return fail("extra load preserves a larger snapshot dependency"); }
            load = op;
        } else if (op->kind() == OperationKind::VIEW_STORE) {
            if (store != nullptr) { return fail("extra store requires a general effect rewrite"); }
            store = op;
        }
    }
    // Shared admission already rejects all other memory effects, loops, nested
    // maps, MMA and unknown operations, including effects inside map bodies.
    if (load == nullptr || store == nullptr || !load->domain() || !store->domain() ||
        load->domain()->rank() != 2u || !same_space(*load->domain(), *store->domain()) ||
        load->bounds_mode() != BoundsMode::ZERO || store->bounds_mode() != BoundsMode::ZERO ||
        load->operand_count() != 3u || store->operand_count() != 4u || load->result_count() != 1u) {
        return fail("requires one zero-padded rank-two load and one matching store with no custom fill");
    }
    auto input = load->operand(0u), output = store->operand(0u);
    if (input->argument_block() != root || output->argument_block() != root || input == output ||
        !input->type().is_view() || !output->type().is_view()) { return fail("requires two distinct direct root view parameters"); }
    auto input_shape = input->type().index_space(), output_shape = output->type().index_space();
    if (input_shape->rank() != 2u || output_shape->rank() != 2u) { return fail("root views must have rank two"); }
    for (auto i = 0u; i < 2u; i++) {
        if (!input_shape->axis(i).extent.is_constant() || input_shape->axis(i).extent != output_shape->axis(i).extent) {
            return fail("input/output logical extents differ or are dynamic");
        }
    }
    plan.storage = input->type().scalar_type();
    if (plan.storage != output->type().scalar_type() ||
        (plan.storage != ScalarType::FLOAT32 && plan.storage != ScalarType::FLOAT16 && plan.storage != ScalarType::BFLOAT16)) {
        return fail("closed prefix subset uses equal FP32/FP16/BF16 storage and FP32 arithmetic");
    }
    plan.collective = admission.collectives.front();
    plan.logical_independent_extent = input_shape->axis(0u).extent.constant_value();
    plan.logical_contribution_extent = input_shape->axis(1u).extent.constant_value();
    if (!load->domain()->axis(0u).extent.is_constant() || !load->domain()->axis(1u).extent.is_constant()) {
        return fail("requires static loaded Tile extents");
    }
    auto row_extent = load->domain()->axis(0u).extent.constant_value();
    auto column_extent = load->domain()->axis(1u).extent.constant_value();
    auto rows = plan.logical_independent_extent;
    auto columns = plan.logical_contribution_extent;
    constexpr auto index_limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (rows == 0u || columns == 0u || row_extent == 0u || column_extent < columns ||
        rows > index_limit || columns > index_limit || row_extent > index_limit || column_extent > index_limit ||
        plan.collective.contribution_extent != column_extent || plan.collective.independent_elements != row_extent) {
        return fail("requires positive static complete row boxes and matching prefix axes");
    }
    auto programs = (rows - 1u) / row_extent + 1u;
    if (programs != admission.programs ||
        !row_origin(load->operand(1u), program->argument(0u), row_extent) ||
        !row_origin(store->operand(1u), program->argument(0u), row_extent) ||
        !integer_constant(load->operand(2u), 0) || !integer_constant(store->operand(2u), 0)) {
        return fail("origins/grid do not prove disjoint complete row boxes with zero column origin");
    }
    // Prove the complete padded index envelope before evaluating any product.
    // CUDA's current power-of-two/chunk caps are deliberately not assumptions.
    uint64_t padded_rows{}, largest_row_offset{}, largest_offset{}, elements{}, bytes{};
    if (!multiply(programs, row_extent, padded_rows) || padded_rows > index_limit ||
        !multiply(padded_rows - 1u, columns, largest_row_offset) ||
        !add(largest_row_offset, column_extent - 1u, largest_offset) || largest_offset > index_limit ||
        !multiply(rows, columns, elements) ||
        !multiply(elements, scalar_type_size(plan.storage), bytes) || bytes > index_limit) {
        return fail("static touched index or byte range overflows int64");
    }
    plan.original = CollectiveCandidateGeometry{
        programs, row_extent, rows / row_extent, rows % row_extent};
    plan.independent_axis = load->domain()->axis(0u).dimension;
    plan.contribution_axis = load->domain()->axis(1u).dimension;
    plan.disjoint = DisjointRequirement{
        RootViewInterval{static_cast<uint32_t>(input->index()), 0u, bytes},
        RootViewInterval{static_cast<uint32_t>(output->index()), 0u, bytes}};
    auto prefix = find(*root, admission.collectives.front().operation_id);
    if (prefix == nullptr) { return fail("missing admitted prefix operation"); }
    auto map = prefix->parent_block()->parent_region()->parent_operation();
    if (map == nullptr || map->kind() != OperationKind::TILE_MAP || map->parent_block() != program ||
        map->result_count() != 1u || !map->domain() || !same_space(*map->domain(), *load->domain())) {
        return fail("prefix is not a direct matching full-shape map");
    }
    auto map_body = map->region(0u)->block(0u);
    auto yield = map_body->operations().back();
    if (yield->kind() != OperationKind::YIELD || yield->operand_count() != 1u || yield->operand(0u) != prefix->result(0u) ||
        prefix->result(0u)->use_count() != 1u) { return fail("scan has an extra scalar epilogue or use"); }
    const Operation *extract = nullptr;
    for (auto op : prefix->region(0u)->block(0u)->operations()) {
        if (op->kind() == OperationKind::TILE_EXTRACT) { extract = op; }
    }
    if (extract == nullptr || extract->operand_count() != 3u ||
        extract->operand(2u) != prefix->region(0u)->block(0u)->argument(0u)) { return fail("prefix must contribute along the contiguous final axis"); }
    auto source = extract->operand(0u);
    auto loaded = load->result(0u);
    if (source != loaded && one_tile_cast(source, plan.storage, ScalarType::FLOAT32, program) != loaded) {
        return fail("scan input is not the original snapshot or its sole FP32 storage conversion");
    }
    auto stored = store->operand(3u), scanned = map->result(0u);
    if (stored != scanned && one_tile_cast(stored, ScalarType::FLOAT32, plan.storage, program) != scanned) {
        return fail("store value is not the scan or its sole storage conversion");
    }
    for (auto value : {loaded, source, scanned, stored}) {
        if (value->use_count() != 1u) { return fail("an intermediate snapshot has another use"); }
    }
    plan.parallel_operation_id = parallel->id();
    plan.load_operation_id = load->id();
    plan.map_operation_id = map->id();
    plan.store_operation_id = store->id();
    plan.function = &function;
    return plan;
}
}
}// namespace ::closed_prefix_detail

ClosedPrefixAnalysis analyze_closed_prefix(const Function &function) noexcept {
    return closed_prefix_detail::analyze(function);
}
}// namespace luisa::compute::tile
