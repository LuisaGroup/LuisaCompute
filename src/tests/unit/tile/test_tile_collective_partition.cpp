#include "ut/ut.hpp"
#include <luisa/tile/algorithms.h>
#include <luisa/tile/collective_partition.h>
#include <luisa/tile/verifier.h>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::tile;
using namespace boost::ut;

namespace {

enum class Change { NONE,
                    MASK_BOUND,
                    MASK_AXIS,
                    MASK_IDENTITY,
                    NEGATIVE_ZERO,
                    EXTRA_INPUT_USE,
                    EPILOGUE,
                    EXTRA_STORE,
                    FIXED_ORIGIN,
                    CUSTOM_FILL };

template<typename T>
[[nodiscard]] tile::Kernel capture(CollectiveKind kind, bool masked, bool transposed = false,
                                   int64_t rows = 17, int64_t columns = 61, int64_t width = 64,
                                   int64_t block_rows = 4, Change change = Change::NONE,
                                   ReductionPolicy policy = reduction::unordered_tree) {
    return tile_kernel("unrelated_program_partition", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
               auto independent = axis(transposed ? "tail" : "renamed", block_rows);
               auto contribution = axis(transposed ? "first" : "unrelated", width);
               auto domain = transposed ? shape(contribution, independent) : shape(independent, contribution);
               for (auto &program : parallel(shape((rows + block_rows - 1) / block_rows))) {
                   auto row = change == Change::FIXED_ORIGIN ? Scalar<int64_t>{0} : program.index() * block_rows;
                   auto memory = input.tile(transposed ? coord(0, row) : coord(row, 0), domain);
                   auto loaded = cast<float>(change == Change::CUSTOM_FILL ? memory.load(static_cast<T>(1.0f)) : memory.load());
                   // Ordinary SUM frontends also contain a dead coordinate mask.
                   auto coordinate = iota(change == Change::MASK_AXIS ? independent : contribution);
                   auto valid = coordinate < (change == Change::MASK_BOUND ? columns - 1 : columns);
                   auto padding = kind == CollectiveKind::SUM ? 0.0f : -std::numeric_limits<float>::infinity();
                   if (change == Change::MASK_IDENTITY) { padding = 1.0f; }
                   if (change == Change::NEGATIVE_ZERO) { padding = -0.0f; }
                   auto source = masked ? ite(valid, loaded, padding) : loaded;
                   if (change == Change::EXTRA_INPUT_USE) {
                       auto extra = loaded + loaded;
                       static_cast<void>(extra);
                   }
                   auto reduced = kind == CollectiveKind::SUM ? reduce(source, contribution, add, policy) :
                                                                reduce(source, contribution, maximum, policy);
                   if (change == Change::EPILOGUE) { reduced = reduced + 1.0f; }
                   auto stored = cast<T>(reduced);
                   auto single = axis("projection", 1);
                   auto destination = output(transposed ? coord(0, row) : coord(row, 0),
                                             transposed ? shape(single, independent) : shape(independent, single));
                   destination.store(stored);
                   if (change == Change::EXTRA_STORE) { destination.store(stored); }
               }
           })
        .capture(transposed ? tensor_shape(columns, rows) : tensor_shape(rows, columns), transposed ? tensor_shape(1, rows) : tensor_shape(rows, 1));
}

[[nodiscard]] Operation *find(Block &block, OperationKind kind) {
    for (auto op : block.operations()) {
        if (op->kind() == kind) { return op; }
        for (auto &&region : op->regions()) {
            for (auto child : region->blocks()) {
                if (auto found = find(*child, kind)) { return found; }
            }
        }
    }
    return nullptr;
}

void geometry_and_storage() {
    auto check = []<typename T>() {
        for (auto kind : {CollectiveKind::SUM, CollectiveKind::MAXIMUM}) {
            for (auto masked : {false, true}) {
                for (auto transposed : {false, true}) {
                    auto kernel = capture<T>(kind, masked, transposed);
                    expect(kernel.valid());
                    if (!kernel.valid()) { continue; }
                    auto plan = plan_independent_collective(kernel.function());
                    expect(plan.ok()) << plan.error;
                    if (!plan.ok()) { continue; }
                    expect(plan.function == &kernel.function());
                    expect(plan.kind == kind);
                    expect(plan.input_storage == scalar_type_v<T> && plan.output_storage == scalar_type_v<T>);
                    expect(plan.logical_independent_extent == 17u && plan.logical_contribution_extent == 61u);
                    expect(plan.tile_contribution_extent == 64u);
                    expect(plan.input_independent_axis == (transposed ? 1u : 0u));
                    expect(plan.input_contribution_axis == (transposed ? 0u : 1u));
                    expect(plan.output_independent_axis == (transposed ? 1u : 0u) && plan.output_rank == 2u);
                    expect(plan.independent_axis != plan.contribution_axis);
                    expect(plan.contribution_identity_mask == masked);
                    expect(plan.original.programs == 5u && plan.original.independent_extent_per_program == 4u);
                    expect(plan.original.full_programs == 4u && plan.original.tail_valid_extent == 1u);
                    expect(plan.candidate.programs == 17u && plan.candidate.independent_extent_per_program == 1u);
                    expect(plan.candidate.full_programs == 17u && plan.candidate.tail_valid_extent == 0u);
                    expect(plan.disjoint.input.argument_index == 0u && plan.disjoint.output.argument_index == 1u);
                    expect(plan.disjoint.input.byte_offset == 0u && plan.disjoint.output.byte_offset == 0u);
                    expect(plan.disjoint.input.byte_count == 17u * 61u * sizeof(T));
                    expect(plan.disjoint.output.byte_count == 17u * sizeof(T));
                    IndependentPartitionRequest explicit_request{plan.collective_operation_id, plan.independent_axis, 2u};
                    auto two = plan_independent_collective(kernel.function(), explicit_request);
                    expect(two.ok()) << two.error;
                    expect(two.candidate.programs == 9u && two.candidate.full_programs == 8u && two.candidate.tail_valid_extent == 1u);
                    explicit_request.independent_axis = plan.contribution_axis;
                    expect(!plan_independent_collective(kernel.function(), explicit_request).ok());
                    explicit_request.independent_axis = plan.independent_axis;
                    explicit_request.collective_operation_id = plan.load_operation_id;
                    expect(!plan_independent_collective(kernel.function(), explicit_request).ok());
                }
            }
        }
    };
    check.template operator()<float>();
    check.template operator()<half>();
    check.template operator()<bfloat16>();
    // Neither CUDA power-of-two Tiles nor its BR4/8 subset are semantic rules.
    auto non_power_two = capture<float>(CollectiveKind::SUM, false, false, 17, 5, 7, 6);
    auto plan = plan_independent_collective(non_power_two.function(), {.target_extent_per_program = 3u});
    expect(plan.ok()) << plan.error;
    expect(plan.original.programs == 3u && plan.candidate.programs == 6u);
    expect(plan.candidate.full_programs == 5u && plan.candidate.tail_valid_extent == 2u);
    for (auto target : {0u, 4u, 6u, 8u}) {
        expect(!plan_independent_collective(non_power_two.function(), {.target_extent_per_program = target}).ok());
    }
}

void closure_and_mask_rejections() {
    for (auto change : {Change::MASK_BOUND, Change::MASK_AXIS, Change::MASK_IDENTITY,
                        Change::EXTRA_INPUT_USE, Change::EPILOGUE, Change::EXTRA_STORE,
                        Change::FIXED_ORIGIN, Change::CUSTOM_FILL}) {
        auto kernel = capture<float>(CollectiveKind::MAXIMUM, true, false, 17, 61, 64, 4, change);
        expect(kernel.valid());
        expect(verify(*kernel.function().parent_module()).ok());
        auto plan = plan_independent_collective(kernel.function());
        expect(!plan.ok()) << static_cast<uint32_t>(change);
        expect(!plan.error.empty());
    }
    auto signed_zero = capture<float>(CollectiveKind::SUM, true, false, 17, 61, 64, 4, Change::NEGATIVE_ZERO);
    expect(signed_zero.valid());
    expect(!plan_independent_collective(signed_zero.function()).ok());
    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        auto kernel = capture<float>(CollectiveKind::SUM, false, false, 17, 61, 64, 4, Change::NONE, policy);
        expect(kernel.valid());
        expect(!plan_independent_collective(kernel.function()).ok());
    }
    auto extra = capture<float>(CollectiveKind::SUM, false);
    auto program = find(*extra.function().body().block(0u), OperationKind::PARALLEL);
    program->set_execution_scope_constraint("parallel");
    expect(verify(*extra.function().parent_module()).ok());
    expect(!plan_independent_collective(extra.function()).ok());
}

void rank_one_and_alias_requirements() {
    auto rank_one = tile_kernel("rank_one_output", [](TensorView<const half, 2> input, TensorView<float, 1> output) {
                        auto r = axis("batch", 8), c = axis("reduce", 16);
                        for (auto &p : parallel(shape(2))) {
                            auto loaded = cast<float>(input.tile(coord(p.index() * int64_t{8}, 0), shape(r, c)).load());
                            output(coord(p.index() * int64_t{8}), shape(r)).store(reduce(loaded, c, add));
                        }
                    }).capture(tensor_shape(16, 16), tensor_shape(16));
    expect(rank_one.valid());
    auto plan = plan_independent_collective(rank_one.function());
    expect(plan.ok()) << plan.error;
    expect(plan.output_rank == 1u && plan.output_independent_axis == 0u);
    expect(plan.input_storage == ScalarType::FLOAT16 && plan.output_storage == ScalarType::FLOAT32);
    expect(plan.disjoint.input.byte_count == 16u * 16u * sizeof(half));
    expect(plan.disjoint.output.byte_count == 16u * sizeof(float));
    // Same root resource cannot satisfy the required nonempty disjoint ranges.
    auto aliased = tile_kernel("same_root_snapshot", [](TensorView<float, 2> memory) {
                       auto r = axis("r", 4), c = axis("c", 16);
                       for (auto &p : parallel(shape(1))) {
                           auto x = memory.tile(coord(0, 0), shape(r, c)).load();
                           memory(coord(0, 0), shape(r, axis("one", 1))).store(reduce(x, c, add));
                       }
                   }).capture(tensor_shape(4, 16));
    expect(aliased.valid());
    expect(!plan_independent_collective(aliased.function()).ok());
    // Distinct root arguments are only a conditional plan; their actual bound
    // byte intervals must still be checked by the runtime for every dispatch.
    expect(plan.disjoint.input.argument_index != plan.disjoint.output.argument_index);
}

void overflow_and_projection_rejections() {
    auto huge = capture<float>(CollectiveKind::SUM, false, false, int64_t{1} << 61, 2, 2, 4);
    expect(huge.valid());
    expect(verify(*huge.function().parent_module()).ok());
    auto overflow = plan_independent_collective(huge.function());
    expect(!overflow.ok());
    expect(overflow.error.find("byte interval overflows") != string::npos) << overflow.error;
    auto kernel = capture<float>(CollectiveKind::SUM, false);
    auto root = kernel.function().body().block(0u);
    auto store = find(*root, OperationKind::VIEW_STORE);
    auto projection = store->operand(store->operand_count() - 1u)->defining_operation();
    expect(projection != nullptr && projection->kind() == OperationKind::TILE_MAP);
    if (projection == nullptr || projection->kind() != OperationKind::TILE_MAP) { return; }
    auto extract = find(*projection->region(0u)->block(0u), OperationKind::TILE_EXTRACT);
    expect(extract != nullptr);
    if (extract == nullptr) { return; }
    IRBuilder builder;
    builder.set_insertion_point(extract);
    Value *operands[]{extract->operand(1u), extract->operand(1u)};
    auto computed = builder.create_elementwise(ElementwiseOp::ADD, operands, extract->operand(1u)->type());
    expect(computed != nullptr);
    if (computed == nullptr) { return; }
    extract->set_operand(1u, computed->result(0u));
    expect(verify(*kernel.function().parent_module()).ok());
    expect(!plan_independent_collective(kernel.function()).ok());
}

void candidate_work_facts() {
    auto kernel = capture<half>(CollectiveKind::MAXIMUM, true);
    auto plan = plan_independent_collective(kernel.function(), {.target_extent_per_program = 2u});
    expect(plan.ok()) << plan.error;
    if (!plan.ok()) { return; }
    auto original = analyze_independent_collective_candidate(plan, IndependentCollectiveGeometryKind::ORIGINAL);
    auto candidate = analyze_independent_collective_candidate(plan);
    expect(original.ok()) << original.error;
    expect(candidate.ok()) << candidate.error;
    expect(original.geometry.programs == 5u && candidate.geometry.programs == 9u);
    expect(original.collective_input_elements_per_program == 256u && candidate.collective_input_elements_per_program == 128u);
    expect(original.collective_input_elements_total == 1280u && candidate.collective_input_elements_total == 1152u);
    expect(candidate.padded_independent_elements == 18u);
    expect(candidate.valid_input_elements == 17u * 61u && candidate.valid_output_elements == 17u);
    expect(candidate.valid_input_bytes == 17u * 61u * 2u && candidate.valid_output_bytes == 34u);
    expect(candidate.input_snapshot_bytes_per_program == 256u && candidate.fp32_source_bytes_per_program == 512u);
    expect(candidate.fp32_result_bytes_per_program == 8u && candidate.output_value_bytes_per_program == 4u);
    expect(!candidate.independent_bounds_elidable && !candidate.contribution_bounds_elidable);
    expect(candidate.contribution_identity_mask && candidate.kind == CollectiveKind::MAXIMUM);
    expect(candidate.independent_axis == plan.independent_axis && candidate.contribution_axis == plan.contribution_axis);
    expect(candidate.collective_operation_id == plan.collective_operation_id);
    auto one = plan_independent_collective(kernel.function());
    expect(analyze_independent_collective_candidate(one).independent_bounds_elidable);
    auto full = capture<float>(CollectiveKind::SUM, false, true, 16, 64, 64, 4);
    auto full_plan = plan_independent_collective(full.function());
    auto full_facts = analyze_independent_collective_candidate(full_plan);
    expect(full_facts.ok()) << full_facts.error;
    expect(full_facts.independent_bounds_elidable && full_facts.contribution_bounds_elidable);
    expect(!full_facts.contribution_identity_mask);
    auto non_power_two = capture<float>(CollectiveKind::SUM, false, false, 17, 5, 7, 6);
    auto non_power_plan = plan_independent_collective(non_power_two.function(), {.target_extent_per_program = 3u});
    expect(analyze_independent_collective_candidate(non_power_plan).ok());
    expect(!analyze_independent_collective_candidate(plan, static_cast<IndependentCollectiveGeometryKind>(255u)).ok());
    for (auto variant = 0u; variant < 8u; variant++) {
        auto invalid = plan;
        switch (variant) {
            case 0u: invalid.error = "unsupported"; break;
            case 1u: invalid.candidate.programs++; break;
            case 2u: invalid.candidate.independent_extent_per_program = 0u; break;
            case 3u: invalid.logical_contribution_extent = 65u; break;
            case 4u: invalid.disjoint.output.byte_count++; break;
            case 5u: invalid.disjoint.input.byte_offset = std::numeric_limits<uint64_t>::max(); break;
            case 6u: invalid.disjoint.output.argument_index = invalid.disjoint.input.argument_index; break;
            case 7u: invalid.input_storage = ScalarType::INT32; break;
        }
        expect(!analyze_independent_collective_candidate(invalid).ok()) << variant;
    }
    auto overflow = plan;
    overflow.tile_contribution_extent = std::numeric_limits<uint64_t>::max();
    auto rejected = analyze_independent_collective_candidate(overflow);
    expect(!rejected.ok());
    expect(rejected.error.find("arithmetic overflows") != string::npos);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_collective_partition_geometry_storage"_test = [] { geometry_and_storage(); };
    "tile_collective_partition_closure_mask"_test = [] { closure_and_mask_rejections(); };
    "tile_collective_partition_rank_alias"_test = [] { rank_one_and_alias_requirements(); };
    "tile_collective_partition_overflow_projection"_test = [] { overflow_and_projection_rejections(); };
    "tile_collective_partition_work_facts"_test = [] { candidate_work_facts(); };
}
