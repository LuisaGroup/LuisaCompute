#include "ut/ut.hpp"
#include <luisa/tile/algorithms.h>
#include <luisa/tile/collective_prefix.h>
#include <luisa/tile/verifier.h>
#include <array>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::tile;
using namespace boost::ut;

namespace {
enum class Change { NONE,
                    RENAME,
                    TRANSPOSE,
                    FIXED_ORIGIN,
                    EXTRA_INPUT_USE,
                    EPILOGUE,
                    EXTRA_LOAD,
                    EXTRA_STORE,
                    CUSTOM_FILL,
                    ASSUME_BOUNDS,
                    OUTPUT_SHAPE };

template<typename T>
[[nodiscard]] tile::Kernel capture(int64_t rows = 17, int64_t columns = 61,
                                   int64_t width = 64, int64_t block_rows = 4,
                                   Change change = Change::NONE) {
    auto transposed = change == Change::TRANSPOSE;
    auto output_columns = columns + (change == Change::OUTPUT_SHAPE ? 1 : 0);
    return tile_kernel("unrelated_closed_dataflow", [=](TensorView<const T, 1> unused0,
                                                        TensorView<const T, 2> input,
                                                        TensorView<const T, 1> unused2,
                                                        TensorView<T, 2> output) {
               static_cast<void>(unused0);
               static_cast<void>(unused2);
               auto r = axis(change == Change::RENAME ? "other" : "independent", block_rows);
               auto c = axis(change == Change::RENAME ? "renamed" : "contribution", width);
               auto domain = transposed ? shape(c, r) : shape(r, c);
               for (auto &program : parallel(shape((rows - 1) / block_rows + 1))) {
                   auto row = change == Change::FIXED_ORIGIN ? Scalar<int64_t>{0} : program.index() * block_rows;
                   auto origin = transposed ? coord(0, row) : coord(row, 0);
                   auto memory = input.tile(origin, domain, change == Change::ASSUME_BOUNDS ? bounds::assume : bounds::zero);
                   auto x = cast<float>(change == Change::CUSTOM_FILL ? memory.load(static_cast<T>(1.0f)) : memory.load());
                   auto unused_mask = iota(c) < columns;
                   static_cast<void>(unused_mask);
                   if (change == Change::EXTRA_INPUT_USE) {
                       auto extra = x + x;
                       static_cast<void>(extra);
                   }
                   if (change == Change::EXTRA_LOAD) {
                       auto extra = input.tile(origin, domain).load();
                       static_cast<void>(extra);
                   }
                   auto scanned = inclusive_sum(x, c, reduction::unordered_tree);
                   if (change == Change::EPILOGUE) { scanned = scanned + 1.0f; }
                   auto stored = cast<T>(scanned);
                   output(origin, domain).store(stored);
                   if (change == Change::EXTRA_STORE) { output(origin, domain).store(stored); }
               }
           })
        .capture(tensor_shape(1), transposed ? tensor_shape(columns, rows) : tensor_shape(rows, columns), tensor_shape(1), transposed ? tensor_shape(output_columns, rows) : tensor_shape(rows, output_columns));
}

[[nodiscard]] Operation *find(Block &block, OperationKind kind) noexcept {
    for (auto op : block.operations()) {
        if (op->kind() == kind) { return op; }
        for (auto &&region : op->regions()) {
            for (auto child : region->blocks()) {
                if (auto result = find(*child, kind)) { return result; }
            }
        }
    }
    return nullptr;
}

void geometry_and_slots() {
    auto check = []<typename T>() {
        for (auto change : {Change::NONE, Change::RENAME}) {
            for (auto rows : {int64_t{3}, int64_t{17}, int64_t{24}}) {
                for (auto block_rows : {int64_t{1}, int64_t{3}, int64_t{4}, int64_t{8}, int64_t{12}}) {
                    auto kernel = capture<T>(rows, 61, 64, block_rows, change);
                    expect(kernel.valid());
                    if (!kernel.valid()) { continue; }
                    auto p = analyze_closed_prefix(kernel.function());
                    expect(p.ok()) << p.error;
                    if (!p.ok()) { continue; }
                    expect(p.function == &kernel.function());
                    expect(p.collective.kind == CollectiveKind::INCLUSIVE_SUM);
                    expect(p.collective.element == ScalarType::FLOAT32 && p.storage == scalar_type_v<T>);
                    expect(p.collective.contribution_extent == 64u);
                    expect(p.collective.independent_elements == static_cast<uint64_t>(block_rows));
                    expect(p.collective.input_elements == static_cast<uint64_t>(block_rows * 64));
                    expect(p.logical_independent_extent == static_cast<uint64_t>(rows));
                    expect(p.logical_contribution_extent == 61u);
                    expect(p.original.programs == static_cast<uint64_t>((rows - 1) / block_rows + 1));
                    expect(p.original.independent_extent_per_program == static_cast<uint64_t>(block_rows));
                    expect(p.original.full_programs == static_cast<uint64_t>(rows / block_rows));
                    expect(p.original.tail_valid_extent == static_cast<uint64_t>(rows % block_rows));
                    expect(p.disjoint.input.argument_index == 1u && p.disjoint.output.argument_index == 3u);
                    expect(p.disjoint.input.byte_offset == 0u && p.disjoint.output.byte_offset == 0u);
                    expect(p.disjoint.input.byte_count == static_cast<uint64_t>(rows * 61 * sizeof(T)));
                    expect(p.disjoint.output.byte_count == p.disjoint.input.byte_count);
                    auto load = find(*kernel.function().body().block(0u), OperationKind::VIEW_LOAD);
                    expect(load != nullptr);
                    if (load != nullptr) {
                        expect(p.load_operation_id == load->id());
                        expect(p.independent_axis == load->domain()->axis(0u).dimension);
                        expect(p.contribution_axis == load->domain()->axis(1u).dimension);
                    }
                }
            }
        }
    };
    check.template operator()<float>();
    check.template operator()<half>();
    check.template operator()<bfloat16>();
    // No CUDA power-of-two width, BR<=8 or width<=65536 gate in shared proof.
    for (auto geometry : {std::array<int64_t, 3u>{5, 7, 6}, std::array<int64_t, 3u>{65537, 65539, 16}}) {
        auto kernel = capture<float>(17, geometry[0u], geometry[1u], geometry[2u]);
        expect(kernel.valid());
        auto p = analyze_closed_prefix(kernel.function());
        expect(p.ok()) << p.error;
        expect(p.collective.contribution_extent == static_cast<uint64_t>(geometry[1u]));
    }
}

void closure_and_numerical_rejections() {
    for (auto change : {Change::TRANSPOSE, Change::FIXED_ORIGIN, Change::EXTRA_INPUT_USE,
                        Change::EPILOGUE, Change::EXTRA_LOAD, Change::EXTRA_STORE,
                        Change::CUSTOM_FILL, Change::ASSUME_BOUNDS, Change::OUTPUT_SHAPE}) {
        auto kernel = capture<float>(17, 61, 64, 4, change);
        expect(kernel.valid());
        expect(verify(*kernel.function().parent_module()).ok());
        auto p = analyze_closed_prefix(kernel.function());
        expect(!p.ok()) << static_cast<uint32_t>(change);
        expect(!p.error.empty());
        // Outer padding is checked by this closure matcher, not established
        // by the reducer's own +0 seed / prefix-select false value proof.
        if (change == Change::CUSTOM_FILL || change == Change::ASSUME_BOUNDS) {
            auto work = analyze_collective_work(kernel.function());
            expect(work.ok()) << work.error;
        }
    }
    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        auto kernel = capture<float>();
        auto reducer = find(*kernel.function().body().block(0u), OperationKind::REDUCE);
        expect(reducer != nullptr);
        if (reducer == nullptr) { continue; }
        reducer->set_reduction_policy(policy);
        expect(verify(*kernel.function().parent_module()).ok());
        expect(!analyze_closed_prefix(kernel.function()).ok());
    }
    for (auto seed : {1.0, -0.0, std::numeric_limits<double>::quiet_NaN()}) {
        auto kernel = capture<float>();
        auto reducer = find(*kernel.function().body().block(0u), OperationKind::REDUCE);
        expect(reducer != nullptr);
        if (reducer == nullptr) { continue; }
        reducer->operand(0u)->defining_operation()->set_attribute("value", tile::Attribute{seed});
        expect(verify(*kernel.function().parent_module()).ok());
        expect(!analyze_closed_prefix(kernel.function()).ok());
    }
    auto incomplete = capture<float>(17, 65, 64, 4);
    expect(incomplete.valid());
    expect(analyze_collective_work(incomplete.function()).ok());
    expect(!analyze_closed_prefix(incomplete.function()).ok());

    auto aliased = tile_kernel("same_root_prefix", [](TensorView<float, 2> memory) {
                       auto r = axis("r", 4), c = axis("c", 16);
                       for (auto &p : parallel(shape(1))) {
                           auto row = p.index() * int64_t{4};
                           auto value = memory.tile(coord(row, 0), shape(r, c)).load();
                           memory(coord(row, 0), shape(r, c)).store(inclusive_sum(value, c));
                       }
                   }).capture(tensor_shape(4, 16));
    expect(aliased.valid());
    expect(!analyze_closed_prefix(aliased.function()).ok());
}

void overflow_and_coordinates() {
    auto huge = capture<float>(int64_t{1} << 61, 2, 2, 4);
    expect(huge.valid());
    expect(verify(*huge.function().parent_module()).ok());
    expect(analyze_collective_work(huge.function()).ok());
    auto p = analyze_closed_prefix(huge.function());
    expect(!p.ok());
    expect(p.error.find("overflows int64") != string::npos) << p.error;
    auto kernel = capture<float>();
    auto extract = find(*kernel.function().body().block(0u), OperationKind::TILE_EXTRACT);
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
    expect(!analyze_closed_prefix(kernel.function()).ok());
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_closed_prefix_geometry_and_slots"_test = geometry_and_slots;
    "tile_closed_prefix_closure_and_numerical_rejections"_test = closure_and_numerical_rejections;
    "tile_closed_prefix_overflow_and_coordinates"_test = overflow_and_coordinates;
}
