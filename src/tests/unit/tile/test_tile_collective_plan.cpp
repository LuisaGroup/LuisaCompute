#include "ut/ut.hpp"
#include <luisa/tile/collective_plan.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/verifier.h>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::tile;
using namespace boost::ut;

namespace {

enum class TestKind { SUM,
                      MAXIMUM,
                      SCAN,
                      RMS,
                      SOFTMAX,
                      ORDERED_SCAN };

[[nodiscard]] tile::Kernel capture(TestKind kind, int64_t rows, int64_t width, int64_t block_rows = 1) {
    return tile_kernel("renamed_collective", [=](TensorView<const half, 2> input, TensorView<half, 2> output) {
               auto row = axis("u", block_rows), column = axis("v", width);
               for (auto &program : parallel(shape((rows + block_rows - 1) / block_rows))) {
                   auto origin = program.index() * block_rows;
                   auto value = cast<float>(input.tile(coord(origin, 0), shape(row, column)).load());
                   if (kind == TestKind::SUM || kind == TestKind::MAXIMUM) {
                       auto result = kind == TestKind::SUM ? reduce(value, column, add) : reduce(value, column, maximum);
                       output(coord(origin, 0), shape(row, axis("out", 1))).store(cast<half>(result));
                   } else {
                       auto result = value;
                       if (kind == TestKind::SCAN || kind == TestKind::ORDERED_SCAN) {
                           result = inclusive_sum(value, column, kind == TestKind::SCAN ? reduction::unordered_tree : reduction::fold_left);
                       } else if (kind == TestKind::RMS) {
                           auto scale = 1.0f / sqrt(reduce(value * value, column, add) / static_cast<float>(width) + 1e-5f);
                           result = value * scale;
                       } else {
                           auto shifted = value - reduce(value, column, maximum);
                           auto exponent = exp(shifted);
                           result = exponent / reduce(exponent, column, add);
                       }
                       output(coord(origin, 0), shape(row, column)).store(cast<half>(result));
                   }
               }
           })
        .capture(tensor_shape(rows, width), tensor_shape(rows, width));
}

[[nodiscard]] Operation *find(Block &block, OperationKind kind) {
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

void features() {
    for (auto width : {int64_t{64}, int64_t{513}, int64_t{8192}}) {
        for (auto block_rows : {int64_t{1}, int64_t{4}, int64_t{8}}) {
            for (auto kind : {TestKind::SUM, TestKind::MAXIMUM, TestKind::SCAN}) {
                auto kernel = capture(kind, 17, width, block_rows);
                expect(kernel.valid());
                auto result = analyze_collective_work(kernel.function());
                expect(result.ok()) << result.error;
                if (!result.ok()) { continue; }
                expect(result.programs == static_cast<uint64_t>((17 + block_rows - 1) / block_rows));
                expect(result.collectives.size() == 1u);
                auto &&work = result.collectives.front();
                expect(work.element == ScalarType::FLOAT32);
                expect(work.contribution_extent == static_cast<uint64_t>(width));
                expect(work.independent_elements == static_cast<uint64_t>(block_rows));
                expect(work.input_elements == static_cast<uint64_t>(block_rows * width));
                expect(work.kind == (kind == TestKind::SCAN ? CollectiveKind::INCLUSIVE_SUM : kind == TestKind::SUM ? CollectiveKind::SUM :
                                                                                                                      CollectiveKind::MAXIMUM));
                auto input_elements = static_cast<uint64_t>(block_rows * width);
                auto output_elements = static_cast<uint64_t>(kind == TestKind::SCAN ? block_rows * width : block_rows);
                expect(result.global_read_bytes_per_program == input_elements * sizeof(half));
                expect(result.global_write_bytes_per_program == output_elements * sizeof(half));
                expect(result.elementwise_elements_per_program == input_elements + output_elements + 1u);
                expect(result.materialized_tile_peak_bytes == input_elements * (kind == TestKind::SCAN ? 8u : 6u));
                expect(result.largest_materialized_tile_elements == input_elements);
                expect(result.materialized_tile_total_bytes >= result.materialized_tile_peak_bytes);
            }
        }
    }
    for (auto kind : {TestKind::RMS, TestKind::SOFTMAX}) {
        auto kernel = capture(kind, 128, 8192);
        auto result = analyze_collective_work(kernel.function());
        expect(result.ok()) << result.error;
        expect(result.collectives.size() == (kind == TestKind::RMS ? 1u : 2u));
        expect(result.global_read_bytes_per_program == 16384u);
        expect(result.global_write_bytes_per_program == 16384u);
        expect(result.materialized_tile_peak_bytes > 65536u);
    }
}

void extended_admission() {
    // Axis spelling and physical position must not identify the operation.
    auto interior = tile_kernel("unrelated_interior_prefix", [](TensorView<const float, 3> input,
                                                                TensorView<float, 3> output) {
                        auto first = axis("right", 2), reduced = axis("unrelated", 64), last = axis("left", 4);
                        for (auto &program : parallel(shape(3))) {
                            auto origin = program.index() * int64_t{2};
                            auto input_tile = input.tile(coord(origin, 0, 0), shape(first, reduced, last)).load();
                            auto scanned = inclusive_sum(input_tile, reduced, reduction::unordered_tree);
                            output(coord(origin, 0, 0), shape(first, reduced, last)).store(scanned);
                        }
                    }).capture(tensor_shape(6, 64, 4), tensor_shape(6, 64, 4));
    expect(interior.valid());
    auto interior_work = analyze_collective_work(interior.function());
    expect(interior_work.ok()) << interior_work.error;
    if (interior_work.ok()) {
        expect(interior_work.programs == 3u);
        expect(interior_work.collectives.size() == 1u);
        auto &&work = interior_work.collectives.front();
        expect(work.kind == CollectiveKind::INCLUSIVE_SUM);
        expect(work.contribution_extent == 64u);
        expect(work.independent_elements == 8u);
        expect(work.input_elements == 512u);
        expect(interior_work.global_read_bytes_per_program == 2048u);
        expect(interior_work.global_write_bytes_per_program == 2048u);
    }

    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        auto ordered = capture(TestKind::SCAN, 2, 64);
        auto reducer = find(*ordered.function().body().block(0u), OperationKind::REDUCE);
        expect(reducer != nullptr);
        if (reducer == nullptr) { continue; }
        reducer->set_reduction_policy(policy);
        expect(verify(*ordered.function().parent_module()).ok());
        expect(!analyze_collective_work(ordered.function()).ok());
    }

    // Values below remain well-typed legal IR, but do not establish the
    // recognized +0 seed. In particular -0 cannot be silently canonicalized.
    for (auto seed : {1.0, -0.0, std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity()}) {
        auto nonidentity = capture(TestKind::SUM, 2, 64);
        auto reducer = find(*nonidentity.function().body().block(0u), OperationKind::REDUCE);
        expect(reducer != nullptr);
        if (reducer == nullptr) { continue; }
        auto initializer = reducer->operand(0u)->defining_operation();
        expect(initializer != nullptr);
        if (initializer == nullptr) { continue; }
        initializer->set_attribute("value", tile::Attribute{seed});
        expect(verify(*nonidentity.function().parent_module()).ok());
        expect(!analyze_collective_work(nonidentity.function()).ok());
    }

    auto wrong_max_identity = capture(TestKind::MAXIMUM, 2, 64);
    auto maximum = find(*wrong_max_identity.function().body().block(0u), OperationKind::REDUCE);
    expect(maximum != nullptr);
    if (maximum != nullptr) {
        maximum->operand(0u)->defining_operation()->set_attribute(
            "value", tile::Attribute{std::numeric_limits<double>::infinity()});
        expect(verify(*wrong_max_identity.function().parent_module()).ok());
        expect(!analyze_collective_work(wrong_max_identity.function()).ok());
    }

    // Place a legal explicit scope on PARALLEL, not on REDUCE (where verifier
    // itself would reject it). A cost analysis cannot ignore this constraint.
    auto scoped = capture(TestKind::SUM, 2, 64);
    auto parallel_op = find(*scoped.function().body().block(0u), OperationKind::PARALLEL);
    expect(parallel_op != nullptr);
    if (parallel_op != nullptr) {
        parallel_op->set_execution_scope_constraint("subgroup");
        expect(verify(*scoped.function().parent_module()).ok());
        expect(!analyze_collective_work(scoped.function()).ok());
    }

    // An additional pure instruction still breaks the closed merge proof;
    // numerical permission alone cannot identify an arbitrary reducer body.
    auto extra_pure_work = capture(TestKind::SUM, 2, 64);
    auto reducer = find(*extra_pure_work.function().body().block(0u), OperationKind::REDUCE);
    expect(reducer != nullptr);
    if (reducer != nullptr) {
        auto body = reducer->region(0u)->block(0u);
        auto extract = find(*body, OperationKind::TILE_EXTRACT);
        expect(extract != nullptr);
        if (extract != nullptr) {
            IRBuilder builder;
            builder.set_insertion_point(body->operations().back());
            Value *operands[]{extract->result(0u), extract->result(0u)};
            static_cast<void>(builder.create_elementwise(ElementwiseOp::MUL, operands, extract->result(0u)->type()));
            expect(verify(*extra_pure_work.function().parent_module()).ok());
            expect(!analyze_collective_work(extra_pure_work.function()).ok());
        }
    }
}

void mapped_projections() {
    // A rank-one reduction value stored into (row, singleton) becomes a
    // second map containing a pure projection. This is normal DSL broadcast.
    for (auto block_rows : {int64_t{1}, int64_t{4}}) {
        auto kernel = capture(TestKind::SUM, 8, 64, block_rows);
        expect(verify(*kernel.function().parent_module()).ok());
        auto baseline = analyze_collective_work(kernel.function());
        expect(baseline.ok()) << baseline.error;
        auto program = find(*kernel.function().body().block(0u), OperationKind::PARALLEL)->region(0u)->block(0u);
        Operation *projection = nullptr;
        for (auto op : program->operations()) {
            if (op->kind() != OperationKind::TILE_MAP) { continue; }
            for (auto child : op->region(0u)->block(0u)->operations()) {
                if (child->kind() == OperationKind::TILE_EXTRACT) { projection = child; }
            }
        }
        expect(projection != nullptr);
        if (projection == nullptr) { continue; }
        // A cast between INDEX and int64 does not change the projection.
        auto original = projection->operand(1u);
        IRBuilder builder;
        builder.set_insertion_point(projection);
        Value *cast_operands[]{original};
        auto converted = builder.create_elementwise(ElementwiseOp::CAST, cast_operands, tile::Type::scalar(ScalarType::INT64));
        expect(converted != nullptr);
        if (converted == nullptr) { continue; }
        projection->set_operand(1u, converted->result(0u));
        expect(verify(*kernel.function().parent_module()).ok());
        expect(analyze_collective_work(kernel.function()).ok());
        // Even x + x is unproved computed indexing; it must not gain admission
        // merely because its zero/small current extent happens to be safe.
        Value *computed_operands[]{converted->result(0u), converted->result(0u)};
        auto computed = builder.create_elementwise(ElementwiseOp::ADD, computed_operands, tile::Type::scalar(ScalarType::INT64));
        expect(computed != nullptr);
        if (computed == nullptr) { continue; }
        projection->set_operand(1u, computed->result(0u));
        expect(verify(*kernel.function().parent_module()).ok());
        expect(!analyze_collective_work(kernel.function()).ok());
    }
}

void rejections() {
    extended_admission();
    mapped_projections();
    auto ordered = capture(TestKind::ORDERED_SCAN, 2, 64);
    expect(ordered.valid());
    expect(!analyze_collective_work(ordered.function()).ok());
    auto attribute = capture(TestKind::SUM, 2, 64);
    find(*attribute.function().body().block(0u), OperationKind::REDUCE)->set_attribute("unknown", tile::Attribute{true});
    expect(!analyze_collective_work(attribute.function()).ok());
    auto extra_use = capture(TestKind::SUM, 2, 64);
    auto reducer = find(*extra_use.function().body().block(0u), OperationKind::REDUCE);
    auto body = reducer->region(0u)->block(0u);
    IRBuilder builder;
    builder.set_insertion_point(body->operations().back());
    Value *operands[]{body->argument(1u)};
    static_cast<void>(builder.create_elementwise(ElementwiseOp::CAST, operands, body->argument(1u)->type()));
    expect(extra_use.valid());
    expect(!analyze_collective_work(extra_use.function()).ok());
    auto detached = capture(TestKind::SUM, 2, 64);
    auto owned = detached.function().remove_self();
    expect(!analyze_collective_work(*owned).ok());
    tile::Function orphan{nullptr, 0u, "orphan", IRForm::CANDIDATE};
    expect(!analyze_collective_work(orphan).ok());
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_collective_work_logical_features"_test = [] { features(); };
    "tile_collective_work_admission"_test = [] { rejections(); };
}
