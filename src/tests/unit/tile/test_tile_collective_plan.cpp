#include "ut/ut.hpp"
#include <luisa/tile/collective_plan.h>
#include <luisa/tile/collective_cost.h>
#include <cmath>
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

void cost_features() {
    CollectiveWorkAnalysis facts;
    facts.programs = 80u;
    facts.elementwise_elements_per_program = 1600u;
    facts.global_read_bytes_per_program = facts.global_write_bytes_per_program = 800u;
    facts.materialized_tile_total_bytes = 8192u;
    facts.materialized_tile_peak_bytes = 4096u;
    facts.largest_materialized_tile_elements = 256u;
    facts.collectives = {{1u, CollectiveKind::SUM, ScalarType::FLOAT32, 64u, 2u, 128u},
                         {2u, CollectiveKind::MAXIMUM, ScalarType::FLOAT32, 32u, 4u, 128u},
                         {3u, CollectiveKind::INCLUSIVE_SUM, ScalarType::FLOAT32, 128u, 1u, 128u},
                         {4u, CollectiveKind::MINIMUM, ScalarType::FLOAT32, 16u, 1u, 16u}};
    auto result = collective_cost_features(facts, {20u, 32u});
    expect(result.ok()) << result.error;
    // Frozen binary64 Python feature_vector parity, C=400. MINIMUM remains
    // representable here; a fitted backend may separately exclude that family.
    constexpr std::array expected{2.321928094887362, 8.005624549193879, 5.044394119358453,
                                  2.321928094887362, 2.321928094887362, 3.169925001442312,
                                  0.32, 0.32, 1.0};
    for (auto i = size_t{0u}; i < expected.size(); i++) { expect(std::abs(result.values[i] - expected[i]) < 2e-14); }
    expect(!collective_cost_features(facts, {0u, 32u}).ok());
    expect(!collective_cost_features(facts, {20u, 0u}).ok());
    auto bad = facts;
    bad.error = "not admitted";
    expect(!collective_cost_features(bad, {20u, 32u}).ok());
    bad = facts;
    bad.collectives[0u].input_elements++;
    expect(!collective_cost_features(bad, {20u, 32u}).ok());
    bad = facts;
    bad.collectives[0u].contribution_extent = std::numeric_limits<uint64_t>::max();
    expect(!collective_cost_features(bad, {20u, 32u}).ok());
    bad = facts;
    bad.collectives = {{0u, CollectiveKind::SUM, ScalarType::FLOAT32, std::numeric_limits<uint64_t>::max(), 1u, std::numeric_limits<uint64_t>::max()},
                       {1u, CollectiveKind::MAXIMUM, ScalarType::FLOAT32, 1u, 1u, 1u}};
    expect(!collective_cost_features(bad, {20u, 32u}).ok());
    bad = facts;
    bad.global_read_bytes_per_program = std::numeric_limits<uint64_t>::max();
    expect(!collective_cost_features(bad, {20u, 32u}).ok());
    bad = facts;
    bad.collectives[0u].kind = static_cast<CollectiveKind>(255u);
    expect(!collective_cost_features(bad, {20u, 32u}).ok());
    auto kernel = capture(TestKind::SCAN, 17, 8192, 4);
    auto admitted = analyze_collective_work(kernel.function());
    expect(admitted.ok());
    expect(collective_cost_features(admitted, {76u, 32u}).ok());
}

void cost_tree() {
    std::array<double, 9u> features{};
    features[0u] = .5;
    std::array<CollectiveCostTreeNode, 3u> tree{{{0, .5, 1u, 2u, 0.0}, {-1, 0.0, 0u, 0u, -.1}, {-1, 0.0, 0u, 0u, .2}}};
    auto evaluate = [&](auto &&nodes) { return evaluate_collective_cost_tree(span<const CollectiveCostTreeNode>{nodes}, span<const double>{features}); };
    auto result = evaluate(tree);
    expect(result.ok());
    expect(result.log_score == -.1);
    features[0u] = std::nextafter(.5, 1.0);
    expect(evaluate(tree).log_score == .2);
    auto bad = tree;
    bad[0u].feature = 9;
    expect(!evaluate(bad).ok());
    bad[0u].feature = -2;
    expect(!evaluate(bad).ok());
    bad = tree;
    bad[0u].right = 3u;
    expect(!evaluate(bad).ok());
    bad = tree;
    bad[0u].left = 0u;
    expect(!evaluate(bad).ok());
    bad = tree;
    bad[0u].right = 1u;// The unselected node still must be well-formed.
    bad[2u] = {0, 0.0, 2u, 2u, 0.0};
    expect(!evaluate(bad).ok());
    bad = tree;
    bad[2u].log_score = std::numeric_limits<double>::quiet_NaN();
    expect(!evaluate(bad).ok());
    bad = tree;
    bad[0u].threshold = std::numeric_limits<double>::infinity();
    expect(!evaluate(bad).ok());
    features[8u] = std::numeric_limits<double>::quiet_NaN();
    expect(!evaluate(tree).ok());
    features[8u] = 0.0;
    expect(!evaluate_collective_cost_tree({}, span<const double>{features}).ok());
    expect(!evaluate_collective_cost_tree(span<const CollectiveCostTreeNode>{tree}, {}).ok());
    std::array<CollectiveCostTreeNode, kCollectiveCostMaxNodes> chain{};
    for (auto i = uint32_t{0u}; i + 1u < chain.size(); i++) { chain[i] = {0, 0.0, i + 1u, i + 1u, 0.0}; }
    chain.back().log_score = -.25;
    expect(evaluate(chain).ok());
    expect(evaluate(chain).log_score == -.25);
    std::array<CollectiveCostTreeNode, kCollectiveCostMaxNodes + 1u> oversized{};
    expect(!evaluate(oversized).ok());
    std::array<double, kCollectiveCostMaxFeatures + 1u> extra_features{};
    expect(!evaluate_collective_cost_tree(span<const CollectiveCostTreeNode>{tree}, span<const double>{extra_features}).ok());
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_collective_work_logical_features"_test = [] { features(); };
    "tile_collective_work_admission"_test = [] { rejections(); };
    "tile_collective_cost_features_v3"_test = [] { cost_features(); };
    "tile_collective_cost_bounded_tree"_test = [] { cost_tree(); };
}
