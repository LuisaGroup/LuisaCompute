// Test for the execution-first TileIR core.
// This test covers:
// - owned structured regions, lexical SSA, and use-def rewriting
// - semantic MMA dimension verification and explicit MemoryState effects
// - target execution containment and independent resource access capabilities
// - RTTI-free cached analyses and invalidation

#include "ut/ut.hpp"

#include <luisa/tile/analysis.h>
#include <luisa/tile/verifier.h>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_operations(const Region &region) noexcept {
    size_t count = 0u;
    for (auto &&block : region.blocks()) {
        count += block->operation_count();
        for (auto &&operation : block->operations()) {
            for (auto &&child : operation->regions()) { count += count_operations(*child); }
        }
    }
    return count;
}

struct OperationCountAnalysis {
    using Result = size_t;
    static uint32_t runs;
    [[nodiscard]] static Result run(const Function &function) noexcept {
        runs++;
        return count_operations(function.body());
    }
};

uint32_t OperationCountAnalysis::runs = 0u;

struct GemmTypes {
    Dim m;
    Dim n;
    Dim k;
    IndexSpace a;
    IndexSpace b;
    IndexSpace c;
};

[[nodiscard]] GemmTypes make_gemm_types(Module &module) noexcept {
    auto &dimensions = module.dimensions();
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto k = dimensions.create_dimension("reduction");
    IndexSpace a;
    static_cast<void>(a.add(m, 2u));
    static_cast<void>(a.add(k, 4u));
    IndexSpace b;
    static_cast<void>(b.add(k, 4u));
    static_cast<void>(b.add(n, 3u));
    IndexSpace c;
    static_cast<void>(c.add(m, 2u));
    static_cast<void>(c.add(n, 3u));
    return GemmTypes{m, n, k, std::move(a), std::move(b), std::move(c)};
}

[[nodiscard]] Operation *make_constant(IRBuilder &builder, const Type &type) noexcept {
    Type results[]{type};
    return builder.create(OperationKind::CONSTANT, {}, results);
}

[[nodiscard]] bool has_diagnostic(const VerificationResult &result, luisa::string_view needle) noexcept {
    for (auto &&diagnostic : result.diagnostics()) {
        if (diagnostic.message.find(needle.data(), 0u, needle.size()) != luisa::string::npos) { return true; }
    }
    return false;
}

}// namespace

void test_valid_structured_mma() {
    Module module;
    auto types = make_gemm_types(module);
    auto function = module.create_function("gemm");
    auto root = function->body().append_block();
    IRBuilder builder{root};

    auto a = make_constant(builder, Type::tile(ScalarType::BFLOAT16, types.a))->result(0u);
    auto b = make_constant(builder, Type::tile(ScalarType::BFLOAT16, types.b))->result(0u);
    auto c = make_constant(builder, Type::tile(ScalarType::FLOAT32, types.c))->result(0u);

    auto group = module.dimensions().create_dimension("group");
    IndexSpace groups;
    expect(groups.add(group, 1u));
    Type parallel_results[]{c->type()};
    auto parallel = builder.create_structured(OperationKind::PARALLEL, groups, {}, parallel_results);
    parallel->set_execution_scope_constraint("block");
    auto body = parallel->region(0u)->block(0u);
    builder.set_insertion_block(body);
    auto mma = builder.create_mma(a, b, c);
    Value *yield_operands[]{mma->result(0u)};
    static_cast<void>(builder.create(OperationKind::YIELD, yield_operands));

    TargetModel target;
    auto block = target.add_execution_scope("block");
    expect(static_cast<bool>(block));
    auto result = verify(module, &target);
    expect(result.ok());
    expect(eq(a->use_count(), 1u));
    expect(eq(b->use_count(), 1u));
    expect(eq(c->use_count(), 1u));
    expect(eq(mma->result(0u)->use_count(), 1u));
}

void test_pipeline_regions_and_memory_effects() {
    Module module;
    auto types = make_gemm_types(module);
    auto function = module.create_function("pipeline_memory");
    auto root = function->body().append_block();
    IRBuilder builder{root};
    auto input = make_constant(builder, Type::tile(ScalarType::BFLOAT16, types.a))->result(0u);

    auto group = module.dimensions().create_dimension("group");
    IndexSpace groups;
    expect(groups.add(group, 1u));
    auto parallel = builder.create_structured(OperationKind::PARALLEL, groups);
    parallel->set_execution_scope_constraint("block");
    auto body = parallel->region(0u)->block(0u);
    builder.set_insertion_block(body);
    auto allocation = builder.create_memory_alloc(Type::memory(ScalarType::BFLOAT16, types.a), "shared");
    auto store = builder.create_memory_store(allocation->result(0u), allocation->result(1u), input);
    auto load = builder.create_memory_load(allocation->result(0u), store->result(0u));
    expect(load != nullptr);

    auto iteration = module.dimensions().create_dimension("iteration");
    IndexSpace iterations;
    expect(iterations.add(iteration, 4u));
    luisa::string_view stages[]{"load", "compute"};
    auto pipeline = builder.create_structured(OperationKind::PIPELINE, iterations, stages);
    expect(eq(pipeline->region_count(), 2u));
    expect(pipeline->region(0u)->label() == "load");
    expect(pipeline->region(1u)->label() == "compute");

    TargetModel permitted;
    auto permitted_block = permitted.add_execution_scope("block");
    auto permitted_shared = permitted.add_resource_class("shared");
    expect(permitted.allow_access(permitted_block, permitted_shared, MemoryAccessKind::LOAD));
    expect(permitted.allow_access(permitted_block, permitted_shared, MemoryAccessKind::STORE));
    expect(verify(module, &permitted).ok());

    TargetModel load_only;
    auto load_only_block = load_only.add_execution_scope("block");
    auto load_only_shared = load_only.add_resource_class("shared");
    expect(load_only.allow_access(load_only_block, load_only_shared, MemoryAccessKind::LOAD));
    auto rejected = verify(module, &load_only);
    expect(!rejected.ok());
    expect(has_diagnostic(rejected, "cannot store to"));
}

void test_execution_scope_partial_order() {
    Module module;
    auto function = module.create_function("nested_scopes");
    auto root = function->body().append_block();
    IRBuilder builder{root};
    auto outer_dim = module.dimensions().create_dimension("outer");
    auto inner_dim = module.dimensions().create_dimension("inner");
    IndexSpace outer_space;
    IndexSpace inner_space;
    expect(outer_space.add(outer_dim, 2u));
    expect(inner_space.add(inner_dim, 4u));
    auto outer = builder.create_structured(OperationKind::PARALLEL, outer_space);
    outer->set_execution_scope_constraint("block");
    builder.set_insertion_block(outer->region(0u)->block(0u));
    auto inner = builder.create_structured(OperationKind::PARALLEL, inner_space);
    inner->set_execution_scope_constraint("warp");

    TargetModel target;
    auto block = target.add_execution_scope("block");
    auto warp = target.add_execution_scope("warp");
    auto dma = target.add_execution_scope("dma");
    expect(target.add_contains(block, warp));
    expect(!target.add_contains(warp, block)) << "execution containment must remain acyclic";
    expect(target.contains(block, warp));
    expect(!target.contains(warp, block));
    expect(!target.contains(block, dma));
    expect(verify(module, &target).ok());

    outer->set_execution_scope_constraint("warp");
    inner->set_execution_scope_constraint("block");
    auto reversed = verify(module, &target);
    expect(!reversed.ok());
    expect(has_diagnostic(reversed, "does not contain"));

    outer->set_execution_scope_constraint("block");
    inner->set_execution_scope_constraint("dma");
    auto incomparable = verify(module, &target);
    expect(!incomparable.ok());
    expect(has_diagnostic(incomparable, "does not contain"));
}

void test_ssa_rewriter_and_analysis_cache() {
    Module module;
    auto types = make_gemm_types(module);
    auto function = module.create_function("rewrite");
    auto root = function->body().append_block();
    IRBuilder builder{root};
    auto first = make_constant(builder, Type::tile(ScalarType::FLOAT32, types.c))->result(0u);
    auto second = make_constant(builder, Type::tile(ScalarType::FLOAT32, types.c))->result(0u);
    Value *operands[]{first};
    Type results[]{first->type()};
    auto consumer = builder.create(OperationKind::CUSTOM, operands, results, "test.identity");
    expect(eq(first->use_count(), 1u));
    expect(eq(second->use_count(), 0u));

    OperationCountAnalysis::runs = 0u;
    AnalysisManager analyses{function};
    auto count0 = analyses.get<OperationCountAnalysis>();
    auto count1 = analyses.get<OperationCountAnalysis>();
    expect(count0 != nullptr);
    expect(count1 != nullptr);
    expect(eq(*count0, 3u));
    expect(eq(OperationCountAnalysis::runs, 1u));

    IRRewriter rewriter{&analyses};
    expect(rewriter.replace_all_uses(first, second));
    expect(eq(first->use_count(), 0u));
    expect(eq(second->use_count(), 1u));
    expect(consumer->operand(0u) == second);
    auto count2 = analyses.get<OperationCountAnalysis>();
    expect(count2 != nullptr);
    expect(eq(*count2, 3u));
    expect(eq(OperationCountAnalysis::runs, 2u));
    expect(verify(module).ok());
}

void test_invalid_region_escape_and_mma_contract() {
    Module module;
    auto types = make_gemm_types(module);
    auto function = module.create_function("invalid_escape");
    auto root = function->body().append_block();
    IRBuilder builder{root};
    auto group = module.dimensions().create_dimension("group");
    IndexSpace groups;
    expect(groups.add(group, 1u));
    auto parallel = builder.create_structured(OperationKind::PARALLEL, groups);
    builder.set_insertion_block(parallel->region(0u)->block(0u));
    auto inner = make_constant(builder, Type::tile(ScalarType::FLOAT32, types.c))->result(0u);
    builder.set_insertion_block(root);
    Value *escaped_operands[]{inner};
    Type escaped_results[]{inner->type()};
    static_cast<void>(builder.create(OperationKind::CUSTOM, escaped_operands, escaped_results, "test.escape"));
    auto escaped = verify(module);
    expect(!escaped.ok());
    expect(has_diagnostic(escaped, "does not lexically dominate"));

    Module bad_mma_module;
    auto bad_types = make_gemm_types(bad_mma_module);
    auto bad_function = bad_mma_module.create_function("bad_mma");
    auto bad_root = bad_function->body().append_block();
    IRBuilder bad_builder{bad_root};
    auto a = make_constant(bad_builder, Type::tile(ScalarType::BFLOAT16, bad_types.a))->result(0u);
    auto b = make_constant(bad_builder, Type::tile(ScalarType::BFLOAT16, bad_types.b))->result(0u);
    auto wrong = bad_mma_module.dimensions().create_dimension("wrong");
    IndexSpace wrong_c;
    expect(wrong_c.add(bad_types.m, 2u));
    expect(wrong_c.add(wrong, 3u));
    auto c = make_constant(bad_builder, Type::tile(ScalarType::FLOAT32, wrong_c))->result(0u);
    static_cast<void>(bad_builder.create_mma(a, b, c));
    auto bad_mma = verify(bad_mma_module);
    expect(!bad_mma.ok());
    expect(has_diagnostic(bad_mma, "accumulator dimension"));
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_ir_valid_structured_mma"_test = test_valid_structured_mma;
    "tile_ir_pipeline_and_memory"_test = test_pipeline_regions_and_memory_effects;
    "tile_ir_execution_scope_partial_order"_test = test_execution_scope_partial_order;
    "tile_ir_ssa_rewriter_and_analysis"_test = test_ssa_rewriter_and_analysis_cache;
    "tile_ir_rejects_invalid_programs"_test = test_invalid_region_escape_and_mma_contract;
}
