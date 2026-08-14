#include "hip_llvm_pipeline.h"
#include "hip_codegen_llvm.h"
#include "ut/ut.hpp"

#include <string>
#include <vector>

using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

static auto suite = [] {
    "HIP generated callable inlining has a structural hard limit"_test = [] {
        expect(!preserve_generated_callable_boundary(0u));
        expect(!preserve_generated_callable_boundary(
            generated_callable_inline_instruction_budget));
        expect(preserve_generated_callable_boundary(
            generated_callable_inline_instruction_budget + 1u));
    };

    "HIP generated callable linear reuse remains an LLVM decision"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count = 100000u},
            {.instruction_count = 10u,
             .callees = {0u, 0u, 0u, 0u, 0u, 0u}},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        expect(boundaries.size() == graph.size());
        expect(boundaries[0u] == 0u);
        expect(boundaries[1u] == 0u);
    };

    "HIP generated callable alternative expansion preserves the complete frontier"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count = 200000u},
            {.instruction_count = 180000u},
            {.instruction_count = 160000u},
            {.instruction_count = 100u,
             .callees = {0u, 1u, 2u},
             .alternative_call_groups = {{0u, 1u, 2u}}},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        expect(boundaries[0u] != 0u);
        expect(boundaries[1u] != 0u);
        expect(boundaries[2u] != 0u);
        expect(boundaries[3u] == 0u);
    };

    "HIP generated callable outlining keeps leaf internals inline"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count = 100000u},
            {.instruction_count = 100000u, .callees = {0u}},
            {.instruction_count = 100000u, .callees = {0u}},
            {.instruction_count = 100000u, .callees = {0u}},
            {.instruction_count = 100u,
             .callees = {1u, 2u, 3u},
             .alternative_call_groups = {{0u, 1u, 2u}}},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        expect(boundaries[0u] == 0u);
        expect(boundaries[1u] != 0u);
        expect(boundaries[2u] != 0u);
        expect(boundaries[3u] != 0u);
        expect(boundaries[4u] == 0u);
    };

    "HIP generated callable inner boundaries prevent redundant outer outlining"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count =
                 generated_callable_inline_instruction_budget + 1u},
            {.instruction_count = 100000u},
            {.instruction_count = 100u,
             .callees = {0u, 1u},
             .alternative_call_groups = {{0u, 1u}}},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        expect(boundaries[0u] != 0u);
        expect(boundaries[1u] == 0u);
        expect(boundaries[2u] == 0u);
    };

    "HIP generated callable expansion is independent of graph order"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count = 100u,
             .callees = {1u, 2u, 3u},
             .alternative_call_groups = {{0u, 1u, 2u}}},
            {.instruction_count = 200000u},
            {.instruction_count = 180000u},
            {.instruction_count = 160000u},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        expect(boundaries[0u] == 0u);
        expect(boundaries[1u] != 0u);
        expect(boundaries[2u] != 0u);
        expect(boundaries[3u] != 0u);
    };

    "HIP generated callable chooses the largest frontier above budget"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count = 220000u},
            {.instruction_count = 210000u},
            {.instruction_count = 300000u},
            {.instruction_count = 290000u},
            {.instruction_count = 100u,
             .callees = {0u, 1u, 2u, 3u},
             .alternative_call_groups = {{0u, 1u}, {2u, 3u}}},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        // Both groups individually exceed the 500k budget when fully
        // expanded, but the second contributes the greater exact expansion.
        // The choice must therefore not depend on group enumeration order.
        expect(boundaries[0u] == 0u);
        expect(boundaries[1u] == 0u);
        expect(boundaries[2u] != 0u);
        expect(boundaries[3u] != 0u);
        expect(boundaries[4u] == 0u);
    };

    "HIP generated callable recursion is a mandatory boundary"_test = [] {
        auto graph = std::vector<GeneratedCallableInlineGraphNode>{
            {.instruction_count = 10u, .callees = {1u}},
            {.instruction_count = 10u, .callees = {0u}},
            {.instruction_count = 10u, .callees = {2u}},
        };
        auto boundaries = select_generated_callable_boundaries(graph);
        expect(boundaries[0u] != 0u);
        expect(boundaries[1u] != 0u);
        expect(boundaries[2u] != 0u);
    };

    "HIP ray-query pipeline preserves only the canonical-loop option"_test = [] {
        auto pipeline = std::string{
            "module(function(loop-vectorize),"
            "function(simplifycfg<forward-switch-cond;switch-range;"
            "switch-to-arithmetic;switch-to-lookup;no-keep-loops;hoist;sink>),"
            "function(slp-vectorizer))"};
        const auto expected = std::string{
            "module(function(loop-vectorize),"
            "function(simplifycfg<forward-switch-cond;switch-range;"
            "switch-to-arithmetic;switch-to-lookup;keep-loops;hoist;sink>),"
            "function(slp-vectorizer))"};

        expect(preserve_hardware_ray_query_loop_form(pipeline) == 1u);
        expect(pipeline == expected);
    };

    "HIP ray-query pipeline does not rewrite substrings"_test = [] {
        auto pipeline = std::string{
            "module(function(simplifycfg<no-keep-loops;hoist>),"
            "function(fake<foo-no-keep-loops-bar>))"};
        const auto original = pipeline;

        expect(preserve_hardware_ray_query_loop_form(pipeline) == 0u);
        expect(pipeline == original);
    };

    "HIP ray-query pipeline reports cardinality to the caller"_test = [] {
        auto pipeline = std::string{
            "function(simplifycfg<a;no-keep-loops;b>),"
            "function(simplifycfg<c;no-keep-loops;d>)"};

        expect(preserve_hardware_ray_query_loop_form(pipeline) == 2u);
        expect(pipeline.find("no-keep-loops") == std::string::npos);
    };

    "HIP synchronous ray-query environment budget is a closed boundary"_test = [] {
        expect(hip_synchronous_ray_query_environment_is_profitable(0u));
        expect(hip_synchronous_ray_query_environment_is_profitable(
            hip_synchronous_ray_query_environment_budget));
        expect(!hip_synchronous_ray_query_environment_is_profitable(
            hip_synchronous_ray_query_environment_budget + 1u));
    };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
