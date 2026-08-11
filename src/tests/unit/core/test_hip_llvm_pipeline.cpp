#include "hip_llvm_pipeline.h"
#include "ut/ut.hpp"

#include <string>

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
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
