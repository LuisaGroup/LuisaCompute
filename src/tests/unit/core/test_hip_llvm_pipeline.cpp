#include "hip_llvm_pipeline.h"
#include "hip_codegen_llvm.h"
#include "ut/ut.hpp"

#include <string>

using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

static auto suite = [] {
    "HIP post-IPO cleanup never promotes ordinary callables to ABI boundaries"_test = [] {
        expect(!preserve_hip_backend_noinline_boundary(
            "callable", false));
        expect(!preserve_hip_backend_noinline_boundary(
            "callable", true));
        expect(!preserve_hip_backend_noinline_boundary(
            "luisa_pipeline_ray_query_trace_surface", true));
        expect(!preserve_hip_backend_noinline_boundary(
            "luisa_ray_query_proceed", false));
        expect(preserve_hip_backend_noinline_boundary(
            "luisa_ray_query_proceed", true));
        expect(preserve_hip_backend_noinline_boundary(
            "luisa_motion_ray_query_proceed", true));
        expect(preserve_hip_backend_noinline_boundary(
            "luisa_hiprt_stack_overflow_fallback_trace", true));
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
