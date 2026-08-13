#include "ut/ut.hpp"

#include "user_compute_codegen_route.h"

using namespace boost::ut;
using namespace boost::ut::literals;

namespace vk_route = lc::vk::detail;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_codegen_route_keeps_native_and_compatibility_paths_disjoint"_test = [] {
        constexpr auto native =
            vk_route::plan_user_compute_codegen_route({});
        static_assert(native.uses_native_xir_spirv());
        expect(native.uses_native_xir_spirv());

        constexpr auto native_include =
            vk_route::plan_user_compute_codegen_route(
                {.native_include = true});
        constexpr auto printing =
            vk_route::plan_user_compute_codegen_route(
                {.printing = true});
        constexpr auto cooperative =
            vk_route::plan_user_compute_codegen_route(
                {.cooperative_operations = true});
        constexpr auto async_copy =
            vk_route::plan_user_compute_codegen_route(
                {.async_copy = true});
        constexpr auto motion_blur =
            vk_route::plan_user_compute_codegen_route(
                {.motion_blur = true});

        expect(native_include.contains(
            vk_route::UserComputeHlslFallbackReason::NATIVE_INCLUDE));
        expect(printing.contains(
            vk_route::UserComputeHlslFallbackReason::PRINTING));
        expect(cooperative.contains(
            vk_route::UserComputeHlslFallbackReason::COOPERATIVE_OPERATIONS));
        expect(async_copy.contains(
            vk_route::UserComputeHlslFallbackReason::ASYNC_COPY));
        expect(motion_blur.contains(
            vk_route::UserComputeHlslFallbackReason::MOTION_BLUR));
        expect(native_include.requires_hlsl_fallback());
        expect(printing.requires_hlsl_fallback());
        expect(cooperative.requires_hlsl_fallback());
        expect(async_copy.requires_hlsl_fallback());
        expect(motion_blur.requires_hlsl_fallback());
    };

    "vk_required_native_route_fails_closed_without_removing_dxc_fallback"_test = [] {
        constexpr auto fallback =
            vk_route::plan_user_compute_codegen_route(
                {.native_include = true});
        constexpr auto unavailable =
            vk_route::plan_required_native_xir_spirv(
                true, false, {});
        constexpr auto forbidden_fallback =
            vk_route::plan_required_native_xir_spirv(
                true, true, fallback);
        constexpr auto native =
            vk_route::plan_required_native_xir_spirv(
                true, true, {});
        constexpr auto compatibility_allowed =
            vk_route::plan_required_native_xir_spirv(
                false, true, fallback);

        expect(unavailable.status ==
               vk_route::RequiredNativeXirSpirvStatus::
                   NATIVE_CODEGEN_UNAVAILABLE);
        expect(forbidden_fallback.status ==
               vk_route::RequiredNativeXirSpirvStatus::
                   HLSL_FALLBACK_REQUIRED);
        expect(native.satisfied());
        expect(compatibility_allowed.satisfied())
            << "when strict native mode is off, the established DXC "
               "compatibility fallback remains selectable";
    };
}
