// Tests for the immutable Vulkan runtime-feature plan that precedes binding.

#include "ut/ut.hpp"

#include <array>
#include <cstdint>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

#include "spirv_codegen/runtime_target_plan.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;

namespace {

[[nodiscard]] lc::spirv::SpirvRuntimeTargetPlanResult plan_one(
    const xir::Function *function,
    lc::spirv::SpirvBindlessResourceUsage xir_bindless,
    lc::spirv::SpirvTargetFeatures features = {}) noexcept {
    std::array functions{function};
    return lc::spirv::plan_spirv_runtime_target_contract(
        luisa::span<const xir::Function *const>{
            functions.data(), functions.size()},
        xir_bindless, features);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_runtime_plan_uses_reachable_xir_for_bindless_heaps"_test = [] {
        constexpr lc::spirv::SpirvBindlessResourceUsage xir_usage{
            .buffer_heap = true,
            .buffer_metadata = true};
        constexpr auto expected =
            lc::spirv::target_feature::descriptor_indexing |
            lc::spirv::target_feature::runtime_descriptor_array |
            lc::spirv::target_feature::descriptor_binding_partially_bound |
            lc::spirv::target_feature::descriptor_binding_storage_buffer_update_after_bind;
        auto missing = plan_one(
            nullptr, xir_usage);
        expect(!missing.succeeded());
        expect(eq(missing.plan.required_features, expected));
        expect(eq(missing.missing_features, expected));
        expect(missing.plan.bindless_resources.buffer_heap);
        expect(missing.plan.bindless_resources.buffer_metadata);
        expect(!missing.plan.bindless_resources.texture_2d);
        expect(!missing.plan.bindless_resources.texture_3d);
        expect(eq(missing.diagnostics.size(), 4u));

        auto supported = plan_one(
            nullptr, xir_usage,
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                expected));
        expect(supported.succeeded());
        expect(eq(supported.plan.required_features, expected));
        expect(eq(supported.missing_features, 0u));
    };

    "spirv_runtime_plan_keeps_metadata_only_queries_local"_test = [] {
        constexpr lc::spirv::SpirvBindlessResourceUsage xir_usage{
            .buffer_metadata = true};
        auto result = plan_one(nullptr, xir_usage);
        expect(result.succeeded());
        expect(!result.plan.bindless_resources.buffer_heap);
        expect(result.plan.bindless_resources.buffer_metadata);
        expect(!result.plan.bindless_resources.texture_2d);
        expect(!result.plan.bindless_resources.texture_3d);
        expect(eq(result.plan.required_features, 0u));
        expect(eq(result.missing_features, 0u));
    };

    "spirv_runtime_plan_preflights_semantic_ray_query_only"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *accel = kernel->create_resource_argument(
            Type::of<Accel>());
        auto *ray = module.create_constant_zero(Type::of<Ray>());
        auto *mask = module.create_constant_one(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *trace = builder.call(
            Type::of<bool>(), ResourceQueryOp::RAY_TRACING_TRACE_ANY,
            {accel, ray, mask});
        builder.return_void();

        auto missing = plan_one(kernel, {});
        expect(!missing.succeeded());
        expect(eq(missing.missing_features,
                  lc::spirv::target_feature::ray_query));
        expect(missing.plan.uses_semantic_ray_query);
        expect(eq(missing.diagnostics.size(), 1u));
        if (!missing.diagnostics.empty()) {
            expect(missing.diagnostics.front().function == kernel);
            expect(missing.diagnostics.front().instruction == trace);
        }

        auto supported = plan_one(
            kernel, {},
            lc::spirv::SpirvTargetFeatures{
                .ray_query = true});
        expect(supported.succeeded());
        expect(eq(supported.plan.required_features,
                  lc::spirv::target_feature::ray_query));
    };

    "spirv_runtime_plan_does_not_treat_instance_queries_as_ray_traversal"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *accel = kernel->create_resource_argument(
            Type::of<Accel>());
        auto *instance = module.create_constant_zero(
            Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *user_id = builder.call(
            Type::of<uint32_t>(),
            ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance});
        builder.return_void();

        auto result = plan_one(kernel, {});
        expect(result.succeeded());
        expect(!result.plan.uses_semantic_ray_query);
        expect(eq(result.plan.required_features, 0u));
        expect(eq(result.missing_features, 0u));
        expect(result.diagnostics.empty());
        expect(user_id != nullptr);
    };

    "spirv_runtime_plan_preflights_subgroup_extended_value_types"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *value = kernel->create_value_argument(
            Type::of<int16_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *sum = builder.call(
            Type::of<int16_t>(), ThreadGroupOp::WARP_ACTIVE_SUM,
            {value});
        builder.return_void();

        auto missing = plan_one(kernel, {});
        expect(!missing.succeeded());
        expect(eq(missing.missing_features,
                  lc::spirv::target_feature::subgroup_extended_types));
        expect(missing.plan.uses_subgroup_extended_types);
        expect(eq(missing.diagnostics.size(), 1u));
        if (!missing.diagnostics.empty()) {
            expect(missing.diagnostics.front().function == kernel);
            expect(missing.diagnostics.front().instruction == sum);
        }

        auto supported = plan_one(
            kernel, {},
            lc::spirv::SpirvTargetFeatures{
                .subgroup_extended_types = true});
        expect(supported.succeeded());
        expect(eq(supported.plan.required_features,
                  lc::spirv::target_feature::subgroup_extended_types));
    };

    "spirv_runtime_plan_preflights_device_clock"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *clock = builder.clock();
        builder.return_void();

        auto missing = plan_one(kernel, {});
        expect(!missing.succeeded());
        expect(missing.plan.uses_shader_device_clock);
        expect(eq(missing.plan.required_features,
                  lc::spirv::target_feature::shader_device_clock));
        expect(eq(missing.missing_features,
                  lc::spirv::target_feature::shader_device_clock));
        expect(eq(missing.diagnostics.size(), 1u));
        if (!missing.diagnostics.empty()) {
            expect(missing.diagnostics.front().function == kernel);
            expect(missing.diagnostics.front().instruction == clock);
        }

        auto supported = plan_one(
            kernel, {},
            lc::spirv::SpirvTargetFeatures{
                .shader_device_clock = true});
        expect(supported.succeeded());
        expect(supported.plan.uses_shader_device_clock);
        expect(eq(supported.plan.required_features,
                  lc::spirv::target_feature::shader_device_clock));
    };

    "spirv_runtime_plan_preflights_buffer_device_address"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *address = builder.call(
            Type::of<uint64_t>(),
            ResourceQueryOp::BUFFER_DEVICE_ADDRESS, {buffer});
        builder.return_void();

        constexpr auto required =
            lc::spirv::target_feature::buffer_device_address |
            lc::spirv::target_feature::shader_int64;
        auto missing = plan_one(kernel, {});
        expect(!missing.succeeded());
        expect(missing.plan.uses_buffer_device_address);
        expect(eq(missing.plan.required_features, required));
        expect(eq(missing.missing_features, required));
        expect(eq(missing.diagnostics.size(), 2u));
        for (auto &&diagnostic : missing.diagnostics) {
            expect(diagnostic.function == kernel);
            expect(diagnostic.instruction == address);
        }

        auto supported = plan_one(
            kernel, {},
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(required));
        expect(supported.succeeded());
        expect(supported.plan.uses_buffer_device_address);
        expect(eq(supported.plan.required_features, required));
    };
}
