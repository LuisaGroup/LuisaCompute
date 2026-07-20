// Pure regressions for Vulkan feature, loader, sampler, and descriptor-limit planning.

#include "ut/ut.hpp"

#include "device_feature_plan.h"
#include "float_atomic_policy.h"
#include "descriptor_interface_plan.h"
#include "command_buffer_ownership.h"
#include "sampler_anisotropy.h"
#include "shader_interface_plan.h"
#include "queue_family_contract.h"
#include "sparse_binding_plan.h"
#include "timeline_semaphore_plan.h"
#include "vulkan_loader_identity.h"
#include "user_compute_codegen_route.h"
#include "../../../backends/common/spirv_llvm/vulkan_binding_properties.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include <luisa/ast/function_builder.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rhi/sampler.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;
namespace hlsl = lc::hlsl;

namespace {

[[nodiscard]] constexpr auto generous_descriptor_interface_limits() noexcept {
    return lc::vk::detail::DescriptorInterfaceLimits{
        .max_bound_descriptor_sets = 5u,
        .max_per_stage_descriptor_samplers = 1024u,
        .max_per_stage_descriptor_uniform_buffers = 1024u,
        .max_per_stage_descriptor_storage_buffers = 1024u,
        .max_per_stage_descriptor_sampled_images = 1024u,
        .max_per_stage_descriptor_storage_images = 1024u,
        .max_per_stage_resources = 4096u,
        .max_descriptor_set_samplers = 1024u,
        .max_descriptor_set_uniform_buffers = 1024u,
        .max_descriptor_set_storage_buffers = 1024u,
        .max_descriptor_set_sampled_images = 1024u,
        .max_descriptor_set_storage_images = 1024u,
        .max_per_stage_descriptor_acceleration_structures = 1024u,
        .max_descriptor_set_acceleration_structures = 1024u,
        .max_per_stage_descriptor_update_after_bind_samplers = 1024u,
        .max_per_stage_descriptor_update_after_bind_storage_buffers = 1024u,
        .max_per_stage_descriptor_update_after_bind_sampled_images = 1024u,
        .max_per_stage_update_after_bind_resources = 4096u,
        .max_descriptor_set_update_after_bind_samplers = 1024u,
        .max_descriptor_set_update_after_bind_storage_buffers = 1024u,
        .max_descriptor_set_update_after_bind_sampled_images = 1024u,
        .max_per_stage_descriptor_update_after_bind_acceleration_structures =
            1024u,
        .max_descriptor_set_update_after_bind_acceleration_structures =
            1024u};
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_user_compute_codegen_route_is_explicit_and_complete"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto native = plan_user_compute_codegen_route({});
        static_assert(native.uses_native_xir_spirv());
        static_assert(!native.requires_hlsl_fallback());
        static_assert(plan_required_native_xir_spirv(
                          false, false, native)
                          .satisfied());
        static_assert(plan_required_native_xir_spirv(
                          true, true, native)
                          .satisfied());
        static_assert(
            plan_required_native_xir_spirv(true, false, native).status ==
            RequiredNativeXirSpirvStatus::NATIVE_CODEGEN_UNAVAILABLE);

        constexpr std::array cases{
            std::pair{UserComputeCodegenRequirements{
                          .native_include = true},
                      UserComputeHlslFallbackReason::NATIVE_INCLUDE},
            std::pair{UserComputeCodegenRequirements{
                          .printing = true},
                      UserComputeHlslFallbackReason::PRINTING},
            std::pair{UserComputeCodegenRequirements{
                          .cooperative_operations = true},
                      UserComputeHlslFallbackReason::COOPERATIVE_OPERATIONS},
            std::pair{UserComputeCodegenRequirements{
                          .async_copy = true},
                      UserComputeHlslFallbackReason::ASYNC_COPY},
            std::pair{UserComputeCodegenRequirements{
                          .typed_bindless_resources = true},
                      UserComputeHlslFallbackReason::TYPED_BINDLESS_RESOURCES},
            std::pair{UserComputeCodegenRequirements{
                          .uniform_bindless_resources = true},
                      UserComputeHlslFallbackReason::UNIFORM_BINDLESS_RESOURCES},
            std::pair{UserComputeCodegenRequirements{
                          .motion_blur = true},
                      UserComputeHlslFallbackReason::MOTION_BLUR}};
        for (auto [requirements, expected_reason] : cases) {
            auto route = plan_user_compute_codegen_route(requirements);
            expect(route.requires_hlsl_fallback());
            expect(route.contains(expected_reason));
            expect(eq(route.hlsl_fallback_reasons,
                      static_cast<uint32_t>(expected_reason)));
            expect(!user_compute_hlsl_fallback_reason_name(expected_reason)
                        .empty());
        }

        constexpr auto combined = plan_user_compute_codegen_route({
            .native_include = true,
            .printing = true,
            .cooperative_operations = true,
            .async_copy = true,
            .typed_bindless_resources = true,
            .uniform_bindless_resources = true,
            .motion_blur = true});
        static_assert(combined.requires_hlsl_fallback());
        static_assert(
            plan_required_native_xir_spirv(true, true, combined).status ==
            RequiredNativeXirSpirvStatus::HLSL_FALLBACK_REQUIRED);
        for (auto [requirements, expected_reason] : cases) {
            static_cast<void>(requirements);
            expect(combined.contains(expected_reason));
        }
        expect(eq(combined.hlsl_fallback_reasons, 0x7fu));
    };

    "vk_float_atomic_codegen_policy_is_vendor_specific"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto amd = plan_vulkan_float_atomic_codegen(
            amd_pci_vendor_id);
        constexpr auto nvidia = plan_vulkan_float_atomic_codegen(0x10deu);
        constexpr auto intel = plan_vulkan_float_atomic_codegen(0x8086u);
        static_assert(
            amd.native_xir_spirv_prefers_software_buffer_float32_rmw);
        static_assert(
            !nvidia.native_xir_spirv_prefers_software_buffer_float32_rmw);
        static_assert(
            !intel.native_xir_spirv_prefers_software_buffer_float32_rmw);
        static_assert(amd.cache_key() != nvidia.cache_key());
        expect(amd.native_xir_spirv_prefers_software_buffer_float32_rmw);
        expect(!nvidia.native_xir_spirv_prefers_software_buffer_float32_rmw);
    };

    "vk_loader_identity_is_pinned_independently_of_module_ownership"_test = [] {
        using namespace lc::vk::detail;
        constexpr VulkanLoaderIdentityView default_loader{
            .source = VulkanLoaderSource::DEFAULT_LOADER,
            .search_path = {},
            .library_name = {}};
        constexpr VulkanLoaderIdentityView custom_loader{
            .source = VulkanLoaderSource::CUSTOM_LOADER,
            .search_path = "/opt/vulkan-a",
            .library_name = "libvulkan-a.so"};

        constexpr auto first_default =
            plan_vulkan_loader_initialization(
                false, default_loader, default_loader);
        static_assert(first_default.should_initialize());
        constexpr auto repeated_default =
            plan_vulkan_loader_initialization(
                true, default_loader, default_loader);
        static_assert(static_cast<bool>(repeated_default));
        static_assert(!repeated_default.should_initialize());
        static_assert(repeated_default.status ==
                      VulkanLoaderInitializationStatus::REUSE);

        constexpr auto first_custom =
            plan_vulkan_loader_initialization(
                false, default_loader, custom_loader);
        static_assert(first_custom.should_initialize());
        constexpr auto repeated_custom =
            plan_vulkan_loader_initialization(
                true, custom_loader, custom_loader);
        static_assert(repeated_custom.status ==
                      VulkanLoaderInitializationStatus::REUSE);

        expect(plan_vulkan_loader_initialization(
                   true, default_loader, custom_loader)
                   .status ==
               VulkanLoaderInitializationStatus::SOURCE_MISMATCH);
        expect(plan_vulkan_loader_initialization(
                   true, custom_loader, default_loader)
                   .status ==
               VulkanLoaderInitializationStatus::SOURCE_MISMATCH);
        expect(plan_vulkan_loader_initialization(
                   true, custom_loader,
                   {.source = VulkanLoaderSource::CUSTOM_LOADER,
                    .search_path = "/opt/vulkan-b",
                    .library_name = "libvulkan-a.so"})
                   .status ==
               VulkanLoaderInitializationStatus::SEARCH_PATH_MISMATCH);
        expect(plan_vulkan_loader_initialization(
                   true, custom_loader,
                   {.source = VulkanLoaderSource::CUSTOM_LOADER,
                    .search_path = "/opt/vulkan-a",
                    .library_name = "libvulkan-b.so"})
                   .status ==
               VulkanLoaderInitializationStatus::LIBRARY_NAME_MISMATCH);
    };

    "vk_imported_device_queue_family_contract_is_exact"_test = [] {
        std::array<VkQueueFamilyProperties, 3u> families{};
        families[0].queueFlags =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        families[0].queueCount = 1u;
        families[1].queueFlags = VK_QUEUE_COMPUTE_BIT;
        families[1].queueCount = 1u;
        families[2].queueFlags = VK_QUEUE_TRANSFER_BIT;
        families[2].queueCount = 1u;
        auto family_span = luisa::span<const VkQueueFamilyProperties>{
            families.data(), families.size()};
        using namespace lc::vk::detail;

        expect(static_cast<bool>(validate_external_queue_family_contract(
            true, {0u, 1u, 2u}, family_span)));
        expect(static_cast<bool>(validate_external_queue_family_contract(
            true, {0u, 0u, 0u}, family_span)));
        expect(static_cast<bool>(validate_external_queue_family_contract(
            true, {0u, 1u, 1u}, family_span)))
            << "graphics/compute queue families implicitly support transfer";
        expect(static_cast<bool>(validate_external_queue_family_contract(
            false,
            {VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             VK_QUEUE_FAMILY_IGNORED},
            {})));

        auto missing = validate_external_queue_family_contract(
            true,
            {VK_QUEUE_FAMILY_IGNORED, 1u, 2u}, family_span);
        expect(missing.status ==
               QueueFamilyContractStatus::MISSING_INDEX);
        expect(missing.role == 0u);

        auto out_of_range = validate_external_queue_family_contract(
            true, {0u, 1u, 3u}, family_span);
        expect(out_of_range.status ==
               QueueFamilyContractStatus::INDEX_OUT_OF_RANGE);
        expect(out_of_range.role == 2u);

        auto missing_compute = validate_external_queue_family_contract(
            true, {2u, 1u, 2u}, family_span);
        expect(missing_compute.status ==
               QueueFamilyContractStatus::MISSING_CAPABILITY);
        expect(missing_compute.role == 0u)
            << "graphics streams require both graphics and compute support";

        families[2].queueCount = 0u;
        auto empty = validate_external_queue_family_contract(
            true, {0u, 1u, 2u}, family_span);
        expect(empty.status == QueueFamilyContractStatus::EMPTY_FAMILY);
        expect(empty.role == 2u);
    };

    "vk_queue_family_sharing_and_sparse_binding_contracts_are_exact"_test = [] {
        using namespace lc::vk::detail;

        constexpr auto one_family =
            plan_queue_family_sharing({4u, 4u, 4u});
        static_assert(one_family.family_count == 1u);
        static_assert(one_family.family_indices[0] == 4u);
        static_assert(!one_family.concurrent());
        static_assert(one_family.sharing_mode() ==
                      VK_SHARING_MODE_EXCLUSIVE);
        static_assert(one_family.create_info_family_count() == 0u);

        constexpr auto two_families =
            plan_queue_family_sharing({4u, 7u, 4u});
        static_assert(two_families.family_count == 2u);
        static_assert(two_families.family_indices[0] == 4u);
        static_assert(two_families.family_indices[1] == 7u);
        static_assert(two_families.concurrent());
        static_assert(two_families.sharing_mode() ==
                      VK_SHARING_MODE_CONCURRENT);
        static_assert(two_families.create_info_family_count() == 2u);

        constexpr auto three_families =
            plan_queue_family_sharing({4u, 7u, 9u});
        static_assert(three_families.family_count == 3u);
        static_assert(three_families.family_indices ==
                      std::array<uint32_t, 3u>{4u, 7u, 9u});
        expect(one_family.create_info_family_indices() == nullptr);
        expect(two_families.create_info_family_indices() != nullptr);
        expect(eq(two_families.family_count, 2u));
        expect(eq(three_families.family_count, 3u));

        std::array<VkQueueFamilyProperties, 3u> families{};
        families[0].queueFlags =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        families[0].queueCount = 1u;
        families[1].queueFlags =
            VK_QUEUE_COMPUTE_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
        families[1].queueCount = 1u;
        families[2].queueFlags = VK_QUEUE_SPARSE_BINDING_BIT;
        families[2].queueCount = 1u;
        auto family_span = luisa::span<const VkQueueFamilyProperties>{
            families.data(), families.size()};

        auto dedicated_sparse =
            select_sparse_binding_queue_family(family_span);
        expect(static_cast<bool>(dedicated_sparse));
        expect(eq(dedicated_sparse.family_index, 2u))
            << "a non-graphics/compute sparse family avoids command-queue serialization";

        families[2].queueCount = 0u;
        auto shared_sparse =
            select_sparse_binding_queue_family(family_span);
        expect(static_cast<bool>(shared_sparse));
        expect(eq(shared_sparse.family_index, 1u))
            << "a compute family is a valid sparse-binding fallback";
        expect(static_cast<bool>(
            validate_sparse_binding_queue_family(1u, family_span)));
        expect(validate_sparse_binding_queue_family(
                   VK_QUEUE_FAMILY_IGNORED, family_span)
                   .status == SparseBindingQueueStatus::MISSING_INDEX);
        expect(validate_sparse_binding_queue_family(3u, family_span).status ==
               SparseBindingQueueStatus::INDEX_OUT_OF_RANGE);
        expect(validate_sparse_binding_queue_family(2u, family_span).status ==
               SparseBindingQueueStatus::EMPTY_FAMILY);
        auto missing_sparse =
            validate_sparse_binding_queue_family(0u, family_span);
        expect(missing_sparse.status ==
               SparseBindingQueueStatus::MISSING_CAPABILITY);
        expect(missing_sparse.available_flags == families[0].queueFlags);
    };

    "vk_sparse_residency_requires_enabled_features_and_valid_dimensions"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto all_supported = SparseResidencyFeatureSupport{
            .sparse_binding = true,
            .sparse_residency_buffer = true,
            .sparse_residency_image_2d = true,
            .sparse_residency_image_3d = true};
        static_assert(validate_sparse_buffer_features(all_supported));
        static_assert(validate_sparse_texture_features(all_supported, 2u));
        static_assert(validate_sparse_texture_features(all_supported, 3u));
        expect(validate_sparse_buffer_features({}).status ==
               SparseResidencyFeatureStatus::MISSING_SPARSE_BINDING);
        expect(validate_sparse_buffer_features({.sparse_binding = true})
                   .status ==
               SparseResidencyFeatureStatus::MISSING_BUFFER_RESIDENCY);
        expect(validate_sparse_texture_features(all_supported, 1u).status ==
               SparseResidencyFeatureStatus::UNSUPPORTED_IMAGE_DIMENSION)
            << "Vulkan forbids SPARSE_RESIDENCY on 1D images";
        expect(validate_sparse_texture_features(
                   {.sparse_binding = true}, 2u)
                   .status ==
               SparseResidencyFeatureStatus::MISSING_IMAGE_2D_RESIDENCY);
        expect(validate_sparse_texture_features(
                   {.sparse_binding = true}, 3u)
                   .status ==
               SparseResidencyFeatureStatus::MISSING_IMAGE_3D_RESIDENCY);
    };

    "vk_sparse_submission_timeline_orders_adjacent_queue_work"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto first = plan_sparse_submission_timeline(
            0u, 0u, 0u, 100u);
        static_assert(first);
        static_assert(first.bridge_signal_value == 1u &&
                      first.sparse_signal_value == 2u);
        constexpr auto chained = plan_sparse_submission_timeline(
            41u, 41u, 41u, 100u);
        static_assert(chained);
        static_assert(chained.bridge_signal_value == 42u &&
                      chained.sparse_signal_value == 43u);
        constexpr auto skipped_host_only_fences =
            plan_sparse_submission_timeline(
                41u, 37u, 37u, 100u);
        static_assert(skipped_host_only_fences);
        static_assert(
            skipped_host_only_fences.bridge_signal_value == 42u &&
            skipped_host_only_fences.sparse_signal_value == 43u);
        constexpr auto overflow = plan_sparse_submission_timeline(
            std::numeric_limits<uint64_t>::max(), 0u, 0u, 100u);
        constexpr auto sparse_signal_overflow =
            plan_sparse_submission_timeline(
                std::numeric_limits<uint64_t>::max() - 1u,
                0u, 0u, 100u);
        constexpr auto largest_valid_sequence =
            plan_sparse_submission_timeline(
                std::numeric_limits<uint64_t>::max() - 2u,
                std::numeric_limits<uint64_t>::max() - 2u,
                std::numeric_limits<uint64_t>::max() - 2u,
                2u);
        static_assert(!overflow);
        static_assert(!sparse_signal_overflow);
        static_assert(largest_valid_sequence);
        static_assert(
            largest_valid_sequence.bridge_signal_value ==
                std::numeric_limits<uint64_t>::max() - 1u &&
            largest_valid_sequence.sparse_signal_value ==
                std::numeric_limits<uint64_t>::max());
        constexpr auto inconsistent =
            plan_sparse_submission_timeline(
                3u, 4u, 4u, 100u);
        static_assert(!inconsistent);
        constexpr auto counter_ahead =
            plan_sparse_submission_timeline(
                3u, 3u, 4u, 100u);
        static_assert(!counter_ahead);
        constexpr auto excessive_gap =
            plan_sparse_submission_timeline(
                41u, 37u, 37u, 5u);
        static_assert(!excessive_gap);
        constexpr auto exact_gap_limit =
            plan_sparse_submission_timeline(
                41u, 37u, 37u, 6u);
        static_assert(exact_gap_limit);
        constexpr auto pending_signal_inside_gap =
            plan_sparse_submission_timeline(
                41u, 40u, 37u, 6u);
        static_assert(pending_signal_inside_gap);
        constexpr auto zero_limit =
            plan_sparse_submission_timeline(
                0u, 0u, 0u, 0u);
        static_assert(!zero_limit);
        expect(first.bridge_signal_value == 1u &&
               first.sparse_signal_value == 2u);
        expect(chained.bridge_signal_value == 42u &&
               chained.sparse_signal_value == 43u);
        expect(skipped_host_only_fences.bridge_signal_value == 42u &&
               skipped_host_only_fences.sparse_signal_value == 43u)
            << "the ordinary-queue bridge follows external-event waits and "
               "host-only logical gaps before handing off to the sparse queue";
        expect(overflow.status ==
               SparseSubmissionTimelineStatus::SIGNAL_VALUE_OVERFLOW);
        expect(sparse_signal_overflow.status ==
               SparseSubmissionTimelineStatus::SIGNAL_VALUE_OVERFLOW);
        expect(largest_valid_sequence.sparse_signal_value ==
               std::numeric_limits<uint64_t>::max());
        expect(inconsistent.status ==
               SparseSubmissionTimelineStatus::
                   GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE);
        expect(counter_ahead.status ==
               SparseSubmissionTimelineStatus::
                   GPU_COUNTER_AHEAD_OF_SIGNAL);
        expect(excessive_gap.status ==
               SparseSubmissionTimelineStatus::
                   MAX_VALUE_DIFFERENCE_EXCEEDED)
            << "both handoff values must stay within the device timeline limit";
        expect(exact_gap_limit.sparse_signal_value == 43u);
        expect(pending_signal_inside_gap.sparse_signal_value == 43u);
        expect(zero_limit.status ==
               SparseSubmissionTimelineStatus::
                   MAX_VALUE_DIFFERENCE_EXCEEDED);
    };

    "vk_timeline_value_plans_enforce_vulkan_pending_window"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto increment = plan_timeline_value_increment(
            std::numeric_limits<uint64_t>::max() - 1u, 1u);
        constexpr auto increment_overflow =
            plan_timeline_value_increment(
                std::numeric_limits<uint64_t>::max(), 1u);
        static_assert(increment &&
                      increment.value ==
                          std::numeric_limits<uint64_t>::max());
        static_assert(!increment_overflow);

        constexpr auto first_signal =
            plan_timeline_semaphore_signal(0u, 0u, 1u, 100u);
        constexpr auto exact_signal_limit =
            plan_timeline_semaphore_signal(5u, 7u, 8u, 3u);
        constexpr auto largest_signal =
            plan_timeline_semaphore_signal(
                std::numeric_limits<uint64_t>::max() - 2u,
                std::numeric_limits<uint64_t>::max() - 1u,
                std::numeric_limits<uint64_t>::max(), 2u);
        constexpr auto excessive_signal =
            plan_timeline_semaphore_signal(0u, 0u, 101u, 100u);
        constexpr auto non_increasing_signal =
            plan_timeline_semaphore_signal(5u, 7u, 7u, 100u);
        constexpr auto stale_tracking =
            plan_timeline_semaphore_signal(8u, 7u, 9u, 100u);
        constexpr auto invalid_pending_range =
            plan_timeline_semaphore_signal(0u, 11u, 12u, 10u);
        constexpr auto zero_limit_signal =
            plan_timeline_semaphore_signal(0u, 0u, 1u, 0u);
        static_assert(first_signal);
        static_assert(exact_signal_limit);
        static_assert(largest_signal);
        static_assert(!excessive_signal);
        static_assert(!non_increasing_signal);
        static_assert(!stale_tracking);
        static_assert(!invalid_pending_range);
        static_assert(!zero_limit_signal);

        constexpr auto pending_wait =
            plan_timeline_semaphore_wait(5u, 8u, 7u, 3u);
        constexpr auto completed_old_wait =
            plan_timeline_semaphore_wait(100u, 100u, 1u, 1u);
        constexpr auto future_wait =
            plan_timeline_semaphore_wait(5u, 8u, 9u, 100u);
        static_assert(pending_wait && !pending_wait.already_satisfied);
        static_assert(completed_old_wait &&
                      completed_old_wait.already_satisfied);
        static_assert(!future_wait);

        expect(excessive_signal.status ==
               TimelineSemaphoreValueStatus::
                   MAX_VALUE_DIFFERENCE_EXCEEDED);
        expect(non_increasing_signal.status ==
               TimelineSemaphoreValueStatus::
                   SIGNAL_VALUE_NOT_INCREASING);
        expect(stale_tracking.status ==
               TimelineSemaphoreValueStatus::
                   TRACKED_SIGNAL_BEHIND_CURRENT);
        expect(invalid_pending_range.status ==
               TimelineSemaphoreValueStatus::
                   TRACKED_SIGNAL_RANGE_EXCEEDED);
        expect(zero_limit_signal.status ==
               TimelineSemaphoreValueStatus::
                   ZERO_MAX_VALUE_DIFFERENCE);
        expect(future_wait.status ==
               TimelineSemaphoreValueStatus::
                   WAIT_VALUE_AHEAD_OF_TRACKED_SIGNAL);
    };

    "vk_internal_gpu_wait_ignores_host_only_logical_fences"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto callback_only_gap =
            plan_internal_timeline_wait(9u, 6u);
        constexpr auto adjacent_gpu_work =
            plan_internal_timeline_wait(9u, 9u);
        constexpr auto inconsistent =
            plan_internal_timeline_wait(8u, 9u);
        static_assert(callback_only_gap &&
                      callback_only_gap.wait_value == 6u);
        static_assert(adjacent_gpu_work &&
                      adjacent_gpu_work.wait_value == 9u);
        static_assert(!inconsistent);
        expect(callback_only_gap.wait_value == 6u)
            << "present must wait only on a value actually submitted to the "
               "Vulkan semaphore before a later sparse bridge";
        expect(inconsistent.status ==
               InternalTimelineWaitStatus::
                   GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE);
    };

    "vk_sparse_image_requirements_reject_unrepresentable_aspects"_test = [] {
        using namespace lc::vk::detail;
        std::array<VkSparseImageMemoryRequirements, 2u> requirements{};
        requirements[0].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        auto one_color = select_sparse_image_requirements(
            std::span<const VkSparseImageMemoryRequirements>{
                requirements.data(), 1u});
        expect(static_cast<bool>(one_color));
        expect(eq(one_color.color_requirement_index, size_t{0u}));

        expect(select_sparse_image_requirements({}).status ==
               SparseImageRequirementsStatus::EMPTY);
        requirements[0].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_METADATA_BIT;
        expect(select_sparse_image_requirements(
                   std::span<const VkSparseImageMemoryRequirements>{
                       requirements.data(), 1u})
                   .status ==
               SparseImageRequirementsStatus::METADATA_BINDING_UNSUPPORTED);
        requirements[0].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        requirements[1].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_METADATA_BIT;
        expect(select_sparse_image_requirements(requirements).status ==
               SparseImageRequirementsStatus::METADATA_BINDING_UNSUPPORTED)
            << "a normal color requirement does not make its required opaque metadata bind optional";
        requirements[0].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        requirements[1].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        expect(select_sparse_image_requirements(requirements).status ==
               SparseImageRequirementsStatus::MULTIPLE_COLOR_REQUIREMENTS);
        requirements[0].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_DEPTH_BIT;
        requirements[1].formatProperties.aspectMask =
            VK_IMAGE_ASPECT_STENCIL_BIT;
        expect(select_sparse_image_requirements(requirements).status ==
               SparseImageRequirementsStatus::MISSING_COLOR_REQUIREMENTS);
    };

    "vk_sparse_image_creation_rejects_any_requested_mip_tail"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto no_tail = validate_sparse_image_mip_tail(
            6u, std::numeric_limits<uint32_t>::max());
        constexpr auto tail_begins_after_last_mip =
            validate_sparse_image_mip_tail(6u, 6u);
        constexpr auto last_mip_is_in_tail =
            validate_sparse_image_mip_tail(6u, 5u);
        constexpr auto all_mips_are_in_tail =
            validate_sparse_image_mip_tail(1u, 0u);
        static_assert(no_tail);
        static_assert(tail_begins_after_last_mip);
        static_assert(!last_mip_is_in_tail);
        static_assert(!all_mips_are_in_tail);
        expect(static_cast<bool>(no_tail));
        expect(static_cast<bool>(tail_begins_after_last_mip));
        expect(last_mip_is_in_tail.status ==
               SparseImageMipTailStatus::REQUESTED_MIP_TAIL);
        expect(all_mips_are_in_tail.status ==
               SparseImageMipTailStatus::REQUESTED_MIP_TAIL);
        expect(validate_sparse_image_mip_tail(0u, 0u).status ==
               SparseImageMipTailStatus::ZERO_MIP_LEVELS);
    };

    "vk_sparse_buffer_plan_uses_actual_alignment_and_full_native_blocks"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto alignment = VkDeviceSize{128u * 1024u};
        constexpr auto plan = plan_sparse_buffer_binding({.physical_resource_size = 3u * alignment,
                                                          .alignment = alignment,
                                                          .start_tile = 2u,
                                                          .tile_count = 1u});
        static_assert(plan);
        static_assert(plan.grid_tile_count == 3u);
        static_assert(plan.resource_offset == 2u * alignment);
        static_assert(plan.binding_size == alignment);
        static_assert(plan.required_heap_size == alignment);
        expect(plan.binding_size == alignment)
            << "buffer bindings consume complete native sparse blocks";

        constexpr auto crosses_last_tile = plan_sparse_buffer_binding({.physical_resource_size = 3u * alignment,
                                                                       .alignment = alignment,
                                                                       .start_tile = 1u,
                                                                       .tile_count = 2u});
        static_assert(crosses_last_tile);
        static_assert(crosses_last_tile.binding_size == 2u * alignment);
    };

    "vk_sparse_buffer_plan_rejects_zero_misaligned_and_grid_errors"_test = [] {
        using namespace lc::vk::detail;
        expect(plan_sparse_buffer_binding({}).status ==
               SparseBufferBindingStatus::ZERO_PHYSICAL_RESOURCE_SIZE);
        expect(plan_sparse_buffer_binding({.physical_resource_size = 1u,
                                           .alignment = 0u,
                                           .tile_count = 1u})
                   .status == SparseBufferBindingStatus::ZERO_ALIGNMENT);
        expect(plan_sparse_buffer_binding({.physical_resource_size = 1000u,
                                           .alignment = 256u,
                                           .tile_count = 1u})
                   .status ==
               SparseBufferBindingStatus::MISALIGNED_PHYSICAL_RESOURCE_SIZE);
        expect(plan_sparse_buffer_binding({.physical_resource_size = 256u,
                                           .alignment = 256u,
                                           .tile_count = 0u})
                   .status == SparseBufferBindingStatus::ZERO_TILE_COUNT);
        expect(plan_sparse_buffer_binding({.physical_resource_size = 3u * 256u,
                                           .alignment = 256u,
                                           .start_tile = 3u,
                                           .tile_count = 1u})
                   .status == SparseBufferBindingStatus::OUT_OF_GRID);
    };

    "vk_sparse_buffer_batch_overlap_detection_distinguishes_adjacent_ranges"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto first = plan_sparse_buffer_binding({.physical_resource_size = 4u * 65536u,
                                                           .alignment = 65536u,
                                                           .start_tile = 0u,
                                                           .tile_count = 2u});
        constexpr auto adjacent = plan_sparse_buffer_binding({.physical_resource_size = 4u * 65536u,
                                                              .alignment = 65536u,
                                                              .start_tile = 2u,
                                                              .tile_count = 1u});
        constexpr auto overlapping = plan_sparse_buffer_binding({.physical_resource_size = 4u * 65536u,
                                                                 .alignment = 65536u,
                                                                 .start_tile = 1u,
                                                                 .tile_count = 2u});
        static_assert(first && adjacent && overlapping);
        static_assert(!sparse_buffer_bindings_overlap(first, adjacent));
        static_assert(sparse_buffer_bindings_overlap(first, overlapping));
        expect(!sparse_buffer_bindings_overlap(first, adjacent));
        expect(sparse_buffer_bindings_overlap(first, overlapping));
    };

    "vk_sparse_image_plan_uses_mip_geometry_and_clips_edge_tiles"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto tile_bytes = VkDeviceSize{128u * 1024u};
        constexpr auto plan = plan_sparse_image_binding({.mip_extent = {250u, 130u, 1u},
                                                         .granularity = {64u, 64u, 1u},
                                                         .tile_byte_size = tile_bytes,
                                                         .mip_level = 2u,
                                                         .mip_tail_first_lod = 5u,
                                                         .start_tile = {3u, 1u, 0u},
                                                         .tile_count = {1u, 2u, 1u}});
        static_assert(plan);
        static_assert(plan.grid_extent ==
                      std::array<uint32_t, 3u>{4u, 3u, 1u});
        static_assert(plan.offset.x == 192 && plan.offset.y == 64 &&
                      plan.offset.z == 0);
        static_assert(plan.extent.width == 58u &&
                      plan.extent.height == 66u &&
                      plan.extent.depth == 1u);
        static_assert(plan.tile_count == 2u);
        static_assert(plan.required_heap_size == 2u * tile_bytes);
        expect(plan.extent.width == 58u && plan.extent.height == 66u)
            << "edge tiles are clipped against the selected mip extent";
    };

    "vk_sparse_image_plan_rejects_mip_tail_zero_grid_and_overflow_errors"_test = [] {
        using namespace lc::vk::detail;
        auto base = SparseImageBindingRequest{
            .mip_extent = {128u, 64u, 1u},
            .granularity = {64u, 64u, 1u},
            .tile_byte_size = 128u * 1024u,
            .mip_level = 0u,
            .mip_tail_first_lod = 3u,
            .start_tile = {0u, 0u, 0u},
            .tile_count = {1u, 1u, 1u}};
        auto zero_extent = base;
        zero_extent.mip_extent.width = 0u;
        expect(plan_sparse_image_binding(zero_extent).status ==
               SparseImageBindingStatus::ZERO_MIP_EXTENT);
        auto zero_granularity = base;
        zero_granularity.granularity.height = 0u;
        expect(plan_sparse_image_binding(zero_granularity).status ==
               SparseImageBindingStatus::ZERO_GRANULARITY);
        auto zero_tile_bytes = base;
        zero_tile_bytes.tile_byte_size = 0u;
        expect(plan_sparse_image_binding(zero_tile_bytes).status ==
               SparseImageBindingStatus::ZERO_TILE_BYTES);
        auto zero_tile_count = base;
        zero_tile_count.tile_count[1u] = 0u;
        expect(plan_sparse_image_binding(zero_tile_count).status ==
               SparseImageBindingStatus::ZERO_TILE_COUNT);
        auto mip_tail = base;
        mip_tail.mip_level = 3u;
        expect(plan_sparse_image_binding(mip_tail).status ==
               SparseImageBindingStatus::MIP_TAIL);
        auto out_of_grid = base;
        out_of_grid.start_tile[0u] = 2u;
        expect(plan_sparse_image_binding(out_of_grid).status ==
               SparseImageBindingStatus::OUT_OF_GRID);

        constexpr auto max_u32 = std::numeric_limits<uint32_t>::max();
        auto offset_overflow = SparseImageBindingRequest{
            .mip_extent = {max_u32, 1u, 1u},
            .granularity = {1u, 1u, 1u},
            .tile_byte_size = 1u,
            .mip_level = 0u,
            .mip_tail_first_lod = max_u32,
            .start_tile = {uint32_t{1u} << 31u, 0u, 0u},
            .tile_count = {1u, 1u, 1u}};
        expect(plan_sparse_image_binding(offset_overflow).status ==
               SparseImageBindingStatus::OFFSET_OUT_OF_RANGE);

        auto arithmetic_overflow = SparseImageBindingRequest{
            .mip_extent = {max_u32, max_u32, max_u32},
            .granularity = {1u, 1u, 1u},
            .tile_byte_size = 1u,
            .mip_level = 0u,
            .mip_tail_first_lod = max_u32,
            .start_tile = {0u, 0u, 0u},
            .tile_count = {max_u32, max_u32, max_u32}};
        expect(plan_sparse_image_binding(arithmetic_overflow).status ==
               SparseImageBindingStatus::ARITHMETIC_OVERFLOW);
    };

    "vk_sparse_image_batch_overlap_detection_respects_boxes_and_mips"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto request = SparseImageBindingRequest{
            .mip_extent = {256u, 128u, 1u},
            .granularity = {64u, 64u, 1u},
            .tile_byte_size = 65536u,
            .mip_level = 0u,
            .mip_tail_first_lod = 4u,
            .start_tile = {0u, 0u, 0u},
            .tile_count = {2u, 1u, 1u}};
        constexpr auto first = plan_sparse_image_binding(request);
        constexpr auto adjacent = plan_sparse_image_binding({.mip_extent = request.mip_extent,
                                                             .granularity = request.granularity,
                                                             .tile_byte_size = request.tile_byte_size,
                                                             .mip_level = request.mip_level,
                                                             .mip_tail_first_lod = request.mip_tail_first_lod,
                                                             .start_tile = {2u, 0u, 0u},
                                                             .tile_count = {1u, 1u, 1u}});
        constexpr auto overlapping = plan_sparse_image_binding({.mip_extent = request.mip_extent,
                                                                .granularity = request.granularity,
                                                                .tile_byte_size = request.tile_byte_size,
                                                                .mip_level = request.mip_level,
                                                                .mip_tail_first_lod = request.mip_tail_first_lod,
                                                                .start_tile = {1u, 0u, 0u},
                                                                .tile_count = request.tile_count});
        static_assert(first && adjacent && overlapping);
        static_assert(!sparse_image_bindings_overlap(
            first, 0u, adjacent, 0u));
        static_assert(sparse_image_bindings_overlap(
            first, 0u, overlapping, 0u));
        static_assert(!sparse_image_bindings_overlap(
            first, 0u, overlapping, 1u));
        expect(!sparse_image_bindings_overlap(
            first, 0u, adjacent, 0u));
        expect(sparse_image_bindings_overlap(
            first, 0u, overlapping, 0u));
        expect(!sparse_image_bindings_overlap(
            first, 0u, overlapping, 1u));
    };

    "vk_sparse_heap_compatibility_checks_size_alignment_and_memory_type"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto alignment = VkDeviceSize{128u * 1024u};
        constexpr auto exact_size = validate_sparse_heap_compatibility({.required_binding_size = 2u * alignment,
                                                                        .logical_heap_size = 2u * alignment,
                                                                        .allocation_offset = 3u * alignment,
                                                                        .required_alignment = alignment,
                                                                        .memory_type_index = 3u,
                                                                        .memory_type_bits = 1u << 3u});
        static_assert(exact_size);
        static_assert(exact_size.allocation_end == 5u * alignment);
        expect(static_cast<bool>(exact_size))
            << "an aligned whole-block binding may exactly consume its logical heap";

        expect(validate_sparse_heap_compatibility({}).status ==
               SparseHeapCompatibilityStatus::ZERO_BINDING_SIZE);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 1u})
                   .status == SparseHeapCompatibilityStatus::ZERO_HEAP_SIZE);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 1u,
                                                   .logical_heap_size = 1u})
                   .status == SparseHeapCompatibilityStatus::ZERO_ALIGNMENT);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 2u * alignment,
                                                   .logical_heap_size = 2u * alignment - 1u,
                                                   .allocation_offset = 0u,
                                                   .required_alignment = alignment,
                                                   .memory_type_index = 3u,
                                                   .memory_type_bits = 1u << 3u})
                   .status == SparseHeapCompatibilityStatus::HEAP_TOO_SMALL);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 1u,
                                                   .logical_heap_size = alignment,
                                                   .allocation_offset = 64u * 1024u,
                                                   .required_alignment = alignment,
                                                   .memory_type_index = 3u,
                                                   .memory_type_bits = 1u << 3u})
                   .status ==
               SparseHeapCompatibilityStatus::MISALIGNED_ALLOCATION_OFFSET);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 1u,
                                                   .logical_heap_size = alignment,
                                                   .allocation_offset = 0u,
                                                   .required_alignment = alignment,
                                                   .memory_type_index = 3u,
                                                   .memory_type_bits = 1u << 2u})
                   .status ==
               SparseHeapCompatibilityStatus::INCOMPATIBLE_MEMORY_TYPE);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 1u,
                                                   .logical_heap_size = alignment,
                                                   .allocation_offset = 0u,
                                                   .required_alignment = alignment,
                                                   .memory_type_index = 32u,
                                                   .memory_type_bits = std::numeric_limits<uint32_t>::max()})
                   .status ==
               SparseHeapCompatibilityStatus::MEMORY_TYPE_INDEX_OUT_OF_RANGE);
        expect(validate_sparse_heap_compatibility({.required_binding_size = 1u,
                                                   .logical_heap_size = 2u,
                                                   .allocation_offset =
                                                       std::numeric_limits<uint64_t>::max() - 1u,
                                                   .required_alignment = 1u,
                                                   .memory_type_index = 0u,
                                                   .memory_type_bits = 1u})
                   .status ==
               SparseHeapCompatibilityStatus::ALLOCATION_RANGE_OVERFLOW);
    };

    "vk_external_device_ancestry_and_queue_lock_contracts_are_explicit"_test = [] {
        using namespace lc::vk::detail;
        expect(static_cast<bool>(validate_external_device_ancestry(
            false, false, false, false)));
        expect(static_cast<bool>(validate_external_device_ancestry(
            true, false, false, false)))
            << "an imported instance may still let the backend create the device";
        expect(validate_external_device_ancestry(
                   false, true, true, true)
                   .status ==
               ExternalDeviceAncestryStatus::MISSING_INSTANCE);
        expect(validate_external_device_ancestry(
                   true, false, true, true)
                   .status ==
               ExternalDeviceAncestryStatus::MISSING_PHYSICAL_DEVICE);
        expect(validate_external_device_ancestry(
                   true, true, false, false)
                   .status ==
               ExternalDeviceAncestryStatus::MISSING_LOGICAL_DEVICE);
        expect(validate_external_device_ancestry(
                   true, false, false, true)
                   .status ==
               ExternalDeviceAncestryStatus::QUEUE_WITHOUT_LOGICAL_DEVICE);
        expect(static_cast<bool>(validate_external_device_ancestry(
            true, true, true, true)));

        expect(static_cast<bool>(validate_external_queue_handles(
            false, {false, false, false})));
        expect(validate_external_queue_handles(
                   true, {false, true, true})
                   .role == 0u);
        expect(validate_external_queue_handles(
                   true, {true, false, true})
                   .role == 1u);
        expect(validate_external_queue_handles(
                   true, {true, true, false})
                   .role == 2u);
        expect(static_cast<bool>(validate_external_queue_handles(
            true, {true, true, true})));

        constexpr auto one_queue =
            plan_queue_locks({11u, 11u, 11u, 11u});
        static_assert(one_queue.lock_count == 1u);
        static_assert(one_queue.lock_indices ==
                      std::array<uint8_t, 4u>{0u, 0u, 0u, 0u});
        constexpr auto two_queues =
            plan_queue_locks({11u, 22u, 22u, 22u});
        static_assert(two_queues.lock_count == 2u);
        static_assert(two_queues.lock_indices ==
                      std::array<uint8_t, 4u>{0u, 1u, 1u, 1u});
        constexpr auto sparse_reuses_first_queue =
            plan_queue_locks({11u, 22u, 33u, 11u});
        static_assert(sparse_reuses_first_queue.lock_count == 3u);
        static_assert(sparse_reuses_first_queue.lock_indices ==
                      std::array<uint8_t, 4u>{0u, 1u, 2u, 0u});
        constexpr auto four_queues =
            plan_queue_locks({11u, 22u, 33u, 44u});
        static_assert(four_queues.lock_count == 4u);
        static_assert(four_queues.lock_indices ==
                      std::array<uint8_t, 4u>{0u, 1u, 2u, 3u});
        expect(eq(one_queue.lock_count, uint8_t{1u}));
        expect(eq(two_queues.lock_count, uint8_t{2u}));
        expect(eq(sparse_reuses_first_queue.lock_count, uint8_t{3u}));
        expect(eq(four_queues.lock_count, uint8_t{4u}));
    };

    "vk_borrowed_command_buffers_are_one_shot_and_never_owned"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto backend = plan_command_buffer_retirement(
            CommandBufferOwnership::BACKEND);
        static_assert(backend.reset_native_buffer);
        static_assert(backend.recycle_native_buffer);
        static_assert(backend.free_native_buffer);

        constexpr auto borrowed = plan_command_buffer_retirement(
            CommandBufferOwnership::BORROWED);
        static_assert(!borrowed.reset_native_buffer);
        static_assert(!borrowed.recycle_native_buffer);
        static_assert(!borrowed.free_native_buffer);
        expect(backend.recycle_native_buffer);
        expect(!borrowed.recycle_native_buffer);
    };

    "vk_robust_buffer_access_plan_honors_update_after_bind_interaction"_test = [] {
        using namespace lc::vk::detail;
        expect(!plan_robust_buffer_access({}));
        expect(plan_robust_buffer_access({.physical_device_feature = true}));
        expect(!plan_robust_buffer_access({.physical_device_feature = true,
                                           .storage_buffer_update_after_bind = true}));
        expect(plan_robust_buffer_access({.physical_device_feature = true,
                                          .storage_buffer_update_after_bind = true,
                                          .robust_buffer_access_update_after_bind = true}));
        expect(!plan_robust_buffer_access({.storage_buffer_update_after_bind = true,
                                           .robust_buffer_access_update_after_bind = true}));
    };

    "vk_custom_device_feature_chain_allows_unowned_structures"_test = [] {
        using namespace lc::vk::detail;
        VkBaseInStructure tail{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
            .pNext = nullptr};
        VkBaseInStructure head{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            .pNext = &tail};
        expect(static_cast<bool>(
            validate_device_feature_settings_chain(nullptr)));
        expect(static_cast<bool>(
            validate_device_feature_settings_chain(&head)));
    };

    "vk_custom_device_feature_chain_rejects_backend_owned_structures"_test = [] {
        using namespace lc::vk::detail;
        constexpr VkStructureType colliding_types[]{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
        };
        for (auto type : colliding_types) {
            VkBaseInStructure collision{
                .sType = type,
                .pNext = nullptr};
            auto result =
                validate_device_feature_settings_chain(&collision);
            expect(!result);
            expect(result.error ==
                   DeviceFeatureChainValidationError::BACKEND_OWNED_STRUCTURE);
            expect(result.structure_type == type);
            expect(eq(result.node_index, 0u));
        }
    };

    "vk_custom_device_feature_chain_rejects_promoted_alias_collisions"_test = [] {
        using namespace lc::vk::detail;
        constexpr VkStructureType colliding_types[]{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
#ifdef VK_VERSION_1_4
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
#endif
        };
        for (auto type : colliding_types) {
            VkBaseInStructure allowed{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
                .pNext = nullptr};
            VkBaseInStructure collision{
                .sType = type,
                .pNext = nullptr};
            allowed.pNext = &collision;
            auto result = validate_device_feature_settings_chain(&allowed);
            expect(!result);
            expect(result.error ==
                   DeviceFeatureChainValidationError::BACKEND_OWNED_STRUCTURE);
            expect(result.structure_type == type);
            expect(eq(result.node_index, 1u));
        }
    };

    "vk_custom_device_feature_chain_rejects_cycles"_test = [] {
        using namespace lc::vk::detail;
        VkBaseInStructure first{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
            .pNext = nullptr};
        VkBaseInStructure second{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            .pNext = &first};
        first.pNext = &second;
        auto result = validate_device_feature_settings_chain(&first);
        expect(!result);
        expect(result.error == DeviceFeatureChainValidationError::CYCLE);
    };

    "vk_custom_device_feature_chain_rejects_non_head_duplicates"_test = [] {
        using namespace lc::vk::detail;
        VkBaseInStructure duplicate{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
            .pNext = nullptr};
        VkBaseInStructure middle{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            .pNext = &duplicate};
        VkBaseInStructure head{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
            .pNext = &middle};
        auto result = validate_device_feature_settings_chain(&head);
        expect(!result);
        expect(result.error ==
               DeviceFeatureChainValidationError::DUPLICATE_STRUCTURE);
        expect(result.structure_type == head.sType);
        expect(eq(result.node_index, 2u));
        expect(eq(result.related_node_index, 0u));
    };

    "vk_descriptor_interface_accepts_exact_canonical_abi"_test = [] {
        using namespace lc::vk::detail;
        const std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::ConstantValue,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantBuffer,
                           0u, 1u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           0u, 2u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::UAVTextureHeap,
                           0u, 3u, 1u}};
        auto plan = plan_descriptor_interface(
            {.properties = properties,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .has_constant_ubo_payload = true},
            generous_descriptor_interface_limits());
        expect(static_cast<bool>(plan));
        expect(eq(plan.descriptor_set_count, 2u));
        expect(eq(plan.local_binding_count, 4u));
        expect(eq(plan.set_counts[0u].uniform_buffers, 1u));
        expect(eq(plan.set_counts[0u].storage_buffers, 1u));
        expect(eq(plan.set_counts[0u].sampled_images, 1u));
        expect(eq(plan.set_counts[0u].storage_images, 1u));
        expect(eq(plan.set_counts[1u].samplers, 16u));
    };

    "vk_descriptor_interface_checks_set_count_at_exact_boundary"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        const std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::SRVBufferHeap,
                           2u, 0u, unbounded},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           3u, 0u, unbounded},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           4u, 0u, unbounded}};
        auto request = DescriptorInterfaceRequest{
            .properties = properties,
            .stage_mask = DescriptorInterfaceStageMask::RASTER,
            .bindless_heap_capacity = 67u,
            .use_buffer_bindless = true,
            .use_tex2d_bindless = true,
            .use_tex3d_bindless = true,
            .sampled_image_update_after_bind_enabled = true,
            .storage_buffer_update_after_bind_enabled = true};
        auto exact = plan_descriptor_interface(
            request, generous_descriptor_interface_limits());
        expect(static_cast<bool>(exact));
        expect(eq(exact.descriptor_set_count, 5u));
        expect(eq(exact.update_after_bind_set_count, 3u));
        expect(eq(exact.set_counts[2u].storage_buffers, 67u));
        expect(eq(exact.set_counts[3u].sampled_images, 67u));
        expect(eq(exact.set_counts[4u].sampled_images, 67u));

        auto plus_one = [=] {
            auto limits = generous_descriptor_interface_limits();
            limits.max_bound_descriptor_sets = 4u;
            return plan_descriptor_interface(request, limits);
        }();
        expect(plus_one.error ==
               DescriptorInterfaceError::TOO_MANY_DESCRIPTOR_SETS);
    };

    "vk_descriptor_interface_bindless_subset_matrix_matches_vulkan_domains"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        constexpr auto capacity = 17u;
        constexpr auto buffer_bit = 1u << 0u;
        constexpr auto tex2d_bit = 1u << 1u;
        constexpr auto tex3d_bit = 1u << 2u;

        // Exercise every non-empty combination of the three persistent heaps.
        // The expected set order is part of the runtime ABI, while the order of
        // records in the persisted property table is intentionally irrelevant.
        for (auto mask = 1u; mask < 8u; ++mask) {
            auto use_buffer = (mask & buffer_bit) != 0u;
            auto use_tex2d = (mask & tex2d_bit) != 0u;
            auto use_tex3d = (mask & tex3d_bit) != 0u;
            auto texture_heap_count =
                static_cast<uint32_t>(use_tex2d) +
                static_cast<uint32_t>(use_tex3d);
            auto heap_count = static_cast<uint32_t>(use_buffer) +
                              texture_heap_count;
            auto expected_set_count = 2u + heap_count;

            std::array<hlsl::Property, 5u> canonical{};
            auto property_count = 0u;
            canonical[property_count++] = hlsl::Property{
                hlsl::ShaderVariableType::SamplerHeap,
                1u, 0u, descriptor_interface_sampler_count};
            canonical[property_count++] = hlsl::Property{
                hlsl::ShaderVariableType::SPIRVAccel,
                0u, 0u, 1u};
            auto next_set = 2u;
            if (use_buffer) {
                canonical[property_count++] = hlsl::Property{
                    hlsl::ShaderVariableType::SRVBufferHeap,
                    next_set++, 0u, unbounded};
            }
            if (use_tex2d) {
                canonical[property_count++] = hlsl::Property{
                    hlsl::ShaderVariableType::SRVTextureHeap,
                    next_set++, 0u, unbounded};
            }
            if (use_tex3d) {
                canonical[property_count++] = hlsl::Property{
                    hlsl::ShaderVariableType::SRVTextureHeap,
                    next_set++, 0u, unbounded};
            }

            auto exact_limits = generous_descriptor_interface_limits();
            exact_limits.max_bound_descriptor_sets = expected_set_count;
            exact_limits.max_per_stage_descriptor_samplers =
                descriptor_interface_sampler_count;
            exact_limits.max_per_stage_descriptor_uniform_buffers = 0u;
            exact_limits.max_per_stage_descriptor_storage_buffers = 0u;
            exact_limits.max_per_stage_descriptor_sampled_images = 0u;
            exact_limits.max_per_stage_descriptor_storage_images = 0u;
            exact_limits.max_per_stage_resources = 0u;
            exact_limits.max_descriptor_set_samplers =
                descriptor_interface_sampler_count;
            exact_limits.max_descriptor_set_uniform_buffers = 0u;
            exact_limits.max_descriptor_set_storage_buffers = 0u;
            exact_limits.max_descriptor_set_sampled_images = 0u;
            exact_limits.max_descriptor_set_storage_images = 0u;
            exact_limits.max_per_stage_descriptor_acceleration_structures = 1u;
            exact_limits.max_descriptor_set_acceleration_structures = 1u;
            exact_limits.max_per_stage_descriptor_update_after_bind_samplers =
                descriptor_interface_sampler_count;
            exact_limits.max_descriptor_set_update_after_bind_samplers =
                descriptor_interface_sampler_count;
            exact_limits.max_per_stage_descriptor_update_after_bind_storage_buffers =
                use_buffer ? capacity : 0u;
            exact_limits.max_descriptor_set_update_after_bind_storage_buffers =
                use_buffer ? capacity : 0u;
            exact_limits.max_per_stage_descriptor_update_after_bind_sampled_images =
                texture_heap_count * capacity;
            exact_limits.max_descriptor_set_update_after_bind_sampled_images =
                texture_heap_count * capacity;
            exact_limits.max_per_stage_update_after_bind_resources =
                heap_count * capacity;
            exact_limits.max_per_stage_descriptor_update_after_bind_acceleration_structures = 1u;
            exact_limits.max_descriptor_set_update_after_bind_acceleration_structures = 1u;

            std::array<uint32_t, 5u> order{0u, 1u, 2u, 3u, 4u};
            do {
                std::array<hlsl::Property, 5u> properties{};
                for (auto i = 0u; i < property_count; ++i) {
                    properties[i] = canonical[order[i]];
                }
                auto property_span = luisa::span<const hlsl::Property>{
                    properties.data(), property_count};
                auto request = DescriptorInterfaceRequest{
                    .properties = property_span,
                    .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
                    .bindless_heap_capacity = capacity,
                    .use_buffer_bindless = use_buffer,
                    .use_tex2d_bindless = use_tex2d,
                    .use_tex3d_bindless = use_tex3d,
                    .acceleration_structure_available = true,
                    // Device-owned bindless devices enable both feature
                    // families, even when this shader uses only one heap kind.
                    .sampled_image_update_after_bind_enabled = true,
                    .storage_buffer_update_after_bind_enabled = true};
                auto plan = plan_descriptor_interface(
                    request, exact_limits);
                expect(static_cast<bool>(plan));
                expect(eq(plan.descriptor_set_count,
                          expected_set_count));
                expect(eq(plan.update_after_bind_set_count,
                          heap_count));
                expect(eq(plan.local_binding_count, 1u));
                expect(eq(plan.acceleration_structure_binding_count, 1u));
                expect(eq(plan.set_counts[0u].acceleration_structures, 1u));
                expect(eq(plan.set_counts[0u].resources(), 0u))
                    << "acceleration structures are not maxPerStageResources";
                expect(eq(plan.set_counts[1u].samplers,
                          uint64_t{descriptor_interface_sampler_count}));

                auto set = 2u;
                if (use_buffer) {
                    expect(eq(plan.set_counts[set].storage_buffers,
                              uint64_t{capacity}));
                    ++set;
                }
                if (use_tex2d) {
                    expect(eq(plan.set_counts[set].sampled_images,
                              uint64_t{capacity}));
                    ++set;
                }
                if (use_tex3d) {
                    expect(eq(plan.set_counts[set].sampled_images,
                              uint64_t{capacity}));
                    ++set;
                }
                expect(eq(set, expected_set_count));

                expect(static_cast<bool>(
                    plan_persisted_descriptor_interface(
                        property_span,
                        DescriptorInterfaceStageMask::COMPUTE,
                        use_buffer, use_tex2d, use_tex3d, false)))
                    << "property record order must not alter the persisted ABI";
            } while (std::next_permutation(
                order.begin(), order.begin() + property_count));

            auto too_few_sets = exact_limits;
            too_few_sets.max_bound_descriptor_sets =
                expected_set_count - 1u;
            auto set_limit_plan = plan_descriptor_interface(
                {.properties = luisa::span<const hlsl::Property>{
                     canonical.data(), property_count},
                 .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
                 .bindless_heap_capacity = capacity,
                 .use_buffer_bindless = use_buffer,
                 .use_tex2d_bindless = use_tex2d,
                 .use_tex3d_bindless = use_tex3d,
                 .acceleration_structure_available = true,
                 .sampled_image_update_after_bind_enabled = true,
                 .storage_buffer_update_after_bind_enabled = true},
                too_few_sets);
            expect(set_limit_plan.error ==
                   DescriptorInterfaceError::TOO_MANY_DESCRIPTOR_SETS);
        }
    };

    "vk_descriptor_interface_checks_each_ordinary_limit_exactly"_test = [] {
        using namespace lc::vk::detail;
        const std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantBuffer,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 1u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           0u, 2u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::UAVTextureHeap,
                           0u, 3u, 1u}};
        auto request = DescriptorInterfaceRequest{
            .properties = properties,
            .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
            .has_constant_ubo_payload = true};
        constexpr auto exact_limits = [] {
            auto limits = generous_descriptor_interface_limits();
            limits.max_per_stage_descriptor_samplers = 16u;
            limits.max_per_stage_descriptor_uniform_buffers = 1u;
            limits.max_per_stage_descriptor_storage_buffers = 1u;
            limits.max_per_stage_descriptor_sampled_images = 1u;
            limits.max_per_stage_descriptor_storage_images = 1u;
            limits.max_per_stage_resources = 4u;
            limits.max_descriptor_set_samplers = 16u;
            limits.max_descriptor_set_uniform_buffers = 1u;
            limits.max_descriptor_set_storage_buffers = 1u;
            limits.max_descriptor_set_sampled_images = 1u;
            limits.max_descriptor_set_storage_images = 1u;
            return limits;
        }();
        auto exact = plan_descriptor_interface(
            request, exact_limits);
        expect(static_cast<bool>(exact));

        auto expect_error = [&](DescriptorInterfaceLimits limits,
                                DescriptorInterfaceError error) {
            auto plan = plan_descriptor_interface(request, limits);
            expect(!static_cast<bool>(plan));
            expect(plan.error == error);
        };
        auto limits = exact_limits;
        limits.max_per_stage_descriptor_samplers = 15u;
        expect_error(limits,
                     DescriptorInterfaceError::MAX_PER_STAGE_SAMPLERS);
        limits = exact_limits;
        limits.max_per_stage_descriptor_uniform_buffers = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_UNIFORM_BUFFERS);
        limits = exact_limits;
        limits.max_per_stage_descriptor_storage_buffers = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_STORAGE_BUFFERS);
        limits = exact_limits;
        limits.max_per_stage_descriptor_sampled_images = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_SAMPLED_IMAGES);
        limits = exact_limits;
        limits.max_per_stage_descriptor_storage_images = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_STORAGE_IMAGES);
        limits = exact_limits;
        limits.max_per_stage_resources = 3u;
        expect_error(limits,
                     DescriptorInterfaceError::MAX_PER_STAGE_RESOURCES);
        limits = exact_limits;
        limits.max_descriptor_set_samplers = 15u;
        expect_error(limits,
                     DescriptorInterfaceError::MAX_DESCRIPTOR_SET_SAMPLERS);
        limits = exact_limits;
        limits.max_descriptor_set_uniform_buffers = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UNIFORM_BUFFERS);
        limits = exact_limits;
        limits.max_descriptor_set_storage_buffers = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_STORAGE_BUFFERS);
        limits = exact_limits;
        limits.max_descriptor_set_sampled_images = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_SAMPLED_IMAGES);
        limits = exact_limits;
        limits.max_descriptor_set_storage_images = 0u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_STORAGE_IMAGES);
    };

    "vk_descriptor_interface_keeps_update_after_bind_limits_separate"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        const std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           2u, 0u, unbounded}};
        auto plan = [=] {
            auto limits = generous_descriptor_interface_limits();
            limits.max_per_stage_descriptor_sampled_images = 0u;
            limits.max_descriptor_set_sampled_images = 0u;
            return plan_descriptor_interface(
                {.properties = properties,
                 .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
                 .bindless_heap_capacity = 257u,
                 .use_tex2d_bindless = true,
                 .sampled_image_update_after_bind_enabled = true,
                 .storage_buffer_update_after_bind_enabled = true},
                limits);
        }();
        expect(static_cast<bool>(plan))
            << "ordinary VkPhysicalDeviceLimits do not govern UAB sets";
        expect(eq(plan.set_counts[2u].sampled_images, 257u));
    };

    "vk_descriptor_interface_checks_update_after_bind_aggregates_exactly"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        const std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           0u, 1u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SRVBufferHeap,
                           2u, 0u, unbounded},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           3u, 0u, unbounded},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           4u, 0u, unbounded}};
        auto request = DescriptorInterfaceRequest{
            .properties = properties,
            .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
            .bindless_heap_capacity = 17u,
            .use_buffer_bindless = true,
            .use_tex2d_bindless = true,
            .use_tex3d_bindless = true,
            .sampled_image_update_after_bind_enabled = true,
            .storage_buffer_update_after_bind_enabled = true};
        constexpr auto exact_limits = [] {
            auto limits = generous_descriptor_interface_limits();
            limits.max_per_stage_descriptor_samplers = 16u;
            limits.max_per_stage_descriptor_storage_buffers = 1u;
            limits.max_per_stage_descriptor_sampled_images = 1u;
            limits.max_per_stage_resources = 2u;
            limits.max_descriptor_set_samplers = 16u;
            limits.max_descriptor_set_storage_buffers = 1u;
            limits.max_descriptor_set_sampled_images = 1u;
            limits.max_per_stage_descriptor_update_after_bind_samplers = 16u;
            limits.max_descriptor_set_update_after_bind_samplers = 16u;
            limits.max_per_stage_descriptor_update_after_bind_storage_buffers = 18u;
            limits.max_descriptor_set_update_after_bind_storage_buffers = 18u;
            limits.max_per_stage_descriptor_update_after_bind_sampled_images = 35u;
            limits.max_descriptor_set_update_after_bind_sampled_images = 35u;
            limits.max_per_stage_update_after_bind_resources = 53u;
            return limits;
        }();
        auto exact = plan_descriptor_interface(
            request, exact_limits);
        expect(static_cast<bool>(exact));
        expect(eq(exact.set_counts[0u].storage_buffers, 1u));
        expect(eq(exact.set_counts[0u].sampled_images, 1u));
        expect(eq(exact.set_counts[2u].storage_buffers, 17u));
        expect(eq(exact.set_counts[3u].sampled_images, 17u));
        expect(eq(exact.set_counts[4u].sampled_images, 17u));

        auto expect_error = [&](DescriptorInterfaceLimits limits,
                                DescriptorInterfaceError error) {
            auto plan = plan_descriptor_interface(request, limits);
            expect(!static_cast<bool>(plan));
            expect(plan.error == error);
        };
        auto limits = exact_limits;
        limits.max_per_stage_descriptor_update_after_bind_samplers = 15u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLERS);
        limits = exact_limits;
        limits.max_descriptor_set_update_after_bind_samplers = 15u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS);
        limits = exact_limits;
        limits.max_per_stage_descriptor_update_after_bind_storage_buffers = 17u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_STORAGE_BUFFERS);
        limits = exact_limits;
        limits.max_descriptor_set_update_after_bind_storage_buffers = 17u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_BUFFERS);
        limits = exact_limits;
        limits.max_per_stage_descriptor_update_after_bind_sampled_images = 34u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLED_IMAGES);
        limits = exact_limits;
        limits.max_descriptor_set_update_after_bind_sampled_images = 34u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLED_IMAGES);
        limits = exact_limits;
        limits.max_per_stage_update_after_bind_resources = 52u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES);
    };

    "vk_descriptor_interface_rejects_malformed_persisted_roles"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        constexpr auto limits = generous_descriptor_interface_limits();
        auto plan = [=](auto &&properties,
                        bool buffer = false,
                        bool tex2d = false,
                        bool tex3d = false,
                        bool ubo = false,
                        DescriptorInterfaceStageMask stage =
                            DescriptorInterfaceStageMask::COMPUTE) {
            return plan_descriptor_interface(
                {.properties = properties,
                 .stage_mask = stage,
                 .bindless_heap_capacity = 64u,
                 .use_buffer_bindless = buffer,
                 .use_tex2d_bindless = tex2d,
                 .use_tex3d_bindless = tex3d,
                 .has_constant_ubo_payload = ubo,
                 .sampled_image_update_after_bind_enabled = true,
                 .storage_buffer_update_after_bind_enabled = true},
                limits);
        };

        std::array missing_sampler{
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u}};
        expect(plan(missing_sampler).error ==
               DescriptorInterfaceError::MISSING_OR_DUPLICATE_SAMPLER);

        std::array local_hole{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 1u, 1u}};
        expect(plan(local_hole).error ==
               DescriptorInterfaceError::NONDENSE_LOCAL_BINDINGS);

        std::array duplicate_local{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::RWStructuredBuffer,
                           0u, 0u, 1u}};
        expect(plan(duplicate_local).error ==
               DescriptorInterfaceError::DUPLICATE_LOCAL_BINDING);

        std::array wrong_heap_order{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           2u, 0u, unbounded},
            hlsl::Property{hlsl::ShaderVariableType::SRVBufferHeap,
                           3u, 0u, unbounded}};
        expect(plan(wrong_heap_order, true, true).error ==
               DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP);

        std::array missing_heap{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u}};
        expect(plan(missing_heap, true).error ==
               DescriptorInterfaceError::MISSING_OR_DUPLICATE_GLOBAL_HEAP);

        std::array ubo_property{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantBuffer,
                           0u, 0u, 1u}};
        expect(plan(ubo_property).error ==
               DescriptorInterfaceError::CONSTANT_UBO_MISMATCH);
        expect(plan(missing_heap, false, false, false, true).error ==
               DescriptorInterfaceError::CONSTANT_UBO_MISMATCH);

        std::array noncanonical_push_constant{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantValue,
                           4u, 0u, 1u}};
        expect(plan(noncanonical_push_constant).error ==
               DescriptorInterfaceError::NONCANONICAL_PUSH_CONSTANT);

        std::array duplicate_push_constant{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantValue,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantValue,
                           0u, 0u, 1u}};
        expect(plan(duplicate_push_constant).error ==
               DescriptorInterfaceError::DUPLICATE_PUSH_CONSTANT);

        std::array push_constant_binding_overlap{
            hlsl::Property{hlsl::ShaderVariableType::ConstantValue,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u}};
        auto overlap_plan = plan(push_constant_binding_overlap);
        expect(static_cast<bool>(overlap_plan));
        expect(eq(overlap_plan.local_binding_count, 1u))
            << "ConstantValue is a pseudo-property, not descriptor binding 0";
        expect(static_cast<bool>(plan_persisted_descriptor_interface(
            push_constant_binding_overlap,
            DescriptorInterfaceStageMask::COMPUTE,
            false, false, false, false)));
        expect(!static_cast<bool>(plan_persisted_descriptor_interface(
            noncanonical_push_constant,
            DescriptorInterfaceStageMask::COMPUTE,
            false, false, false, false)))
            << "writer-side validation must reject noncanonical pseudo-properties";

        std::array raster_indirect{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 0u, 1u}};
        expect(plan(raster_indirect, false, false, false, false,
                    DescriptorInterfaceStageMask::RASTER)
                   .error ==
               DescriptorInterfaceError::INDIRECT_DISPATCH_NOT_SUPPORTED);
    };

    "vk_descriptor_lookup_never_treats_push_constants_as_descriptors"_test = [] {
        using namespace lc::vk::detail;
        using lc::hlsl::Property;
        using lc::hlsl::ShaderVariableType;
        constexpr Property descriptor{
            ShaderVariableType::StructuredBuffer, 0u, 0u, 1u};
        constexpr Property push_constant{
            ShaderVariableType::ConstantValue, 0u, 0u, 1u};
        constexpr std::array descriptor_then_push{
            descriptor, push_constant};
        constexpr std::array push_then_descriptor{
            push_constant, descriptor};

        auto *lhs = find_local_descriptor_property(
            descriptor_then_push, 0u);
        auto *rhs = find_local_descriptor_property(
            push_then_descriptor, 0u);
        expect(lhs != nullptr &&
               lhs->type == ShaderVariableType::StructuredBuffer);
        expect(rhs != nullptr &&
               rhs->type == ShaderVariableType::StructuredBuffer);
        expect(find_local_descriptor_property(
                   descriptor_then_push, 1u) == nullptr);
    };

    "vk_shader_interface_owns_cross_boundary_binding_order"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        std::array<SavedArgument, 3u> arguments{};
        arguments[0u].tag = Type::Tag::UINT32;
        arguments[0u].var_usage = Usage::READ;
        arguments[0u].struct_size = sizeof(uint32_t);
        arguments[1u].tag = Type::Tag::BUFFER;
        arguments[1u].var_usage = Usage::READ;
        arguments[1u].struct_size = sizeof(uint32_t);
        arguments[1u].set_buffer_metadata_index(0u);
        arguments[2u].tag = Type::Tag::BINDLESS_ARRAY;
        arguments[2u].var_usage = Usage::READ;
        constexpr std::array canonical{
            hlsl::Property{hlsl::ShaderVariableType::ConstantValue,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::ConstantBuffer,
                           0u, 1u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 2u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 3u, 1u},
            hlsl::Property{
                hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata,
                0u, 4u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 5u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::RWStructuredBuffer,
                           0u, 6u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::RWStructuredBuffer,
                           0u, 7u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SRVBufferHeap,
                           2u, 0u, unbounded}};
        auto request = ShaderInterfaceRequest{
            .properties = canonical,
            .arguments = arguments,
            .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
            .dialect = ShaderCodegenDialect::XIR_SPIRV,
            .printer_count = 1u,
            .use_buffer_bindless = true,
            .has_constant_ubo_payload = true};
        auto components_are_valid = [&](auto properties) {
            expect(static_cast<bool>(
                plan_persisted_descriptor_interface(
                    properties, DescriptorInterfaceStageMask::COMPUTE,
                    true, false, false, true)));
            expect(static_cast<bool>(
                plan_saved_argument_contract(arguments, 0u)));
        };

        auto valid = plan_shader_interface(request);
        expect(static_cast<bool>(valid));
        expect(eq(valid.argument_buffer_binding_count, 1u));
        expect(eq(valid.constant_ubo_binding_count, 1u));
        expect(eq(valid.resource_binding_count, 3u));
        expect(eq(valid.indirect_binding_count, 1u));
        expect(eq(valid.printer_binding_count, 2u));
        expect(eq(valid.local_binding_count, 8u));

        auto shifted_ubo = canonical;
        std::swap(shifted_ubo[2u].type, shifted_ubo[3u].type);
        components_are_valid(shifted_ubo);
        auto shifted_ubo_request = request;
        shifted_ubo_request.properties = shifted_ubo;
        auto shifted_ubo_plan = plan_shader_interface(
            shifted_ubo_request);
        expect(shifted_ubo_plan.error ==
               ShaderInterfaceError::CONSTANT_UBO_BINDING_MISMATCH);

        auto unsupported_local_role = canonical;
        unsupported_local_role[4u].type =
            hlsl::ShaderVariableType::SRVBufferHeap;
        components_are_valid(unsupported_local_role);
        auto unsupported_role_request = request;
        unsupported_role_request.properties = unsupported_local_role;
        auto unsupported_role_plan = plan_shader_interface(
            unsupported_role_request);
        expect(unsupported_role_plan.error ==
               ShaderInterfaceError::UNSUPPORTED_LOCAL_BUFFER_ROLE);

        auto wrong_native_metadata = canonical;
        wrong_native_metadata[6u].type =
            hlsl::ShaderVariableType::StructuredBuffer;
        components_are_valid(wrong_native_metadata);
        auto metadata_request = request;
        metadata_request.properties = wrong_native_metadata;
        auto metadata_plan = plan_shader_interface(metadata_request);
        expect(metadata_plan.error ==
               ShaderInterfaceError::NATIVE_BINDLESS_METADATA_MISMATCH);

        auto misplaced_indirect = canonical;
        std::swap(misplaced_indirect[6u].type,
                  misplaced_indirect[7u].type);
        components_are_valid(misplaced_indirect);
        auto indirect_request = request;
        indirect_request.properties = misplaced_indirect;
        auto indirect_plan = plan_shader_interface(indirect_request);
        expect(indirect_plan.error ==
               ShaderInterfaceError::INDIRECT_BINDING_MISMATCH);

        auto malformed_printer_tail = canonical;
        malformed_printer_tail[9u].type =
            hlsl::ShaderVariableType::StructuredBuffer;
        components_are_valid(malformed_printer_tail);
        auto printer_request = request;
        printer_request.properties = malformed_printer_tail;
        auto printer_plan = plan_shader_interface(printer_request);
        expect(printer_plan.error ==
               ShaderInterfaceError::PRINTER_TAIL_MISMATCH);
    };

    "vk_shader_interface_handles_dialect_specific_empty_argument_buffers"_test = [] {
        using namespace lc::vk::detail;
        constexpr std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u}};
        auto hlsl_debug_placeholder = plan_shader_interface(
            {.properties = properties,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::HLSL_SPIRV});
        expect(static_cast<bool>(hlsl_debug_placeholder));
        expect(eq(hlsl_debug_placeholder.argument_buffer_binding_count, 1u));

        auto llvm_extra_binding = plan_shader_interface(
            {.properties = properties,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::LLVM_SPIRV});
        expect(!static_cast<bool>(llvm_extra_binding));
        expect(llvm_extra_binding.error ==
               ShaderInterfaceError::ARGUMENT_BUFFER_BINDING_MISMATCH);

        auto raster_printer = plan_shader_interface(
            {.properties = properties,
             .stage_mask = DescriptorInterfaceStageMask::RASTER,
             .dialect = ShaderCodegenDialect::HLSL_SPIRV,
             .printer_count = 1u});
        expect(!static_cast<bool>(raster_printer));
        expect(raster_printer.error ==
               ShaderInterfaceError::PRINTER_TAIL_MISMATCH);
    };

    "vk_shader_interface_preserves_exact_native_accel_roles"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;
        std::array<SavedArgument, 1u> arguments{};
        arguments[0u].tag = Type::Tag::ACCEL;
        arguments[0u].var_usage = Usage::READ;
        arguments[0u].set_native_accel_roles(
            SavedArgument::native_accel_role_instance);

        constexpr std::array native_instance_only{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccelInstance,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 1u, 1u}};
        auto native = plan_shader_interface(
            {.properties = native_instance_only,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(static_cast<bool>(native));
        expect(eq(native.argument_buffer_binding_count, 0u));
        expect(eq(native.resource_binding_count, 1u));
        expect(eq(native.indirect_binding_count, 1u));
        expect(eq(native.local_binding_count, 2u));

        // Legacy Usage::READ keeps its historical traversal meaning. An
        // instance-only table is therefore invalid for HLSL/LLVM artifacts.
        constexpr std::array legacy_instance_only{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccelInstance,
                           0u, 0u, 1u}};
        auto legacy_arguments = arguments;
        legacy_arguments[0u].set_native_accel_roles(
            SavedArgument::unspecified_native_resource_roles);
        auto legacy_invalid = plan_shader_interface(
            {.properties = legacy_instance_only,
             .arguments = legacy_arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::LLVM_SPIRV});
        expect(!static_cast<bool>(legacy_invalid));
        expect(legacy_invalid.error ==
               ShaderInterfaceError::RESOURCE_BINDING_MISMATCH);

        constexpr std::array legacy_traversal_only{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccel,
                           0u, 0u, 1u}};
        auto legacy = plan_shader_interface(
            {.properties = legacy_traversal_only,
             .arguments = legacy_arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::LLVM_SPIRV});
        expect(static_cast<bool>(legacy));
        expect(eq(legacy.resource_binding_count, 1u));
        expect(eq(legacy.local_binding_count, 1u));
    };

    "vk_texture_descriptor_roles_delimit_adjacent_arguments"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;

        constexpr auto read_roles =
            texture_descriptor_roles(Usage::READ);
        constexpr auto write_roles =
            texture_descriptor_roles(Usage::WRITE);
        constexpr auto read_write_roles =
            texture_descriptor_roles(Usage::READ_WRITE);
        constexpr auto default_roles =
            texture_descriptor_roles(Usage::NONE);
        static_assert(read_roles.sampled && !read_roles.storage);
        static_assert(!write_roles.sampled && write_roles.storage);
        static_assert(read_write_roles.sampled &&
                      read_write_roles.storage);
        static_assert(default_roles.sampled && !default_roles.storage);

        std::array<SavedArgument, 2u> arguments{};
        arguments[0u].tag = Type::Tag::TEXTURE;
        arguments[0u].var_usage = Usage::READ;
        arguments[1u].tag = Type::Tag::TEXTURE;
        arguments[1u].var_usage = Usage::WRITE;
        constexpr std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::SRVTextureHeap,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::UAVTextureHeap,
                           0u, 1u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 2u, 1u}};
        auto adjacent = plan_shader_interface(
            {.properties = properties,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(static_cast<bool>(adjacent));
        expect(eq(adjacent.resource_binding_count, 2u));
        expect(eq(adjacent.indirect_binding_count, 1u));
        expect(eq(adjacent.local_binding_count, 3u));

        // The UAV belongs to the second argument. Claiming it for the first
        // read/write texture must leave the second argument without a role
        // instead of silently crossing the argument boundary.
        arguments[0u].var_usage = Usage::READ_WRITE;
        auto crossed = plan_shader_interface(
            {.properties = properties,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(!static_cast<bool>(crossed));
        expect(crossed.error ==
               ShaderInterfaceError::LOCAL_BINDING_COUNT_MISMATCH);
    };

    "vk_shader_interface_native_accel_roles_delimit_adjacent_arguments"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;
        std::array<SavedArgument, 2u> arguments{};
        arguments[0u].tag = Type::Tag::ACCEL;
        arguments[0u].var_usage = Usage::READ;
        arguments[0u].set_native_accel_roles(
            SavedArgument::native_accel_role_traversal);
        arguments[1u].tag = Type::Tag::ACCEL;
        arguments[1u].var_usage = Usage::READ;
        arguments[1u].set_native_accel_roles(
            SavedArgument::native_accel_role_instance);
        constexpr std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccel,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccelInstance,
                           0u, 1u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 2u, 1u}};
        auto exact = plan_shader_interface(
            {.properties = properties,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(static_cast<bool>(exact));
        expect(eq(exact.resource_binding_count, 2u));
        expect(eq(exact.indirect_binding_count, 1u));
        expect(eq(exact.local_binding_count, 3u));

        auto unspecified = arguments;
        unspecified[0u].set_native_accel_roles(
            SavedArgument::unspecified_native_resource_roles);
        auto missing_exact_role = plan_shader_interface(
            {.properties = properties,
             .arguments = unspecified,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(missing_exact_role.error ==
               ShaderInterfaceError::ACCEL_ROLE_MISMATCH);

        auto legacy_with_exact_roles = plan_shader_interface(
            {.properties = properties,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::LLVM_SPIRV});
        expect(legacy_with_exact_roles.error ==
               ShaderInterfaceError::ACCEL_ROLE_MISMATCH);

        std::array<SavedArgument, 1u> unused_argument{};
        unused_argument[0u].tag = Type::Tag::ACCEL;
        unused_argument[0u].var_usage = Usage::NONE;
        unused_argument[0u].set_native_accel_roles(0u);
        constexpr std::array unused_properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 0u, 1u}};
        auto unused = plan_shader_interface(
            {.properties = unused_properties,
             .arguments = unused_argument,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(static_cast<bool>(unused));
        expect(eq(unused.resource_binding_count, 0u));
        expect(eq(unused.indirect_binding_count, 1u));

        unused_argument[0u].var_usage = Usage::READ;
        auto used_with_zero_roles = plan_shader_interface(
            {.properties = unused_properties,
             .arguments = unused_argument,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV});
        expect(used_with_zero_roles.error ==
               ShaderInterfaceError::ACCEL_ROLE_MISMATCH);
    };

    "vk_shader_interface_binds_native_metadata_without_buffer_heap"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;
        std::array<SavedArgument, 2u> arguments{};
        arguments[0u].tag = Type::Tag::BINDLESS_ARRAY;
        arguments[0u].var_usage = Usage::NONE;
        arguments[1u].tag = Type::Tag::BINDLESS_ARRAY;
        arguments[1u].var_usage = Usage::READ;
        constexpr std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, descriptor_interface_sampler_count},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                           0u, 1u, 1u},
            hlsl::Property{
                hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata,
                0u, 2u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                           0u, 3u, 1u}};
        auto plan = plan_shader_interface(
            {.properties = properties,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV,
             .use_buffer_bindless = false});
        expect(static_cast<bool>(plan));
        expect(eq(plan.argument_buffer_binding_count, 0u));
        expect(eq(plan.resource_binding_count, 3u));
        expect(eq(plan.indirect_binding_count, 1u));
        expect(eq(plan.local_binding_count, 4u));
        expect(eq(plan.descriptor_interface.descriptor_set_count, 2u));

        constexpr auto unbounded =
            std::numeric_limits<uint32_t>::max();
        constexpr std::array heap_properties{
            properties[0u], properties[1u], properties[2u],
            properties[3u], properties[4u],
            hlsl::Property{hlsl::ShaderVariableType::SRVBufferHeap,
                           2u, 0u, unbounded}};
        auto heap_plan = plan_shader_interface(
            {.properties = heap_properties,
             .arguments = arguments,
             .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
             .dialect = ShaderCodegenDialect::XIR_SPIRV,
             .use_buffer_bindless = true});
        expect(static_cast<bool>(heap_plan));
        expect(eq(heap_plan.resource_binding_count, 3u));
        expect(eq(heap_plan.local_binding_count, 4u));
        expect(eq(heap_plan.descriptor_interface.descriptor_set_count, 3u));
    };

    "vk_llvm_property_plan_matches_the_declared_runtime_binding_abi"_test = [] {
        using namespace lc::llvm_codegen;
        using namespace lc::vk;
        using namespace lc::vk::detail;
        constexpr auto unbounded = std::numeric_limits<uint32_t>::max();
        constexpr std::array llvm_arguments{
            LLVMVulkanBindingArgument{Type::Tag::UINT32, Usage::READ},
            LLVMVulkanBindingArgument{Type::Tag::BUFFER, Usage::READ},
            LLVMVulkanBindingArgument{Type::Tag::TEXTURE, Usage::READ_WRITE},
            LLVMVulkanBindingArgument{Type::Tag::BINDLESS_ARRAY, Usage::READ},
            LLVMVulkanBindingArgument{Type::Tag::ACCEL, Usage::READ_WRITE}};
        auto property_plan = plan_llvm_vulkan_binding_properties({.arguments = llvm_arguments,
                                                                  .use_buffer_bindless = true,
                                                                  .use_tex2d_bindless = true,
                                                                  .use_tex3d_bindless = true,
                                                                  .printer_count = 1u});
        expect(property_plan.has_argument_buffer);
        expect(eq(property_plan.local_binding_count, 9u));
        expect(eq(property_plan.properties.size(), size_t{13u}));

        auto expect_property = [&](size_t index,
                                   hlsl::ShaderVariableType type,
                                   uint32_t set, uint32_t binding,
                                   uint32_t array_size) {
            auto property = property_plan.properties[index];
            expect(property.type == type);
            expect(eq(property.space_index, set));
            expect(eq(property.register_index, binding));
            expect(eq(property.array_size, array_size));
        };
        expect_property(0u, hlsl::ShaderVariableType::SamplerHeap,
                        1u, 0u, descriptor_interface_sampler_count);
        expect_property(1u, hlsl::ShaderVariableType::StructuredBuffer,
                        0u, 0u, 1u);
        expect_property(2u, hlsl::ShaderVariableType::SRVBufferHeap,
                        2u, 0u, unbounded);
        expect_property(3u, hlsl::ShaderVariableType::SRVTextureHeap,
                        3u, 0u, unbounded);
        expect_property(4u, hlsl::ShaderVariableType::SRVTextureHeap,
                        4u, 0u, unbounded);
        expect_property(5u, hlsl::ShaderVariableType::StructuredBuffer,
                        0u, 1u, 1u);
        expect_property(6u, hlsl::ShaderVariableType::SRVTextureHeap,
                        0u, 2u, 1u);
        expect_property(7u, hlsl::ShaderVariableType::UAVTextureHeap,
                        0u, 3u, 1u);
        expect_property(8u, hlsl::ShaderVariableType::StructuredBuffer,
                        0u, 4u, 1u);
        expect_property(9u, hlsl::ShaderVariableType::SPIRVAccel,
                        0u, 5u, 1u);
        expect_property(10u, hlsl::ShaderVariableType::SPIRVAccelInstanceRW,
                        0u, 6u, 1u);
        expect_property(11u, hlsl::ShaderVariableType::RWStructuredBuffer,
                        0u, 7u, 1u);
        expect_property(12u, hlsl::ShaderVariableType::RWStructuredBuffer,
                        0u, 8u, 1u);

        std::array<SavedArgument, llvm_arguments.size()> saved_arguments{};
        saved_arguments[0u].tag = Type::Tag::UINT32;
        saved_arguments[0u].var_usage = Usage::READ;
        saved_arguments[0u].struct_size = sizeof(uint32_t);
        saved_arguments[1u].tag = Type::Tag::BUFFER;
        saved_arguments[1u].var_usage = Usage::READ;
        saved_arguments[1u].struct_size = sizeof(uint32_t);
        saved_arguments[2u].tag = Type::Tag::TEXTURE;
        saved_arguments[2u].var_usage = Usage::READ_WRITE;
        saved_arguments[3u].tag = Type::Tag::BINDLESS_ARRAY;
        saved_arguments[3u].var_usage = Usage::READ;
        saved_arguments[4u].tag = Type::Tag::ACCEL;
        saved_arguments[4u].var_usage = Usage::READ_WRITE;
        auto runtime_plan = plan_shader_interface({.properties = property_plan.properties,
                                                   .arguments = saved_arguments,
                                                   .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
                                                   .dialect = ShaderCodegenDialect::LLVM_SPIRV,
                                                   .printer_count = 1u,
                                                   .use_buffer_bindless = true,
                                                   .use_tex2d_bindless = true,
                                                   .use_tex3d_bindless = true});
        expect(static_cast<bool>(runtime_plan));
        expect(eq(runtime_plan.argument_buffer_binding_count, 1u));
        expect(eq(runtime_plan.constant_ubo_binding_count, 0u));
        expect(eq(runtime_plan.resource_binding_count, 6u));
        expect(eq(runtime_plan.printer_binding_count, 2u));
        expect(eq(runtime_plan.local_binding_count, 9u));
    };

    "vk_llvm_property_plan_only_maps_specialized_custom_arguments"_test = [] {
        using namespace lc::llvm_codegen;
        auto plan_one = [](bool indirect_dispatch_buffer) {
            const std::array arguments{
                LLVMVulkanBindingArgument{
                    Type::Tag::CUSTOM,
                    Usage::READ_WRITE,
                    indirect_dispatch_buffer}};
            return plan_llvm_vulkan_binding_properties({.arguments = arguments});
        };

        auto indirect = plan_one(true);
        expect(!indirect.has_argument_buffer);
        expect(eq(indirect.local_binding_count, 1u));
        expect(eq(indirect.properties.size(), size_t{2u}));
        expect(indirect.properties[1u].type ==
               hlsl::ShaderVariableType::RWStructuredBuffer);

        auto arbitrary_custom = plan_one(false);
        expect(!arbitrary_custom.has_argument_buffer);
        expect(eq(arbitrary_custom.local_binding_count, 0u));
        expect(eq(arbitrary_custom.properties.size(), size_t{1u}));
    };

    "vk_llvm_codegen_rejects_unimplemented_resource_models_explicitly"_test = [] {
        using namespace lc::llvm_codegen;
        using Error = LLVMVulkanResourceModelError;
        auto validate_one = [](LLVMVulkanBindingArgument argument) {
            std::array arguments{argument};
            return validate_llvm_vulkan_resource_model({.arguments = arguments});
        };

        expect(static_cast<bool>(validate_llvm_vulkan_resource_model({})));
        expect(validate_llvm_vulkan_resource_model({.printer_count = 1u})
                   .error == Error::PRINTING_NOT_IMPLEMENTED);
        expect(validate_llvm_vulkan_resource_model({.use_buffer_bindless = true})
                   .error == Error::BINDLESS_RESOURCES_NOT_IMPLEMENTED);
        expect(validate_one({Type::Tag::BUFFER, Usage::READ}).error ==
               Error::DIRECT_BUFFER_DESCRIPTORS_NOT_IMPLEMENTED);
        expect(validate_one({Type::Tag::TEXTURE, Usage::READ}).error ==
               Error::TEXTURE_RESOURCES_NOT_IMPLEMENTED);
        expect(validate_one({Type::Tag::BINDLESS_ARRAY, Usage::READ}).error ==
               Error::BINDLESS_RESOURCES_NOT_IMPLEMENTED);
        expect(validate_one({Type::Tag::ACCEL, Usage::READ}).error ==
               Error::ACCEL_RESOURCES_NOT_IMPLEMENTED);
        expect(validate_one(
                   {Type::Tag::CUSTOM, Usage::READ_WRITE, true})
                   .error ==
               Error::INDIRECT_DISPATCH_BUFFER_NOT_IMPLEMENTED);
        expect(validate_one(
                   {Type::Tag::CUSTOM, Usage::READ_WRITE, false})
                   .error ==
               Error::CUSTOM_ARGUMENT_NOT_IMPLEMENTED);
        expect(validate_one({Type::Tag::UINT32, Usage::READ}).error ==
               Error::ARGUMENT_BUFFER_DESCRIPTOR_NOT_IMPLEMENTED);
    };

    "vk_descriptor_interface_checks_accel_limit_domains_exactly"_test = [] {
        using namespace lc::vk::detail;
        const std::array properties{
            hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                           1u, 0u, 16u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccel,
                           0u, 0u, 1u},
            hlsl::Property{hlsl::ShaderVariableType::SPIRVAccel,
                           0u, 1u, 1u}};
        auto request = DescriptorInterfaceRequest{
            .properties = properties,
            .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
            .acceleration_structure_available = true};
        constexpr auto exact_limits = [] {
            auto limits = generous_descriptor_interface_limits();
            limits.max_per_stage_resources = 0u;
            limits.max_per_stage_descriptor_acceleration_structures = 2u;
            limits.max_descriptor_set_acceleration_structures = 2u;
            limits.max_per_stage_descriptor_update_after_bind_acceleration_structures = 2u;
            limits.max_descriptor_set_update_after_bind_acceleration_structures = 2u;
            return limits;
        }();
        auto exact = plan_descriptor_interface(
            request, exact_limits);
        expect(eq(exact.acceleration_structure_binding_count, 2u));
        expect(eq(exact.set_counts[0u].resources(), 0u));

        auto expect_error = [&](DescriptorInterfaceLimits limits,
                                DescriptorInterfaceError error) {
            auto plan = plan_descriptor_interface(request, limits);
            expect(!static_cast<bool>(plan));
            expect(plan.error == error);
        };
        auto limits = exact_limits;
        limits.max_descriptor_set_acceleration_structures = 1u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES);
        limits = exact_limits;
        limits.max_per_stage_descriptor_acceleration_structures = 1u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_ACCELERATION_STRUCTURES);
        limits = exact_limits;
        limits.max_per_stage_descriptor_update_after_bind_acceleration_structures = 1u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES);
        limits = exact_limits;
        limits.max_descriptor_set_update_after_bind_acceleration_structures = 1u;
        expect_error(
            limits,
            DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES);
    };

    "vk_push_constant_write_guard_checks_exact_end_and_alignment"_test = [] {
        using lc::vk::detail::valid_push_constant_write;
        static_assert(valid_push_constant_write(0u, 32u, 32u));
        static_assert(valid_push_constant_write(16u, 16u, 32u));
        static_assert(!valid_push_constant_write(16u, 20u, 32u));
        static_assert(!valid_push_constant_write(1u, 4u, 32u));
        static_assert(!valid_push_constant_write(0u, 2u, 32u));
        static_assert(!valid_push_constant_write(
            std::numeric_limits<uint32_t>::max() - 3u, 4u, 32u));
        expect(valid_push_constant_write(0u, 32u, 32u));
        expect(!valid_push_constant_write(0u, 36u, 32u));
    };

    "vk_bindless_heap_capacity_reserves_local_budget_before_per_stage_clamps"_test = [] {
        using namespace lc::vk::detail;
        constexpr BindlessHeapLimits generous{
            .max_per_stage_update_after_bind_samplers = 1'000'000u,
            .max_descriptor_set_update_after_bind_samplers = 1'000'000u,
            .max_per_stage_update_after_bind_storage_buffers = 1'000'000u,
            .max_descriptor_set_update_after_bind_storage_buffers = 1'000'000u,
            .max_per_stage_update_after_bind_sampled_images = 1'000'000u,
            .max_descriptor_set_update_after_bind_sampled_images = 1'000'000u,
            .max_per_stage_update_after_bind_resources = 1'000'000u,
            .max_update_after_bind_descriptors_in_all_pools = 1'000'000u};

        constexpr auto storage_limited = [=] {
            auto limits = generous;
            limits.max_per_stage_update_after_bind_storage_buffers =
                local_per_stage_descriptor_budget + 29u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(storage_limited, 29u));

        constexpr auto sampled_limited = [=] {
            auto limits = generous;
            limits.max_per_stage_update_after_bind_sampled_images =
                local_per_stage_descriptor_budget + 2u * 31u + 1u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(sampled_limited, 31u))
            << "local sampled-image headroom must be reserved before the two-heap division";

        constexpr auto aggregate_limited = [=] {
            auto limits = generous;
            limits.max_per_stage_update_after_bind_resources =
                local_per_stage_descriptor_budget + 3u * 37u + 2u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(aggregate_limited, 37u))
            << "local aggregate headroom must be reserved before the three-heap division";

        constexpr auto no_local_headroom = [=] {
            auto limits = generous;
            limits.max_per_stage_update_after_bind_storage_buffers =
                local_per_stage_descriptor_budget;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(no_local_headroom, 0u));
    };

    "vk_bindless_heap_capacity_applies_set_pool_and_request_limits_exactly"_test = [] {
        using namespace lc::vk::detail;
        constexpr BindlessHeapLimits generous{
            .max_per_stage_update_after_bind_samplers = 1'000'000u,
            .max_descriptor_set_update_after_bind_samplers = 1'000'000u,
            .max_per_stage_update_after_bind_storage_buffers = 1'000'000u,
            .max_descriptor_set_update_after_bind_storage_buffers = 1'000'000u,
            .max_per_stage_update_after_bind_sampled_images = 1'000'000u,
            .max_descriptor_set_update_after_bind_sampled_images = 1'000'000u,
            .max_per_stage_update_after_bind_resources = 1'000'000u,
            .max_update_after_bind_descriptors_in_all_pools = 1'000'000u};
        expect(eq(plan_bindless_heap_capacity(generous),
                  requested_bindless_heap_capacity));

        constexpr auto sampled_set_limited = [=] {
            auto limits = generous;
            limits.max_descriptor_set_update_after_bind_sampled_images =
                local_per_stage_descriptor_budget + 2u * 53u + 1u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(sampled_set_limited, 53u))
            << "the aggregate limit sees local sampled images and both texture heaps";

        constexpr auto storage_set_limited = [=] {
            auto limits = generous;
            limits.max_descriptor_set_update_after_bind_storage_buffers =
                local_per_stage_descriptor_budget + 47u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(storage_set_limited, 47u));

        constexpr auto pool_limited = [=] {
            auto limits = generous;
            limits.max_update_after_bind_descriptors_in_all_pools =
                3u * 41u + 2u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(pool_limited, 41u));

        constexpr auto sampler_limited = [=] {
            auto limits = generous;
            limits.max_descriptor_set_update_after_bind_samplers =
                fixed_sampler_descriptor_count - 1u;
            return plan_bindless_heap_capacity(limits);
        }();
        expect(eq(sampler_limited, 0u))
            << "the fixed ordinary sampler set participates in UAB aggregates";
    };

    "vk_required_features_use_core_without_redundant_extensions"_test = [] {
        constexpr auto plan = lc::vk::detail::plan_required_device_features({.timeline_semaphore_core = true,
                                                                             .timeline_semaphore_extension = true,
                                                                             .timeline_semaphore_feature = true,
                                                                             .synchronization2_core = true,
                                                                             .synchronization2_extension = true,
                                                                             .synchronization2_feature = true,
                                                                             .physical_device_api_1_3 = true});
        expect(plan.supported);
        expect(!plan.enable_timeline_semaphore_extension);
        expect(!plan.enable_synchronization2_extension);
    };

    "vk_imported_device_requires_enabled_feature_attestations"_test = [] {
        using namespace lc::vk::detail;
        expect(static_cast<bool>(validate_external_required_features(
            false, false, false)));
        expect(validate_external_required_features(true, false, true).status ==
               ExternalRequiredFeatureStatus::MISSING_TIMELINE_SEMAPHORE);
        expect(validate_external_required_features(true, true, false).status ==
               ExternalRequiredFeatureStatus::MISSING_SYNCHRONIZATION2);
        expect(static_cast<bool>(validate_external_required_features(
            true, true, true)));
    };

    "vk_borrowed_instance_extension_paths_fail_closed"_test = [] {
        using namespace lc::vk::detail;
        constexpr auto owned = plan_instance_runtime_capabilities(
            false, true, true);
        static_assert(owned.surface && owned.debug_utils);
        constexpr auto borrowed = plan_instance_runtime_capabilities(
            true, true, true);
        static_assert(!borrowed.surface && !borrowed.debug_utils);
        expect(owned.surface);
        expect(owned.debug_utils);
        expect(!borrowed.surface);
        expect(!borrowed.debug_utils);
    };

    "vk_imported_instance_requires_effective_vulkan_1_3_api"_test = [] {
        using namespace lc::vk::detail;
        expect(static_cast<bool>(validate_external_instance_api(
            false, 0u)))
            << "an internally created instance selects the backend API";
        expect(validate_external_instance_api(true, 0u).status ==
               ExternalInstanceApiStatus::MISSING_VERSION);
        expect(validate_external_instance_api(
                   true, VK_API_VERSION_1_2)
                   .status ==
               ExternalInstanceApiStatus::VERSION_TOO_OLD);
        expect(validate_external_instance_api(
                   true, VK_MAKE_API_VERSION(1u, 1u, 3u, 0u))
                   .status ==
               ExternalInstanceApiStatus::UNSUPPORTED_VARIANT);
        expect(static_cast<bool>(validate_external_instance_api(
            true, VK_API_VERSION_1_3)));
        expect(static_cast<bool>(validate_external_instance_api(
            true, VK_MAKE_API_VERSION(0u, 1u, 4u, 7u))));
    };

    "vk_required_features_reject_pre_1_3_extension_only_devices"_test = [] {
        constexpr auto plan = lc::vk::detail::plan_required_device_features({.timeline_semaphore_extension = true,
                                                                             .timeline_semaphore_feature = true,
                                                                             .synchronization2_extension = true,
                                                                             .synchronization2_feature = true});
        expect(!plan.supported)
            << "core copy_commands2 is a backend-wide Vulkan 1.3 contract";
        expect(!plan.enable_timeline_semaphore_extension);
        expect(!plan.enable_synchronization2_extension);
    };

    "vk_required_extension_names_do_not_imply_features"_test = [] {
        constexpr auto missing_timeline =
            lc::vk::detail::plan_required_device_features({.timeline_semaphore_extension = true,
                                                           .synchronization2_extension = true,
                                                           .synchronization2_feature = true,
                                                           .physical_device_api_1_3 = true});
        expect(!missing_timeline.supported);
        expect(!missing_timeline.enable_timeline_semaphore_extension);

        constexpr auto missing_synchronization2 =
            lc::vk::detail::plan_required_device_features({.timeline_semaphore_core = true,
                                                           .timeline_semaphore_feature = true,
                                                           .synchronization2_extension = true,
                                                           .physical_device_api_1_3 = true});
        expect(!missing_synchronization2.supported);
        expect(!missing_synchronization2.enable_synchronization2_extension);
    };

    "vk_narrow_numeric_features_remain_independent"_test = [] {
        using namespace lc::vk::detail;

        constexpr auto storage_only_16 = plan_narrow_numeric_features({.storage_buffer_16bit_access = true});
        static_assert(storage_only_16.storage_buffer_16bit_access);
        static_assert(!storage_only_16.shader_float16);
        expect(storage_only_16.storage_buffer_16bit_access);
        expect(!storage_only_16.uniform_storage_buffer_16bit_access);
        expect(!storage_only_16.shader_float16);

        constexpr auto arithmetic_only = plan_narrow_numeric_features({.shader_float16 = true,
                                                                       .shader_int8 = true});
        expect(arithmetic_only.shader_float16);
        expect(arithmetic_only.shader_int8);
        expect(!arithmetic_only.storage_buffer_8bit_access);
        expect(!arithmetic_only.uniform_storage_buffer_8bit_access);
        expect(!arithmetic_only.storage_buffer_16bit_access);
        expect(!arithmetic_only.uniform_storage_buffer_16bit_access);

        constexpr auto asymmetric_8 = plan_narrow_numeric_features({.storage_buffer_8bit_access = true});
        expect(asymmetric_8.storage_buffer_8bit_access);
        expect(!asymmetric_8.uniform_storage_buffer_8bit_access);
        expect(!asymmetric_8.shader_int8);

        constexpr auto all = plan_narrow_numeric_features({.shader_float16 = true,
                                                           .shader_int8 = true,
                                                           .storage_buffer_8bit_access = true,
                                                           .uniform_storage_buffer_8bit_access = true,
                                                           .storage_buffer_16bit_access = true,
                                                           .uniform_storage_buffer_16bit_access = true});
        expect(all.shader_float16);
        expect(all.shader_int8);
        expect(all.storage_buffer_8bit_access);
        expect(all.uniform_storage_buffer_8bit_access);
        expect(all.storage_buffer_16bit_access);
        expect(all.uniform_storage_buffer_16bit_access);
    };

    "vk_optional_extension_names_do_not_imply_features"_test = [] {
        constexpr auto plan = lc::vk::detail::plan_optional_device_features(
            {.fragment_shader_barycentric_extension = true,
             .ray_query_extension = true,
             .acceleration_structure_extension = true,
             .deferred_host_operations_extension = true,
             .buffer_device_address = true,
             .ray_tracing_pipeline_extension = true,
             .ray_tracing_motion_blur_extension = true},
            {.ray_query = true,
             .ray_tracing_motion_blur = true});
        expect(!plan.fragment_shader_barycentric);
        expect(!plan.ray_query);
        expect(!plan.acceleration_structure);
        expect(!plan.ray_tracing_pipeline);
        expect(!plan.ray_tracing_motion_blur);
    };

    "vk_ray_query_does_not_require_ray_tracing_pipeline_features"_test = [] {
        constexpr auto plan = lc::vk::detail::plan_optional_device_features(
            {.ray_query_extension = true,
             .ray_query_feature = true,
             .acceleration_structure_extension = true,
             .acceleration_structure_feature = true,
             .deferred_host_operations_extension = true,
             .buffer_device_address = true},
            {.ray_query = true,
             .ray_tracing_motion_blur = false});
        expect(plan.ray_query);
        expect(plan.acceleration_structure);
        expect(!plan.ray_tracing_pipeline);
        expect(!plan.ray_tracing_motion_blur);
        // Vulkan's SPIR-V capability table authorizes
        // RayTraversalPrimitiveCullingKHR through rayQuery itself for
        // OpRayQueryInitializeKHR. Do not accidentally make ordinary ray
        // queries depend on the separate ray-tracing-pipeline feature struct.
    };

    "vk_ray_query_requires_the_complete_feature_contract"_test = [] {
        constexpr lc::vk::detail::OptionalDeviceFeatureSupport supported{
            .fragment_shader_barycentric_extension = true,
            .fragment_shader_barycentric_feature = true,
            .ray_query_extension = true,
            .ray_query_feature = true,
            .acceleration_structure_extension = true,
            .acceleration_structure_feature = true,
            .deferred_host_operations_extension = true,
            .buffer_device_address = true,
            .ray_tracing_pipeline_extension = true,
            .ray_tracing_pipeline_feature = true,
            .ray_traversal_primitive_culling_feature = true,
            .ray_tracing_motion_blur_extension = true,
            .ray_tracing_motion_blur_feature = true};
        constexpr auto plan = lc::vk::detail::plan_optional_device_features(
            supported,
            {.ray_query = true,
             .ray_tracing_motion_blur = true});
        expect(plan.fragment_shader_barycentric);
        expect(plan.ray_query);
        expect(plan.acceleration_structure);
        expect(plan.ray_tracing_pipeline);
        expect(plan.ray_tracing_motion_blur);

        constexpr auto no_device_address = [=] {
            auto value = supported;
            value.buffer_device_address = false;
            return lc::vk::detail::plan_optional_device_features(
                value,
                {.ray_query = true,
                 .ray_tracing_motion_blur = true});
        }();
        expect(!no_device_address.ray_query);
        expect(!no_device_address.ray_tracing_motion_blur);

        constexpr auto no_acceleration_structure = [=] {
            auto value = supported;
            value.acceleration_structure_feature = false;
            return lc::vk::detail::plan_optional_device_features(
                value,
                {.ray_query = true,
                 .ray_tracing_motion_blur = true});
        }();
        expect(!no_acceleration_structure.ray_query);
        expect(!no_acceleration_structure.acceleration_structure);
        expect(!no_acceleration_structure.ray_tracing_motion_blur);

        constexpr auto no_motion_feature = [=] {
            auto value = supported;
            value.ray_tracing_motion_blur_feature = false;
            return lc::vk::detail::plan_optional_device_features(
                value,
                {.ray_query = true,
                 .ray_tracing_motion_blur = true});
        }();
        expect(no_motion_feature.ray_query);
        expect(!no_motion_feature.ray_tracing_pipeline);
        expect(!no_motion_feature.ray_tracing_motion_blur);

        constexpr auto no_primitive_culling = [=] {
            auto value = supported;
            value.ray_traversal_primitive_culling_feature = false;
            return lc::vk::detail::plan_optional_device_features(
                value,
                {.ray_query = true,
                 .ray_tracing_motion_blur = true});
        }();
        expect(no_primitive_culling.ray_query)
            << "rayQuery independently authorizes primitive-culling ray flags";
        expect(!no_primitive_culling.ray_tracing_pipeline);
        expect(!no_primitive_culling.ray_tracing_motion_blur);
    };

    "vk_sampler_anisotropy_plan_is_feature_exact_and_clamped"_test = [] {
        constexpr auto unsupported =
            lc::vk::detail::plan_sampler_anisotropy(
                {.max_sampler_anisotropy = 64.0f});
        expect(!unsupported.enabled);
        expect(eq(unsupported.max_anisotropy, 1.0f));

        constexpr auto imported =
            lc::vk::detail::plan_sampler_anisotropy({.physical_device_feature = true,
                                                     .imported_device = true,
                                                     .max_sampler_anisotropy = 64.0f});
        expect(!imported.enabled)
            << "physical support does not reveal imported VkDevice features";
        expect(eq(imported.max_anisotropy, 1.0f));

        constexpr auto limited =
            lc::vk::detail::plan_sampler_anisotropy({.physical_device_feature = true,
                                                     .max_sampler_anisotropy = 8.0f});
        expect(limited.enabled);
        expect(eq(limited.max_anisotropy, 8.0f));

        constexpr auto requested_limit =
            lc::vk::detail::plan_sampler_anisotropy({.physical_device_feature = true,
                                                     .max_sampler_anisotropy = 32.0f});
        expect(requested_limit.enabled);
        expect(eq(requested_limit.max_anisotropy, 16.0f));

        expect(lc::vk::detail::sampler_requirement_is_supported(
            false, false));
        expect(!lc::vk::detail::sampler_requirement_is_supported(
            true, false));
        expect(lc::vk::detail::sampler_requirement_is_supported(
            true, true));
    };

    "vk_sampler_heap_index_is_address_major_filter_minor"_test = [] {
        using Filter = Sampler::Filter;
        using Address = Sampler::Address;
        expect(eq(lc::vk::detail::sampler_heap_size, 16u));
        expect(eq(lc::vk::detail::descriptor_interface_sampler_count,
                  lc::vk::detail::sampler_heap_size));
        expect(eq(lc::vk::detail::fixed_sampler_descriptor_count,
                  lc::vk::detail::sampler_heap_size));
        expect(eq(lc::vk::detail::sampler_heap_index(
                      to_underlying(Filter::POINT),
                      to_underlying(Address::EDGE)),
                  0u));
        expect(eq(lc::vk::detail::sampler_heap_index(
                      to_underlying(Filter::ANISOTROPIC),
                      to_underlying(Address::EDGE)),
                  3u));
        expect(eq(lc::vk::detail::sampler_heap_index(
                      to_underlying(Filter::POINT),
                      to_underlying(Address::REPEAT)),
                  4u));
        expect(eq(lc::vk::detail::sampler_heap_index(
                      to_underlying(Filter::ANISOTROPIC),
                      to_underlying(Address::ZERO)),
                  15u));
        expect(Sampler{Filter::POINT, Address::REPEAT}.code() !=
               lc::vk::detail::sampler_heap_index(
                   to_underlying(Filter::POINT),
                   to_underlying(Address::REPEAT)))
            << "public Sampler::code and the Vulkan heap index are distinct ABIs";

        std::array<bool, lc::vk::detail::sampler_heap_size> seen{};
        for (auto address = 0u;
             address < lc::vk::detail::sampler_address_count;
             ++address) {
            for (auto filter = 0u;
                 filter < lc::vk::detail::sampler_filter_count;
                 ++filter) {
                auto index = lc::vk::detail::sampler_heap_index(
                    filter, address);
                expect(index < seen.size());
                expect(!seen[index]);
                seen[index] = true;
            }
        }
        for (auto value : seen) { expect(value); }
    };

    "vk_explicit_sampler_call_classifier_covers_all_frontends"_test = [] {
        constexpr std::array explicit_sampler_ops{
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TEXTURE2D_SAMPLE,
            CallOp::TEXTURE2D_SAMPLE_LEVEL,
            CallOp::TEXTURE2D_SAMPLE_GRAD,
            CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL,
            CallOp::TEXTURE3D_SAMPLE,
            CallOp::TEXTURE3D_SAMPLE_LEVEL,
            CallOp::TEXTURE3D_SAMPLE_GRAD,
            CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL};
        for (auto op : explicit_sampler_ops) {
            expect(lc::vk::detail::call_has_explicit_sampler_filter(op));
        }
        expect(!lc::vk::detail::call_has_explicit_sampler_filter(
            CallOp::BINDLESS_TEXTURE2D_SAMPLE));
        expect(!lc::vk::detail::call_has_explicit_sampler_filter(
            CallOp::TEXTURE_READ));
    };

    "vk_sampler_ast_usage_distinguishes_static_and_dynamic_filters"_test = [] {
        auto make_kernel = [](bool dynamic, uint32_t static_filter) {
            Kernel1D kernel = [=](ImageFloat image, BufferFloat4 output,
                                  UInt filter_argument) noexcept {
                auto builder =
                    luisa::compute::detail::FunctionBuilder::current();
                auto literal = [&](auto value) noexcept {
                    return builder->literal(
                        Type::of<decltype(value)>(), value);
                };
                auto filter = dynamic ?
                                  filter_argument.expression() :
                                  literal(static_filter);
                auto sample = builder->call(
                    Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
                    {image.expression(), literal(make_float2(0.5f)),
                     filter, literal(0u)});
                output.write(0u, def<float4>(sample));
            };
            return kernel;
        };

        auto non_anisotropic = make_kernel(false, 2u);
        auto non_anisotropic_usage =
            lc::vk::detail::analyze_sampler_usage(
                non_anisotropic.function()->function());
        expect(non_anisotropic_usage.uses_explicit_sampler);
        expect(!non_anisotropic_usage.requires_anisotropy);
        expect(!non_anisotropic_usage.has_dynamic_filter);
        expect(!non_anisotropic_usage.has_invalid_filter);

        auto anisotropic = make_kernel(false, 3u);
        auto anisotropic_usage =
            lc::vk::detail::analyze_sampler_usage(
                anisotropic.function()->function());
        expect(anisotropic_usage.requires_anisotropy);
        expect(!anisotropic_usage.has_dynamic_filter);
        expect(!anisotropic_usage.has_invalid_filter);

        auto dynamic = make_kernel(true, 0u);
        auto dynamic_usage = lc::vk::detail::analyze_sampler_usage(
            dynamic.function()->function());
        expect(dynamic_usage.requires_anisotropy);
        expect(dynamic_usage.has_dynamic_filter);
        expect(!dynamic_usage.has_invalid_filter);

        auto invalid = make_kernel(false, 4u);
        auto invalid_usage = lc::vk::detail::analyze_sampler_usage(
            invalid.function()->function());
        expect(invalid_usage.has_invalid_filter);
    };
}
