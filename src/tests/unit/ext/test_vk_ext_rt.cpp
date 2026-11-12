// Test for the Vulkan ray tracing extension planning of the backend
// (VK_EXT_ray_tracing_invocation_reorder, VK_KHR_opacity_micromap,
// VK_NV_ray_tracing_linear_swept_spheres, VK_AMDX_dense_geometry_format,
// VK_NV_partitioned_acceleration_structure).
//
// This test covers the host-visible planning helpers of
// `src/backends/vk/rt_extension_plan.h`, which are the single source of truth
// for every decision the backend makes about these five extensions:
//
// - the opacity micromap build plan: the "extension off -> nothing is chained,
//   nothing is sized" rejection, the "enabled but the public accel API cannot
//   request a micromap" default answer, the KHR-revision object model that this
//   backend cannot create a VkMicromapEXT for, both build modes (host build
//   behind micromapHostCommands, device build behind buffer device addresses)
//   and the exact data/triangle-array sizes the ABI demands;
// - the opacity micromap attach plan: every reason a legal-looking request is
//   still refused (AABB/motion geometry, UINT8 index type without the index
//   extension, misaligned index stride, the instance that disables micromaps,
//   the OpaqueKHR ray flag that would make the micromap a no-op, and the
//   OpacityMicromapIdKHR execution mode that only EXT_opacity_micromap lifts);
// - the geometry flag decision, including the invariant that an offered-but-
//   rejected micromap leaves the flags byte-identical to today's build;
// - the SER (VK_EXT_ray_tracing_invocation_reorder) applicability: the
//   extension has no ray-query surface at all, so it stays enabled-but-inactive
//   for every traversal this backend generates;
// - the linear swept sphere geometry mapping (indexing mode -> primitive count,
//   stride/format checks) and the one reason the backend cannot reach it (no
//   curve resource);
// - the DGF1 blob validation (128-byte block alignment/size, 64 triangles and
//   64 vertices per block) of the provisional VK_AMDX_dense_geometry_format;
// - the PTLAS instance-input limits (partitionCount + global partition count <=
//   maxPartitionCount) of VK_NV_partitioned_acceleration_structure;
// - every status-name mapping, so a log line can never print "unknown".
//
// No Vulkan object is created and no device is required: the helpers are pure,
// the backend only ever walks the branch they select, and these cases are the
// only evidence available for the four extensions the test GPU does not expose.
#include "ut/ut.hpp"
#include "rt_extension_plan.h"
#include <cstdint>
#include <string_view>
using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace test_vk_ext_rt_detail {

// boost::ut prints both operands of a failed `expect(lhs == rhs)`, which needs an
// `operator<<` for the enum. Comparing the underlying value keeps the enum classes
// strongly typed while giving readable failures.
template<typename E>
[[nodiscard]] bool same(E lhs, E rhs) noexcept {
    return static_cast<int>(lhs) == static_cast<int>(rhs);
}

[[nodiscard]] bool known_name(const char *name) noexcept {
    return name != nullptr && std::string_view{name} != "unknown";
}

// A request that *would* build a legal micromap on a fully capable device:
// every "rejected" case below flips exactly one field of this.
[[nodiscard]] OpacityMicromapBuildRequest capable_build_request() noexcept {
    OpacityMicromapBuildRequest request;
    request.extension_enabled = true;
    request.object_model = MicromapObjectModel::EXT_MICROMAP_OBJECT;
    request.host_commands_enabled = true;
    request.device_address_supported = true;
    request.storage_host_visible = true;
    request.public_api_request = true;
    request.triangle_count = 10u;
    request.format = OpacityMicromapFormat::TWO_STATE;
    request.subdivision_level = 0u;
    request.max_two_state_subdivision_level = 6u;
    request.max_four_state_subdivision_level = 4u;
    request.scratch_offset_alignment = 256u;
    return request;
}

// A request that *would* attach a legal micromap.
[[nodiscard]] OpacityMicromapAttachRequest capable_attach_request() noexcept {
    OpacityMicromapAttachRequest request;
    request.extension_enabled = true;
    request.public_api_request = true;
    request.micromap_available = true;
    request.triangle_geometry = true;
    request.ext_opacity_micromap_enabled = false;
    request.ray_query_execution_mode_emitted = true;
    request.index_type = OpacityMicromapIndexType::UINT32;
    request.index_stride = 4u;
    request.base_triangle = 0u;
    request.usage_count = 1u;
    request.triangle_count = 10u;
    return request;
}

// A curve build the swept-sphere geometry mapping accepts.
[[nodiscard]] LssRequest capable_lss_request() noexcept {
    LssRequest request;
    request.extension_enabled = true;
    request.curve_resource_available = true;
    request.vertex_format_supported = true;
    request.radius_format_supported = true;
    request.index_buffer_present = true;
    request.indexing_mode = LssIndexingMode::LIST;
    request.end_caps_mode = LssEndCapsMode::CHAINED;
    request.index_type = OpacityMicromapIndexType::UINT32;
    request.vertex_count = 8u;
    request.radius_count = 8u;
    request.index_count = 8u;
    request.vertex_stride = 12u;
    request.radius_stride = 4u;
    request.index_stride = 4u;
    request.vertex_format_size = 12u;
    request.radius_format_size = 4u;
    return request;
}

// A DGF1 blob of two blocks that exactly fills its triangle budget.
[[nodiscard]] DgfRequest capable_dgf_request() noexcept {
    DgfRequest request;
    request.extension_enabled = true;
    request.beta_extensions_enabled = true;
    request.compressed_data_offered = true;
    request.public_api_request = true;
    request.compressed_data_address = 4u * dgf1_block_size;
    request.data_size = 2u * dgf1_block_size;
    request.triangle_count = 128u;
    request.vertex_count = 128u;
    return request;
}

[[nodiscard]] PtlasRequest capable_ptlas_request() noexcept {
    PtlasRequest request;
    request.extension_enabled = true;
    request.public_api_request = true;
    request.partition_count = 4u;
    request.max_instance_per_partition_count = 64u;
    request.max_instance_in_global_partition_count = 4u;
    request.instance_count = 100u;
    request.max_partition_count = 8u;
    return request;
}

}// namespace test_vk_ext_rt_detail

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using namespace test_vk_ext_rt_detail;

    // -----------------------------------------------------------------------
    // Opacity micromap: build plan
    // -----------------------------------------------------------------------
    "vk_rt_micromap_build_extension_off_rejects_offered_request"_test = [] {
        auto request = capable_build_request();
        request.extension_enabled = false;
        auto plan = plan_opacity_micromap_build(request);
        expect(same(plan.status, OpacityMicromapBuildStatus::EXTENSION_DISABLED));
        expect(!plan.build);
        expect(same(plan.mode, MicromapBuildMode::NONE));
        // Nothing may be sized: no vkGetMicromapBuildSizesEXT is legal here.
        expect(plan.data_bytes == 0_u);
        expect(plan.data_bytes_padded == 0_u);
        expect(plan.triangle_array_bytes == 0_u);
        expect(plan.triangle_array_stride == 0_u);
        expect(plan.usage_count == 0_u);
    };

    "vk_rt_micromap_build_without_public_api_request_keeps_default_build"_test = [] {
        auto request = capable_build_request();
        request.public_api_request = false;
        auto plan = plan_opacity_micromap_build(request);
        expect(same(plan.status, OpacityMicromapBuildStatus::NO_PUBLIC_API_REQUEST));
        expect(!plan.build);
        expect(plan.data_bytes == 0_u);
    };

    "vk_rt_micromap_build_khr_acceleration_structure_model_is_unsupported"_test = [] {
        auto request = capable_build_request();
        request.object_model = MicromapObjectModel::KHR_ACCELERATION_STRUCTURE;
        auto plan = plan_opacity_micromap_build(request);
        expect(same(plan.status, OpacityMicromapBuildStatus::OBJECT_MODEL_UNSUPPORTED));
        expect(!plan.build);
        expect(same(plan.mode, MicromapBuildMode::NONE));
        expect(same(plan.object_model, MicromapObjectModel::KHR_ACCELERATION_STRUCTURE))
            << "the negotiated model is reported so the caller can log it";
    };

    "vk_rt_micromap_build_device_mode_sizes"_test = [] {
        auto request = capable_build_request();
        request.host_build_preferred = false;
        request.triangle_count = 10u;
        request.subdivision_level = 0u;
        auto plan = plan_opacity_micromap_build(request);
        expect(same(plan.status, OpacityMicromapBuildStatus::BUILD));
        expect(plan.build);
        expect(same(plan.mode, MicromapBuildMode::DEVICE));
        // Level 0 has a single micro triangle, i.e. one bit -> one byte.
        expect(plan.triangle_data_bytes == 1_u);
        expect(plan.data_bytes == 10_u);
        expect(plan.data_bytes_padded == 256_u)
            << "a device build needs 256-byte aligned build inputs";
        expect(plan.triangle_array_stride == opacity_micromap_triangle_stride);
        expect(plan.triangle_array_stride == 8_u);
        expect(plan.triangle_array_bytes == 80_u);
        expect(plan.usage_count == 1_u);
        expect(same(plan.usage_format, OpacityMicromapFormat::TWO_STATE));
        expect(plan.usage_subdivision_level == 0_u);
        expect(plan.storage_offset_alignment == 256_u);
        expect(plan.scratch_offset_alignment == 256_u);
    };

    "vk_rt_micromap_build_mode_selection_respects_host_commands"_test = [] {
        {
            auto request = capable_build_request();
            request.host_build_preferred = true;
            request.host_commands_enabled = true;
            auto plan = plan_opacity_micromap_build(request);
            expect(same(plan.mode, MicromapBuildMode::HOST))
                << "vkBuildMicromapsEXT needs the micromapHostCommands feature";
            expect(plan.build);
        }
        {
            auto request = capable_build_request();
            request.host_build_preferred = true;
            request.host_commands_enabled = false;
            auto plan = plan_opacity_micromap_build(request);
            expect(same(plan.mode, MicromapBuildMode::DEVICE))
                << "the device build produces the identical micromap";
            expect(plan.build);
        }
        {
            auto request = capable_build_request();
            request.host_build_preferred = true;
            request.host_commands_enabled = false;
            request.device_address_supported = false;
            auto plan = plan_opacity_micromap_build(request);
            expect(same(plan.status, OpacityMicromapBuildStatus::NO_HOST_COMMANDS));
            expect(!plan.build);
        }
        {
            auto request = capable_build_request();
            request.host_build_preferred = true;
            request.host_commands_enabled = true;
            request.storage_host_visible = false;
            auto plan = plan_opacity_micromap_build(request);
            expect(same(plan.status, OpacityMicromapBuildStatus::NO_HOST_VISIBLE_STORAGE));
            expect(!plan.build);
        }
        {
            auto request = capable_build_request();
            request.host_build_preferred = false;
            request.device_address_supported = false;
            auto plan = plan_opacity_micromap_build(request);
            expect(same(plan.status, OpacityMicromapBuildStatus::DEVICE_ADDRESS_UNSUPPORTED));
            expect(!plan.build);
        }
    };

    "vk_rt_micromap_build_rejects_illegal_counts_and_levels"_test = [] {
        {
            auto request = capable_build_request();
            request.triangle_count = 0u;
            expect(same(plan_opacity_micromap_build(request).status,
                        OpacityMicromapBuildStatus::TRIANGLE_COUNT_ZERO));
        }
        {
            auto request = capable_build_request();
            request.format = OpacityMicromapFormat::TWO_STATE;
            request.subdivision_level = 7u;
            request.max_two_state_subdivision_level = 6u;
            expect(same(plan_opacity_micromap_build(request).status,
                        OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_TOO_HIGH));
            // The 4-state limit is a separate property bit.
            request.format = OpacityMicromapFormat::FOUR_STATE;
            request.subdivision_level = 4u;
            request.max_four_state_subdivision_level = 4u;
            expect(plan_opacity_micromap_build(request).build)
                << "the 4-state limit is queried, not the 2-state one";
            request.subdivision_level = 5u;
            expect(same(plan_opacity_micromap_build(request).status,
                        OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_TOO_HIGH));
        }
        {
            auto request = capable_build_request();
            request.subdivision_level = 16u;
            request.max_two_state_subdivision_level = 32u;
            expect(same(plan_opacity_micromap_build(request).status,
                        OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_UNREPRESENTABLE));
        }
        {
            // A zero limit means "not queried", so level 0 must still be legal.
            auto request = capable_build_request();
            request.max_two_state_subdivision_level = 0u;
            request.subdivision_level = 0u;
            expect(plan_opacity_micromap_build(request).build);
            request.subdivision_level = 1u;
            expect(plan_opacity_micromap_build(request).build)
                << "an unqueried limit must not reject legal levels";
        }
    };

    "vk_rt_micromap_build_data_size_heuristics"_test = [] {
        // 4^0 = 1 micro triangle.
        expect(opacity_micromap_micro_triangle_count(0u) == 1_u);
        // 4^1 = 4, 4^2 = 16, 4^3 = 64.
        expect(opacity_micromap_micro_triangle_count(1u) == 4_u);
        expect(opacity_micromap_micro_triangle_count(2u) == 16_u);
        expect(opacity_micromap_micro_triangle_count(3u) == 64_u);
        // Beyond the representable level the helper saturates instead of
        // shifting into undefined behaviour.
        expect(opacity_micromap_micro_triangle_count(200u) ==
               opacity_micromap_micro_triangle_count(opacity_micromap_max_representable_level));
        // 2-state: 1 bit per micro triangle, rounded up to whole bytes.
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::TWO_STATE, 0u) == 1_u);
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::TWO_STATE, 1u) == 1_u);
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::TWO_STATE, 2u) == 2_u);
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::TWO_STATE, 3u) == 8_u);
        // 4-state: 2 bits per micro triangle.
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::FOUR_STATE, 0u) == 1_u);
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::FOUR_STATE, 1u) == 1_u);
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::FOUR_STATE, 2u) == 4_u);
        expect(opacity_micromap_triangle_data_bytes(OpacityMicromapFormat::FOUR_STATE, 3u) == 16_u);
        // Total = count * per-triangle, and the padding is the 256 boundary.
        auto request = capable_build_request();
        request.triangle_count = 3u;
        request.format = OpacityMicromapFormat::FOUR_STATE;
        request.subdivision_level = 2u;
        auto plan = plan_opacity_micromap_build(request);
        expect(plan.triangle_data_bytes == 4_u);
        expect(plan.data_bytes == 12_u);
        expect(plan.data_bytes_padded == 256_u);
        expect(plan.triangle_array_bytes == 24_u);
        // A 256-byte multiple needs no padding.
        expect(round_up_to_multiple(512u, 256u) == 512_u);
        expect(round_up_to_multiple(513u, 256u) == 768_u);
        expect(round_up_to_multiple(7u, 0u) == 7_u);
    };

    // -----------------------------------------------------------------------
    // Opacity micromap: attach plan
    // -----------------------------------------------------------------------
    "vk_rt_micromap_attach_extension_off_rejects_offered_struct"_test = [] {
        auto request = capable_attach_request();
        request.extension_enabled = false;
        auto plan = plan_opacity_micromap_attach(request);
        expect(same(plan.status, OpacityMicromapAttachStatus::EXTENSION_DISABLED));
        expect(!plan.chain) << "VkAccelerationStructureTrianglesOpacityMicromapEXT is never chained";
        expect(!plan.clear_geometry_opaque_flag);
        expect(plan.usage_count == 0_u);
        expect(plan.index_stride == 0_u);
        expect(plan.base_triangle == 0_u);
    };

    "vk_rt_micromap_attach_default_build_never_chains"_test = [] {
        auto request = capable_attach_request();
        request.public_api_request = false;
        expect(same(plan_opacity_micromap_attach(request).status,
                    OpacityMicromapAttachStatus::NO_PUBLIC_API_REQUEST));
        request.public_api_request = true;
        request.micromap_available = false;
        expect(same(plan_opacity_micromap_attach(request).status,
                    OpacityMicromapAttachStatus::NO_MICROMAP))
            << "an index-typed attach without a built micromap";
        // VUID-...-micromap-11579: INDEX_TYPE_NONE needs a valid handle, and that
        // has its own answer.
        request.index_type = OpacityMicromapIndexType::NONE;
        request.index_stride = 0u;
        expect(same(plan_opacity_micromap_attach(request).status,
                    OpacityMicromapAttachStatus::INDEX_NONE_REQUIRES_MICROMAP));
    };

    "vk_rt_micromap_attach_legal_case_carries_the_struct_fields"_test = [] {
        auto request = capable_attach_request();
        request.ext_opacity_micromap_enabled = false;
        request.ray_query_execution_mode_emitted = true;
        request.index_type = OpacityMicromapIndexType::UINT16;
        request.index_stride = 6u;// a multiple of 2, as the VU requires
        request.base_triangle = 4u;
        request.usage_count = 1u;
        auto plan = plan_opacity_micromap_attach(request);
        expect(same(plan.status, OpacityMicromapAttachStatus::ATTACH));
        expect(plan.chain);
        expect(plan.index_type == static_cast<uint8_t>(OpacityMicromapIndexType::UINT16));
        expect(plan.index_stride == 6_u);
        expect(plan.base_triangle == 4_u);
        expect(plan.usage_count == 1_u);
        expect(plan.needs_ray_query_execution_mode)
            << "KHR-only traversal needs OpacityMicromapIdKHR";
        expect(plan.clear_geometry_opaque_flag)
            << "the micromap decides opacity instead of VK_GEOMETRY_OPAQUE_BIT_KHR";
        // EXT_opacity_micromap lifts the execution-mode requirement.
        request.ext_opacity_micromap_enabled = true;
        request.ray_query_execution_mode_emitted = false;
        plan = plan_opacity_micromap_attach(request);
        expect(same(plan.status, OpacityMicromapAttachStatus::ATTACH));
        expect(!plan.needs_ray_query_execution_mode);
    };

    "vk_rt_micromap_attach_rejects_illegal_geometry_and_index"_test = [] {
        {
            auto request = capable_attach_request();
            request.triangle_geometry = false;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::GEOMETRY_NOT_TRIANGLES));
        }
        {
            auto request = capable_attach_request();
            request.motion_triangles = true;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::GEOMETRY_MOTION));
        }
        {
            auto request = capable_attach_request();
            request.index_type = OpacityMicromapIndexType::UINT8;
            request.index_stride = 1u;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::INDEX_TYPE_UNSUPPORTED));
            request.index_type_uint8_supported = true;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::ATTACH));
        }
        {
            auto request = capable_attach_request();
            request.index_type = OpacityMicromapIndexType::UINT32;
            request.index_stride = 6u;// not a multiple of 4
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::INDEX_STRIDE_MISALIGNED));
            request.index_stride = 0u;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::INDEX_STRIDE_MISALIGNED));
            // INDEX_TYPE_NONE ignores the stride entirely.
            request.index_type = OpacityMicromapIndexType::NONE;
            auto plan = plan_opacity_micromap_attach(request);
            expect(same(plan.status, OpacityMicromapAttachStatus::ATTACH));
            expect(plan.index_stride == 0_u);
            expect(plan.index_type == static_cast<uint8_t>(OpacityMicromapIndexType::NONE));
        }
    };

    "vk_rt_micromap_attach_rejects_traversal_that_ignores_micromaps"_test = [] {
        {
            auto request = capable_attach_request();
            request.opaque_ray_flags = true;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::RAY_FLAGS_FORCE_OPAQUE))
                << "OpaqueKHR makes the micromap a no-op, so it must not be chained";
        }
        {
            auto request = capable_attach_request();
            request.instance_disables_micromaps = true;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::INSTANCE_DISABLES_MICROMAPS));
        }
        {
            auto request = capable_attach_request();
            request.ray_query_execution_mode_emitted = false;
            expect(same(plan_opacity_micromap_attach(request).status,
                        OpacityMicromapAttachStatus::RAY_QUERY_EXECUTION_MODE_MISSING));
        }
    };

    // -----------------------------------------------------------------------
    // Geometry flags
    // -----------------------------------------------------------------------
    "vk_rt_geometry_flags_preserve_the_current_build_exactly"_test = [] {
        // Today's mesh build: VK_GEOMETRY_OPAQUE_BIT_KHR.
        auto mesh = plan_geometry_flags({.triangle_geometry = true,
                                        .opaque_requested = true,
                                        .micromap_chained = false});
        expect(mesh.flags == geometry_flag_opaque);
        expect(!mesh.micromap_overrides_opaque);
        // Today's procedural build: flags = 0.
        auto aabb = plan_geometry_flags({.triangle_geometry = false,
                                        .opaque_requested = false,
                                        .micromap_chained = false});
        expect(aabb.flags == 0_u);
        // An AABB geometry never becomes opaque, whatever it asks for.
        auto aabb_opaque = plan_geometry_flags({.triangle_geometry = false,
                                               .opaque_requested = true,
                                               .micromap_chained = false});
        expect(aabb_opaque.flags == 0_u);
    };

    "vk_rt_geometry_flags_drop_opaque_only_for_a_legal_micromap"_test = [] {
        auto with_micromap = plan_geometry_flags({.triangle_geometry = true,
                                                 .opaque_requested = true,
                                                 .micromap_chained = true});
        expect(with_micromap.flags == 0_u);
        expect(with_micromap.micromap_overrides_opaque);
        // The "extension off but a micromap was offered" case: the attach plan
        // refuses the struct, so `micromap_chained` stays false and the flags are
        // bit-identical to the current build.
        auto request = capable_attach_request();
        request.extension_enabled = false;
        auto attach = plan_opacity_micromap_attach(request);
        auto flags = plan_geometry_flags({.triangle_geometry = true,
                                         .opaque_requested = true,
                                         .micromap_chained = attach.chain});
        expect(flags.flags == geometry_flag_opaque);
        expect(!flags.micromap_overrides_opaque);
    };

    // -----------------------------------------------------------------------
    // SER (VK_EXT_ray_tracing_invocation_reorder)
    // -----------------------------------------------------------------------
    "vk_rt_ser_extension_off_never_declares_the_capability"_test = [] {
        SerRequest request;
        request.extension_enabled = false;
        request.traversal = SerTraversal::RAY_TRACING_PIPELINE;
        request.shader_uses_hit_objects = true;
        request.reorder_mode_reorder = true;
        auto plan = plan_ray_tracing_invocation_reorder(request);
        expect(same(plan.status, SerStatus::EXTENSION_DISABLED));
        expect(!plan.enable_reorder);
        expect(!plan.declare_shader_capability)
            << "SPV_EXT_shader_invocation_reorder needs the enabled feature bit";
        expect(!plan.reorder_hint_active);
    };

    "vk_rt_ser_ray_query_traversal_is_not_expressible"_test = [] {
        // The backend's user-shader traversal: OpRayQueryProceedKHR in compute.
        SerRequest request;
        request.extension_enabled = true;
        request.reorder_mode_reorder = true;
        request.traversal = SerTraversal::RAY_QUERY;
        // Even a shader that somehow used hit objects cannot reorder a ray query.
        request.shader_uses_hit_objects = true;
        auto plan = plan_ray_tracing_invocation_reorder(request);
        expect(same(plan.status, SerStatus::NOT_EXPRESSIBLE_FOR_RAY_QUERY));
        expect(!plan.enable_reorder);
        expect(!plan.declare_shader_capability);
        expect(plan.reorder_hint_active)
            << "the device hint is still reported even though nothing uses it";
    };

    "vk_rt_ser_pipeline_traversal_needs_hit_object_shaders"_test = [] {
        SerRequest request;
        request.extension_enabled = true;
        request.traversal = SerTraversal::RAY_TRACING_PIPELINE;
        request.shader_uses_hit_objects = false;
        auto plan = plan_ray_tracing_invocation_reorder(request);
        expect(same(plan.status, SerStatus::PIPELINE_WITHOUT_HIT_OBJECT_SHADER));
        expect(!plan.enable_reorder);
        request.shader_uses_hit_objects = true;
        request.reorder_mode_reorder = true;
        plan = plan_ray_tracing_invocation_reorder(request);
        expect(same(plan.status, SerStatus::REORDER_ENABLED));
        expect(plan.enable_reorder);
        expect(plan.declare_shader_capability);
        expect(plan.reorder_hint_active);
        // A device that exposes the extension without the reorder hint is legal.
        request.reorder_mode_reorder = false;
        plan = plan_ray_tracing_invocation_reorder(request);
        expect(same(plan.status, SerStatus::REORDER_ENABLED));
        expect(!plan.reorder_hint_active);
    };

    "vk_rt_ser_no_traversal_is_a_distinct_answer"_test = [] {
        SerRequest request;
        request.extension_enabled = true;
        request.traversal = SerTraversal::NONE;
        auto plan = plan_ray_tracing_invocation_reorder(request);
        expect(same(plan.status, SerStatus::NO_TRAVERSAL));
        expect(!plan.declare_shader_capability);
    };

    // -----------------------------------------------------------------------
    // Linear swept spheres
    // -----------------------------------------------------------------------
    "vk_rt_lss_extension_off_rejects_offered_geometry"_test = [] {
        auto request = capable_lss_request();
        request.extension_enabled = false;
        auto plan = plan_linear_swept_spheres(request);
        expect(same(plan.status, LssStatus::EXTENSION_DISABLED));
        expect(!plan.usable);
        expect(plan.primitive_count == 0_u);
    };

    "vk_rt_lss_without_a_curve_resource_is_unreachable"_test = [] {
        auto request = capable_lss_request();
        request.curve_resource_available = false;
        auto plan = plan_linear_swept_spheres(request);
        expect(same(plan.status, LssStatus::NO_CURVE_RESOURCE))
            << "DeviceInterface::create_curve is not implemented on Vulkan";
        expect(!plan.usable);
    };

    "vk_rt_lss_primitive_count_follows_the_indexing_mode"_test = [] {
        {
            // No index buffer: pairs of vertices, the trailing one is dropped.
            auto request = capable_lss_request();
            request.index_buffer_present = false;
            request.index_count = 0u;
            request.vertex_count = 7u;
            request.radius_count = 7u;
            auto plan = plan_linear_swept_spheres(request);
            expect(same(plan.status, LssStatus::USABLE));
            expect(plan.usable);
            expect(plan.primitive_count == 3_u);
            expect(plan.index_type == static_cast<uint8_t>(OpacityMicromapIndexType::NONE));
            expect(plan.index_stride == 0_u);
        }
        {
            // LIST: consecutive pairs of indices.
            auto request = capable_lss_request();
            request.indexing_mode = LssIndexingMode::LIST;
            request.index_count = 8u;
            auto plan = plan_linear_swept_spheres(request);
            expect(plan.primitive_count == 4_u);
            expect(plan.index_type == static_cast<uint8_t>(OpacityMicromapIndexType::UINT32));
            expect(plan.index_stride == 4_u);
            expect(same(plan.indexing_mode, LssIndexingMode::LIST));
            expect(same(plan.end_caps_mode, LssEndCapsMode::CHAINED));
        }
        {
            // SUCCESSIVE: one primitive per index.
            auto request = capable_lss_request();
            request.indexing_mode = LssIndexingMode::SUCCESSIVE;
            request.index_count = 5u;
            auto plan = plan_linear_swept_spheres(request);
            expect(plan.primitive_count == 5_u);
        }
    };

    "vk_rt_lss_rejects_illegal_strides_and_counts"_test = [] {
        {
            auto request = capable_lss_request();
            request.vertex_format_supported = false;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::VERTEX_FORMAT_UNSUPPORTED));
        }
        {
            auto request = capable_lss_request();
            request.radius_format_supported = false;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::RADIUS_FORMAT_UNSUPPORTED));
        }
        {
            auto request = capable_lss_request();
            request.vertex_stride = 10u;// not a multiple of the 12-byte format
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::VERTEX_STRIDE_MISALIGNED));
            request.vertex_stride = 24u;
            expect(plan_linear_swept_spheres(request).usable);
        }
        {
            auto request = capable_lss_request();
            request.radius_stride = 6u;// not a multiple of 4
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::RADIUS_STRIDE_MISALIGNED));
        }
        {
            auto request = capable_lss_request();
            request.radius_count = 7u;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::RADIUS_COUNT_MISMATCH));
        }
        {
            auto request = capable_lss_request();
            request.indexing_mode = LssIndexingMode::SUCCESSIVE;
            request.index_buffer_present = false;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::SUCCESSIVE_WITHOUT_INDEX));
        }
        {
            auto request = capable_lss_request();
            request.index_type = OpacityMicromapIndexType::NONE;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::INDEX_TYPE_UNSUPPORTED));
            request.index_type = OpacityMicromapIndexType::UINT8;
            request.index_stride = 1u;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::INDEX_TYPE_UNSUPPORTED));
            request.index_type_uint8_supported = true;
            expect(plan_linear_swept_spheres(request).usable);
        }
        {
            auto request = capable_lss_request();
            request.index_stride = 6u;// not a multiple of 4
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::INDEX_STRIDE_MISALIGNED));
        }
        {
            auto request = capable_lss_request();
            request.index_buffer_present = false;
            request.vertex_count = 1u;
            request.radius_count = 1u;
            expect(same(plan_linear_swept_spheres(request).status,
                        LssStatus::NO_PRIMITIVES));
        }
    };

    // -----------------------------------------------------------------------
    // Dense geometry format
    // -----------------------------------------------------------------------
    "vk_rt_dgf_extension_off_rejects_offered_blob"_test = [] {
        auto request = capable_dgf_request();
        request.extension_enabled = false;
        auto plan = plan_dense_geometry_format(request);
        expect(same(plan.status, DgfStatus::EXTENSION_DISABLED));
        expect(!plan.usable);
        expect(plan.block_count == 0_u);
        expect(plan.max_triangles == 0_u);
    };

    "vk_rt_dgf_requires_the_beta_header_and_a_public_request"_test = [] {
        auto request = capable_dgf_request();
        request.beta_extensions_enabled = false;
        expect(same(plan_dense_geometry_format(request).status,
                    DgfStatus::BETA_EXTENSIONS_DISABLED));
        request.beta_extensions_enabled = true;
        request.public_api_request = false;
        expect(same(plan_dense_geometry_format(request).status,
                    DgfStatus::NO_PUBLIC_API_REQUEST));
        request.public_api_request = true;
        request.compressed_data_offered = false;
        expect(same(plan_dense_geometry_format(request).status,
                    DgfStatus::NO_PUBLIC_API_REQUEST));
    };

    "vk_rt_dgf_validates_block_alignment_and_counts"_test = [] {
        {
            auto request = capable_dgf_request();
            request.data_size = 0u;
            expect(same(plan_dense_geometry_format(request).status, DgfStatus::DATA_SIZE_ZERO));
            request.data_size = 192u;// 1.5 blocks
            expect(same(plan_dense_geometry_format(request).status, DgfStatus::DATA_SIZE_MISALIGNED));
            request.data_size = 2u * dgf1_block_size;
            request.compressed_data_address = 4u * dgf1_block_size + 8u;
            expect(same(plan_dense_geometry_format(request).status, DgfStatus::DATA_ADDRESS_MISALIGNED));
            request.compressed_data_address = 4u * dgf1_block_size;
            request.triangle_count = 0u;
            expect(same(plan_dense_geometry_format(request).status, DgfStatus::TRIANGLE_COUNT_ZERO));
        }
        {
            auto request = capable_dgf_request();
            auto plan = plan_dense_geometry_format(request);
            expect(same(plan.status, DgfStatus::USABLE));
            expect(plan.usable);
            expect(plan.block_count == 2_u);
            expect(plan.max_triangles == 128_u);
            expect(plan.max_vertices == 128_u);
            // One triangle over the block budget is a malformed blob.
            request.triangle_count = 129u;
            expect(same(plan_dense_geometry_format(request).status,
                        DgfStatus::TRIANGLE_COUNT_EXCEEDS_BLOCKS));
            request.triangle_count = 1u;
            request.vertex_count = 129u;
            expect(same(plan_dense_geometry_format(request).status,
                        DgfStatus::TRIANGLE_COUNT_EXCEEDS_BLOCKS));
        }
    };

    // -----------------------------------------------------------------------
    // Partitioned acceleration structure
    // -----------------------------------------------------------------------
    "vk_rt_ptlas_extension_off_rejects_offered_build"_test = [] {
        auto request = capable_ptlas_request();
        request.extension_enabled = false;
        auto plan = plan_partitioned_acceleration_structure(request);
        expect(same(plan.status, PtlasStatus::EXTENSION_DISABLED));
        expect(!plan.usable);
        expect(plan.partition_count == 0_u);
    };

    "vk_rt_ptlas_needs_a_partition_aware_public_api"_test = [] {
        auto request = capable_ptlas_request();
        request.public_api_request = false;
        expect(same(plan_partitioned_acceleration_structure(request).status,
                    PtlasStatus::NO_PUBLIC_API_REQUEST))
            << "Tlas writes plain VkAccelerationStructureInstanceKHR records";
    };

    "vk_rt_ptlas_enforces_the_partition_count_limit"_test = [] {
        {
            auto request = capable_ptlas_request();
            request.partition_count = 0u;
            expect(same(plan_partitioned_acceleration_structure(request).status,
                        PtlasStatus::PARTITION_COUNT_ZERO));
        }
        {
            auto request = capable_ptlas_request();
            request.instance_count = 0u;
            expect(same(plan_partitioned_acceleration_structure(request).status,
                        PtlasStatus::INSTANCE_COUNT_ZERO));
        }
        {
            auto request = capable_ptlas_request();
            request.max_instance_per_partition_count = 0u;
            expect(same(plan_partitioned_acceleration_structure(request).status,
                        PtlasStatus::PER_PARTITION_COUNT_ZERO));
        }
        {
            auto request = capable_ptlas_request();
            // 4 partitions + 5 global partitions > maxPartitionCount (8).
            request.max_instance_in_global_partition_count = 5u;
            expect(same(plan_partitioned_acceleration_structure(request).status,
                        PtlasStatus::PARTITION_COUNT_EXCEEDED));
            // Exactly at the limit is legal.
            request.max_instance_in_global_partition_count = 4u;
            auto plan = plan_partitioned_acceleration_structure(request);
            expect(same(plan.status, PtlasStatus::USABLE));
            expect(plan.usable);
            expect(plan.partition_count == 4_u);
            expect(plan.max_instance_per_partition_count == 64_u);
            expect(plan.max_instance_in_global_partition_count == 4_u);
            expect(plan.scratch_alignment == 256_u);
            // An unqueried maxPartitionCount (0) cannot reject anything.
            request.max_partition_count = 0u;
            expect(plan_partitioned_acceleration_structure(request).usable);
            // Partition translation is carried through.
            request.partition_translation = true;
            expect(plan_partitioned_acceleration_structure(request).partition_translation);
        }
    };

    // -----------------------------------------------------------------------
    // Status names
    // -----------------------------------------------------------------------
    "vk_rt_status_names_are_never_unknown"_test = [] {
        expect(known_name(ser_status_name(SerStatus::REORDER_ENABLED)));
        expect(known_name(ser_status_name(SerStatus::EXTENSION_DISABLED)));
        expect(known_name(ser_status_name(SerStatus::NOT_EXPRESSIBLE_FOR_RAY_QUERY)));
        expect(known_name(ser_status_name(SerStatus::NO_TRAVERSAL)));
        expect(known_name(ser_status_name(SerStatus::PIPELINE_WITHOUT_HIT_OBJECT_SHADER)));

        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::BUILD)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::EXTENSION_DISABLED)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::NO_PUBLIC_API_REQUEST)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::OBJECT_MODEL_UNSUPPORTED)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::DEVICE_ADDRESS_UNSUPPORTED)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::TRIANGLE_COUNT_ZERO)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_TOO_HIGH)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_UNREPRESENTABLE)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::NO_HOST_COMMANDS)));
        expect(known_name(opacity_micromap_build_status_name(OpacityMicromapBuildStatus::NO_HOST_VISIBLE_STORAGE)));

        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::ATTACH)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::EXTENSION_DISABLED)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::NO_PUBLIC_API_REQUEST)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::NO_MICROMAP)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::GEOMETRY_NOT_TRIANGLES)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::GEOMETRY_MOTION)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::INDEX_TYPE_UNSUPPORTED)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::INDEX_STRIDE_MISALIGNED)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::INDEX_NONE_REQUIRES_MICROMAP)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::RAY_FLAGS_FORCE_OPAQUE)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::INSTANCE_DISABLES_MICROMAPS)));
        expect(known_name(opacity_micromap_attach_status_name(OpacityMicromapAttachStatus::RAY_QUERY_EXECUTION_MODE_MISSING)));

        expect(known_name(lss_status_name(LssStatus::USABLE)));
        expect(known_name(lss_status_name(LssStatus::EXTENSION_DISABLED)));
        expect(known_name(lss_status_name(LssStatus::NO_CURVE_RESOURCE)));
        expect(known_name(lss_status_name(LssStatus::VERTEX_FORMAT_UNSUPPORTED)));
        expect(known_name(lss_status_name(LssStatus::RADIUS_FORMAT_UNSUPPORTED)));
        expect(known_name(lss_status_name(LssStatus::VERTEX_STRIDE_MISALIGNED)));
        expect(known_name(lss_status_name(LssStatus::RADIUS_STRIDE_MISALIGNED)));
        expect(known_name(lss_status_name(LssStatus::RADIUS_COUNT_MISMATCH)));
        expect(known_name(lss_status_name(LssStatus::SUCCESSIVE_WITHOUT_INDEX)));
        expect(known_name(lss_status_name(LssStatus::INDEX_TYPE_UNSUPPORTED)));
        expect(known_name(lss_status_name(LssStatus::INDEX_STRIDE_MISALIGNED)));
        expect(known_name(lss_status_name(LssStatus::NO_PRIMITIVES)));

        expect(known_name(dgf_status_name(DgfStatus::USABLE)));
        expect(known_name(dgf_status_name(DgfStatus::EXTENSION_DISABLED)));
        expect(known_name(dgf_status_name(DgfStatus::BETA_EXTENSIONS_DISABLED)));
        expect(known_name(dgf_status_name(DgfStatus::NO_PUBLIC_API_REQUEST)));
        expect(known_name(dgf_status_name(DgfStatus::DATA_ADDRESS_MISALIGNED)));
        expect(known_name(dgf_status_name(DgfStatus::DATA_SIZE_MISALIGNED)));
        expect(known_name(dgf_status_name(DgfStatus::DATA_SIZE_ZERO)));
        expect(known_name(dgf_status_name(DgfStatus::TRIANGLE_COUNT_ZERO)));
        expect(known_name(dgf_status_name(DgfStatus::TRIANGLE_COUNT_EXCEEDS_BLOCKS)));

        expect(known_name(ptlas_status_name(PtlasStatus::USABLE)));
        expect(known_name(ptlas_status_name(PtlasStatus::EXTENSION_DISABLED)));
        expect(known_name(ptlas_status_name(PtlasStatus::NO_PUBLIC_API_REQUEST)));
        expect(known_name(ptlas_status_name(PtlasStatus::PARTITION_COUNT_EXCEEDED)));
        expect(known_name(ptlas_status_name(PtlasStatus::PARTITION_COUNT_ZERO)));
        expect(known_name(ptlas_status_name(PtlasStatus::INSTANCE_COUNT_ZERO)));
        expect(known_name(ptlas_status_name(PtlasStatus::PER_PARTITION_COUNT_ZERO)));
    };
}
