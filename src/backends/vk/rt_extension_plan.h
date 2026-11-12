#pragma once
#include <cstdint>
#include <cstddef>

namespace lc::vk::detail {

// Host-side planning for the ray tracing optional extensions of the Vulkan
// backend:
//
//   * VK_EXT_ray_tracing_invocation_reorder (cross-vendor SER)
//   * VK_KHR_opacity_micromap               (the EXT alias is accepted)
//   * VK_NV_ray_tracing_linear_swept_spheres
//   * VK_AMDX_dense_geometry_format         (provisional, VK_ENABLE_BETA_EXTENSIONS)
//   * VK_NV_partitioned_acceleration_structure
//
// Nothing here touches a Vulkan entry point or a `Device`: every helper answers
// "is this fast path legal *and* actually usable for this build, or must the
// existing code run instead" from booleans the caller already negotiated
// (`Device::optional_extension_enabled`). That keeps the decision testable on a
// machine without the extension - which is the only evidence available for the
// four extensions this backend cannot reach through its public API (see the
// per-helper comments) - and it guarantees that a disabled or unsupported
// extension can never leak a struct, a flag, or an entry point into the recorded
// command stream.
//
// Everything is header-only (`inline`/`constexpr`) and deliberately free of
// Vulkan and `luisa` includes: the unit test compiles this header directly and
// the backend converts the PODs below into `Vk*` structures at the one place
// that records them (`blas.cpp`, `opacity_micromap.cpp`).
//
// The invariant every helper repeats: a request that is *offered* while its
// extension is off (or unsupported, or on an imported device) is rejected and
// reported as such - it never becomes "chain the struct and hope". Each status
// enum therefore has an explicit `*_DISABLED` value next to the legal one, and
// every plan carries a `bool` that the caller must test before touching Vulkan.

// ---------------------------------------------------------------------------
// VK_EXT_ray_tracing_invocation_reorder (shader execution reordering)
// ---------------------------------------------------------------------------
// What the extension actually adds (Vulkan 1.4.368):
//   * VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT::rayTracingInvocationReorder,
//     which only says "this implementation supports
//     SPV_EXT_shader_invocation_reorder";
//   * VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT
//     (rayTracingInvocationReorderReorderingHint +
//     maxShaderBindingTableRecordIndex);
//   * the VkRayTracingInvocationReorderModeEXT hint enum.
// There is *no* create-info struct, no pipeline flag, and no ray-query surface:
// vk.xml declares the extension as `depends="VK_KHR_ray_tracing_pipeline"` and
// its whole <require> block is the two physical-device structs plus the hint
// enum. The reorder operations themselves are the SPIR-V instructions
// OpReorderThreadWithHintEXT / OpReorderThreadWithHitObjectEXT /
// OpHitObjectTraceRayEXT / OpHitObjectTraceRayMotionEXT /
// OpHitObjectReorderExecuteShaderEXT / OpHitObjectTraceReorderExecuteEXT /
// OpHitObjectTraceMotionReorderExecuteEXT, i.e. they operate on *hit objects*.
// Hit objects only exist inside ray tracing pipeline stages (table "Shader Call
// Instructions" in raytracing.adoc: OpHitObject* may only be reached from
// intersection/any-hit/closest-hit/miss/raygen), and SPV_EXT_shader_invocation
// _reorder requires SPV_KHR_ray_tracing.
//
// Consequence for this backend: user shaders trace with OpRayQueryInitializeKHR
// / OpRayQueryProceedKHR inside compute kernels, and neither the SPIR-V
// extension nor VK_EXT_ray_tracing_invocation_reorder define a "reorder" ray
// flag or any ray-query-facing struct. A hint is therefore *not expressible*
// for a ray query, which is why the extension is negotiated (so the capability
// is visible through VulkanExtensionStatusExt) but stays inactive. The only
// place it could ever be requested is the internal VK_KHR_ray_tracing_pipeline
// path (RayTracingShader) and even there it needs hit-object shader code, which
// no codegen route in this backend emits today.
enum class SerTraversal : uint8_t {
    NONE,
    // OpRayQueryInitializeKHR / OpRayQueryProceedKHR inside a compute kernel:
    // the traversal every user shader uses.
    RAY_QUERY,
    // The internal VK_KHR_ray_tracing_pipeline raygen/miss/closest-hit pipeline
    // (RayTracingShader, used for motion blurred tracing).
    RAY_TRACING_PIPELINE
};

enum class SerStatus : uint8_t {
    // The shader declares ShaderInvocationReorderEXT / builds hit objects and
    // the device enabled the feature. No code path reaches this value today.
    REORDER_ENABLED,
    // Negotiated off, unsupported, or an imported logical device.
    EXTENSION_DISABLED,
    // Legal answer for the ray-query traversal: the extension has no ray-query
    // surface at all, so there is nothing to hint.
    NOT_EXPRESSIBLE_FOR_RAY_QUERY,
    // No traversal was requested (e.g. a shader without any trace).
    NO_TRAVERSAL,
    // Pipeline trace is in use, but no shader of the module uses hit objects, so
    // there is nothing to reorder.
    PIPELINE_WITHOUT_HIT_OBJECT_SHADER
};

[[nodiscard]] inline const char *ser_status_name(SerStatus status) noexcept {
    switch (status) {
        case SerStatus::REORDER_ENABLED: return "reorder-enabled";
        case SerStatus::EXTENSION_DISABLED: return "extension-disabled";
        case SerStatus::NOT_EXPRESSIBLE_FOR_RAY_QUERY: return "not-expressible-for-ray-query";
        case SerStatus::NO_TRAVERSAL: return "no-traversal";
        case SerStatus::PIPELINE_WITHOUT_HIT_OBJECT_SHADER: return "pipeline-without-hit-object-shader";
    }
    return "unknown";
}

struct SerRequest {
    // Device::optional_extension_enabled(RAY_TRACING_INVOCATION_REORDER). The
    // registry only enables the extension after the
    // `rayTracingInvocationReorder` feature bit was probed VK_TRUE, so this one
    // boolean is the complete feature test.
    bool extension_enabled{false};
    // VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT::
    // rayTracingInvocationReorderReorderingHint ==
    // VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT. Informational: the
    // spec lets an implementation expose the extension while never reordering.
    bool reorder_mode_reorder{false};
    SerTraversal traversal{SerTraversal::NONE};
    // Whether any reachable shader of the module declares
    // ShaderInvocationReorderEXT / uses OpHitObject* instructions.
    bool shader_uses_hit_objects{false};
};

struct SerPlan {
    SerStatus status{SerStatus::EXTENSION_DISABLED};
    // Whether the backend may request the extension's state for this traversal.
    // False for every status except REORDER_ENABLED, and the only value a
    // caller is allowed to turn into a pipeline flag, a chained struct, or a
    // SPIR-V capability declaration.
    bool enable_reorder{false};
    // Whether the emitted module may declare
    // OpExtension "SPV_EXT_shader_invocation_reorder" + OpCapability
    // ShaderInvocationReorderEXT.
    bool declare_shader_capability{false};
    // Whether the driver would actually reorder (the hint, not a promise).
    bool reorder_hint_active{false};
};

[[nodiscard]] inline SerPlan plan_ray_tracing_invocation_reorder(SerRequest const &request) noexcept {
    SerPlan plan{};
    if (!request.extension_enabled) {
        // Nothing below may be set: without the extension the feature bit was
        // never enabled at vkCreateDevice, so both the SPIR-V capability and any
        // reorder state would be invalid.
        plan.status = SerStatus::EXTENSION_DISABLED;
        return plan;
    }
    plan.reorder_hint_active = request.reorder_mode_reorder;
    switch (request.traversal) {
        case SerTraversal::RAY_QUERY:
            // The extension adds no ray-query surface: a ray query has no hit
            // object, no reorder flag and no reorder instruction.
            plan.status = SerStatus::NOT_EXPRESSIBLE_FOR_RAY_QUERY;
            break;
        case SerTraversal::RAY_TRACING_PIPELINE:
            if (request.shader_uses_hit_objects) {
                plan.status = SerStatus::REORDER_ENABLED;
                plan.enable_reorder = true;
                plan.declare_shader_capability = true;
            } else {
                plan.status = SerStatus::PIPELINE_WITHOUT_HIT_OBJECT_SHADER;
            }
            break;
        case SerTraversal::NONE:
        default:
            plan.status = SerStatus::NO_TRAVERSAL;
            break;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// VK_KHR_opacity_micromap (VK_EXT_opacity_micromap alias)
// ---------------------------------------------------------------------------
// Two object models are reachable in this spec revision and they do not mix:
//
//   * EXT_MICROMAP_OBJECT - VkMicromapEXT created with vkCreateMicromapEXT from
//     a buffer with VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT, sized by
//     vkGetMicromapBuildSizesEXT, built with vkCmdBuildMicromapsEXT (device
//     build, device addresses) or vkBuildMicromapsEXT (host build, host
//     addresses, additionally requires the `micromapHostCommands` feature which
//     the registry does not enable).
//   * KHR_ACCELERATION_STRUCTURE - micromaps.adoc: "Micromaps created with
//     VK_KHR_opacity_micromap, which are represented by
//     VkAccelerationStructureKHR objects, are built with device build commands"
//     (VK_GEOMETRY_TYPE_MICROMAP_KHR geometry + VkMicromapBuildInfoEXT-less
//     VkAccelerationStructureGeometryMicromapDataKHR), which needs
//     VK_KHR_device_address_commands on top. The backend negotiates that
//     extension for its copy/build commands already; the micromap shape itself
//     is not implemented here.
//
// The machinery in `opacity_micromap.{h,cpp}` implements the first model, which
// is the one whose build inputs (VkAccelerationStructureTrianglesOpacityMicromap
// EXT + VkMicromapBuildInfoEXT) this work package asked for. The plan reports
// the model so a device that only exposes the KHR revision is refused instead of
// fed a VkMicromapEXT it cannot create.
enum class MicromapObjectModel : uint8_t {
    EXT_MICROMAP_OBJECT,
    KHR_ACCELERATION_STRUCTURE
};

enum class MicromapBuildMode : uint8_t {
    NONE,
    // vkCmdBuildMicromapsEXT, device addresses, memcpy'd into mapped storage.
    DEVICE,
    // vkBuildMicromapsEXT, host addresses, requires `micromapHostCommands`.
    HOST
};

// VkOpacityMicromapFormatEXT without the Vulkan include.
enum class OpacityMicromapFormat : uint8_t {
    TWO_STATE = 0,
    FOUR_STATE = 1
};

// VkIndexType subset VkAccelerationStructureTrianglesOpacityMicromapEXT allows
// (VUID-VkAccelerationStructureTrianglesOpacityMicromapKHR-indexType-11570):
// UINT8 (only with VK_KHR/EXT_index_type_uint8), UINT16, UINT32, or NONE.
//
// The swept-sphere planner reuses it because VkAccelerationStructureGeometryLinear
// SweptSpheresDataNV::indexType is the same VkIndexType with the same legal subset
// (VUID-...-indexData-10428: UINT16, UINT32 or NONE).
enum class OpacityMicromapIndexType : uint8_t {
    NONE = 0,
    UINT8 = 1,
    UINT16 = 2,
    UINT32 = 3
};

// One VkMicromapTriangleEXT / VkMicromapTriangleKHR record: a 4-byte data
// offset, a 2-byte subdivision level and a 2-byte format, i.e. exactly 8 bytes
// and the smallest legal triangleArrayStride ("must be >= 8 bytes and a
// multiple of 4").
struct OpacityMicromapTriangleRecord {
    uint32_t data_offset{0u};
    uint16_t subdivision_level{0u};
    uint16_t format{0u};
};
inline constexpr size_t opacity_micromap_triangle_stride = sizeof(OpacityMicromapTriangleRecord);
static_assert(opacity_micromap_triangle_stride == 8u,
              "VkMicromapTriangleEXT is a 4+2+2 byte record");

// VkMicromapCreateInfoEXT::offset "must be a multiple of 256".
inline constexpr uint64_t opacity_micromap_storage_offset_alignment = 256u;
// vkCmdBuildMicromapsEXT: data/triangleArray device addresses "must be a
// multiple of 256".
inline constexpr uint64_t opacity_micromap_build_input_alignment = 256u;
// Geometry-level attach (VkAccelerationStructureGeometryMicromapDataKHR and the
// triangles pNext chain): both addresses must be a multiple of 128.
inline constexpr uint64_t opacity_micromap_attach_input_alignment = 128u;
// 4^level micro triangles; level 15 is 2^30, beyond which the 4-state data of a
// single triangle alone would exceed a 32-bit byte count.
inline constexpr uint8_t opacity_micromap_max_representable_level = 15u;

// Micro triangles produced by subdividing one triangle `level` times: 4^level.
[[nodiscard]] inline constexpr uint64_t opacity_micromap_micro_triangle_count(uint8_t level) noexcept {
    auto level_clamped = level > opacity_micromap_max_representable_level ?
                             opacity_micromap_max_representable_level :
                             level;
    return uint64_t{1u} << (2u * static_cast<uint32_t>(level_clamped));
}

// Bytes of micromap data one triangle consumes, "calculated using its format and
// subdivisionLevel rounded up to the nearest byte" (VUID-...-micromap-11553):
// 1 bit per micro triangle for the 2-state format, 2 bits for the 4-state one.
[[nodiscard]] inline constexpr uint64_t opacity_micromap_triangle_data_bytes(
    OpacityMicromapFormat format, uint8_t level) noexcept {
    auto micro = opacity_micromap_micro_triangle_count(level);
    auto bits = format == OpacityMicromapFormat::FOUR_STATE ? micro * 2u : micro;
    return (bits + 7u) / 8u;
}

[[nodiscard]] inline uint64_t round_up_to_multiple(uint64_t value, uint64_t alignment) noexcept {
    if (alignment == 0u) { return value; }
    auto remainder = value % alignment;
    return remainder == 0u ? value : value + (alignment - remainder);
}

enum class OpacityMicromapBuildStatus : uint8_t {
    // Legal: `mode` names the command that has to be recorded.
    BUILD,
    // Negotiated off, unsupported, or an imported logical device.
    EXTENSION_DISABLED,
    // Nothing asked for a micromap, so the existing BLAS build must run
    // unchanged. This is what the default build always reports: the public
    // LuisaCompute accelerator API (AccelOption / MeshBuildCommand) has no
    // micromap input.
    NO_PUBLIC_API_REQUEST,
    // The KHR revision was negotiated: a micromap is a VkAccelerationStructureKHR
    // built through VK_KHR_device_address_commands, not a VkMicromapEXT.
    OBJECT_MODEL_UNSUPPORTED,
    // A device build needs buffer device addresses; without them the micromap
    // build inputs cannot be expressed at all.
    DEVICE_ADDRESS_UNSUPPORTED,
    TRIANGLE_COUNT_ZERO,
    // Implementation limit (VkPhysicalDeviceOpacityMicromapPropertiesEXT::
    // maxOpacity{2,4}StateSubdivisionLevel). The EXT properties structure has no
    // maxMicromapTriangles member (only the KHR one does), so the triangle count
    // has no queryable upper bound in the object model this backend implements.
    SUBDIVISION_LEVEL_TOO_HIGH,
    // Above 4^15 micro triangles a 32-bit byte count cannot describe one
    // triangle's data.
    SUBDIVISION_LEVEL_UNREPRESENTABLE,
    // A host build was asked for but `micromapHostCommands` is not enabled, so
    // vkBuildMicromapsEXT must not be called.
    NO_HOST_COMMANDS,
    // The micromap storage buffer would not be host-visible, which
    // vkBuildMicromapsEXT requires.
    NO_HOST_VISIBLE_STORAGE
};

[[nodiscard]] inline const char *opacity_micromap_build_status_name(
    OpacityMicromapBuildStatus status) noexcept {
    switch (status) {
        case OpacityMicromapBuildStatus::BUILD: return "build";
        case OpacityMicromapBuildStatus::EXTENSION_DISABLED: return "extension-disabled";
        case OpacityMicromapBuildStatus::NO_PUBLIC_API_REQUEST: return "no-public-api-request";
        case OpacityMicromapBuildStatus::OBJECT_MODEL_UNSUPPORTED: return "object-model-unsupported";
        case OpacityMicromapBuildStatus::DEVICE_ADDRESS_UNSUPPORTED: return "device-address-unsupported";
        case OpacityMicromapBuildStatus::TRIANGLE_COUNT_ZERO: return "triangle-count-zero";
        case OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_TOO_HIGH: return "subdivision-level-too-high";
        case OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_UNREPRESENTABLE: return "subdivision-level-unrepresentable";
        case OpacityMicromapBuildStatus::NO_HOST_COMMANDS: return "no-host-commands";
        case OpacityMicromapBuildStatus::NO_HOST_VISIBLE_STORAGE: return "no-host-visible-storage";
    }
    return "unknown";
}

struct OpacityMicromapBuildRequest {
    // Device::optional_extension_enabled(OPACITY_MICROMAP): the only gate.
    bool extension_enabled{false};
    // Which name the registry resolved (KHR preferred, EXT alias).
    MicromapObjectModel object_model{MicromapObjectModel::EXT_MICROMAP_OBJECT};
    // VkPhysicalDeviceOpacityMicromapFeaturesEXT::micromapHostCommands. The
    // registry enables only `micromap`, so this is false on every device this
    // backend creates today and the host build is unreachable by construction.
    bool host_commands_enabled{false};
    // bufferDeviceAddress, required by the device build.
    bool device_address_supported{false};
    // The storage buffer the backend would allocate is host-visible.
    bool storage_host_visible{true};
    // The caller prefers the host build (its micromap content already lives in
    // host memory). Only honoured when `host_commands_enabled` is set.
    bool host_build_preferred{false};
    // The caller hands over a DGF-style pre-built blob instead of the raw
    // usage/data/triangle arrays. Informational for the caller's upload path.
    bool public_api_request{false};
    uint32_t triangle_count{0};
    OpacityMicromapFormat format{OpacityMicromapFormat::TWO_STATE};
    uint8_t subdivision_level{0};
    // VkPhysicalDeviceOpacityMicromapPropertiesEXT limits (0 = not queried).
    uint8_t max_two_state_subdivision_level{0};
    uint8_t max_four_state_subdivision_level{0};
    // VkPhysicalDeviceAccelerationStructurePropertiesKHR::
    // minAccelerationStructureScratchOffsetAlignment.
    uint32_t scratch_offset_alignment{0};
};

struct OpacityMicromapBuildPlan {
    OpacityMicromapBuildStatus status{OpacityMicromapBuildStatus::EXTENSION_DISABLED};
    MicromapObjectModel object_model{MicromapObjectModel::EXT_MICROMAP_OBJECT};
    MicromapBuildMode mode{MicromapBuildMode::NONE};
    OpacityMicromapFormat format{OpacityMicromapFormat::TWO_STATE};
    uint8_t subdivision_level{0};
    // Canonical aggregated usage entry: one VkMicromapUsageEXT describing every
    // triangle of this micromap, which is legal because "multiple elements with
    // the same format and subdivisionLevel are allowed" and the total count is
    // what the build consumes.
    uint32_t triangle_count{0};
    uint32_t usage_count{0};
    OpacityMicromapFormat usage_format{OpacityMicromapFormat::TWO_STATE};
    uint8_t usage_subdivision_level{0};
    // Sizes the caller has to provide, all in bytes.
    uint64_t triangle_data_bytes{0};       // one triangle's micromap data
    uint64_t data_bytes{0};                // triangle_count * triangle_data_bytes
    uint64_t data_bytes_padded{0};         // data_bytes rounded to 256
    uint64_t triangle_array_stride{0};     // 0 or opacity_micromap_triangle_stride
    uint64_t triangle_array_bytes{0};      // triangle_count * stride
    uint64_t storage_offset_alignment{opacity_micromap_storage_offset_alignment};
    uint64_t scratch_offset_alignment{0};
    // True only for status == BUILD.
    bool build{false};
};

[[nodiscard]] inline OpacityMicromapBuildPlan plan_opacity_micromap_build(
    OpacityMicromapBuildRequest const &request) noexcept {
    OpacityMicromapBuildPlan plan{};
    plan.object_model = request.object_model;
    plan.format = request.format;
    plan.subdivision_level = request.subdivision_level;
    plan.triangle_count = request.triangle_count;
    if (!request.extension_enabled) {
        // No extension -> no vkGetMicromapBuildSizesEXT, no VkMicromapEXT and no
        // VkMicromapBuildInfoEXT for this build. The default path stays.
        plan.status = OpacityMicromapBuildStatus::EXTENSION_DISABLED;
        return plan;
    }
    if (!request.public_api_request) {
        // Enabled but not requested: the public accelerator API cannot describe
        // a micromap, so the BLAS build keeps its current inputs.
        plan.status = OpacityMicromapBuildStatus::NO_PUBLIC_API_REQUEST;
        return plan;
    }
    if (request.object_model != MicromapObjectModel::EXT_MICROMAP_OBJECT) {
        plan.status = OpacityMicromapBuildStatus::OBJECT_MODEL_UNSUPPORTED;
        return plan;
    }
    if (request.triangle_count == 0u) {
        plan.status = OpacityMicromapBuildStatus::TRIANGLE_COUNT_ZERO;
        return plan;
    }
    if (request.subdivision_level > opacity_micromap_max_representable_level) {
        plan.status = OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_UNREPRESENTABLE;
        return plan;
    }
    auto limit = request.format == OpacityMicromapFormat::FOUR_STATE ?
                     request.max_four_state_subdivision_level :
                     request.max_two_state_subdivision_level;
    // A zero limit means "not queried": skip the comparison rather than
    // rejecting every level 0 request.
    if (limit != 0u && request.subdivision_level > limit) {
        plan.status = OpacityMicromapBuildStatus::SUBDIVISION_LEVEL_TOO_HIGH;
        return plan;
    }
    plan.triangle_data_bytes = opacity_micromap_triangle_data_bytes(
        request.format, request.subdivision_level);
    // Both factors are bounded (triangle_count <= 2^32-1 and one triangle's data
    // <= 2^28 bytes at level 15), so the 64-bit product cannot overflow.
    plan.data_bytes = static_cast<uint64_t>(request.triangle_count) *
                      plan.triangle_data_bytes;
    plan.data_bytes_padded = round_up_to_multiple(plan.data_bytes,
                                                 opacity_micromap_build_input_alignment);
    plan.triangle_array_stride = opacity_micromap_triangle_stride;
    plan.triangle_array_bytes = static_cast<uint64_t>(request.triangle_count) *
                                plan.triangle_array_stride;
    plan.scratch_offset_alignment = request.scratch_offset_alignment;
    plan.usage_count = 1u;
    plan.usage_format = request.format;
    plan.usage_subdivision_level = request.subdivision_level;
    // Build mode. A host build is only legal with `micromapHostCommands`; a
    // device build needs device addresses.
    if (request.host_build_preferred) {
        if (!request.host_commands_enabled) {
            // Falling back to the device build would need device addresses; do
            // that when they are available instead of failing, because the
            // device build produces the identical micromap.
            if (!request.device_address_supported) {
                plan.status = OpacityMicromapBuildStatus::NO_HOST_COMMANDS;
                return plan;
            }
            plan.mode = MicromapBuildMode::DEVICE;
        } else if (!request.storage_host_visible) {
            plan.status = OpacityMicromapBuildStatus::NO_HOST_VISIBLE_STORAGE;
            return plan;
        } else {
            plan.mode = MicromapBuildMode::HOST;
        }
    } else {
        if (!request.device_address_supported) {
            plan.status = OpacityMicromapBuildStatus::DEVICE_ADDRESS_UNSUPPORTED;
            return plan;
        }
        plan.mode = MicromapBuildMode::DEVICE;
    }
    plan.status = OpacityMicromapBuildStatus::BUILD;
    plan.build = true;
    return plan;
}

enum class OpacityMicromapAttachStatus : uint8_t {
    // Legal: the caller may chain VkAccelerationStructureTrianglesOpacityMicromap
    // EXT into VkAccelerationStructureGeometryTrianglesDataKHR::pNext.
    ATTACH,
    EXTENSION_DISABLED,
    NO_PUBLIC_API_REQUEST,
    // No built micromap (or the object model does not produce one).
    NO_MICROMAP,
    // AABB/procedural geometry cannot carry a micromap.
    GEOMETRY_NOT_TRIANGLES,
    // VkAccelerationStructureGeometryMotionTrianglesDataNV already occupies the
    // triangles pNext chain; a micromap on a motion geometry is not supported by
    // this backend's build shape.
    GEOMETRY_MOTION,
    // VK_INDEX_TYPE_UINT8 without VK_KHR_index_type_uint8 / VK_EXT_index_type_uint8,
    // or a value outside the four the spec allows.
    INDEX_TYPE_UNSUPPORTED,
    // indexStride must be a non-zero multiple of the index size.
    INDEX_STRIDE_MISALIGNED,
    // VUID-...-micromap-11579: with INDEX_TYPE_NONE the micromap handle must be
    // valid, i.e. NONE is only legal for a *built* micromap.
    INDEX_NONE_REQUIRES_MICROMAP,
    // The emitted ray flags include OpaqueKHR, which "overrides all non-opaque
    // geometries to behave as if they are opaque" (raytraversal.adoc) and would
    // silently make the micromap a no-op. The backend's direct closest-hit ray
    // query emits exactly that flag.
    RAY_FLAGS_FORCE_OPAQUE,
    // VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_BIT_KHR was set on the
    // instancing TLAS entry, so the micromap would be ignored.
    INSTANCE_DISABLES_MICROMAPS,
    // With only VK_KHR_opacity_micromap (no EXT_opacity_micromap) a ray query
    // must declare the OpacityMicromapIdKHR execution mode with its
    // specialization constant VK_TRUE; no codegen route in this backend emits it.
    RAY_QUERY_EXECUTION_MODE_MISSING
};

[[nodiscard]] inline const char *opacity_micromap_attach_status_name(
    OpacityMicromapAttachStatus status) noexcept {
    switch (status) {
        case OpacityMicromapAttachStatus::ATTACH: return "attach";
        case OpacityMicromapAttachStatus::EXTENSION_DISABLED: return "extension-disabled";
        case OpacityMicromapAttachStatus::NO_PUBLIC_API_REQUEST: return "no-public-api-request";
        case OpacityMicromapAttachStatus::NO_MICROMAP: return "no-micromap";
        case OpacityMicromapAttachStatus::GEOMETRY_NOT_TRIANGLES: return "geometry-not-triangles";
        case OpacityMicromapAttachStatus::GEOMETRY_MOTION: return "geometry-motion";
        case OpacityMicromapAttachStatus::INDEX_TYPE_UNSUPPORTED: return "index-type-unsupported";
        case OpacityMicromapAttachStatus::INDEX_STRIDE_MISALIGNED: return "index-stride-misaligned";
        case OpacityMicromapAttachStatus::INDEX_NONE_REQUIRES_MICROMAP: return "index-none-requires-micromap";
        case OpacityMicromapAttachStatus::RAY_FLAGS_FORCE_OPAQUE: return "ray-flags-force-opaque";
        case OpacityMicromapAttachStatus::INSTANCE_DISABLES_MICROMAPS: return "instance-disables-micromaps";
        case OpacityMicromapAttachStatus::RAY_QUERY_EXECUTION_MODE_MISSING: return "ray-query-execution-mode-missing";
    }
    return "unknown";
}

struct OpacityMicromapAttachRequest {
    bool extension_enabled{false};
    bool public_api_request{false};
    bool micromap_available{false};
    bool triangle_geometry{true};
    bool motion_triangles{false};
    // VK_EXT_opacity_micromap was the negotiated name. When it is, ray-query
    // traversal uses micromap data without any execution mode; with KHR only the
    // shader has to declare one.
    bool ext_opacity_micromap_enabled{false};
    bool ray_query_execution_mode_emitted{false};
    // The ray flags the emitted traversal would use include OpaqueKHR.
    bool opaque_ray_flags{false};
    bool instance_disables_micromaps{false};
    bool index_type_uint8_supported{false};
    OpacityMicromapIndexType index_type{OpacityMicromapIndexType::NONE};
    uint64_t index_stride{0};
    uint32_t base_triangle{0};
    uint32_t usage_count{0};
    uint32_t triangle_count{0};
};

struct OpacityMicromapAttachPlan {
    OpacityMicromapAttachStatus status{OpacityMicromapAttachStatus::EXTENSION_DISABLED};
    // The only field a caller may act on: chain the attach struct iff `chain`.
    bool chain{false};
    // The attach struct's fields, ready for the Vk* conversion.
    uint8_t index_type{static_cast<uint8_t>(OpacityMicromapIndexType::NONE)};
    uint64_t index_stride{0};
    uint32_t base_triangle{0};
    uint32_t usage_count{0};
    // Whether the geometry's OPAQUE flag must be dropped: "the micromap is used
    // to determine geometry opacity instead of the VK_GEOMETRY_OPAQUE_BIT_KHR
    // flag" (raytraversal.adoc), so keeping it only misleads the driver.
    bool clear_geometry_opaque_flag{false};
    // Whether the ray-query module still needs the OpacityMicromapIdKHR
    // execution mode that the attach makes mandatory.
    bool needs_ray_query_execution_mode{false};
};

[[nodiscard]] inline OpacityMicromapAttachPlan plan_opacity_micromap_attach(
    OpacityMicromapAttachRequest const &request) noexcept {
    OpacityMicromapAttachPlan plan{};
    if (!request.extension_enabled) {
        plan.status = OpacityMicromapAttachStatus::EXTENSION_DISABLED;
        return plan;
    }
    if (!request.public_api_request) {
        plan.status = OpacityMicromapAttachStatus::NO_PUBLIC_API_REQUEST;
        return plan;
    }
    if (!request.triangle_geometry) {
        plan.status = OpacityMicromapAttachStatus::GEOMETRY_NOT_TRIANGLES;
        return plan;
    }
    if (request.motion_triangles) {
        plan.status = OpacityMicromapAttachStatus::GEOMETRY_MOTION;
        return plan;
    }
    // Index type: NONE, UINT16, UINT32 always; UINT8 only with index_type_uint8.
    switch (request.index_type) {
        case OpacityMicromapIndexType::NONE:
            // VUID-VkAccelerationStructureTrianglesOpacityMicromapKHR-micromap
            // -11579: with INDEX_TYPE_NONE the micromap handle must be valid, so
            // the "no micromap" answer is only reachable through an explicit
            // index type.
            if (!request.micromap_available) {
                plan.status = OpacityMicromapAttachStatus::INDEX_NONE_REQUIRES_MICROMAP;
                return plan;
            }
            break;
        case OpacityMicromapIndexType::UINT16:
        case OpacityMicromapIndexType::UINT32:
            break;
        case OpacityMicromapIndexType::UINT8:
            if (!request.index_type_uint8_supported) {
                plan.status = OpacityMicromapAttachStatus::INDEX_TYPE_UNSUPPORTED;
                return plan;
            }
            break;
        default:
            plan.status = OpacityMicromapAttachStatus::INDEX_TYPE_UNSUPPORTED;
            return plan;
    }
    if (request.index_type != OpacityMicromapIndexType::NONE) {
        auto index_size = request.index_type == OpacityMicromapIndexType::UINT8 ?
                              uint64_t{1u} :
                          request.index_type == OpacityMicromapIndexType::UINT16 ?
                              uint64_t{2u} :
                              uint64_t{4u};
        if (request.index_stride == 0u || request.index_stride % index_size != 0u) {
            plan.status = OpacityMicromapAttachStatus::INDEX_STRIDE_MISALIGNED;
            return plan;
        }
        if (request.index_stride > 0xFFFFFFFFull) {
            // VUID-VkAccelerationStructureTrianglesOpacityMicromapKHR-indexStride-11574
            plan.status = OpacityMicromapAttachStatus::INDEX_STRIDE_MISALIGNED;
            return plan;
        }
    }
    if (!request.micromap_available) {
        // Nothing to attach: an index-typed request without a built micromap.
        plan.status = OpacityMicromapAttachStatus::NO_MICROMAP;
        return plan;
    }
    if (request.instance_disables_micromaps) {
        plan.status = OpacityMicromapAttachStatus::INSTANCE_DISABLES_MICROMAPS;
        return plan;
    }
    if (request.opaque_ray_flags) {
        // OpaqueKHR (and NoOpaqueKHR) override every geometry flag *and* the
        // micromap: attaching one would build a micromap the traversal never
        // reads.
        plan.status = OpacityMicromapAttachStatus::RAY_FLAGS_FORCE_OPAQUE;
        return plan;
    }
    plan.needs_ray_query_execution_mode =
        !request.ext_opacity_micromap_enabled;
    if (plan.needs_ray_query_execution_mode &&
        !request.ray_query_execution_mode_emitted) {
        plan.status = OpacityMicromapAttachStatus::RAY_QUERY_EXECUTION_MODE_MISSING;
        return plan;
    }
    plan.status = OpacityMicromapAttachStatus::ATTACH;
    plan.chain = true;
    plan.index_type = static_cast<uint8_t>(request.index_type);
    plan.index_stride = request.index_type == OpacityMicromapIndexType::NONE ?
                            uint64_t{0u} :
                            request.index_stride;
    plan.base_triangle = request.base_triangle;
    plan.usage_count = request.usage_count;
    plan.clear_geometry_opaque_flag = true;
    return plan;
}

// ---------------------------------------------------------------------------
// Geometry flags
// ---------------------------------------------------------------------------
// Bits mirror VkGeometryFlagBitsKHR so the header stays Vulkan-free;
// `blas.cpp` static_asserts both values against the real enums.
inline constexpr uint32_t geometry_flag_opaque = 0x00000001u;
inline constexpr uint32_t geometry_flag_no_duplicate_any_hit = 0x00000002u;

struct GeometryFlagRequest {
    // VK_GEOMETRY_TYPE_TRIANGLES_KHR vs VK_GEOMETRY_TYPE_AABBS_KHR. The AABB
    // geometry of this backend always leaves the flags at zero.
    bool triangle_geometry{true};
    // The flag the geometry kind would set without any optional extension. The
    // mesh path passes `true` (VK_GEOMETRY_OPAQUE_BIT_KHR, today's behaviour),
    // the procedural path `false`.
    bool opaque_requested{true};
    // OpacityMicromapAttachPlan::chain: only a *legal* attach may influence the
    // flags, so an off/unsupported extension cannot change the geometry.
    bool micromap_chained{false};
};

struct GeometryFlagPlan {
    uint32_t flags{0};
    // True when the OPAQUE bit was deliberately dropped because a micromap now
    // decides opacity. `flags == plan_geometry_flags(...).flags` is what the
    // caller passes to VkAccelerationStructureGeometryKHR::flags verbatim.
    bool micromap_overrides_opaque{false};
};

[[nodiscard]] inline GeometryFlagPlan plan_geometry_flags(
    GeometryFlagRequest const &request) noexcept {
    GeometryFlagPlan plan{};
    if (!request.triangle_geometry) {
        // AABB geometry: this backend never sets a flag (today's behaviour).
        return plan;
    }
    if (request.micromap_chained) {
        plan.micromap_overrides_opaque = true;
        return plan;
    }
    if (request.opaque_requested) {
        plan.flags = geometry_flag_opaque;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// VK_NV_ray_tracing_linear_swept_spheres
// ---------------------------------------------------------------------------
// VkAccelerationStructureGeometryLinearSweptSpheresDataNV is chained into
// VkAccelerationStructureGeometryKHR::pNext with
// geometryType == VK_GEOMETRY_TYPE_LINEAR_SWEPT_SPHERES_NV, and needs
// vertex + radius buffers (a radius per vertex), an optional index buffer, an
// indexing mode, and an end-caps mode. The primitive count of the BLAS build
// range follows from the indexing mode:
//   * no index buffer          -> floor(vertexCount / 2) line-list pairs
//   * LIST_NV                  -> floor(indexCount / 2) pairs
//   * SUCCESSIVE_NV            -> indexCount (1:1 with the primitive index)
// Formats are constrained by format features, not by a fixed list:
// vertexFormat needs VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR
// and radiusFormat needs VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_RADIUS_BUFFER_BIT_NV.
//
// This backend cannot produce that shape today: `DeviceInterface::create_curve`
// is not implemented on Vulkan (`stream.cpp` ignores CurveBuildCommand), so
// there is no curve resource, no control-point/segment build path and no BLAS
// holding curve geometry. The planner below is the exact ABI mapping that path
// will need, so the day `create_curve` lands the primitive count and the stride
// checks are already proven instead of being re-derived from the spec text.
enum class LssIndexingMode : uint8_t {
    LIST = 0,
    SUCCESSIVE = 1
};

enum class LssEndCapsMode : uint8_t {
    NONE = 0,
    CHAINED = 1
};

enum class LssStatus : uint8_t {
    // Legal: the caller may chain VkAccelerationStructureGeometryLinearSweptSpheresDataNV.
    USABLE,
    EXTENSION_DISABLED,
    // No curve resource exists in this backend, so no build can reach the
    // geometry. Missing API: DeviceInterface::create_curve on Vulkan plus a
    // CurveBuildCommand handler (see the report).
    NO_CURVE_RESOURCE,
    // VkFormatProperties3 of the vertex/radius format lacks the required bit.
    VERTEX_FORMAT_UNSUPPORTED,
    RADIUS_FORMAT_UNSUPPORTED,
    // vertexStride/radiusStride must be a multiple of the format's component
    // size and <= 2^32-1.
    VERTEX_STRIDE_MISALIGNED,
    RADIUS_STRIDE_MISALIGNED,
    // One radius per vertex is mandatory.
    RADIUS_COUNT_MISMATCH,
    // VUID-...-indexingMode-10427: SUCCESSIVE needs an index buffer.
    SUCCESSIVE_WITHOUT_INDEX,
    // indexType must be UINT16, UINT32 or NONE (VUID-...-indexData-10428).
    INDEX_TYPE_UNSUPPORTED,
    INDEX_STRIDE_MISALIGNED,
    // Fewer than two vertices/indices: nothing to sweep.
    NO_PRIMITIVES
};

[[nodiscard]] inline const char *lss_status_name(LssStatus status) noexcept {
    switch (status) {
        case LssStatus::USABLE: return "usable";
        case LssStatus::EXTENSION_DISABLED: return "extension-disabled";
        case LssStatus::NO_CURVE_RESOURCE: return "no-curve-resource";
        case LssStatus::VERTEX_FORMAT_UNSUPPORTED: return "vertex-format-unsupported";
        case LssStatus::RADIUS_FORMAT_UNSUPPORTED: return "radius-format-unsupported";
        case LssStatus::VERTEX_STRIDE_MISALIGNED: return "vertex-stride-misaligned";
        case LssStatus::RADIUS_STRIDE_MISALIGNED: return "radius-stride-misaligned";
        case LssStatus::RADIUS_COUNT_MISMATCH: return "radius-count-mismatch";
        case LssStatus::SUCCESSIVE_WITHOUT_INDEX: return "successive-without-index";
        case LssStatus::INDEX_TYPE_UNSUPPORTED: return "index-type-unsupported";
        case LssStatus::INDEX_STRIDE_MISALIGNED: return "index-stride-misaligned";
        case LssStatus::NO_PRIMITIVES: return "no-primitives";
    }
    return "unknown";
}

struct LssRequest {
    bool extension_enabled{false};
    // False on this backend: no curve resource / no curve BLAS.
    bool curve_resource_available{false};
    bool vertex_format_supported{false};
    bool radius_format_supported{false};
    bool index_buffer_present{false};
    bool index_type_uint8_supported{false};
    LssIndexingMode indexing_mode{LssIndexingMode::LIST};
    LssEndCapsMode end_caps_mode{LssEndCapsMode::NONE};
    OpacityMicromapIndexType index_type{OpacityMicromapIndexType::NONE};
    uint32_t vertex_count{0};
    uint32_t radius_count{0};
    uint32_t index_count{0};
    uint64_t vertex_stride{0};
    uint64_t radius_stride{0};
    uint64_t index_stride{0};
    // Size in bytes of the declared vertex/radius format (e.g. 12 for
    // VK_FORMAT_R32G32B32_SFLOAT, 4 for VK_FORMAT_R32_SFLOAT). Strides have to
    // be multiples of the component size; using the whole format size keeps the
    // check conservative and format-table free.
    uint32_t vertex_format_size{0};
    uint32_t radius_format_size{0};
};

struct LssPlan {
    LssStatus status{LssStatus::EXTENSION_DISABLED};
    bool usable{false};
    LssIndexingMode indexing_mode{LssIndexingMode::LIST};
    LssEndCapsMode end_caps_mode{LssEndCapsMode::NONE};
    uint8_t index_type{static_cast<uint8_t>(OpacityMicromapIndexType::NONE)};
    uint64_t index_stride{0};
    // VkAccelerationStructureBuildRangeInfoKHR::primitiveCount for the geometry.
    uint32_t primitive_count{0};
};

[[nodiscard]] inline LssPlan plan_linear_swept_spheres(LssRequest const &request) noexcept {
    LssPlan plan{};
    if (!request.extension_enabled) {
        plan.status = LssStatus::EXTENSION_DISABLED;
        return plan;
    }
    if (!request.curve_resource_available) {
        plan.status = LssStatus::NO_CURVE_RESOURCE;
        return plan;
    }
    if (!request.vertex_format_supported) {
        plan.status = LssStatus::VERTEX_FORMAT_UNSUPPORTED;
        return plan;
    }
    if (!request.radius_format_supported) {
        plan.status = LssStatus::RADIUS_FORMAT_UNSUPPORTED;
        return plan;
    }
    if (request.vertex_format_size == 0u ||
        request.vertex_stride == 0u ||
        request.vertex_stride % request.vertex_format_size != 0u ||
        request.vertex_stride > 0xFFFFFFFFull) {
        plan.status = LssStatus::VERTEX_STRIDE_MISALIGNED;
        return plan;
    }
    if (request.radius_format_size == 0u ||
        request.radius_stride == 0u ||
        request.radius_stride % request.radius_format_size != 0u ||
        request.radius_stride > 0xFFFFFFFFull) {
        plan.status = LssStatus::RADIUS_STRIDE_MISALIGNED;
        return plan;
    }
    if (request.radius_count != request.vertex_count) {
        plan.status = LssStatus::RADIUS_COUNT_MISMATCH;
        return plan;
    }
    if (request.indexing_mode == LssIndexingMode::SUCCESSIVE &&
        !request.index_buffer_present) {
        plan.status = LssStatus::SUCCESSIVE_WITHOUT_INDEX;
        return plan;
    }
    if (request.index_buffer_present) {
        switch (request.index_type) {
            case OpacityMicromapIndexType::UINT16:
            case OpacityMicromapIndexType::UINT32:
                break;
            case OpacityMicromapIndexType::UINT8:
                if (!request.index_type_uint8_supported) {
                    plan.status = LssStatus::INDEX_TYPE_UNSUPPORTED;
                    return plan;
                }
                break;
            default:
                plan.status = LssStatus::INDEX_TYPE_UNSUPPORTED;
                return plan;
        }
        auto index_size = request.index_type == OpacityMicromapIndexType::UINT8 ?
                              uint64_t{1u} :
                          request.index_type == OpacityMicromapIndexType::UINT16 ?
                              uint64_t{2u} :
                              uint64_t{4u};
        if (request.index_stride < index_size ||
            request.index_stride % index_size != 0u ||
            request.index_stride > 0xFFFFFFFFull) {
            plan.status = LssStatus::INDEX_STRIDE_MISALIGNED;
            return plan;
        }
    }
    uint32_t primitive_count = 0u;
    if (!request.index_buffer_present) {
        primitive_count = request.vertex_count / 2u;
    } else if (request.indexing_mode == LssIndexingMode::SUCCESSIVE) {
        primitive_count = request.index_count;
    } else {
        primitive_count = request.index_count / 2u;
    }
    if (primitive_count == 0u) {
        plan.status = LssStatus::NO_PRIMITIVES;
        return plan;
    }
    plan.status = LssStatus::USABLE;
    plan.usable = true;
    plan.indexing_mode = request.indexing_mode;
    plan.end_caps_mode = request.end_caps_mode;
    plan.index_type = request.index_buffer_present ?
                          static_cast<uint8_t>(request.index_type) :
                          static_cast<uint8_t>(OpacityMicromapIndexType::NONE);
    plan.index_stride = request.index_buffer_present ? request.index_stride : uint64_t{0u};
    plan.primitive_count = primitive_count;
    return plan;
}

// ---------------------------------------------------------------------------
// VK_AMDX_dense_geometry_format (provisional, VK_ENABLE_BETA_EXTENSIONS)
// ---------------------------------------------------------------------------
// The extension builds an acceleration structure from pre-compressed DGF1
// triangle data: VkAccelerationStructureDenseGeometryFormatTrianglesDataAMDX is
// chained into VkAccelerationStructureGeometryKHR::pNext with
// geometryType == VK_GEOMETRY_TYPE_DENSE_GEOMETRY_FORMAT_TRIANGLES_AMDX, and the
// compressed data must come from a buffer created with
// VK_BUFFER_USAGE_2_COMPRESSED_DATA_DGF1_BIT_AMDX. DGF1 blocks are 128 bytes and
// hold at most 64 triangles / 64 vertices each, the base address must be
// 128-byte aligned and dataSize must be a multiple of 128.
//
// Capability-only here: the backend's BLAS build takes raw vertex + index
// buffers (Blas::pre_build(MeshBuildCommand)), the public API has no compressed
// geometry input, and no compressor exists in the tree. The planner pins the
// alignment/size/count rules so a future DGF input does not have to rediscover
// them - and so an offered blob is rejected, not chained, while the extension is
// off.
inline constexpr uint64_t dgf1_block_size = 128u;
inline constexpr uint32_t dgf1_max_triangles_per_block = 64u;
inline constexpr uint32_t dgf1_max_vertices_per_block = 64u;

enum class DgfStatus : uint8_t {
    USABLE,
    EXTENSION_DISABLED,
    // The header was compiled without VK_ENABLE_BETA_EXTENSIONS, so the
    // structures do not exist and nothing may be referenced.
    BETA_EXTENSIONS_DISABLED,
    // No public-API input carries compressed geometry.
    NO_PUBLIC_API_REQUEST,
    DATA_ADDRESS_MISALIGNED,
    DATA_SIZE_MISALIGNED,
    DATA_SIZE_ZERO,
    TRIANGLE_COUNT_ZERO,
    // 64 triangles per block is the hard DGF1 limit; a blob claiming more
    // triangles than its blocks can hold is malformed.
    TRIANGLE_COUNT_EXCEEDS_BLOCKS
};

[[nodiscard]] inline const char *dgf_status_name(DgfStatus status) noexcept {
    switch (status) {
        case DgfStatus::USABLE: return "usable";
        case DgfStatus::EXTENSION_DISABLED: return "extension-disabled";
        case DgfStatus::BETA_EXTENSIONS_DISABLED: return "beta-extensions-disabled";
        case DgfStatus::NO_PUBLIC_API_REQUEST: return "no-public-api-request";
        case DgfStatus::DATA_ADDRESS_MISALIGNED: return "data-address-misaligned";
        case DgfStatus::DATA_SIZE_MISALIGNED: return "data-size-misaligned";
        case DgfStatus::DATA_SIZE_ZERO: return "data-size-zero";
        case DgfStatus::TRIANGLE_COUNT_ZERO: return "triangle-count-zero";
        case DgfStatus::TRIANGLE_COUNT_EXCEEDS_BLOCKS: return "triangle-count-exceeds-blocks";
    }
    return "unknown";
}

struct DgfRequest {
    bool extension_enabled{false};
    // Compiled with VK_ENABLE_BETA_EXTENSIONS and the header exposes the
    // provisional structures.
    bool beta_extensions_enabled{false};
    // The caller offers a DGF1 blob for this build.
    bool compressed_data_offered{false};
    // The public API can express that offer.
    bool public_api_request{false};
    uint64_t compressed_data_address{0};
    uint64_t data_size{0};
    uint32_t triangle_count{0};
    uint32_t vertex_count{0};
};

struct DgfPlan {
    DgfStatus status{DgfStatus::EXTENSION_DISABLED};
    bool usable{false};
    uint32_t block_count{0};
    uint32_t max_triangles{0};
    uint32_t max_vertices{0};
};

[[nodiscard]] inline DgfPlan plan_dense_geometry_format(DgfRequest const &request) noexcept {
    DgfPlan plan{};
    if (!request.extension_enabled) {
        plan.status = DgfStatus::EXTENSION_DISABLED;
        return plan;
    }
    if (!request.beta_extensions_enabled) {
        plan.status = DgfStatus::BETA_EXTENSIONS_DISABLED;
        return plan;
    }
    if (!request.compressed_data_offered || !request.public_api_request) {
        plan.status = DgfStatus::NO_PUBLIC_API_REQUEST;
        return plan;
    }
    if (request.data_size == 0u) {
        plan.status = DgfStatus::DATA_SIZE_ZERO;
        return plan;
    }
    if (request.data_size % dgf1_block_size != 0u) {
        plan.status = DgfStatus::DATA_SIZE_MISALIGNED;
        return plan;
    }
    if (request.compressed_data_address % dgf1_block_size != 0u) {
        plan.status = DgfStatus::DATA_ADDRESS_MISALIGNED;
        return plan;
    }
    if (request.triangle_count == 0u) {
        plan.status = DgfStatus::TRIANGLE_COUNT_ZERO;
        return plan;
    }
    auto blocks = static_cast<uint32_t>(request.data_size / dgf1_block_size);
    // 64 triangles and 64 vertices per block, computed in 64 bits so a huge
    // blob cannot wrap the limit.
    auto max_triangles_64 = static_cast<uint64_t>(blocks) * dgf1_max_triangles_per_block;
    auto max_vertices_64 = static_cast<uint64_t>(blocks) * dgf1_max_vertices_per_block;
    plan.block_count = blocks;
    plan.max_triangles = max_triangles_64 > 0xFFFFFFFFull ?
                            0xFFFFFFFFu :
                            static_cast<uint32_t>(max_triangles_64);
    plan.max_vertices = max_vertices_64 > 0xFFFFFFFFull ?
                            0xFFFFFFFFu :
                            static_cast<uint32_t>(max_vertices_64);
    if (static_cast<uint64_t>(request.triangle_count) > max_triangles_64 ||
        static_cast<uint64_t>(request.vertex_count) > max_vertices_64) {
        plan.status = DgfStatus::TRIANGLE_COUNT_EXCEEDS_BLOCKS;
        return plan;
    }
    plan.status = DgfStatus::USABLE;
    plan.usable = true;
    return plan;
}

// ---------------------------------------------------------------------------
// VK_NV_partitioned_acceleration_structure
// ---------------------------------------------------------------------------
// A partitioned (PTLAS) build is not a vkCmdBuildAccelerationStructuresKHR
// variant: it takes VkBuildPartitionedAccelerationStructureInfoNV with
// device-address inputs (srcAccelerationStructureData /
// dstAccelerationStructureData / scratchData / srcInfos / srcInfosCount) and the
// instance records live in the caller's own
// VkPartitionedAccelerationStructureWriteInstanceDataNV array, which carries the
// per-instance partition index and flags. A PTLAS is *not* a
// VkAccelerationStructureKHR, is not created with
// vkCreateAccelerationStructureKHR, and cannot be written into a
// VkAccelerationStructureInstanceKHR::accelerationStructureReference the way
// this backend's Tlas packs its instance buffer.
//
// Capability-only: Tlas::pre_build writes plain VkAccelerationStructureInstanceKHR
// records and has no partition id, no per-instance partition flags, and no
// VkPartitionedAccelerationStructure* buffer. The planner pins the two limits the
// spec states - the sum check against maxPartitionCount
// (VUID-VkPartitionedAccelerationStructureInstancesInputNV-partitionCount-10535)
// and the phase-count/scratch alignment - so a future PTLAS path is machine
// checked, and so an offered PTLAS request is refused while the extension is off.
inline constexpr uint64_t ptlas_scratch_alignment = 256u;

enum class PtlasStatus : uint8_t {
    USABLE,
    EXTENSION_DISABLED,
    // Tlas has no partition-aware instance layout.
    NO_PUBLIC_API_REQUEST,
    // partitionCount + maxInstanceInGlobalPartitionCount >
    // maxPartitionCount from VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV.
    PARTITION_COUNT_EXCEEDED,
    PARTITION_COUNT_ZERO,
    INSTANCE_COUNT_ZERO,
    // "The number of instances in a partition must not exceed
    // maxInstancePerPartitionCount" is a build-time VU; a zero per-partition
    // maximum with a non-zero instance count can never satisfy it.
    PER_PARTITION_COUNT_ZERO
};

[[nodiscard]] inline const char *ptlas_status_name(PtlasStatus status) noexcept {
    switch (status) {
        case PtlasStatus::USABLE: return "usable";
        case PtlasStatus::EXTENSION_DISABLED: return "extension-disabled";
        case PtlasStatus::NO_PUBLIC_API_REQUEST: return "no-public-api-request";
        case PtlasStatus::PARTITION_COUNT_EXCEEDED: return "partition-count-exceeded";
        case PtlasStatus::PARTITION_COUNT_ZERO: return "partition-count-zero";
        case PtlasStatus::INSTANCE_COUNT_ZERO: return "instance-count-zero";
        case PtlasStatus::PER_PARTITION_COUNT_ZERO: return "per-partition-count-zero";
    }
    return "unknown";
}

struct PtlasRequest {
    bool extension_enabled{false};
    // The public accel API can describe partitions.
    bool public_api_request{false};
    bool partition_translation{false};
    uint32_t partition_count{0};
    uint32_t max_instance_per_partition_count{0};
    uint32_t max_instance_in_global_partition_count{0};
    uint32_t instance_count{0};
    // VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV::maxPartitionCount.
    uint32_t max_partition_count{0};
};

struct PtlasPlan {
    PtlasStatus status{PtlasStatus::EXTENSION_DISABLED};
    bool usable{false};
    bool partition_translation{false};
    uint32_t partition_count{0};
    uint32_t max_instance_per_partition_count{0};
    uint32_t max_instance_in_global_partition_count{0};
    uint64_t scratch_alignment{ptlas_scratch_alignment};
};

[[nodiscard]] inline PtlasPlan plan_partitioned_acceleration_structure(
    PtlasRequest const &request) noexcept {
    PtlasPlan plan{};
    if (!request.extension_enabled) {
        plan.status = PtlasStatus::EXTENSION_DISABLED;
        return plan;
    }
    if (!request.public_api_request) {
        plan.status = PtlasStatus::NO_PUBLIC_API_REQUEST;
        return plan;
    }
    if (request.partition_count == 0u) {
        plan.status = PtlasStatus::PARTITION_COUNT_ZERO;
        return plan;
    }
    if (request.instance_count == 0u) {
        plan.status = PtlasStatus::INSTANCE_COUNT_ZERO;
        return plan;
    }
    if (request.max_instance_per_partition_count == 0u) {
        plan.status = PtlasStatus::PER_PARTITION_COUNT_ZERO;
        return plan;
    }
    // VUID-VkPartitionedAccelerationStructureInstancesInputNV-partitionCount-10535
    auto partitions_64 = static_cast<uint64_t>(request.partition_count) +
                         static_cast<uint64_t>(request.max_instance_in_global_partition_count);
    if (request.max_partition_count != 0u &&
        partitions_64 > static_cast<uint64_t>(request.max_partition_count)) {
        plan.status = PtlasStatus::PARTITION_COUNT_EXCEEDED;
        return plan;
    }
    plan.status = PtlasStatus::USABLE;
    plan.usable = true;
    plan.partition_translation = request.partition_translation;
    plan.partition_count = request.partition_count;
    plan.max_instance_per_partition_count = request.max_instance_per_partition_count;
    plan.max_instance_in_global_partition_count = request.max_instance_in_global_partition_count;
    return plan;
}

}// namespace lc::vk::detail
