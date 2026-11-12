#pragma once
#include <volk.h>
#include "vk_allocator.h"
#include "rt_extension_plan.h"

namespace lc::vk {

class Device;
class CommandBuffer;

// Backend-internal opacity micromap (VK_KHR_opacity_micromap; the EXT alias is
// accepted by the registry).
//
// ---------------------------------------------------------------------------
// Why this class is not reachable from the default build
// ---------------------------------------------------------------------------
// The planning header (`rt_extension_plan.h`) is the single gate. Even with the
// extension negotiated and enabled, `detail::plan_opacity_micromap_build()` only
// answers BUILD for a request that sets `public_api_request`, and
// `detail::plan_opacity_micromap_attach()` only answers ATTACH for the very same
// reason. Neither is reachable today:
//
//   * `AccelOption` (include/luisa/runtime/rhi/resource.h) has no micromap field,
//     `MeshBuildCommand` (include/luisa/runtime/rhi/command.h) carries only the
//     vertex and index buffers, and `Mesh` has no micromap input. So the public
//     accelerator API cannot *ask* for a micromap, which is why `Blas::_micromap`
//     is always null and `Blas::pre_build` never chains
//     VkAccelerationStructureTrianglesOpacityMicromapEXT.
//   * With only VK_KHR_opacity_micromap negotiated (the registry prefers the KHR
//     name and falls back to EXT), raytraversal.adoc requires the ray query to
//     declare the `OpacityMicromapIdKHR` execution mode with its
//     specialization-constant operand VK_TRUE. No codegen route in this backend
//     emits that execution mode (the HLSL route cannot express it at all, and the
//     native XIR->SPIR-V route has no execution-mode hook for it), so
//     `ray_query_execution_mode_emitted` stays false and the attach plan refuses.
//
// The public API additions needed for the whole feature are therefore, exactly:
//   1. an `AccelOption`/`MeshBuildCommand` micromap input (micromap resource
//      handle + VkAccelerationStructureTrianglesOpacityMicromapEXT's index type /
//      index buffer / stride / base triangle / usage counts), plus a
//      `DeviceInterface` entry point that creates the micromap resource from
//      per-triangle format+level data, and
//   2. `OpacityMicromapIdKHR` emission in the ray-query path (or the
//      VK_EXT_opacity_micromap name being the negotiated one).
//
// Because none of that exists, `Device::mark_optional_extension_active()` is
// never called for OPACITY_MICROMAP: the status extension reports the extension
// as "enabled" but not "active", which is the honest answer.
//
// ---------------------------------------------------------------------------
// What is implemented
// ---------------------------------------------------------------------------
// The micromap *object* model of VK_EXT_opacity_micromap / VK_KHR_opacity_micromap
// (a VkMicromapEXT backed by a VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT buffer)
// with both build commands the spec offers:
//
//   * build_device(): vkCmdBuildMicromapsEXT recorded into the caller's command
//     buffer with device addresses (the path a device without
//     `micromapHostCommands` must use, and the one the registry's negotiation
//     implies because it only enables the `micromap` feature bit);
//   * build_host(): vkBuildMicromapsEXT with host addresses, refused unless
//     `micromapHostCommands` is enabled.
//
// The micromap revision this spec snapshot also documents - a micromap as a
// VkAccelerationStructureKHR of type VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_
// MICROMAP_KHR built through VK_KHR_device_address_commands - is reported as
// `MicromapObjectModel::KHR_ACCELERATION_STRUCTURE` and refused by the plan
// instead of being fed a VkMicromapEXT that could never be created.
class OpacityMicromap {
public:
    // The only gate: `Device::optional_extension_enabled(OPACITY_MICROMAP)`.
    [[nodiscard]] static bool available(Device *device) noexcept;

    // Whether the registry resolved the EXT spelling (VK_EXT_opacity_micromap)
    // rather than the KHR one. Only the EXT name lifts the ray-query
    // `OpacityMicromapIdKHR` execution-mode requirement (raytraversal.adoc: "If
    // VK_EXT_opacity_micromap is enabled, OpRayQueryProceedKHR traversal always
    // uses opacity micromap data without the OpacityMicromapIdKHR execution mode
    // being declared").
    [[nodiscard]] static bool ext_alias_negotiated(Device *device) noexcept;

    // Fills the negotiated half of a build request from the device. Everything
    // that comes from a physical-device property or from the registry lives
    // here, so no caller has to remember `micromapHostCommands` is never
    // enabled.
    [[nodiscard]] static detail::OpacityMicromapBuildRequest make_build_request(
        Device *device, uint32_t triangle_count,
        detail::OpacityMicromapFormat format, uint8_t subdivision_level,
        bool public_api_request, bool host_build_preferred = false) noexcept;

    // Creates the micromap object and its storage/build-input buffers, sizes them
    // with vkGetMicromapBuildSizesEXT and creates the VkMicromapEXT. Returns
    // nullptr - without calling a single Vulkan entry point - whenever the plan
    // does not answer BUILD, i.e. for a disabled/unsupported extension, for a
    // request the public API cannot express, and for every illegal input.
    //
    // `data` is the packed opacity bit data (1 bit per micro triangle for the
    // 2-state format, 2 bits for the 4-state one, packed LSB-first) and
    // `triangles` the per-triangle VkMicromapTriangleEXT records. Both are copied
    // into the object's own buffers.
    [[nodiscard]] static OpacityMicromap *create(
        Device *device,
        detail::OpacityMicromapBuildRequest const &request,
        luisa::span<std::byte const> data,
        luisa::span<detail::OpacityMicromapTriangleRecord const> triangles) noexcept;

    ~OpacityMicromap();
    OpacityMicromap(OpacityMicromap const &) = delete;
    OpacityMicromap &operator=(OpacityMicromap const &) = delete;

    // vkCmdBuildMicromapsEXT. Device addresses, so the micromap can be built on a
    // device that does not offer `micromapHostCommands`.
    void build_device(CommandBuffer &cmdbuffer);

    // vkBuildMicromapsEXT. Requires the plan's mode to be HOST (which requires
    // `micromapHostCommands`); returns false otherwise, without calling the
    // entry point.
    bool build_host() noexcept;

    // Fills a VkAccelerationStructureTrianglesOpacityMicromapEXT for a BLAS
    // build. Returns false - leaving `info` untouched - unless the attach plan
    // answers ATTACH, so a disabled extension can never leak this struct into a
    // VkAccelerationStructureGeometryTrianglesDataKHR::pNext chain.
    bool fill_attach_info(
        VkAccelerationStructureTrianglesOpacityMicromapEXT &info,
        detail::OpacityMicromapAttachRequest const &request,
        uint64_t index_buffer_device_address) const noexcept;

    [[nodiscard]] VkMicromapEXT handle() const noexcept { return _micromap; }
    // Whether a build command really ran: a micromap handles that was created but
    // never built cannot be attached to a BLAS.
    [[nodiscard]] bool built() const noexcept { return _built; }
    [[nodiscard]] auto const &plan() const noexcept { return _plan; }
    // Whether the micromap may be destroyed right after the BLAS build
    // (VkMicromapBuildSizesInfoEXT::discardable). When false the caller has to
    // keep it alive for as long as the built acceleration structure is traced.
    [[nodiscard]] bool discardable() const noexcept { return _discardable; }
    [[nodiscard]] uint64_t data_device_address() const noexcept { return _data.address; }
    [[nodiscard]] uint64_t triangle_array_device_address() const noexcept { return _triangles.address; }

    // Public only because `new_with_allocator` has to reach it; always obtain an
    // instance through `create()`, which is the only caller and runs the plan
    // first.
    OpacityMicromap(Device *device, detail::OpacityMicromapBuildPlan const &plan) noexcept;

private:
    // One VMA buffer with a persistently mapped host pointer and a device
    // address. Host-visible on purpose: the host build takes host addresses, and
    // the device build takes the same memory through its device address, so one
    // allocation serves both build modes.
    struct MappedBuffer {
        VkBuffer buffer{VK_NULL_HANDLE};
        VmaAllocation allocation{};
        void *mapped{nullptr};
        VkDeviceAddress address{0u};
        size_t size{0u};
    };

    Device *_device{nullptr};
    VkMicromapEXT _micromap{VK_NULL_HANDLE};
    VkMicromapBuildSizesInfoEXT _sizes{};
    detail::OpacityMicromapBuildPlan _plan{};
    MappedBuffer _storage{};
    MappedBuffer _data{};
    MappedBuffer _triangles{};
    MappedBuffer _scratch{};
    // The canonical aggregated usage entry the build references. Kept in the
    // object because vkCmdBuildMicromapsEXT only reads the pointer at record
    // time, but vkBuildMicromapsEXT uses it for the duration of the call and a
    // future host-build retry may outlive the caller's stack frame.
    VkMicromapUsageEXT _usage{};
    bool _built{false};
    bool _discardable{false};

    [[nodiscard]] MappedBuffer create_mapped_buffer(
        VkBufferUsageFlags usage, size_t size) noexcept;
    void destroy_mapped_buffer(MappedBuffer &buffer) noexcept;
    // Fills data/scratch/triangleArray of a VkMicromapBuildInfoEXT for the
    // selected build mode.
    void fill_build_info(VkMicromapBuildInfoEXT &info, detail::MicromapBuildMode mode) const noexcept;
};

}// namespace lc::vk
