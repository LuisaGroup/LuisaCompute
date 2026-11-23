#pragma once

// Replayable command recording for the Vulkan backend (plan P4, narrow v1).
//
// What this extension is
// ----------------------
// `record()` encodes a `CommandList` **once** into a dedicated Vulkan primary
// command buffer that is never reset, re-begun or re-recorded, and keeps every
// native object the encoded commands still reference (command pool, descriptor
// pools and sets, argument staging, retained `VkPipeline`s and the interned
// pipeline/descriptor-set layouts) alive in a graph. `replay()` submits that
// command buffer again through the owning stream's ordinary timeline path, with
// exactly one queue submission, no argument planning, no descriptor work and no
// host encode.
//
// The only thing that may change between two replays is the **device-visible
// content** of the buffers the recording reads or writes. Everything else -
// uniform/trailer bytes, resource identities, descriptor-set and pipeline-layout
// identity, and the dispatch shape - is immutable for the graph's lifetime, which
// is why `update()` always refuses (see below).
//
// What v1 accepts (nothing else; every rejection is typed)
// -------------------------------------------------------
//  * commands: a direct, non-indirect compute `ShaderDispatchCommand` (single or
//    batched) and a device-to-device `BufferCopyCommand`;
//  * shader arguments: `UNIFORM` and `BUFFER` only;
//  * shaders: compute shaders that use no printer, no constant UBO, no bindless
//    array (`use_buffer_bindless()` / `use_tex2d_bindless()` /
//    `use_tex3d_bindless()`) and no texture argument.
//
// Rejected: every other command tag (uploads, downloads, texture copies,
// acceleration-structure/mesh/curve/procedural/motion builds, bindless updates,
// indirect dispatch, raster and ray-tracing dispatch, printer dispatch, every
// `CustomCommand` incl. native shaders and CUDA launches), texture/bindless/
// acceleration arguments, command-list callbacks, presents, destructor
// callbacks, and a stream whose `VulkanDeviceConfigExt` injects per-submission
// before/after resource states.
//
// `VulkanDeviceConfigExt::execute_command_buffer()` is deliberately not consulted
// for a replay: a replay's whole contract is "the backend submits this one
// buffer, once", so a config extension cannot take the submission over. Recording
// is refused when the same extension publishes per-submission resource states
// (`before_states()`/`after_states()`), because those describe one submission and
// not a recording, and `borrow_command_buffer()` is never used: the borrow
// contract is one-shot and forbids resetting or freeing the buffer, while a
// recorded buffer is resubmitted and later freed by the backend.
//
// A refusal never throws and never aborts: it returns a status plus a
// `VulkanReplayRejectReason` and a message, consumes the command list, and
// creates no native resource.
//
// Lifetime and threading contract (read this before using the API)
// ----------------------------------------------------------------
//  * `record()` consumes the list: the commands, callbacks, presents and
//    destructor callbacks are taken out of it, so the caller's list is empty
//    afterwards whether the recording succeeded or not. A destructor callback of
//    a *rejected* list is run at the end of `record()` (that is where the list's
//    own destructor would have run it); it is never run during a replay.
//  * A graph keeps no pointer or reference into the `Stream`, the buffers or the
//    shaders it recorded; it only watches their native identities by value.
//    Destroying a watched buffer marks the graph `RESOURCE_DESTROYED`, destroying
//    the recording stream marks it `STREAM_DESTROYED`, and destroying the device
//    drains every graph. A later `replay()` returns that typed error without
//    touching the destroyed resource.
//  * A shader may be destroyed after `record()`: the recorded pipeline and the
//    interned pipeline/descriptor-set layouts are retained explicitly by the
//    graph.
//  * `destroy()` refuses with `BUSY` while the graph's latest replay has not
//    completed on the host. Synchronize the stream and retry; the retry succeeds
//    and frees the native state.
//  * Destroying the stream, a watched buffer or a graph **must not race** a
//    `replay()`/`destroy()` on the same graph from another thread; that is the
//    same rule as destroying any other Luisa resource with work in flight.
//    `replay()` itself is serialized by the stream's dispatch mutex, so several
//    in-flight replays of one graph from one thread are supported.
//
// The header deliberately includes no Vulkan header and names no backend-private
// type: handles are `uint64_t`, statuses are plain enums.

#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

// Why a recording was refused. Every value maps to exactly one check in
// `src/backends/vk/replay_plan.h` (host-testable, see
// src/tests/unit/ext/test_vk_replay_plan.cpp).
enum class VulkanReplayRejectReason : uint32_t {
    NONE = 0u,
    // ---- command level -----------------------------------------------------
    COMMAND_BUFFER_UPLOAD,   // BufferUploadCommand: host staging content
    COMMAND_BUFFER_DOWNLOAD, // BufferDownloadCommand: readback tied to retirement
    COMMAND_TEXTURE,         // any texture copy/upload/download command
    COMMAND_ACCEL_BUILD,     // accel/mesh/curve/procedural/motion-instance build
    COMMAND_BINDLESS_UPDATE, // BindlessArrayUpdateCommand
    COMMAND_CUSTOM,          // every CustomCommand (native shader, RT, raster, ...)
    DISPATCH_INDIRECT,       // the dispatch shape is read from a buffer at execution
    DISPATCH_RASTER,         // raster shader tag
    DISPATCH_RAY_TRACING,    // ray-tracing shader tag
    DISPATCH_PRINTER,        // printer dispatch needs stream scratch + readback
    DISPATCH_CONSTANT_UBO,   // the shader owns a constant-UBO buffer a graph would point at
    DISPATCH_BINDLESS,       // the shader reads/writes the bindless heap
    DISPATCH_UNKNOWN_SHADER, // the shader handle is not a compute shader of this device
    // ---- argument level ----------------------------------------------------
    ARG_TEXTURE,             // a texture argument (no image-view lifetime work in v1)
    ARG_BINDLESS_ARRAY,      // a bindless-array argument (membership must be static)
    ARG_ACCELERATION_STRUCTURE,
    ARG_UNKNOWN_TAG,
    // ---- list level --------------------------------------------------------
    LIST_CALLBACKS,
    LIST_PRESENTS,
    LIST_DTOR_CALLBACKS,
    EMPTY_LIST,
    // ---- device/stream level ----------------------------------------------
    CONFIG_EXT_STATE_INJECTION, // the stream's config ext publishes per-submission state
    // ---- resource level ----------------------------------------------------
    OUT_OF_RESOURCES, // a graph budget (commands / watched buffers) was exceeded
};

[[nodiscard]] constexpr luisa::string_view
vulkan_replay_reject_reason_name(VulkanReplayRejectReason reason) noexcept {
    switch (reason) {
        case VulkanReplayRejectReason::NONE: return "none";
        case VulkanReplayRejectReason::COMMAND_BUFFER_UPLOAD: return "command_buffer_upload";
        case VulkanReplayRejectReason::COMMAND_BUFFER_DOWNLOAD: return "command_buffer_download";
        case VulkanReplayRejectReason::COMMAND_TEXTURE: return "command_texture";
        case VulkanReplayRejectReason::COMMAND_ACCEL_BUILD: return "command_accel_build";
        case VulkanReplayRejectReason::COMMAND_BINDLESS_UPDATE: return "command_bindless_update";
        case VulkanReplayRejectReason::COMMAND_CUSTOM: return "command_custom";
        case VulkanReplayRejectReason::DISPATCH_INDIRECT: return "dispatch_indirect";
        case VulkanReplayRejectReason::DISPATCH_RASTER: return "dispatch_raster";
        case VulkanReplayRejectReason::DISPATCH_RAY_TRACING: return "dispatch_ray_tracing";
        case VulkanReplayRejectReason::DISPATCH_PRINTER: return "dispatch_printer";
        case VulkanReplayRejectReason::DISPATCH_CONSTANT_UBO: return "dispatch_constant_ubo";
        case VulkanReplayRejectReason::DISPATCH_BINDLESS: return "dispatch_bindless";
        case VulkanReplayRejectReason::DISPATCH_UNKNOWN_SHADER: return "dispatch_unknown_shader";
        case VulkanReplayRejectReason::ARG_TEXTURE: return "arg_texture";
        case VulkanReplayRejectReason::ARG_BINDLESS_ARRAY: return "arg_bindless_array";
        case VulkanReplayRejectReason::ARG_ACCELERATION_STRUCTURE: return "arg_acceleration_structure";
        case VulkanReplayRejectReason::ARG_UNKNOWN_TAG: return "arg_unknown_tag";
        case VulkanReplayRejectReason::LIST_CALLBACKS: return "list_callbacks";
        case VulkanReplayRejectReason::LIST_PRESENTS: return "list_presents";
        case VulkanReplayRejectReason::LIST_DTOR_CALLBACKS: return "list_dtor_callbacks";
        case VulkanReplayRejectReason::EMPTY_LIST: return "empty_list";
        case VulkanReplayRejectReason::CONFIG_EXT_STATE_INJECTION: return "config_ext_state_injection";
        case VulkanReplayRejectReason::OUT_OF_RESOURCES: return "out_of_resources";
    }
    return "unknown";
}

// Why a live graph stopped being replayable. Invalidation is permanent: a
// graph's resource keys are native identities, which a driver may hand out again
// to an unrelated resource, so a "revalidated" graph could silently alias.
enum class VulkanReplayInvalidationReason : uint32_t {
    NONE = 0u,
    RESOURCE_DESTROYED, // a buffer the recording names was destroyed
    STREAM_DESTROYED,   // the recording stream was destroyed
    DEVICE_TEARDOWN,    // the device was destroyed
};

[[nodiscard]] constexpr luisa::string_view
vulkan_replay_invalidation_reason_name(VulkanReplayInvalidationReason reason) noexcept {
    switch (reason) {
        case VulkanReplayInvalidationReason::NONE: return "none";
        case VulkanReplayInvalidationReason::RESOURCE_DESTROYED: return "resource_destroyed";
        case VulkanReplayInvalidationReason::STREAM_DESTROYED: return "stream_destroyed";
        case VulkanReplayInvalidationReason::DEVICE_TEARDOWN: return "device_teardown";
    }
    return "unknown";
}

// Outcome of one entry point. `SUCCESS` is 0, so `status != SUCCESS` is the
// failure test; nothing here is fatal and nothing throws.
enum class VulkanReplayStatus : uint32_t {
    SUCCESS = 0u,
    INVALID_HANDLE,   // the replay handle is unknown, or was already destroyed
    INVALID_STREAM,   // the stream handle is not a stream of this device
    CUSTOM_STREAM,    // extension-owned streams (DStorage & co.) cannot record
    INVALID_ARGUMENT, // nothing usable was passed (e.g. a null stream handle)
    REJECTED,         // record only: see VulkanReplayRecordResult::reason
    BUSY,             // destroy only: the latest replay has not completed yet
    RESOURCE_DESTROYED,
    STREAM_DESTROYED,
    NOT_SUPPORTED,
    OUT_OF_RESOURCES,
};

[[nodiscard]] constexpr luisa::string_view
vulkan_replay_status_name(VulkanReplayStatus status) noexcept {
    switch (status) {
        case VulkanReplayStatus::SUCCESS: return "success";
        case VulkanReplayStatus::INVALID_HANDLE: return "invalid_handle";
        case VulkanReplayStatus::INVALID_STREAM: return "invalid_stream";
        case VulkanReplayStatus::CUSTOM_STREAM: return "custom_stream";
        case VulkanReplayStatus::INVALID_ARGUMENT: return "invalid_argument";
        case VulkanReplayStatus::REJECTED: return "rejected";
        case VulkanReplayStatus::BUSY: return "busy";
        case VulkanReplayStatus::RESOURCE_DESTROYED: return "resource_destroyed";
        case VulkanReplayStatus::STREAM_DESTROYED: return "stream_destroyed";
        case VulkanReplayStatus::NOT_SUPPORTED: return "not_supported";
        case VulkanReplayStatus::OUT_OF_RESOURCES: return "out_of_resources";
    }
    return "unknown";
}

enum class VulkanReplayUpdateStatus : uint32_t {
    // The only reachable answer of v1: a replay graph encodes its uniform bytes,
    // descriptor sets, push constants and argument-block offsets, so there is no
    // smaller granularity than a new recording. The graph is left byte-identical
    // and keeps replaying the recorded values.
    UNSUPPORTED_UPDATE = 0u,
    INVALID_HANDLE,
    RESOURCE_DESTROYED,
    STREAM_DESTROYED,
};

[[nodiscard]] constexpr luisa::string_view
vulkan_replay_update_status_name(VulkanReplayUpdateStatus status) noexcept {
    switch (status) {
        case VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE: return "unsupported_update";
        case VulkanReplayUpdateStatus::INVALID_HANDLE: return "invalid_handle";
        case VulkanReplayUpdateStatus::RESOURCE_DESTROYED: return "resource_destroyed";
        case VulkanReplayUpdateStatus::STREAM_DESTROYED: return "stream_destroyed";
    }
    return "unknown";
}

struct VulkanReplayRecordResult {
    // Valid iff `status == VulkanReplayStatus::SUCCESS`.
    uint64_t handle{0u};
    VulkanReplayStatus status{VulkanReplayStatus::SUCCESS};
    VulkanReplayRejectReason reason{VulkanReplayRejectReason::NONE};
    // Always filled when `status != SUCCESS`.
    luisa::string message;
};

struct VulkanReplaySubmitResult {
    VulkanReplayStatus status{VulkanReplayStatus::SUCCESS};
    VulkanReplayInvalidationReason invalidated_by{VulkanReplayInvalidationReason::NONE};
    // The stream timeline value this replay signalled (0 when nothing was
    // submitted). `stream.sync()` / an `Event` covers it like any other batch.
    uint64_t timeline_value{0u};
    luisa::string message;
};

struct VulkanReplayGraphStatus {
    VulkanReplayStatus status{VulkanReplayStatus::INVALID_HANDLE};
    VulkanReplayInvalidationReason invalidation{VulkanReplayInvalidationReason::NONE};
    uint64_t stream_handle{0u};
    uint32_t recorded_dispatches{0u};
    uint32_t recorded_buffer_copies{0u};
    // How many *distinct* interned layouts the graph keeps alive by id. A graph that
    // recorded several compute shaders with structurally different pipeline layouts
    // reports more than one pipeline layout here; the count is what makes "every
    // shader's layout is retained, not just the first" observable from outside.
    uint32_t interned_pipeline_layouts{0u};
    uint32_t interned_descriptor_set_layouts{0u};
    uint64_t submits{0u};
    uint64_t last_timeline_value{0u};
    luisa::string message;
};

// Declarative copy of the v1 policy, so a caller (and a test) can ask what is
// supported instead of discovering it through rejections. The `false` entries are
// part of the contract, not an omission.
struct VulkanReplayCapabilities {
    bool direct_dispatch{true};
    bool multiple_dispatch{true};
    bool buffer_copy{true};
    bool uniform_arguments{true};
    bool buffer_arguments{true};
    bool texture_arguments{false};
    bool bindless_arguments{false};
    bool acceleration_arguments{false};
    bool indirect_dispatch{false};
    bool update{false};
    bool replay_on{false};
    uint32_t max_commands{65536u};
    uint32_t max_dispatches{65536u};
};

// Runtime view of the Vulkan backend's replayable recording (plan P4).
class VulkanReplayExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "VulkanReplayExt";

    [[nodiscard]] virtual VulkanReplayCapabilities capabilities() const noexcept = 0;

    // Record `list` into one dedicated primary command buffer on `stream_handle`
    // (a `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` buffer that is never reset
    // or re-begun) without submitting it. Consumes the list.
    [[nodiscard]] virtual VulkanReplayRecordResult
    record(uint64_t stream_handle, CommandList &&list) noexcept = 0;

    // Submit the recorded batch once on its recording stream: one queue
    // submission through the stream's timeline path (throttle + reserved value +
    // `Event::_signal`), no re-record, no argument planning, no descriptor work.
    [[nodiscard]] virtual VulkanReplaySubmitResult
    replay(uint64_t replay_handle) noexcept = 0;

    // Always answers `UNSUPPORTED_UPDATE` for a live graph (see the enum) and
    // leaves the recording untouched. The parameter exists so a caller gets a
    // defined answer instead of a missing function; it consumes the list.
    [[nodiscard]] virtual VulkanReplayUpdateStatus
    update(uint64_t replay_handle, CommandList &&list) noexcept = 0;

    // Frees the graph. Refuses `BUSY` while its latest replay has not completed;
    // synchronize the stream and call again.
    [[nodiscard]] virtual VulkanReplayStatus destroy(uint64_t replay_handle) noexcept = 0;

    [[nodiscard]] virtual VulkanReplayGraphStatus
    status(uint64_t replay_handle) const noexcept = 0;

    // Number of live graphs of this device (diagnostics; never used to decide).
    [[nodiscard]] virtual size_t live_replay_count() const noexcept = 0;

protected:
    ~VulkanReplayExt() = default;
};

}// namespace luisa::compute