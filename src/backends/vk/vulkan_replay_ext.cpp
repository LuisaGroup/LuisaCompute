// Implementation of the Vulkan backend's replayable recording (plan P4, narrow
// v1). See include/luisa/backends/ext/vk_replay_ext.h for the contract and
// replay_plan.h for the admissibility policy this file applies.
//
// Shape of one call:
//
//   record()  consume the list -> list-level rejection -> per-command/argument
//             classification (pure) -> stream/config-ext validation -> retain the
//             interned layouts by id -> encode once into a SIMULTANEOUS_USE
//             primary command buffer through Stream::record_replay_batch (the
//             same prefix/suffix an ordinary batch uses) -> register the graph.
//             Nothing is submitted.
//
//   replay()  check the graph, look the stream up by handle (never a stored
//             pointer), and submit through Stream::submit_recorded: one reserved
//             timeline value, one Event::_signal (one native submission), the
//             ordinary SyncExt/NotifyEvt retirement. No re-record, no argument
//             planning, no descriptor work, no encode.
//
//   destroy() refuse BUSY while the latest replay has not retired on the host,
//             otherwise free the native buffer + state and release the interned
//             layouts by id.
#include "vulkan_replay_ext.h"

#include "buffer.h"
#include "device.h"
#include "replay_graph.h"
#include "replay_plan.h"
#include "shader.h"
#include "stream.h"
#include "submit_profile.h"

#include <luisa/core/stl/format.h>
// `CommandList::steal_presents()` returns a container of `SwapchainPresent` by
// value, so the complete type is needed here even though a list with a present is
// always rejected.
#include <luisa/runtime/swapchain.h>

namespace lc::vk {

namespace {

using namespace luisa::compute;

// Graph budgets. Exceeding one fails closed *before* a native object is created,
// instead of exhausting the descriptor pool in the middle of a recording.
constexpr auto kMaxCommands = uint32_t{65536u};
constexpr auto kMaxDispatches = uint32_t{65536u};
constexpr auto kMaxWatchedBuffers = size_t{65536u};

// The classifier's view of one shader handle. This is the only place the recorder
// dereferences a shader handle, and it reads exactly the facts the policy needs.
[[nodiscard]] detail::ReplayShaderFacts
replay_shader_facts(uint64_t handle) noexcept {
    detail::ReplayShaderFacts facts{};
    if (handle == 0u) { return facts; }
    auto *shader = reinterpret_cast<Shader *>(handle);
    switch (shader->shader_tag()) {
        case Shader::ShaderTag::kComputeShader:
            facts.tag = detail::ReplayShaderFacts::Tag::COMPUTE;
            break;
        case Shader::ShaderTag::kRasterShader:
            facts.tag = detail::ReplayShaderFacts::Tag::RASTER;
            break;
        case Shader::ShaderTag::kRayTracingShader:
            facts.tag = detail::ReplayShaderFacts::Tag::RAY_TRACING;
            break;
        default: facts.tag = detail::ReplayShaderFacts::Tag::UNKNOWN; break;
    }
    facts.forbidden.printers = !shader->printers().empty();
    facts.forbidden.constant_ubo = shader->has_constant_ubo();
    facts.forbidden.bindless = shader->use_buffer_bindless() ||
                               shader->use_tex2d_bindless() ||
                               shader->use_tex3d_bindless();
    return facts;
}

[[nodiscard]] uint64_t native_buffer_identity(uint64_t buffer_handle) noexcept {
    if (buffer_handle == 0u) { return 0u; }
    auto *buffer = reinterpret_cast<Buffer *>(buffer_handle);
    return detail::handle_to_uint64(buffer->vk_buffer());
}

void add_watched_buffer(vstd::vector<uint64_t> &watched,
                        uint64_t native_identity) noexcept {
    if (native_identity == 0u) { return; }
    for (auto existing : watched) {
        if (existing == native_identity) { return; }
    }
    watched.emplace_back(native_identity);
}

[[nodiscard]] VulkanReplayRecordResult
reject(VulkanReplayRejectReason reason, luisa::string_view detail_message) noexcept {
    VulkanReplayRecordResult result{};
    result.status = VulkanReplayStatus::REJECTED;
    result.reason = reason;
    result.message = luisa::format(
        "VulkanReplayExt::record() refused ({}): {}",
        vulkan_replay_reject_reason_name(reason), detail_message);
    submit_profile::record_replay_rejection();
    return result;
}

[[nodiscard]] VulkanReplaySubmitResult
submit_failure(VulkanReplayStatus status,
               VulkanReplayInvalidationReason invalidation,
               luisa::string_view detail_message) noexcept {
    VulkanReplaySubmitResult result{};
    result.status = status;
    result.invalidated_by = invalidation;
    result.message = luisa::format("VulkanReplayExt::replay() failed ({}): {}",
                                   vulkan_replay_status_name(status),
                                   detail_message);
    return result;
}

}// namespace

VulkanReplayCapabilities VkReplayExtImpl::capabilities() const noexcept {
    // A declarative copy of the policy in replay_plan.h: every `false` is part of
    // the v1 contract, not an omission.
    VulkanReplayCapabilities caps{};
    caps.max_commands = kMaxCommands;
    caps.max_dispatches = kMaxDispatches;
    return caps;
}

VulkanReplayRecordResult
VkReplayExtImpl::record(uint64_t stream_handle, CommandList &&list) noexcept {
    // (1) Consume the list first, on every path. The caller handed us an rvalue
    // reference to *their* list object, so leaving it non-empty would trip
    // `~CommandList`'s "did you forget to commit?" assertion. The destructor
    // callbacks are taken too: this is where the list dies, so this is where they
    // would have run - and they never run during a replay.
    auto commands = list.steal_commands();
    auto callbacks = list.steal_callbacks();
    auto presents = list.steal_presents();
    auto dtor_callbacks = list.steal_dtor_callbacks();
    auto run_dtor_callbacks = vstd::scope_exit([&dtor_callbacks] {
        for (auto &callback : dtor_callbacks) {
            callback();
        }
    });

    // (2) List-level rules. They come before any command inspection so the reason
    // names the real problem, and before any native allocation.
    if (!callbacks.empty()) {
        return reject(VulkanReplayRejectReason::LIST_CALLBACKS,
                      "a replayed batch would run the callback after every replay, "
                      "which is a host-side dynamic channel");
    }
    if (!presents.empty()) {
        return reject(VulkanReplayRejectReason::LIST_PRESENTS,
                      "present() is not a replayable operation");
    }
    if (!dtor_callbacks.empty()) {
        return reject(VulkanReplayRejectReason::LIST_DTOR_CALLBACKS,
                      "the list carries destructor callbacks; they were run at "
                      "record time instead of behind the recording");
    }
    if (commands.empty()) {
        return reject(VulkanReplayRejectReason::EMPTY_LIST,
                      "there is nothing to record");
    }

    // (3) Command- and argument-level classification (pure, no device work).
    luisa::vector<detail::ReplayCommandFacts> facts;
    facts.reserve(commands.size());
    vstd::vector<uint64_t> watched_buffers;
    for (auto &&command : commands) {
        auto command_facts = detail::ReplayCommandFacts{};
        command_facts.tag = command->tag();
        if (command_facts.tag == Command::Tag::EShaderDispatchCommand) {
            auto *dispatch = static_cast<ShaderDispatchCommand *>(command.get());
            command_facts.indirect = dispatch->is_indirect();
            if (!command_facts.indirect) {
                auto shader_handle = dispatch->handle();
                command_facts.shader = replay_shader_facts(shader_handle);
                if (command_facts.shader.tag ==
                    detail::ReplayShaderFacts::Tag::COMPUTE) {
                    // The argument tables of the accepted dispatch. Rejecting here
                    // (rather than while encoding) keeps every refusal free of
                    // native resources.
                    //
                    // Two tables have to be inspected, not one: the encoder keeps a
                    // dispatch's *runtime* arguments (`command->arguments()`, what
                    // the caller passed to `shader(...)`) separate from the shader's
                    // *captured* ones (`shader->captured()`, resources the kernel
                    // closed over), and it validates
                    // `captured.size() + runtime.size() == saved_arguments.size()`.
                    // A rule applied to only one of them would let a captured
                    // texture/bindless/acceleration argument through and would leave
                    // a captured buffer unwatched, so its destruction would not
                    // invalidate the graph.
                    auto *shader = reinterpret_cast<Shader *>(shader_handle);
                    // One rule, applied to both tables. `add_watched_buffer`
                    // deduplicates by native identity, so a buffer that appears in
                    // both tables (or twice in one) is watched once, and a captured
                    // UNIFORM needs no watching at all: its bytes are baked into the
                    // command's argument block exactly like a runtime uniform's.
                    auto classify_and_watch = [&](auto arguments) noexcept {
                        for (auto &&argument : arguments) {
                            if (auto reason =
                                    detail::classify_replay_argument(argument.tag);
                                reason != VulkanReplayRejectReason::NONE) {
                                return reason;
                            }
                            if (argument.tag == Argument::Tag::BUFFER) {
                                add_watched_buffer(
                                    watched_buffers,
                                    native_buffer_identity(argument.buffer.handle));
                            }
                        }
                        return VulkanReplayRejectReason::NONE;
                    };
                    if (auto reason = classify_and_watch(dispatch->arguments());
                        reason != VulkanReplayRejectReason::NONE) {
                        return reject(reason,
                                      "a recorded dispatch may carry only "
                                      "UNIFORM and BUFFER arguments in v1 "
                                      "(runtime argument)");
                    }
                    if (auto reason = classify_and_watch(shader->captured());
                        reason != VulkanReplayRejectReason::NONE) {
                        return reject(reason,
                                      "a recorded dispatch may carry only "
                                      "UNIFORM and BUFFER arguments in v1 "
                                      "(captured argument)");
                    }
                }
            }
        } else if (command_facts.tag == Command::Tag::EBufferCopyCommand) {
            auto *copy = static_cast<BufferCopyCommand *>(command.get());
            add_watched_buffer(watched_buffers,
                               native_buffer_identity(copy->src_handle()));
            add_watched_buffer(watched_buffers,
                               native_buffer_identity(copy->dst_handle()));
        }
        facts.emplace_back(command_facts);
    }
    auto plan = detail::plan_replay_record(
        luisa::span{facts},
        detail::ReplayRecordLimits{
            .max_commands = kMaxCommands,
            .max_dispatches = kMaxDispatches});
    if (!plan.ok()) {
        return reject(plan.reason, "the command list is outside the v1 replay contract");
    }
    if (watched_buffers.size() > kMaxWatchedBuffers) {
        return reject(VulkanReplayRejectReason::OUT_OF_RESOURCES,
                      "the recording names more buffers than a graph may watch");
    }

    // (4) Stream validation. The graph stores the handle value only; this lookup
    // happens once, now.
    if (stream_handle == 0u) {
        auto result = VulkanReplayRecordResult{};
        result.status = VulkanReplayStatus::INVALID_ARGUMENT;
        result.message = "VulkanReplayExt::record(): null stream handle";
        submit_profile::record_replay_rejection();
        return result;
    }
    auto *stream = _device->find_stream(stream_handle);
    if (stream == nullptr) {
        if (_device->find_custom_stream(stream_handle) != nullptr) {
            auto result = VulkanReplayRecordResult{};
            result.status = VulkanReplayStatus::CUSTOM_STREAM;
            result.message = "VulkanReplayExt::record(): an extension-owned stream "
                             "cannot record a replay";
            submit_profile::record_replay_rejection();
            return result;
        }
        auto result = VulkanReplayRecordResult{};
        result.status = VulkanReplayStatus::INVALID_STREAM;
        result.message = "VulkanReplayExt::record(): the stream handle is not a "
                         "stream of this device";
        submit_profile::record_replay_rejection();
        return result;
    }

    // (5) A config ext that publishes per-submission native state cannot be
    // recorded: the recorded barrier structs would describe one submission's
    // state only. Fail closed instead of replaying stale layouts (plan §8.3).
    if (auto *config_ext = _device->config_ext(); config_ext != nullptr) {
        if (!config_ext->before_states(stream_handle).empty() ||
            !config_ext->after_states(stream_handle).empty()) {
            return reject(VulkanReplayRejectReason::CONFIG_EXT_STATE_INJECTION,
                          "the stream's VulkanDeviceConfigExt publishes "
                          "per-submission resource states");
        }
    }

    // (6) Retain the interned layouts **by id** before recording. P1.1 destroys an
    // interned layout when its last shader releases it, and this recording (and
    // the descriptor sets allocated from those layouts) outlives the shader, so
    // relying on the shader's own reference would be a use-after-free.
    auto &interner = _device->layout_interner();
    vstd::vector<uint64_t> desc_set_layout_ids;
    // Every dispatch shader's interned pipeline layout, deduplicated. A graph may
    // hold several compute shaders with structurally different pipeline layouts, so
    // this cannot be a single id: destroying a later shader would otherwise destroy a
    // layout (and, through it, its descriptor-set layouts) that the recorded command
    // buffer still binds.
    vstd::vector<uint64_t> pipeline_layout_ids;
    auto retained_owned_by_graph = false;
    auto release_retained = vstd::scope_exit([&] {
        if (retained_owned_by_graph) {
            // ownership moved to the graph, which releases the same ids exactly
            // once in `~ReplayGraph`
            return;
        }
        auto vk_device = _device->logic_device();
        for (auto id : pipeline_layout_ids) {
            interner.release_pipeline_layout(id, vk_device);
        }
        for (auto id : desc_set_layout_ids) {
            // Destruction-recording release: a final reference dropped here evicts
            // the cached descriptor-set buckets of this layout.
            (void)_device->release_descriptor_set_layout(id);
        }
    });
    for (auto &&command : commands) {
        if (command->tag() != Command::Tag::EShaderDispatchCommand) { continue; }
        auto *dispatch = static_cast<ShaderDispatchCommand *>(command.get());
        if (dispatch->is_indirect()) { continue; }
        auto *shader = reinterpret_cast<Shader *>(dispatch->handle());
        for (auto &&interned : shader->interned_desc_set_layouts()) {
            if (!interned) { continue; }
            auto known = false;
            for (auto id : desc_set_layout_ids) {
                if (id == interned.id) {
                    known = true;
                    break;
                }
            }
            if (known) { continue; }
            if (interner.retain_descriptor_set_layout(interned.id) == 0u) {
                return reject(VulkanReplayRejectReason::DISPATCH_UNKNOWN_SHADER,
                              "an interned descriptor-set layout of a recorded "
                              "shader is no longer live");
            }
            desc_set_layout_ids.emplace_back(interned.id);
        }
        if (auto interned = shader->interned_pipeline_layout(); interned) {
            auto known = false;
            for (auto id : pipeline_layout_ids) {
                if (id == interned.id) {
                    known = true;
                    break;
                }
            }
            if (!known) {
                if (interner.retain_pipeline_layout(interned.id) == 0u) {
                    return reject(VulkanReplayRejectReason::DISPATCH_UNKNOWN_SHADER,
                                  "an interned pipeline layout of a recorded "
                                  "shader is no longer live");
                }
                pipeline_layout_ids.emplace_back(interned.id);
            }
        }
    }

    // (7) Encode once. `record_replay_batch` takes the stream's dispatch mutex and
    // runs the ordinary record prefix/suffix, so barriers, config-ext states,
    // descriptor/argument encoding and the trailing restore are the same ones an
    // ordinary batch emits. It does not submit anything.
    auto batch = stream->record_replay_batch(commands);
    if (batch.empty()) {
        auto result = VulkanReplayRecordResult{};
        result.status = VulkanReplayStatus::NOT_SUPPORTED;
        result.message = "VulkanReplayExt::record(): the stream produced no "
                         "command buffer";
        submit_profile::record_replay_rejection();
        return result;
    }

    // (8) Hand the native state to a graph and register it. From here on the
    // layout references belong to the graph, so the guard above is dismissed.
    auto *graph = new detail::ReplayGraph{};
    graph->device = _device;
    graph->stream_handle = stream_handle;
    graph->dispatches = plan.dispatches;
    graph->buffer_copies = plan.buffer_copies;
    graph->batch = std::move(batch);
    graph->watched_buffers = std::move(watched_buffers);
    graph->desc_set_layout_ids = std::move(desc_set_layout_ids);
    graph->pipeline_layout_ids = std::move(pipeline_layout_ids);
    retained_owned_by_graph = true;
    auto handle = _device->replay_registry().add(graph);
    if (handle == 0u) {
        // `~ReplayGraph` frees the native batch and gives the layout references
        // back, so a failed registration leaks nothing.
        delete graph;
        auto result = VulkanReplayRecordResult{};
        result.status = VulkanReplayStatus::OUT_OF_RESOURCES;
        result.message = "VulkanReplayExt::record(): the replay handle space is "
                         "exhausted";
        submit_profile::record_replay_rejection();
        return result;
    }
    // Counter-only: this thread's mirror was already folded by
    // `record_replay_batch`; nothing else runs here.
    VulkanReplayRecordResult result{};
    result.handle = handle;
    result.status = VulkanReplayStatus::SUCCESS;
    LUISA_INFO(
        "Vulkan replay: recorded graph 0x{:x} on stream 0x{:x} "
        "({} dispatch(es), {} buffer copy(ies), {} watched buffer(s), "
        "{} interned descriptor-set layout(s), {} interned pipeline layout(s)).",
        handle, stream_handle, plan.dispatches, plan.buffer_copies,
        graph->watched_buffers.size(),
        graph->desc_set_layout_ids.size(), graph->pipeline_layout_ids.size());
    return result;
}

VulkanReplaySubmitResult
VkReplayExtImpl::replay(uint64_t replay_handle) noexcept {
    // Phase 3C: the hot path of a device-greedy decode loop (one call per token).
    // `prepare_replay` carries exactly the four fields a submission needs, so a
    // successful replay copies no `luisa::string` and performs no allocation in the
    // registry; the full `snapshot()` (which copies the diagnostic message and the
    // counters) stays on the diagnostic entry points. Every safety check below is
    // the same one the snapshot-based version performed.
    auto prepared = _device->replay_registry().prepare_replay(replay_handle);
    if (!prepared.found) {
        return submit_failure(VulkanReplayStatus::INVALID_HANDLE,
                              VulkanReplayInvalidationReason::NONE,
                              "unknown replay handle");
    }
    if (prepared.invalidation != VulkanReplayInvalidationReason::NONE) {
        return submit_failure(
            prepared.invalidation == VulkanReplayInvalidationReason::RESOURCE_DESTROYED ?
                VulkanReplayStatus::RESOURCE_DESTROYED :
                VulkanReplayStatus::STREAM_DESTROYED,
            prepared.invalidation,
            prepared.message);
    }
    if (prepared.command_buffer == 0u) {
        return submit_failure(VulkanReplayStatus::INVALID_HANDLE,
                              VulkanReplayInvalidationReason::NONE,
                              "the recording no longer owns a command buffer");
    }
    // The stream is looked up by handle on every submission; the graph itself
    // stores no pointer, so a graph that outlives its stream reports an error
    // instead of dereferencing released memory.
    auto *stream = _device->find_stream(prepared.stream_handle);
    if (stream == nullptr) {
        return submit_failure(VulkanReplayStatus::STREAM_DESTROYED,
                              VulkanReplayInvalidationReason::STREAM_DESTROYED,
                              "the recording stream no longer exists");
    }
    auto command_buffer = reinterpret_cast<VkCommandBuffer>(
        static_cast<uintptr_t>(prepared.command_buffer));
    auto timeline_value = stream->submit_recorded(
        command_buffer, _device->inqueue_limit());
    _device->replay_registry().record_submission(replay_handle, timeline_value);
    VulkanReplaySubmitResult result{};
    result.status = VulkanReplayStatus::SUCCESS;
    result.timeline_value = timeline_value;
    return result;
}

VulkanReplayUpdateStatus
VkReplayExtImpl::update(uint64_t replay_handle, CommandList &&list) noexcept {
    // The list is destroyed here whatever the answer, exactly like in record().
    auto commands = list.steal_commands();
    auto callbacks = list.steal_callbacks();
    auto presents = list.steal_presents();
    auto dtor_callbacks = list.steal_dtor_callbacks();
    auto run_dtor_callbacks = vstd::scope_exit([&dtor_callbacks] {
        for (auto &callback : dtor_callbacks) {
            callback();
        }
    });
    static_cast<void>(commands);
    static_cast<void>(callbacks);
    static_cast<void>(presents);
    // Every update() call is a refusal in v1, so it is counted before the handle
    // is examined: the counter describes caller behaviour, not graph state.
    submit_profile::record_replay_update_refusal();
    auto snapshot = _device->replay_registry().snapshot(replay_handle);
    if (!snapshot.found) {
        return VulkanReplayUpdateStatus::INVALID_HANDLE;
    }
    switch (snapshot.invalidation) {
        case VulkanReplayInvalidationReason::NONE:
            // A replay graph encodes its uniform bytes, descriptor sets, push
            // constants and argument-block offsets; there is no granularity below
            // a new recording. The graph is left byte-identical.
            return VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE;
        case VulkanReplayInvalidationReason::RESOURCE_DESTROYED:
            return VulkanReplayUpdateStatus::RESOURCE_DESTROYED;
        case VulkanReplayInvalidationReason::STREAM_DESTROYED:
        case VulkanReplayInvalidationReason::DEVICE_TEARDOWN:
            return VulkanReplayUpdateStatus::STREAM_DESTROYED;
    }
    return VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE;
}

VulkanReplayStatus VkReplayExtImpl::destroy(uint64_t replay_handle) noexcept {
    auto snapshot = _device->replay_registry().snapshot(replay_handle);
    if (!snapshot.found) { return VulkanReplayStatus::INVALID_HANDLE; }
    if (snapshot.submits != 0u) {
        // BUSY while the latest replay has not retired on the host: freeing the
        // command buffer while a submission may still refer to it is illegal, and
        // waiting here would turn a graph destruction into an unbounded stall.
        // The check looks the stream up *outside* the registry lock, so it cannot
        // form a lock cycle with `Device::destroy_stream` (which holds
        // `_stream_mtx` and then invalidates through the registry).
        if (auto *stream = _device->find_stream(snapshot.stream_handle);
            stream != nullptr &&
            !stream->submission_complete(snapshot.last_timeline_value)) {
            return VulkanReplayStatus::BUSY;
        }
        // A stream that is already gone synchronized in its destructor, so its
        // replays are complete by definition.
    }
    auto *graph = _device->replay_registry().take(replay_handle);
    if (graph == nullptr) {
        // Lost a race with another destroy() (or with device teardown).
        return VulkanReplayStatus::INVALID_HANDLE;
    }
    // Frees the recorded command buffer, its state (pools, descriptor sets,
    // staging, retained pipelines) and gives the interned layout references back.
    delete graph;
    return VulkanReplayStatus::SUCCESS;
}

VulkanReplayGraphStatus VkReplayExtImpl::status(uint64_t replay_handle) const noexcept {
    auto snapshot = _device->replay_registry().snapshot(replay_handle);
    VulkanReplayGraphStatus result{};
    if (!snapshot.found) {
        result.status = VulkanReplayStatus::INVALID_HANDLE;
        result.message = "unknown replay handle";
        return result;
    }
    result.invalidation = snapshot.invalidation;
    result.stream_handle = snapshot.stream_handle;
    result.recorded_dispatches = snapshot.dispatches;
    result.recorded_buffer_copies = snapshot.buffer_copies;
    result.interned_pipeline_layouts = snapshot.interned_pipeline_layouts;
    result.interned_descriptor_set_layouts = snapshot.interned_descriptor_set_layouts;
    result.submits = snapshot.submits;
    result.last_timeline_value = snapshot.last_timeline_value;
    switch (snapshot.invalidation) {
        case VulkanReplayInvalidationReason::NONE:
            result.status = VulkanReplayStatus::SUCCESS;
            break;
        case VulkanReplayInvalidationReason::RESOURCE_DESTROYED:
            result.status = VulkanReplayStatus::RESOURCE_DESTROYED;
            result.message = snapshot.message;
            break;
        case VulkanReplayInvalidationReason::STREAM_DESTROYED:
            result.status = VulkanReplayStatus::STREAM_DESTROYED;
            result.message = snapshot.message;
            break;
        case VulkanReplayInvalidationReason::DEVICE_TEARDOWN:
            // Every graph is deleted at device teardown, so this is unreachable;
            // answer it honestly anyway instead of reporting success.
            result.status = VulkanReplayStatus::NOT_SUPPORTED;
            result.message = snapshot.message;
            break;
    }
    return result;
}

size_t VkReplayExtImpl::live_replay_count() const noexcept {
    return _device->replay_registry().size();
}

}// namespace lc::vk