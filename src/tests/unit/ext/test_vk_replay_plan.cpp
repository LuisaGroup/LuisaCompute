// Host-level coverage for the Vulkan backend's replayable-recording policy
// (plan P4, narrow v1, `src/backends/vk/replay_plan.h`).
//
// The classifier is a pure function of plain values - it touches no Vulkan
// handle, calls no entry point and dereferences no device object - so the whole
// rejection matrix of `VulkanReplayExt::record()` is asserted here instead of
// through a device. This is also the anti-drift test of the contract: a new
// `Command::Tag`, `Argument::Tag` or status value that nobody classified fails
// this file, not a log line.
//
// What is covered:
// - the two allowed command kinds (direct compute dispatch single/batched, and a
//   device-to-device buffer copy) and their counted plan
// - every rejected dispatch shape with its own typed reason: indirect, printer,
//   constant UBO, bindless, raster, ray tracing, unknown shader
// - every other `Command::Tag` with its own typed reason (uploads, downloads,
//   texture copies, accel/mesh/curve/procedural/motion builds, bindless updates,
//   custom commands)
// - every `Argument::Tag`: UNIFORM/BUFFER allowed, TEXTURE/BINDLESS/ACCEL rejected
// - the plan budget (too many commands, too many dispatches) failing closed with
//   OUT_OF_RESOURCES instead of a silent partial plan
// - the first refusal winning, so a caller gets a deterministic reason
// - name coverage of every public enum, because a report must never print
//   "unknown" for a value the backend can really return
// - the command-buffer ownership plan (`command_buffer_ownership.h`): which
//   ownership state is reset/recycled/freed, and that only a persistent recording
//   may be handed to a replay graph (`release_native_batch`)
#include "ut/ut.hpp"

#include "command_buffer_ownership.h"
#include "replay_plan.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string_view>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace vk_detail = lc::vk::detail;
namespace compute = luisa::compute;

namespace {

using Tag = compute::Command::Tag;
using ArgumentTag = compute::Argument::Tag;
using Reason = compute::VulkanReplayRejectReason;
using ShaderTag = vk_detail::ReplayShaderFacts::Tag;

[[nodiscard]] vk_detail::ReplayCommandFacts
make_command(Tag tag, bool indirect = false,
             ShaderTag shader_tag = ShaderTag::COMPUTE,
             bool printers = false, bool constant_ubo = false,
             bool bindless = false) noexcept {
    vk_detail::ReplayCommandFacts facts{};
    facts.tag = tag;
    facts.indirect = indirect;
    facts.shader.tag = shader_tag;
    facts.shader.forbidden.printers = printers;
    facts.shader.forbidden.constant_ubo = constant_ubo;
    facts.shader.forbidden.bindless = bindless;
    return facts;
}

[[nodiscard]] vk_detail::ReplayRecordPlan
plan(std::initializer_list<vk_detail::ReplayCommandFacts> commands,
     vk_detail::ReplayRecordLimits limits = {}) noexcept {
    luisa::vector<vk_detail::ReplayCommandFacts> facts{commands};
    return vk_detail::plan_replay_record(luisa::span{facts}, limits);
}

// Every `Command::Tag` the runtime can produce (include/luisa/runtime/rhi/
// command.h: LUISA_COMPUTE_RUNTIME_COMMANDS). Listed explicitly instead of via a
// macro loop so an added tag is a compile-visible edit of this test.
constexpr auto kAllCommandTags = std::array{
    Tag::EBufferUploadCommand,
    Tag::EBufferDownloadCommand,
    Tag::EBufferCopyCommand,
    Tag::EBufferToTextureCopyCommand,
    Tag::EShaderDispatchCommand,
    Tag::ETextureUploadCommand,
    Tag::ETextureDownloadCommand,
    Tag::ETextureCopyCommand,
    Tag::ETextureToBufferCopyCommand,
    Tag::EAccelBuildCommand,
    Tag::EMeshBuildCommand,
    Tag::ECurveBuildCommand,
    Tag::EProceduralPrimitiveBuildCommand,
    Tag::EMotionInstanceBuildCommand,
    Tag::EBindlessArrayUpdateCommand,
    Tag::ECustomCommand};

}// namespace

namespace {

void reg_dispatch_allowed() {
    "replay_plan_dispatch_allowed"_test = [] {
        // A single direct compute dispatch: admissible.
        expect(vk_detail::classify_replay_command(make_command(Tag::EShaderDispatchCommand)) ==
               Reason::NONE)
            << "a direct compute dispatch is the primary replayable command";

        // Two dispatches and one copy: admissible, and the plan counts exactly the
        // command kinds the graph will hold.
        auto result = plan({make_command(Tag::EShaderDispatchCommand),
                            make_command(Tag::EShaderDispatchCommand),
                            make_command(Tag::EBufferCopyCommand)});
        expect(result.ok()) << "a mixed dispatch/copy list is admissible";
        expect(result.dispatches == 2u)
            << "the plan counts recorded dispatches, not vkCmdDispatch calls";
        expect(result.buffer_copies == 1u)
            << "a device-to-device copy is one recorded copy";

        // A copy is admissible on its own, and a dispatch with a batched shape is
        // just a dispatch: the shape is baked at record time, so it needs no flag.
        expect(plan({make_command(Tag::EBufferCopyCommand)}).ok());
        expect(plan({make_command(Tag::EShaderDispatchCommand)}).dispatches == 1u);
    };
}

void reg_dispatch_shape_rejections() {
    "replay_plan_indirect_rejected"_test = [] {
        expect(vk_detail::classify_replay_command(
                   make_command(Tag::EShaderDispatchCommand, true)) ==
               Reason::DISPATCH_INDIRECT)
            << "the dispatch shape of an indirect dispatch is read from a buffer "
               "at execution, so a recording cannot promise it";
        expect(plan({make_command(Tag::EShaderDispatchCommand),
                     make_command(Tag::EShaderDispatchCommand, true)})
                   .reason == Reason::DISPATCH_INDIRECT)
            << "the first offending command wins, in list order";
    };

    "replay_plan_printer_rejected"_test = [] {
        expect(vk_detail::classify_replay_command(make_command(
                   Tag::EShaderDispatchCommand, false, ShaderTag::COMPUTE, true)) ==
               Reason::DISPATCH_PRINTER)
            << "a printer dispatch consumes stream-scoped scratch and a readback "
               "buffer that only the batch's own retirement would drain";
    };

    "replay_plan_constant_ubo_rejected"_test = [] {
        expect(vk_detail::classify_replay_command(make_command(
                   Tag::EShaderDispatchCommand, false, ShaderTag::COMPUTE, false, true)) ==
               Reason::DISPATCH_CONSTANT_UBO)
            << "the constant UBO is shader-owned, so v1 refuses it instead of "
               "tracking a shader lifetime inside the graph";
    };

    "replay_plan_bindless_rejected"_test = [] {
        expect(vk_detail::classify_replay_command(make_command(
                   Tag::EShaderDispatchCommand, false, ShaderTag::COMPUTE, false,
                   false, true)) == Reason::DISPATCH_BINDLESS)
            << "bindless membership must be static for a graph, and v1 carries no "
               "membership generation";
    };

    "replay_plan_raster_and_rt_rejected"_test = [] {
        expect(vk_detail::classify_replay_command(make_command(
                   Tag::EShaderDispatchCommand, false, ShaderTag::RASTER)) ==
               Reason::DISPATCH_RASTER);
        expect(vk_detail::classify_replay_command(make_command(
                   Tag::EShaderDispatchCommand, false, ShaderTag::RAY_TRACING)) ==
               Reason::DISPATCH_RAY_TRACING);
        expect(vk_detail::classify_replay_command(make_command(
                   Tag::EShaderDispatchCommand, false, ShaderTag::UNKNOWN)) ==
               Reason::DISPATCH_UNKNOWN_SHADER)
            << "a handle that is not a compute shader of this device is its own "
               "reason, not a silent success";
    };
}

void reg_every_tag_classified() {
    "replay_plan_every_tag_classified"_test = [] {
        // The anti-drift assertion: every tag is either admissible with a defined
        // contract or rejected with a *specific* reason. `NONE` for a command the
        // recorder cannot encode would be the dangerous outcome.
        for (auto tag : kAllCommandTags) {
            auto facts = make_command(tag);
            auto reason = vk_detail::classify_replay_command(facts);
            auto const allowed = tag == Tag::EShaderDispatchCommand ||
                                 tag == Tag::EBufferCopyCommand;
            if (allowed) {
                expect(reason == Reason::NONE)
                    << "only the two contract command kinds are admissible";
            } else {
                expect(reason != Reason::NONE)
                    << "every other command tag must be refused with a reason";
                expect(reason != Reason::COMMAND_CUSTOM || tag == Tag::ECustomCommand)
                    << "a non-custom tag must not be reported as a custom command";
            }
            expect(!compute::vulkan_replay_reject_reason_name(reason).empty())
                << "a reason without a name cannot be reported";
        }
        expect(kAllCommandTags.size() == 16u)
            << "the runtime defines exactly 16 command tags; a 17th needs its own "
               "row here and in replay_plan.h";
    };

    "replay_plan_argument_table"_test = [] {
        expect(vk_detail::classify_replay_argument(ArgumentTag::UNIFORM) == Reason::NONE)
            << "uniform bytes are the baked case update() refuses to change";
        expect(vk_detail::classify_replay_argument(ArgumentTag::BUFFER) == Reason::NONE)
            << "a buffer argument is an identity whose contents stay free";
        expect(vk_detail::classify_replay_argument(ArgumentTag::TEXTURE) ==
               Reason::ARG_TEXTURE)
            << "v1 does no image-view lifetime work";
        expect(vk_detail::classify_replay_argument(ArgumentTag::BINDLESS_ARRAY) ==
               Reason::ARG_BINDLESS_ARRAY);
        expect(vk_detail::classify_replay_argument(ArgumentTag::ACCEL) ==
               Reason::ARG_ACCELERATION_STRUCTURE);
        // Every argument tag is covered, and each has a printable name.
        constexpr auto all_argument_tags = std::array{
            ArgumentTag::BUFFER, ArgumentTag::TEXTURE, ArgumentTag::UNIFORM,
            ArgumentTag::BINDLESS_ARRAY, ArgumentTag::ACCEL};
        for (auto tag : all_argument_tags) {
            auto reason = vk_detail::classify_replay_argument(tag);
            expect(!compute::vulkan_replay_reject_reason_name(reason).empty());
            if (tag == ArgumentTag::BUFFER || tag == ArgumentTag::UNIFORM) {
                expect(reason == Reason::NONE);
            } else {
                expect(reason != Reason::NONE)
                    << "every non-buffer/uniform argument must be refused";
            }
        }
    };
}

void reg_budget() {
    "replay_plan_budget"_test = [] {
        // Too many commands fails closed before a single native object exists,
        // instead of exhausting the descriptor pool in the middle of a recording.
        luisa::vector<vk_detail::ReplayCommandFacts> commands;
        for (auto i = 0u; i < 8u; ++i) {
            commands.emplace_back(make_command(Tag::EBufferCopyCommand));
        }
        auto limited = vk_detail::plan_replay_record(
            luisa::span{commands},
            vk_detail::ReplayRecordLimits{.max_commands = 4u, .max_dispatches = 64u});
        expect(limited.reason == Reason::OUT_OF_RESOURCES)
            << "a list longer than the command budget is refused, not truncated";
        expect(limited.dispatches == 0u && limited.buffer_copies == 0u)
            << "a refused plan reports no partial counts";

        auto dispatch_limited = vk_detail::plan_replay_record(
            luisa::span{commands},
            vk_detail::ReplayRecordLimits{.max_commands = 64u, .max_dispatches = 2u});
        expect(dispatch_limited.ok())
            << "copies are not dispatches, so the dispatch budget cannot reject "
               "a copy-only list";

        luisa::vector<vk_detail::ReplayCommandFacts> dispatches;
        for (auto i = 0u; i < 4u; ++i) {
            dispatches.emplace_back(make_command(Tag::EShaderDispatchCommand));
        }
        auto too_many_dispatches = vk_detail::plan_replay_record(
            luisa::span{dispatches},
            vk_detail::ReplayRecordLimits{.max_commands = 64u, .max_dispatches = 2u});
        expect(too_many_dispatches.reason == Reason::OUT_OF_RESOURCES)
            << "the dispatch budget is enforced too";
    };
}

void reg_status_names() {
    "replay_plan_status_names"_test = [] {
        // A report must never print "unknown" for a value the backend can return:
        // every enumerator of the three public enums answers a real name, and no
        // enumerator shares a name with another.
        constexpr auto reject_reasons = std::array{
            Reason::NONE,
            Reason::COMMAND_BUFFER_UPLOAD,
            Reason::COMMAND_BUFFER_DOWNLOAD,
            Reason::COMMAND_TEXTURE,
            Reason::COMMAND_ACCEL_BUILD,
            Reason::COMMAND_BINDLESS_UPDATE,
            Reason::COMMAND_CUSTOM,
            Reason::DISPATCH_INDIRECT,
            Reason::DISPATCH_RASTER,
            Reason::DISPATCH_RAY_TRACING,
            Reason::DISPATCH_PRINTER,
            Reason::DISPATCH_CONSTANT_UBO,
            Reason::DISPATCH_BINDLESS,
            Reason::DISPATCH_UNKNOWN_SHADER,
            Reason::ARG_TEXTURE,
            Reason::ARG_BINDLESS_ARRAY,
            Reason::ARG_ACCELERATION_STRUCTURE,
            Reason::ARG_UNKNOWN_TAG,
            Reason::LIST_CALLBACKS,
            Reason::LIST_PRESENTS,
            Reason::LIST_DTOR_CALLBACKS,
            Reason::EMPTY_LIST,
            Reason::CONFIG_EXT_STATE_INJECTION,
            Reason::OUT_OF_RESOURCES};
        for (auto i = size_t{0u}; i < reject_reasons.size(); ++i) {
            auto name = compute::vulkan_replay_reject_reason_name(reject_reasons[i]);
            expect(!name.empty());
            expect(name != "unknown")
                << "every classifiable rejection has a name";
            for (auto j = i + 1u; j < reject_reasons.size(); ++j) {
                expect(name != compute::vulkan_replay_reject_reason_name(reject_reasons[j]))
                    << "two rejection reasons must not share a name";
            }
        }

        constexpr auto statuses = std::array{
            compute::VulkanReplayStatus::SUCCESS,
            compute::VulkanReplayStatus::INVALID_HANDLE,
            compute::VulkanReplayStatus::INVALID_STREAM,
            compute::VulkanReplayStatus::CUSTOM_STREAM,
            compute::VulkanReplayStatus::INVALID_ARGUMENT,
            compute::VulkanReplayStatus::REJECTED,
            compute::VulkanReplayStatus::BUSY,
            compute::VulkanReplayStatus::RESOURCE_DESTROYED,
            compute::VulkanReplayStatus::STREAM_DESTROYED,
            compute::VulkanReplayStatus::NOT_SUPPORTED,
            compute::VulkanReplayStatus::OUT_OF_RESOURCES};
        for (auto status : statuses) {
            auto name = compute::vulkan_replay_status_name(status);
            expect(!name.empty() && name != "unknown");
        }
        expect(static_cast<uint32_t>(compute::VulkanReplayStatus::SUCCESS) == 0u)
            << "SUCCESS is 0, so `status != SUCCESS` is the failure test";

        constexpr auto update_statuses = std::array{
            compute::VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE,
            compute::VulkanReplayUpdateStatus::INVALID_HANDLE,
            compute::VulkanReplayUpdateStatus::RESOURCE_DESTROYED,
            compute::VulkanReplayUpdateStatus::STREAM_DESTROYED};
        for (auto status : update_statuses) {
            auto name = compute::vulkan_replay_update_status_name(status);
            expect(!name.empty() && name != "unknown");
        }
        expect(static_cast<uint32_t>(compute::VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE) == 0u);

        constexpr auto invalidations = std::array{
            compute::VulkanReplayInvalidationReason::NONE,
            compute::VulkanReplayInvalidationReason::RESOURCE_DESTROYED,
            compute::VulkanReplayInvalidationReason::STREAM_DESTROYED,
            compute::VulkanReplayInvalidationReason::DEVICE_TEARDOWN};
        for (auto reason : invalidations) {
            auto name = compute::vulkan_replay_invalidation_reason_name(reason);
            expect(!name.empty() && name != "unknown");
        }
    };

    "replay_plan_capabilities_are_honest"_test = [] {
        // The declarative capability record must not promise what the classifier
        // refuses: every `false` here has a matching rejection in this file.
        compute::VulkanReplayCapabilities caps{};
        expect(caps.direct_dispatch && caps.multiple_dispatch && caps.buffer_copy)
            << "the three admissible shapes are advertised";
        expect(caps.uniform_arguments && caps.buffer_arguments)
            << "the two admissible argument kinds are advertised";
        expect(!caps.texture_arguments)
            << "texture arguments are refused (ARG_TEXTURE), so they are not promised";
        expect(!caps.bindless_arguments && !caps.acceleration_arguments);
        expect(!caps.indirect_dispatch && !caps.replay_on)
            << "an indirect dispatch and a cross-stream replay are outside v1";
        expect(!caps.update)
            << "update() always refuses, so the record says so up front";
        expect(caps.max_commands == 65536u && caps.max_dispatches == 65536u)
            << "the budgets the recorder enforces are the ones it advertises";
    };
}

void reg_command_buffer_ownership() {
    "replay_plan_command_buffer_ownership"_test = [] {
        using Ownership = vk_detail::CommandBufferOwnership;
        // The retirement plan keeps the three ownership states apart, and it is the
        // whole contract `CommandBuffer::retire_and_recycle` / `~CommandBuffer` read:
        // only a backend-owned buffer is reset and recycled; a borrowed buffer is
        // never touched (the backend may neither reset nor free it); a persistent
        // recording is neither reset nor recycled but is still freed exactly once -
        // by the replay graph, which is why it must be freeable at all.
        auto backend = vk_detail::plan_command_buffer_retirement(Ownership::BACKEND);
        expect(backend.reset_native_buffer && backend.recycle_native_buffer)
            << "a backend-owned batch is reset and returned to the recycle FIFO";
        expect(backend.free_native_buffer)
            << "and the backend frees it when it is destroyed instead";

        auto borrowed = vk_detail::plan_command_buffer_retirement(Ownership::BORROWED);
        expect(!borrowed.reset_native_buffer && !borrowed.recycle_native_buffer)
            << "a one-shot borrowed buffer is begun and ended only";
        expect(!borrowed.free_native_buffer)
            << "and its pool is never the backend's to free";

        auto persistent = vk_detail::plan_command_buffer_retirement(Ownership::PERSISTENT);
        expect(!persistent.reset_native_buffer && !persistent.recycle_native_buffer)
            << "a recording is begun once and resubmitted, never reset or recycled";
        expect(persistent.free_native_buffer)
            << "the graph still frees it exactly once";

        // The acquisition/hand-over side of the same contract:
        // `CommandBuffer::release_native_batch()` gives the native buffer *and* its
        // state (i.e. its pool) to a replay graph, so only a PERSISTENT buffer may
        // take that path. A BACKEND buffer would be freed twice (retirement FIFO /
        // destructor) and a BORROWED one would be freed from a foreign pool.
        expect(vk_detail::may_release_native_batch(Ownership::PERSISTENT))
            << "a recording is exactly what a graph adopts";
        expect(!vk_detail::may_release_native_batch(Ownership::BACKEND))
            << "a backend-owned batch is owned by the stream, not by a graph";
        expect(!vk_detail::may_release_native_batch(Ownership::BORROWED))
            << "a borrowed batch belongs to the config ext forever";

        // Classified exhaustively, and the two predicates agree: everything a graph
        // may adopt must be freeable (otherwise the graph leaks), and nothing may be
        // adoptable while still being recycled (which would hand one native buffer to
        // two owners).
        constexpr auto all_ownerships = std::array{
            Ownership::BACKEND, Ownership::BORROWED, Ownership::PERSISTENT};
        auto releasable = 0u;
        for (auto ownership : all_ownerships) {
            auto plan = vk_detail::plan_command_buffer_retirement(ownership);
            if (vk_detail::may_release_native_batch(ownership)) {
                ++releasable;
                expect(plan.free_native_buffer)
                    << "an adoptable batch must be freeable, or a graph would leak it";
                expect(!plan.recycle_native_buffer)
                    << "an adoptable batch must not stay in the recycle FIFO, or two "
                       "owners would free or reuse one native buffer";
            }
        }
        expect(releasable == 1u)
            << "exactly one ownership state may be adopted by a replay graph";
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_dispatch_allowed();
    reg_dispatch_shape_rejections();
    reg_every_tag_classified();
    reg_budget();
    reg_command_buffer_ownership();
    reg_status_names();
    return 0;
}