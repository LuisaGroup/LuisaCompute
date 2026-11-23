// Admissibility rules of replayable recording (plan P4, narrow v1).
//
// Everything in this file is a pure function of plain values: it never touches a
// device, never dereferences a handle and never calls Vulkan. That is what makes
// the rejection matrix of the public API testable on the host (see
// src/tests/unit/ext/test_vk_replay_plan.cpp) and keeps the policy in one
// reviewable place instead of spreading it through the recorder.
//
// The two tables of the contract:
//   * command level - which `Command::Tag` may be recorded, and for the one
//     dispatch tag which shader shape it must have;
//   * argument level - which `Argument::Tag` a recorded dispatch may carry.
// Every value of both enumerations is classified; none is left implicit, which is
// the anti-drift property the host test asserts.
#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/argument.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/vk_replay_ext.h>

namespace lc::vk::detail {

using ReplayRejectReason = luisa::compute::VulkanReplayRejectReason;

// What the classifier needs to know about a dispatch's shader without naming a
// backend type. The recorder fills it in from the shader handle it already has;
// the host test fills it in directly.
struct ReplayShaderFacts {
    enum struct Tag : uint8_t {
        COMPUTE,
        RASTER,
        RAY_TRACING,
        UNKNOWN,// a handle that is not a compute shader of this device
    };
    struct Forbidden {
        bool printers{false};
        bool constant_ubo{false};
        bool bindless{false};
    };
    Tag tag{Tag::UNKNOWN};
    Forbidden forbidden{};
};

// One command of a candidate list, reduced to what the policy looks at.
struct ReplayCommandFacts {
    luisa::compute::Command::Tag tag{};
    // EShaderDispatchCommand only.
    bool indirect{false};
    ReplayShaderFacts shader{};
};

// Command-level rule of §3.2 of the design: the tag decides first, then the
// command's own shape, then the shader's.
[[nodiscard]] inline ReplayRejectReason
classify_replay_command(const ReplayCommandFacts &facts) noexcept {
    using Tag = luisa::compute::Command::Tag;
    switch (facts.tag) {
        case Tag::EShaderDispatchCommand:
            if (facts.indirect) {
                // The record-time plan unrolls the dispatch into the shape it
                // read from the metadata buffer; a replay that must follow a
                // changing shape is a different feature.
                return ReplayRejectReason::DISPATCH_INDIRECT;
            }
            switch (facts.shader.tag) {
                case ReplayShaderFacts::Tag::COMPUTE: break;
                case ReplayShaderFacts::Tag::RASTER:
                    return ReplayRejectReason::DISPATCH_RASTER;
                case ReplayShaderFacts::Tag::RAY_TRACING:
                    return ReplayRejectReason::DISPATCH_RAY_TRACING;
                case ReplayShaderFacts::Tag::UNKNOWN:
                    return ReplayRejectReason::DISPATCH_UNKNOWN_SHADER;
            }
            if (facts.shader.forbidden.printers) {
                // Uses stream-scoped scratch and a readback buffer only the
                // batch's own retirement consumes.
                return ReplayRejectReason::DISPATCH_PRINTER;
            }
            if (facts.shader.forbidden.constant_ubo) {
                // The one shader-owned allocation a recorded buffer would point
                // at; refusing it removes the whole class of shader-lifetime
                // hazards instead of tracking the shader.
                return ReplayRejectReason::DISPATCH_CONSTANT_UBO;
            }
            if (facts.shader.forbidden.bindless) {
                // Bindless membership must be static for a graph, and v1 does not
                // carry a membership generation.
                return ReplayRejectReason::DISPATCH_BINDLESS;
            }
            return ReplayRejectReason::NONE;
        case Tag::EBufferCopyCommand:
            // Device-to-device only; both endpoints are baked identities and only
            // their contents are free.
            return ReplayRejectReason::NONE;
        case Tag::EBufferUploadCommand: return ReplayRejectReason::COMMAND_BUFFER_UPLOAD;
        case Tag::EBufferDownloadCommand: return ReplayRejectReason::COMMAND_BUFFER_DOWNLOAD;
        case Tag::EBufferToTextureCopyCommand:
        case Tag::ETextureUploadCommand:
        case Tag::ETextureDownloadCommand:
        case Tag::ETextureCopyCommand:
        case Tag::ETextureToBufferCopyCommand:
            return ReplayRejectReason::COMMAND_TEXTURE;
        case Tag::EAccelBuildCommand:
        case Tag::EMeshBuildCommand:
        case Tag::ECurveBuildCommand:
        case Tag::EProceduralPrimitiveBuildCommand:
        case Tag::EMotionInstanceBuildCommand:
            return ReplayRejectReason::COMMAND_ACCEL_BUILD;
        case Tag::EBindlessArrayUpdateCommand:
            return ReplayRejectReason::COMMAND_BINDLESS_UPDATE;
        case Tag::ECustomCommand:
            // Every foreign command (native shader, CUDA launch, DStorage, raster
            // clear, denoiser) has its own descriptor/state contract.
            return ReplayRejectReason::COMMAND_CUSTOM;
    }
    return ReplayRejectReason::COMMAND_CUSTOM;
}

// Argument-level rule of §3.3: `UNIFORM` and `BUFFER` only.
[[nodiscard]] inline ReplayRejectReason
classify_replay_argument(luisa::compute::Argument::Tag tag) noexcept {
    using Tag = luisa::compute::Argument::Tag;
    switch (tag) {
        case Tag::UNIFORM: return ReplayRejectReason::NONE;
        case Tag::BUFFER: return ReplayRejectReason::NONE;
        case Tag::TEXTURE: return ReplayRejectReason::ARG_TEXTURE;
        case Tag::BINDLESS_ARRAY: return ReplayRejectReason::ARG_BINDLESS_ARRAY;
        case Tag::ACCEL: return ReplayRejectReason::ARG_ACCELERATION_STRUCTURE;
    }
    return ReplayRejectReason::ARG_UNKNOWN_TAG;
}

// A graph budget, so an oversized list fails closed *before* a single native
// object is created instead of exhausting the descriptor pool mid-recording.
struct ReplayRecordLimits {
    uint32_t max_commands{65536u};
    uint32_t max_dispatches{65536u};
};

struct ReplayRecordPlan {
    ReplayRejectReason reason{ReplayRejectReason::NONE};
    uint32_t dispatches{0u};
    uint32_t buffer_copies{0u};
    [[nodiscard]] bool ok() const noexcept { return reason == ReplayRejectReason::NONE; }
};

// Classifies a whole list in order and stops at the first refusal, so the
// reported reason is the first offending command's (deterministic, and what a
// caller sees when they fix one thing at a time).
[[nodiscard]] inline ReplayRecordPlan plan_replay_record(
    luisa::span<const ReplayCommandFacts> commands,
    ReplayRecordLimits limits = {}) noexcept {
    ReplayRecordPlan plan{};
    if (commands.size() > limits.max_commands) {
        plan.reason = ReplayRejectReason::OUT_OF_RESOURCES;
        return plan;
    }
    for (auto &&command : commands) {
        if (auto reason = classify_replay_command(command);
            reason != ReplayRejectReason::NONE) {
            plan.reason = reason;
            return plan;
        }
        switch (command.tag) {
            case luisa::compute::Command::Tag::EShaderDispatchCommand:
                ++plan.dispatches;
                break;
            case luisa::compute::Command::Tag::EBufferCopyCommand:
                ++plan.buffer_copies;
                break;
            default: break;
        }
    }
    if (plan.dispatches > limits.max_dispatches) {
        plan.reason = ReplayRejectReason::OUT_OF_RESOURCES;
    }
    return plan;
}

}// namespace lc::vk::detail