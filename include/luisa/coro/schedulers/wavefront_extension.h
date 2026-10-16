#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

/// Stable scheduler identity for one Extension at one static suspend site.
///
/// The complete normalized Extension is retained in the scheduler's immutable
/// copy of the CoroGraph boundary descriptors. This view
/// only joins it with the compiler-proved partial-frame dataflow and the
/// scheduler queue that will execute the stage. It never allocates another
/// frame slot. It remains valid after the source Coroutine is destroyed.
struct WavefrontCoroExtensionStage {
    size_t queue_index{0u};
    const CoroGraph::Boundary *boundary{nullptr};
    const CoroSuspendExtension *extension{nullptr};
    const CoroGraph::Stage *dataflow{nullptr};

    [[nodiscard]] const CoroSlotAccess &binding(
        luisa::string_view name) const noexcept {
        LUISA_ASSERT(boundary != nullptr && extension != nullptr,
                     "Invalid wavefront coroutine Extension stage.");
        for (auto &&descriptor : extension->bindings()) {
            if (descriptor.name == name) {
                LUISA_ASSERT(
                    descriptor.index < boundary->bindings.size(),
                    "Coroutine Extension binding '{}' has invalid owner "
                    "index {}.",
                    name, descriptor.index);
                return boundary->bindings[descriptor.index];
            }
        }
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine Extension '{}' has no binding named '{}'.",
            extension->schema(), name);
    }
};

/// Preparation context supplied to an Extension facade for every unclaimed
/// static stage. A facade typically compiles a small frame-indexed kernel and
/// may enqueue one-time resource initialization on stream. If the scheduler
/// is later dispatched on a different stream, the caller is responsible for
/// the required cross-stream synchronization.
struct WavefrontCoroExtensionPrepareContext {
    Device &device;
    Stream &stream;
    const CoroFrameDesc &frame_desc;
    const CoroFrameStorageLayout &frame_layout;
    uint frame_capacity{0u};
    bool global_memory_soa{true};
    const ShaderOption &shader_option;
};

/// Runtime ownership transferred to a selected Extension handler.
///
/// frame_indices is the exact selected queue, not the whole frame pool. Work
/// appended to stream is ordered before the scheduler advances these frames to
/// the next Extension stage or coroutine continuation. logical_dispatch_size
/// is the original user dispatch shape and frame_capacity is the physical
/// scheduler allocation stride used by runtime SoA addressing.
struct WavefrontCoroExtensionDispatchContext {
    Stream &stream;
    ByteBufferView frame_buffer;
    BufferView<uint> frame_indices;
    uint frame_count{0u};
    uint frame_capacity{0u};
    uint3 logical_dispatch_size{};
    const WavefrontCoroExtensionStage &stage;
};

enum class WavefrontCoroExtensionExecution : uint8_t {
    // Independent scheduler queue; only frame effects survive a later gather.
    stage,
    // A read-only annotation suffix, executed together with its continuation.
    before_resume
};

/// One prepared handler for one static suspend Extension stage.
///
/// WavefrontCoroScheduler::register_extension_handler accepts a facade
/// callable with the shape
///
///     (PrepareContext &, const ExtensionStage &)
///         -> unique_ptr<ExtensionHandler>
///
/// for every still-unclaimed stage. Returning nullptr declines the stage and
/// lets the next registered facade try it; returning a handler claims it for
/// the scheduler's lifetime. The facade itself is not retained. One facade
/// may therefore create independent handlers for zero, one, or many static
/// stages without keeping a queue-indexed handler table of its own.
///
/// dispatch() must enqueue all work required to establish
/// Stage::required_writeback_slot_span() before returning. The scheduler owns
/// frame allocation, queue selection, stage ordering, and continuation resume;
/// handlers own only their external operation and explicitly bound resources.
///
/// A before_resume handler must implement a read-only annotation. Such
/// handlers must form a suffix of the handled Extensions at a static suspend
/// boundary. The scheduler selects this suffix and its target continuation as
/// one unit: there is no gather, refill, or relocation between handlers and
/// resume. Different static boundaries retain their own typed binding plans.
///
/// dispatch_queue() may return a different buffer view, but it must contain
/// an exact permutation of the input queue (same membership and cardinality).
/// The view and its storage must stay valid until the enqueued work completes.
/// The returned view is passed directly to the next handler or continuation.
/// Ordinary stage handlers still make no ordering promise across a later
/// gather. Existing handlers can continue to override only dispatch().
class WavefrontCoroSchedulerExtensionHandler {

public:
    virtual ~WavefrontCoroSchedulerExtensionHandler() noexcept = default;

    [[nodiscard]] virtual luisa::string_view name() const noexcept = 0;
    [[nodiscard]] virtual WavefrontCoroExtensionExecution execution() const noexcept {
        return WavefrontCoroExtensionExecution::stage;
    }
    /// Explicit permission to jointly process compatible before-resume queues.
    /// Empty (the default) preserves a distinct invocation for every boundary.
    /// Equal nonempty identities promise that either instance and its captured
    /// resources/policy can process the disjoint union using a representative
    /// Stage descriptor. The operation must not depend on predecessor/queue
    /// identity or on one invocation per source boundary. The scheduler also
    /// checks the complete ordered suffix, normalized metadata, typed physical
    /// bindings and resident certificates; schema equality alone is not enough.
    /// An independent semantic stage is never eligible for this batching.
    [[nodiscard]] virtual luisa::string_view batching_identity() const noexcept {
        return {};
    }
    virtual void dispatch(
        const WavefrontCoroExtensionDispatchContext &) noexcept {
        LUISA_ERROR_WITH_LOCATION("Extension handler must implement dispatch or dispatch_queue.");
    }
    [[nodiscard]] virtual BufferView<uint> dispatch_queue(
        const WavefrontCoroExtensionDispatchContext &context) noexcept {
        dispatch(context);
        return context.frame_indices;
    }
};

}// namespace luisa::compute::coro
