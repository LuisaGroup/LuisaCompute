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
/// The complete normalized Extension remains owned by CoroGraph. This view
/// only joins it with the compiler-proved partial-frame dataflow and the
/// scheduler queue that will execute the stage. It never allocates another
/// copy of an Extension binding or frame slot.
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

/// Immutable compilation context supplied once for every static stage claimed
/// by a handler. A handler typically compiles a small frame-indexed kernel here
/// and stores it under stage.queue_index for later dispatch.
struct WavefrontCoroExtensionPrepareContext {
    Device &device;
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

/// Responsibility-chain element for WavefrontCoroScheduler.
///
/// Registration order is semantic: the first handler whose can_handle()
/// returns true owns that static stage. prepare() runs once before the first
/// scheduler dispatch. dispatch() must enqueue all work required to establish
/// Stage::required_writeback_slot_span() before returning. The scheduler owns
/// frame allocation, queue selection, stage ordering, and continuation resume;
/// handlers own only their external operation.
template<typename... Args>
class WavefrontCoroSchedulerExtensionHandler {

public:
    virtual ~WavefrontCoroSchedulerExtensionHandler() noexcept = default;

    [[nodiscard]] virtual luisa::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool can_handle(
        const WavefrontCoroExtensionStage &stage) const noexcept = 0;
    virtual void prepare(
        const WavefrontCoroExtensionPrepareContext &context,
        const WavefrontCoroExtensionStage &stage) noexcept = 0;
    virtual void dispatch(
        const WavefrontCoroExtensionDispatchContext &context,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept = 0;
};

}// namespace luisa::compute::coro
