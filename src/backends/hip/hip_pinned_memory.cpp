#include <limits>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_device.h"
#include "hip_pinned_memory.h"

namespace luisa::compute::hip {

namespace {

class ScopedHIPContext {

private:
    hipCtx_t _context;

public:
    explicit ScopedHIPContext(hipCtx_t context) noexcept
        : _context{context} {
        LUISA_ASSERT(_context != nullptr, "HIP device context is null.");
        LUISA_CHECK_HIP(hipCtxPushCurrent(_context));
    }
    ~ScopedHIPContext() noexcept {
        hipCtx_t popped_context{nullptr};
        LUISA_CHECK_HIP(hipCtxPopCurrent(&popped_context));
        LUISA_ASSERT(popped_context == _context,
                     "Unexpected HIP context popped from the current thread.");
    }
    ScopedHIPContext(ScopedHIPContext &&) noexcept = delete;
    ScopedHIPContext(const ScopedHIPContext &) noexcept = delete;
    ScopedHIPContext &operator=(ScopedHIPContext &&) noexcept = delete;
    ScopedHIPContext &operator=(const ScopedHIPContext &) noexcept = delete;
};

struct BufferLayout {
    size_t element_stride;
    size_t size_bytes;
};

[[nodiscard]] BufferLayout compute_buffer_layout(
    const Type *elem_type, size_t elem_count) noexcept {
    LUISA_ASSERT(elem_count != 0u,
                 "Pinned memory element count must not be zero.");
    // Type::of<void>() is represented by nullptr. Treat it as byte storage,
    // consistently with DeviceInterface::create_buffer.
    LUISA_ASSERT(elem_type == nullptr ||
                     elem_type->is_basic() || elem_type->is_structure() || elem_type->is_array(),
                 "Invalid pinned memory element type {}.",
                 elem_type == nullptr ? "void" : elem_type->description());
    auto element_stride = elem_type == nullptr ? 1u : elem_type->size();
    LUISA_ASSERT(element_stride != 0u &&
                     elem_count <= std::numeric_limits<size_t>::max() / element_stride,
                 "Pinned memory allocation size overflow (stride = {}, count = {}).",
                 element_stride, elem_count);
    return {.element_stride = element_stride,
            .size_bytes = element_stride * elem_count};
}

[[nodiscard]] BufferCreationInfo make_buffer_creation_info(
    HIPBuffer *buffer, BufferLayout layout) noexcept {
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->native_handle();
    info.element_stride = layout.element_stride;
    info.total_size_bytes = layout.size_bytes;
    return info;
}

}// namespace

BufferCreationInfo HIPPinnedMemoryExt::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option [[maybe_unused]]) noexcept {
    LUISA_ASSERT(host_ptr != nullptr,
                 "Cannot pin a null host memory pointer.");
    auto layout = compute_buffer_layout(elem_type, elem_count);
    ScopedHIPContext guard{static_cast<hipCtx_t>(_device->native_handle())};
    auto buffer = HIPBuffer::register_host_buffer(host_ptr, layout.size_bytes);
    return make_buffer_creation_info(buffer, layout);
}

BufferCreationInfo HIPPinnedMemoryExt::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    auto layout = compute_buffer_layout(elem_type, elem_count);
    ScopedHIPContext guard{static_cast<hipCtx_t>(_device->native_handle())};
    auto buffer = HIPBuffer::create_host_buffer(
        layout.size_bytes, option.write_combined);
    return make_buffer_creation_info(buffer, layout);
}

DeviceInterface *HIPPinnedMemoryExt::device() const noexcept {
    return _device;
}

}// namespace luisa::compute::hip
