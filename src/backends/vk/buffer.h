#pragma once
#include "../common/storage_buffer_metadata.h"
#include "resource.h"
#include <volk.h>
namespace lc::vk {
class Buffer;
struct BufferFlusher {
    std::atomic_size_t begin{std::numeric_limits<size_t>::max()};
    std::atomic_size_t end{};
    void mark_dirty(size_t range_begin, size_t range_end);
    void flush(Device *device, void *alloc);
};
void vma_defragment(Device *device);
class Buffer : public Resource {
    size_t _byte_size;
    size_t _addressable_byte_size;

public:
    Buffer(Device *device, size_t byte_size,
           size_t addressable_byte_size = 0u)
        : Resource{device},
          _byte_size{byte_size},
          _addressable_byte_size{addressable_byte_size == 0u ?
                                     byte_size :
                                     addressable_byte_size} {}
    Buffer(Buffer &&) = default;
    // `byte_size` is the API-visible logical size. `addressable_byte_size`
    // is the physical range that may legally appear in a Vulkan descriptor.
    // They differ for owned buffers with a padded final storage word.
    auto byte_size() const { return _byte_size; }
    auto addressable_byte_size() const { return _addressable_byte_size; }
    virtual ~Buffer() = default;
    virtual VkBuffer vk_buffer() const = 0;
    Tag tag() const override { return Tag::kBuffer; }
    uint64_t get_device_address() const;
    virtual bool flush_host() const { return false; }
    virtual void flush_range(size_t begin, size_t end) {}
    [[nodiscard]] virtual bool is_indirect_dispatch_buffer() const noexcept {
        return false;
    }
    [[nodiscard]] virtual size_t indirect_dispatch_capacity() const noexcept {
        return 0u;
    }
    [[nodiscard]] virtual bool claim_indirect_header_initialization() const noexcept {
        return false;
    }
    [[nodiscard]] virtual bool indirect_header_initialization_claimed() const noexcept {
        return false;
    }
};

struct StorageBufferDescriptorRange {
    VkDeviceSize offset;
    VkDeviceSize range;
    StorageBufferMetadata metadata;
};

// Computes a Vulkan-valid storage-buffer descriptor for a logical buffer
// view. The descriptor base is moved down to satisfy both Vulkan's descriptor
// alignment and the shader-visible storage element alignment; `metadata`
// carries the resulting descriptor-relative bias and the exact logical size.
// A non-zero `logical_element_stride` additionally proves that a typed view is
// made of whole elements. Pass zero for byte-addressed/word-backed views.
[[nodiscard]] StorageBufferDescriptorRange storage_buffer_descriptor_range(
    const Buffer *buffer, size_t view_offset, size_t view_size,
    size_t logical_element_stride = 0u);

class ExternalBuffer : public Buffer {
    VkBuffer _buffer{};

public:
    ExternalBuffer(Device *device, VkBuffer vk_buffer, size_t size_bytes)
        : Buffer{device, size_bytes},
          _buffer(vk_buffer) {
    }
    ExternalBuffer(ExternalBuffer &&rhs) = default;
    ~ExternalBuffer() = default;
    VkBuffer vk_buffer() const override { return _buffer; }
};
class BufferView {
public:
    Buffer const *buffer;
    size_t offset;
    size_t size_bytes;
    BufferView() : buffer(nullptr), offset(0), size_bytes(0) {}
    BufferView(Buffer const *buffer) : buffer(buffer), offset(0), size_bytes(buffer->byte_size()) {
    }
    BufferView(
        Buffer const *buffer,
        size_t offset,
        size_t size_bytes)
        : buffer(buffer),
          offset(offset),
          size_bytes(size_bytes) {}
};

}// namespace lc::vk
