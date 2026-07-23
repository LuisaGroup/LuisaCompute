#pragma once
#include "buffer.h"
namespace lc::vk {
class SparseBuffer : public Buffer {
    VkBuffer _buffer{};
    VkMemoryRequirements _memory_requirements{};
    VkDeviceSize _sparse_binding_size{};

public:
    SparseBuffer(Device *device, size_t size_bytes, bool used_as_accel = false, VkBufferUsageFlagBits extra_bit = static_cast<VkBufferUsageFlagBits>(0));
    SparseBuffer(SparseBuffer &&rhs) noexcept;
    ~SparseBuffer() override;
    VkBuffer vk_buffer() const override { return _buffer; }
    [[nodiscard]] auto const &memory_requirements() const noexcept {
        return _memory_requirements;
    }
    [[nodiscard]] auto sparse_binding_size() const noexcept {
        return _sparse_binding_size;
    }
    [[nodiscard]] auto sparse_block_size() const noexcept {
        return _memory_requirements.alignment;
    }
};
}// namespace lc::vk
