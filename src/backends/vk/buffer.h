#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
namespace lc::vk {
class Buffer : public Resource {
    size_t _byte_size;

public:
    Buffer(Device *device, size_t byte_size)
        : Resource{device},
          _byte_size{byte_size} {};
    auto byte_size() const { return _byte_size; }
    virtual ~Buffer() = default;
    virtual VkBuffer vk_buffer() const = 0;
};
}// namespace lc::vk
