#pragma once
#include <vulkan/vulkan.h>
#include <luisa/vstl/common.h>
namespace lc::vk {
class Buffer;
class ResourceBarrier {
    struct Range {
        int64_t min;
        int64_t max;
        Range() {
            min = std::numeric_limits<int64_t>::min();
            max = std::numeric_limits<int64_t>::max();
        }
        Range(int64_t value) {
            min = value;
            max = value + 1;
        }
        Range(int64_t min, int64_t size)
            : min(min), max(size + min) {}
        bool collide(Range const &r) const {
            return min < r.max && r.min < max;
        }
        bool operator==(Range const &r) const {
            return min == r.min && max == r.max;
        }
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    vstd::unordered_map<Buffer const*, vstd::vector<Range>> buffer_ranges;

public:
    ResourceBarrier();
    void add_buffer(Buffer const *buffer, size_t offset, size_t size);
    ~ResourceBarrier();
};
}// namespace lc::vk
