#pragma once
#include <luisa/vstl/common.h>

namespace lc::validation {
enum class StreamFunc : uint32_t {
    Signal = 1,
    Wait = 2,
    Graphics = 4,
    Compute = 8,
    Copy = 16,
    Custom = 32,
    Swapchain = 64,
    Sync = 128,
    All = 0xffffffff,
};
struct StreamOption {
    StreamFunc func{};
    vstd::unordered_set<uint64_t> supported_custom;
    bool check_stream_func(StreamFunc func, uint64_t custom_id = 0) const {
        if ((luisa::to_underlying(func) & luisa::to_underlying(this->func)) == 0) {
            return false;
        }
        if (custom_id != 0 && (luisa::to_underlying(func) & luisa::to_underlying(StreamFunc::Custom)) != 0) {
            if (supported_custom.find(custom_id) == supported_custom.end()) return false;
        }
        return true;
    }
};
}// namespace lc::validation
