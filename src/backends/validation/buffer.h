#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Buffer : public RWResource {
    uint64_t _tile_size;
    size_t _indirect_dispatch_capacity;
    bool _is_indirect_dispatch;

public:
    Buffer(uint64_t handle, uint64_t tile_size,
           bool is_indirect_dispatch = false,
           size_t indirect_dispatch_capacity = 0u)
        : RWResource(handle, Tag::BUFFER, false),
          _tile_size{tile_size},
          _indirect_dispatch_capacity{indirect_dispatch_capacity},
          _is_indirect_dispatch{is_indirect_dispatch} {
        LUISA_ASSERT(
            _is_indirect_dispatch ==
                (_indirect_dispatch_capacity != 0u),
            "Validation indirect-dispatch buffers require a positive capacity.");
    }
    auto tile_size() const { return _tile_size; }
    [[nodiscard]] bool is_indirect_dispatch_buffer() const noexcept {
        return _is_indirect_dispatch;
    }
    [[nodiscard]] size_t indirect_dispatch_capacity() const noexcept {
        return _indirect_dispatch_capacity;
    }
    static constexpr luisa::string_view validation_res_name{"Buffer"};
};
}// namespace lc::validation
