#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Buffer : public RWResource {
    uint64_t _tile_size;
    size_t _indirect_dispatch_capacity;
    bool _is_indirect_dispatch;
    // Total byte size, when known.  Extensions that address a byte range of a
    // buffer (direct storage) validate their requests against it.
    size_t _size_bytes{0u};

public:
    Buffer(uint64_t handle, uint64_t tile_size,
           bool is_indirect_dispatch = false,
           size_t indirect_dispatch_capacity = 0u,
           size_t size_bytes = 0u)
        : RWResource(handle, Tag::BUFFER, false),
          _tile_size{tile_size},
          _indirect_dispatch_capacity{indirect_dispatch_capacity},
          _is_indirect_dispatch{is_indirect_dispatch},
          _size_bytes{size_bytes} {
        LUISA_ASSERT(
            _is_indirect_dispatch ==
                (_indirect_dispatch_capacity != 0u),
            "Validation indirect-dispatch buffers require a positive capacity.");
    }
          auto tile_size() const { return _tile_size; }
      /// Total size in bytes, or 0 when the size is not tracked by the caller.
      [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    [[nodiscard]] bool is_indirect_dispatch_buffer() const noexcept {
        return _is_indirect_dispatch;
    }
    [[nodiscard]] size_t indirect_dispatch_capacity() const noexcept {
        return _indirect_dispatch_capacity;
    }
    static constexpr luisa::string_view validation_res_name{"Buffer"};
};
}// namespace lc::validation
