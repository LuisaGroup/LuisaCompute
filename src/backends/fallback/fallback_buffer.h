//
// Created by swfly on 2024/11/21.
//

#pragma once

#include <vector>

namespace luisa::compute::fallback {

struct alignas(16) FallbackBufferView {
    void *ptr;
    size_t size;
};

class FallbackBuffer {

private:
    std::byte *_data{};
    size_t _size : 63;
    size_t _is_external : 1;

public:
    explicit FallbackBuffer(size_t size) noexcept;
    FallbackBuffer(std::byte *data, size_t size) noexcept;// create from external memory
    ~FallbackBuffer() noexcept;
    [[nodiscard]] auto data() noexcept { return _data; }
    [[nodiscard]] auto size_bytes() const noexcept { return static_cast<size_t>(_size); }
    [[nodiscard]] FallbackBufferView view(size_t offset, size_t size) noexcept;
    [[nodiscard]] FallbackBufferView view_with_offset(size_t offset) noexcept;
};

}// namespace luisa::compute::fallback
