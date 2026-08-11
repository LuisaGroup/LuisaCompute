#pragma once

#include <cstddef>

#include "../llvm/llvm_schedule_codegen.h"

namespace luisa::compute::simd {

class SIMDBuffer {

private:
    std::byte *_data{nullptr};
    size_t _size{0u};
    bool _external{false};

public:
    explicit SIMDBuffer(size_t size_bytes) noexcept;
    SIMDBuffer(std::byte *data, size_t size_bytes) noexcept;
    ~SIMDBuffer() noexcept;

    [[nodiscard]] SIMDHostBufferView view(
        size_t offset, size_t size) noexcept;
    [[nodiscard]] SIMDHostBufferView view_with_offset(
        size_t offset) noexcept;
    [[nodiscard]] auto data() noexcept { return _data; }
    [[nodiscard]] auto size() const noexcept { return _size; }
};

}// namespace luisa::compute::simd
