//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <span>
#include <cuda.h>

#include <core/spin_mutex.h>
#include <core/mathematics.h>
#include <backends/cuda/cuda_error.h>

namespace luisa::compute::cuda {

class CUDARingBuffer {

public:
    static constexpr auto alignment = static_cast<size_t>(16u);

public:
    class View {

    private:
        std::byte *_handle;
        uint64_t _is_pooled_and_size;
        static constexpr auto size_mask = 0x7fff'ffff'ffff'ffffull;
        static constexpr auto is_pooled_shift = 63u;

    public:
        View(std::byte *handle, size_t size, bool is_pooled) noexcept
            : _handle{handle},
              _is_pooled_and_size{(static_cast<uint64_t>(is_pooled) << is_pooled_shift) | size} {}
        [[nodiscard]] auto address() const noexcept { return _handle; }
        [[nodiscard]] auto size() const noexcept {
            return static_cast<size_t>(_is_pooled_and_size & size_mask);
        }
        [[nodiscard]] auto is_pooled() const noexcept {
            return static_cast<bool>(_is_pooled_and_size >> is_pooled_shift);
        }
    };

private:
    std::byte *_memory{nullptr};
    size_t _size;
    size_t _free_begin;
    size_t _free_end;
    uint _alloc_count;
    bool _write_combined;

public:
    CUDARingBuffer(size_t size, bool write_combined) noexcept;
    ~CUDARingBuffer() noexcept;
    [[nodiscard]] View allocate(size_t size) noexcept;
    void recycle(View buffer) noexcept;
};

}// namespace luisa::compute::cuda
