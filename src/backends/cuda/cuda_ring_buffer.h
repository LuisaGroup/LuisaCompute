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
    [[nodiscard]] std::span<std::byte> allocate(size_t size) noexcept;
    void recycle(std::span<std::byte> buffer) noexcept;
};

}// namespace luisa::compute::cuda
