//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <vector>
#include <memory>
#include <iostream>
#include <string_view>

#include <core/concepts.h>

namespace luisa {

class Arena : Noncopyable {

public:
    static constexpr auto block_size = 256ul * 1024ul;

private:
    std::vector<std::byte *> _blocks;
    uint64_t _ptr{0ul};

public:
    Arena() noexcept = default;
    Arena(Arena &&) noexcept = default;
    Arena &operator=(Arena &&) noexcept = default;
    ~Arena() noexcept { for (auto p : _blocks) { free(p); }}
    
    template<typename T, typename ...Args>
    [[nodiscard]] T *create(Args &&...args) {
        auto memory = allocate<T>(1u);
        return new(memory.data()) T(std::forward<Args>(args)...);
    }
    
    template<typename T = std::byte, uint64_t alignment = alignof(T)>
    [[nodiscard]] std::span<T> allocate(size_t n) {
        
        static_assert(std::is_trivially_destructible_v<T>);
        static constexpr auto size = sizeof(T);
        
        auto byte_size = n * size;
        auto aligned_p = reinterpret_cast<std::byte *>((_ptr + alignment - 1u) / alignment * alignment);
        if (_blocks.empty() || aligned_p + byte_size > _blocks.back() + block_size) {
            static constexpr auto min_alignment = static_cast<uint64_t>(16u);
            aligned_p = static_cast<std::byte *>(aligned_alloc(std::max(alignment, min_alignment), std::max(block_size, byte_size)));
            _blocks.emplace_back(aligned_p);
        }
        _ptr = reinterpret_cast<uint64_t>(aligned_p + byte_size);
        return {reinterpret_cast<T *>(aligned_p), n};
    }
};

}
