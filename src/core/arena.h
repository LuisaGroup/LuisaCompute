//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <vector>
#include <memory>

#include <core/concepts.h>
#include <core/logging.h>

namespace luisa {

class Arena : public Noncopyable {

public:
    static constexpr auto block_size = static_cast<size_t>(256ul * 1024ul);

private:
    std::vector<std::byte *> _blocks;
    uint64_t _ptr{0ul};

public:
    Arena() noexcept = default;
    Arena(Arena &&) noexcept = default;
    Arena &operator=(Arena &&) noexcept = default;
    ~Arena() noexcept {
        for (auto p : _blocks) { free(p); }
    }

    template<typename T, typename... Args>
    [[nodiscard]] T *create(Args &&...args) {
        auto memory = allocate<T>(1u);
        return new (memory.data()) T(std::forward<Args>(args)...);
    }

    template<typename T = std::byte, size_t alignment = alignof(T)>
    [[nodiscard]] std::span<T> allocate(size_t n) {

        static_assert(std::is_trivially_destructible_v<T>);
        static constexpr auto size = sizeof(T);

        auto byte_size = n * size;
        auto aligned_p = reinterpret_cast<std::byte *>((_ptr + alignment - 1u) / alignment * alignment);
        if (_blocks.empty() || aligned_p + byte_size > _blocks.back() + block_size) {
            static constexpr auto alloc_alignment = std::max(alignment, sizeof(void *));
            auto alloc_size = (std::max(block_size, byte_size) + alloc_alignment - 1u) / alloc_alignment * alloc_alignment;
            aligned_p = static_cast<std::byte *>(aligned_alloc(alloc_alignment, alloc_size));
            if (aligned_p == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to allocate memory: size = {}, alignment = {}, count = {}", size, alignment, n); }
            _blocks.emplace_back(aligned_p);
        }
        _ptr = reinterpret_cast<uint64_t>(aligned_p + byte_size);
        return {reinterpret_cast<T *>(aligned_p), n};
    }
};

template<typename T, size_t max_size>
class SmallVector {};

class FixedString {};

}// namespace luisa
