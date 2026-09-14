#pragma once

#include "fallback_codegen.h"

#include <algorithm>
#include <limits>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute::fallback {

inline constexpr size_t luisa_coro_allocation_alignment = 2u * sizeof(intptr_t);

class FallbackCoroutineArena {
private:
    struct alignas(luisa_coro_allocation_alignment) OverflowBlock {
        OverflowBlock *next;
        size_t capacity;
    };

    alignas(luisa_coro_allocation_alignment) std::byte _buffer[initial_thread_frame_buffer_size];
    size_t _offset{};
    size_t _overflow_offset{};
    OverflowBlock *_overflow{};
    OverflowBlock **_next{&_overflow};

public:
    FallbackCoroutineArena() noexcept = default;
    FallbackCoroutineArena(const FallbackCoroutineArena &) = delete;
    FallbackCoroutineArena &operator=(const FallbackCoroutineArena &) = delete;

    // Every lane's frame can be live simultaneously. Grow by stable chunks,
    // never by relocating frames or malloc/free for each lane and dispatch.
    [[nodiscard]] void *allocate(size_t size) noexcept {
        LUISA_ASSERT(size <= std::numeric_limits<size_t>::max() -
                                 sizeof(OverflowBlock) - luisa_coro_allocation_alignment,
                     "Coroutine frame allocation size overflow: {}.", size);
        size = luisa::align(std::max(size, size_t{1u}), luisa_coro_allocation_alignment);
        if (size <= sizeof(_buffer) - _offset) {
            auto p = _buffer + _offset;
            _offset += size;
            return p;
        }
        while (*_next && size > (*_next)->capacity - _overflow_offset) {
            _next = &(*_next)->next;
            _overflow_offset = 0u;
        }
        if (*_next == nullptr) {
            // A cache growth quantum, not a limit on frame size. Multiple
            // frames share each chunk; a larger individual frame fits too.
            const auto capacity = std::max(size, sizeof(_buffer));
            auto p = luisa::detail::allocator_allocate(
                sizeof(OverflowBlock) + capacity, luisa_coro_allocation_alignment);
            LUISA_ASSERT(p != nullptr, "Failed to allocate coroutine storage: {} bytes.", capacity);
            *_next = std::construct_at(static_cast<OverflowBlock *>(p), nullptr, capacity);
        }
        auto p = reinterpret_cast<std::byte *>(*_next) + sizeof(OverflowBlock) + _overflow_offset;
        _overflow_offset += size;
        return p;
    }

    // Called only once the preceding simulated GPU block has completed.
    // Cursors reset without clearing, moving or releasing any frame storage.
    void reset() noexcept {
        _offset = 0u;
        _overflow_offset = 0u;
        _next = &_overflow;
    }

    [[nodiscard]] size_t heap_allocation_count() const noexcept {
        // Exactly one heap allocation per retained chunk, with no freeing or
        // resizing in allocate/reset. No counter work on the allocation path.
        size_t count = 0u;
        for (auto block = _overflow; block; block = block->next) { ++count; }
        return count;
    }

    ~FallbackCoroutineArena() noexcept {
        while (_overflow) {
            auto block = _overflow;
            _overflow = block->next;
            luisa::detail::allocator_deallocate(block, luisa_coro_allocation_alignment);
        }
    }
};

} // namespace luisa::compute::fallback
