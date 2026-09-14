#include "fallback_coro_arena.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace luisa::compute::fallback;

int main() {
    auto arena = std::make_unique<FallbackCoroutineArena>();
    bool passed = true;
    const auto run = [&](const std::vector<size_t> &sizes, unsigned epoch) {
        arena->reset();
        std::vector<std::byte *> pointers;
        for (auto i = 0u; i < sizes.size(); ++i) {
            auto p = static_cast<std::byte *>(arena->allocate(sizes[i]));
            passed &= reinterpret_cast<uintptr_t>(p) % luisa_coro_allocation_alignment == 0u;
            pointers.push_back(p);
            std::memset(p, (i + epoch) & 255u, std::max(sizes[i], size_t{1u}));
        }
        // Check after every allocation has completed: growth must preserve
        // earlier addresses and no live spans may overlap, including size 0.
        for (auto i = 0u; i < sizes.size(); ++i) {
            const auto value = static_cast<std::byte>((i + epoch) & 255u);
            for (auto j = size_t{0u}; j < std::max(sizes[i], size_t{1u}); ++j) {
                if (pointers[i][j] != value) { passed = false; break; }
            }
        }
        return pointers;
    };
    const std::vector<size_t> small{0u, 1u, 17u, 127u};
    run(small, 0u);
    passed &= arena->heap_allocation_count() == 0u;
    const std::vector<size_t> wide(64u, 73728u);
    const std::vector<size_t> narrow(128u, 36864u);
    const auto initial = run(wide, 1u);
    // 4.5 MiB of live frames require just one overflow chunk, not one heap
    // allocation for every overflow lane.
    passed &= arena->heap_allocation_count() == 1u;
    passed &= run(wide, 2u) == initial;
    const std::vector<size_t> oversized{
        initial_thread_frame_buffer_size + 17u,
        initial_thread_frame_buffer_size + 33u, 0u, 31u};
    run(oversized, 3u); // Individual frames larger than the initial arena.
    run(narrow, 4u);
    const auto allocations = arena->heap_allocation_count();
    for (auto epoch = 5u; epoch < 13u; ++epoch) {
        for (const auto &sizes : {wide, oversized, narrow, small}) { run(sizes, epoch); }
        passed &= arena->heap_allocation_count() == allocations;
    }
    if (!passed) { std::cerr << "Fallback coroutine arena lifetime/reuse contract failed\n"; }
    return passed ? 0 : 1;
}
