#pragma once

#include <cstddef>
#include <string>

namespace luisa::compute::xir {
class Function;
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

struct BlockBarrierCanonicalizationResult {
    size_t barrier_count{0u};
    size_t split_block_count{0u};
    std::string error{};

    [[nodiscard]] bool succeeded() const noexcept {
        return error.empty();
    }
};

// Makes every block barrier the final non-terminator of its XIR block. The
// original suffix (including its terminator) is moved to a fresh resume block,
// and successor PHI labels follow the moved outgoing edges. Schedule lowering
// can then represent the barrier as a suspension terminator without retaining
// source-XIR instruction pointers.
[[nodiscard]] BlockBarrierCanonicalizationResult
canonicalize_block_barriers(xir::Function *function) noexcept;

}// namespace luisa::compute::simd::schedule
