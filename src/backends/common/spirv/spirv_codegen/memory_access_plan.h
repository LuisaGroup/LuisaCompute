#pragma once

#include <cstddef>
#include <cstdint>

#include <SPIRV/spirv.hpp11>

namespace lc::spirv {

// Frozen memory-access mask of one word-ABI load/store.
//
// `mask` is the caller's original mask plus an optional `Aligned` bit, and
// `literal` is the operand that bit requires (0 when no literal is emitted).
// `alignment` is the byte alignment the proof actually established.
struct SpirvAlignedMemoryAccessPlan {
    spv::MemoryAccessMask mask{spv::MemoryAccessMask::MaskNone};
    uint32_t literal{0u};
    size_t alignment{1u};

    [[nodiscard]] bool aligned() const noexcept { return literal != 0u; }
};

// The uint32 word ABI stores every value in 32-bit words, so the largest scalar
// size an ordinary word load/store can be annotated with is one word.
inline constexpr size_t spirv_word_access_scalar_size = 4u;

// Adds `Aligned 4` to `memory_access` when, and only when, the existing
// byte-alignment proof proves at least a whole word for an ordinary word-ABI
// access.
//
// Every other proof keeps the current masked access path:
//   - one- and two-byte proofs are not rounded up;
//   - a proof that is not a power of two is not rounded up;
//   - `access_scalar_size` above one word is never annotated with a literal that
//     is smaller than the access;
//   - the original bits, in particular `Volatile`, are always preserved.
//
// The function never invents alignment: `byte_alignment` is the caller's exact
// proof (the gcd of the parent proof and every relative aggregate offset, which
// already folds in the direct-buffer descriptor bias proof).
[[nodiscard]] SpirvAlignedMemoryAccessPlan plan_spirv_aligned_word_access(
    spv::MemoryAccessMask memory_access, size_t byte_alignment,
    size_t access_scalar_size = spirv_word_access_scalar_size) noexcept;

}// namespace lc::spirv