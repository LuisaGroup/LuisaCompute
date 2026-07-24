#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct FuseConsecutiveBufferReadsInfo {
    size_t fused_group_count{0u};
    size_t fused_read_count{0u};
    size_t fused_write_count{0u};
};

// This pass coalesces runs of consecutive byte-addressed buffer accesses
// (BYTE_BUFFER_READ / BYTE_BUFFER_WRITE) with adjacent constant byte offsets
// into a single wider vector transaction plus extract/aggregate chains.
//
// Typed BUFFER_READ / BUFFER_WRITE accesses are deliberately left untouched:
// the typed-buffer ABI requires the access type to equal the buffer element
// type, so a scalar-to-vector rewrite of typed accesses is not legal without
// a byte-addressed lowering and backend-independent proofs for alignment,
// bounds, aliasing, and volatility.

[[nodiscard]] LUISA_XIR_API FuseConsecutiveBufferReadsInfo
fuse_consecutive_buffer_reads_pass_run_on_function(Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API FuseConsecutiveBufferReadsInfo
fuse_consecutive_buffer_reads_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
