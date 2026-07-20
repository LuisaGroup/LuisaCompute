#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct FuseConsecutiveBufferReadsInfo {
    size_t fused_group_count{0u};
    size_t fused_read_count{0u};
};

// This pass is intentionally quarantined as a no-op. XIR typed buffer
// operations require the access type to equal the buffer element type, so a
// scalar-to-vector rewrite is not legal without a byte-addressed lowering and
// backend-independent proofs for alignment, bounds, aliasing, and volatility.
// The entry points remain available for pipeline/API compatibility and report
// zero changes.

[[nodiscard]] LUISA_XIR_API FuseConsecutiveBufferReadsInfo
fuse_consecutive_buffer_reads_pass_run_on_function(Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API FuseConsecutiveBufferReadsInfo
fuse_consecutive_buffer_reads_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
