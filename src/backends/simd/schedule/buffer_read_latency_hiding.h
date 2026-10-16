#pragma once

#include <cstddef>

namespace luisa::compute::xir {
class Function;
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

struct BufferReadLatencyHidingInfo {
    size_t hidden_diamond_count{0u};
    size_t moved_instruction_count{0u};
    size_t generated_select_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return hidden_diamond_count != 0u;
    }
};

// Overlap a varying direct 32-bit buffer read in an innermost natural loop
// with a bounded, total continuation diamond. The diamond is first converted
// to selects, then its pure computations are issued before the read while the
// selected state remains committed only on the original continuation edge.
//
// This is deliberately a target-policy pass rather than generic LICM. The
// caller is responsible for selecting a measured SIMD width; every structural,
// dominance, speculation-safety, and register-pressure condition fails closed.
[[nodiscard]] BufferReadLatencyHidingInfo
hide_innermost_buffer_read_latency(xir::Function *function) noexcept;

}// namespace luisa::compute::simd::schedule
