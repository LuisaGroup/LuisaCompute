#pragma once

#include <cstddef>

namespace llvm {
class Module;
}

namespace luisa::compute::hip {

struct HIPSwitchTableLoweringStats {
    size_t rewritten_switch_count{};
    size_t rewritten_phi_count{};
};

// Replace a small divergent switch whose arms only select constant integer
// payloads with a constant-address-space lookup. Unknown selector values keep
// the original default payload. The pattern is deliberately structural and
// applies only when all switch arms are forwarding blocks and the merge PHIs
// are immutable integer constants.
[[nodiscard]] HIPSwitchTableLoweringStats
lower_hip_constant_switch_tables(llvm::Module &module) noexcept;

}// namespace luisa::compute::hip
