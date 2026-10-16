#pragma once

#include <cstddef>

namespace llvm {
class Module;
class TargetMachine;
}// namespace llvm

namespace luisa::compute::hip {

struct HIPPrivateMemoryOptimizationStats {
    size_t analyzed_allocas{};
    size_t eliminated_self_reference_stores{};
};

// Eliminate an alloca's stored self-address only when a complete, constant-
// offset access proof shows that the stored bytes are never read and that no
// other pointer or integer derived from the alloca escapes. The transformation
// is representation-independent: it applies to any private aggregate with the
// same access relation, not to a particular RayQuery layout or field offset.
// A short scalar cleanup then exposes the now non-escaping aggregate to SROA.
[[nodiscard]] HIPPrivateMemoryOptimizationStats
optimize_hip_private_memory(llvm::Module &module,
                            llvm::TargetMachine *target_machine) noexcept;

}// namespace luisa::compute::hip
