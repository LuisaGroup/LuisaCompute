//
// Post-IPO ABI specialization for retained HIP generated callables.
//

#pragma once

#include <cstddef>

#include <llvm/ADT/StringRef.h>

namespace llvm {
class Module;
}// namespace llvm

namespace luisa::compute::hip {

inline constexpr auto llvm_generated_callable_attribute =
    "luisa-generated-callable";

struct AggregateArgumentSpecializationStats {
    size_t rewritten_function_count{};
    size_t removed_aggregate_bytes{};
};

// Replaces aggregate value arguments of retained generated callables with the
// statically known leaf values observed by the callable body. An unused
// aggregate becomes zero parameters. An argument is rewritten only when every
// use is understood; see the implementation for the analysis lattice and
// semantic preconditions.
[[nodiscard]] AggregateArgumentSpecializationStats
specialize_generated_callable_aggregate_arguments(
    llvm::Module &module,
    llvm::StringRef callable_attribute =
        llvm_generated_callable_attribute) noexcept;

}// namespace luisa::compute::hip
