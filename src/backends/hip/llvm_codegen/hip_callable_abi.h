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

// RetCC_AMDGPU_Func assigns at most VGPR0--VGPR31 to returned legalized
// values. A larger return is demoted by GlobalISel to a caller-owned stack
// object. Keep this number next to the transform that makes that implicit ABI
// explicit; it is a calling-convention limit, not a tuning parameter.
inline constexpr size_t amdgpu_callable_return_vgpr_limit = 32u;

struct AggregateArgumentSpecializationStats {
    size_t rewritten_function_count{};
    size_t removed_aggregate_bytes{};
};

struct LargeReturnDemotionStats {
    size_t rewritten_function_count{};
    size_t rewritten_call_count{};
    size_t shared_result_slot_count{};
    size_t demoted_return_bytes{};
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

// Makes an AMDGPU generated-callable return that exceeds the 32-VGPR return
// convention explicit as
//
//   Ret f(args...)  ->  void f(private Ret *result, args...).
//
// This pass runs after IPO and considers every generated callable that remains
// in the module: the HIP cleanup stage retains exactly those surviving
// boundaries. Thus the original callable body is optimized in SSA form before
// stores are introduced. Each caller owns one private result slot
// per exact return type and reuses it across calls; the store/load pair around
// every call makes this safe even when calls are not mutually exclusive. This
// prevents AMDGPU instruction selection from allocating one hidden result
// frame object per static call site. Uses whose ABI cannot be remapped without
// additional semantic information (for example external linkage, calling
// conventions other than C/Fast, COMDAT membership, semantic metadata,
// operand bundles, tail-call annotations, call-site fast-math assumptions, or
// allocsize) are rejected atomically.
[[nodiscard]] LargeReturnDemotionStats
demote_generated_callable_large_returns(
    llvm::Module &module,
    llvm::StringRef callable_attribute =
        llvm_generated_callable_attribute) noexcept;

}// namespace luisa::compute::hip
