//
// Post-IPO ABI specialization for retained HIP generated callables.
//

#pragma once

#include <cstddef>

#include <llvm/ADT/StringRef.h>

namespace llvm {
class Function;
class Module;
}// namespace llvm

namespace luisa::compute::hip {

inline constexpr auto llvm_generated_callable_attribute =
    "luisa-generated-callable";
inline constexpr auto llvm_explicit_noinline_attribute =
    "luisa-explicit-noinline";
inline constexpr auto llvm_constant_argument_specialization_attribute =
    "luisa-specialize-constant-argument";

// Marks a generated callable and applies its source-owned inlining policy.
// The explicit marker survives IPO so final cleanup can distinguish a
// semantic noinline request from incidental policy on an ordinary Callable.
void mark_hip_generated_callable(
    llvm::Function &function,
    bool requires_noinline) noexcept;

// Removes optimizer-owned inline policy before IPO while preserving an
// explicit source noinline request on generated callables.
void prepare_hip_generated_callable_for_ipo(
    llvm::Function &function) noexcept;

// RetCC_AMDGPU_Func assigns at most VGPR0--VGPR31 to returned legalized
// values. A larger return is demoted by GlobalISel to a caller-owned stack
// object. Keep this number next to the transform that makes that implicit ABI
// explicit; it is a calling-convention limit, not a tuning parameter.
inline constexpr size_t amdgpu_callable_return_vgpr_limit = 32u;

// CC_AMDGPU_Func exposes the same fixed VGPR0--VGPR31 window for legalized
// value arguments. Keep the argument and return constants distinct: although
// the numerical limit is currently identical, the two transforms model
// independent halves of the ABI.
inline constexpr size_t amdgpu_callable_argument_vgpr_limit = 32u;

// Finalizes the attributes of an IPO-optimized function without discarding
// any semantic, ABI, or optimizer-proven facts. Luisa's temporary provenance
// marker and optimizer-owned inlining directives attached to an ordinary
// generated Callable are removed; an explicit source noinline request and
// source-owned low-level wrapper attributes are preserved. Target controls are
// replaced by the final shader configuration for every definition.
void finalize_hip_function_attributes(
    llvm::Function &function,
    llvm::StringRef target_cpu,
    llvm::StringRef target_features,
    llvm::StringRef max_vgpr_count) noexcept;

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

struct LargeArgumentDemotionStats {
    size_t rewritten_function_count{};
    size_t rewritten_call_count{};
    size_t shared_argument_slot_count{};
    size_t packed_argument_count{};
    size_t argument_record_bytes{};
};

struct ConstantArgumentSpecializationStats {
    size_t rewritten_function_count{};
    size_t cloned_function_count{};
    size_t merged_clone_count{};
    size_t rewritten_call_count{};
};

// Specializes an internal function on every integer parameter carrying
// `argument_attribute`, provided every use is a direct non-musttail call with
// a constant integer at every selected position. One clone is made per
// distinct tuple; all selected formals are replaced simultaneously and
// removed from the clone's ABI. The transformation is the SSA beta-reduction
//
//   call F(..., c_0, ..., c_n, ...) ==
//       call F[p_0 := c_0, ..., p_n := c_n](..., ...)
//
// and is applied atomically per function. Address-taken functions, dynamic
// actuals, recursion, or ABI features whose parameter indices require a richer
// model make the complete function fail closed. Identical specialized clones
// are deduplicated with LLVM's semantic function comparator. The internal
// marker is always removed, including on rejected functions, so it cannot
// escape to target code generation.
[[nodiscard]] ConstantArgumentSpecializationStats
specialize_marked_constant_integer_arguments(
    llvm::Module &module,
    llvm::StringRef argument_attribute =
        llvm_constant_argument_specialization_attribute) noexcept;

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

// Normalizes a generated callable whose legalized arguments exceed
// CC_AMDGPU_Func's 32-VGPR input window. Let p be the longest parameter prefix
// for which locations(p) + locations(private-record-pointer) <= 32. The exact
// rewrite is
//
//   Ret f(p..., s...)  ->  Ret f(p..., private {s...} *suffix),
//
// with one max-sized caller-owned record slot, populated immediately before
// each synchronous direct call. A suffix record is alive until that call
// returns; these intervals are pairwise disjoint within one caller, while
// nested and recursive invocations are isolated by their machine frames. The pass runs after
// large-return demotion so an explicit result pointer participates in the same
// argument budget. Unsupported indirect/musttail/ABI-bearing uses reject the
// complete function atomically.
[[nodiscard]] LargeArgumentDemotionStats
demote_generated_callable_large_arguments(
    llvm::Module &module,
    llvm::StringRef callable_attribute =
        llvm_generated_callable_attribute) noexcept;

}// namespace luisa::compute::hip
