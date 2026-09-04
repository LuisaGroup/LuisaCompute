#pragma once

#include <cstddef>

#include <llvm/Analysis/InlineCost.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>

namespace llvm {
class Module;
struct PipelineTuningOptions;
}// namespace llvm

namespace luisa::compute::hip {

// HIP LLVM inlining is split into two disjoint cost domains for every
// generated-callable edge e:
//
//   profitable-growth: inline_cost(e) < T(level)
//   canonicalization:  inline_cost(e) < 1
//
// LLVM's module inliner owns the first domain and orders all module call sites
// in one priority queue over the frontend call graph, before any pass can
// specialize formerly equivalent call sites through caller context. The
// ordinary bottom-up CGSCC pipeline is retained for target-aware
// simplification, but its default threshold is restricted to the second
// domain. Generated callables carry neither inline hints nor profile metadata,
// so CGSCC cannot later make an order-dependent positive-growth copy of one
// consumer while retaining the shared callee for another. Last-call bonuses
// remain legal because deleting the final private body is modeled as a
// non-growing transformation.
//
// This is a pipeline ownership rule, not an inline/noinline annotation. LLVM
// remains responsible for every call-edge profitability decision.
void configure_hip_cgscc_canonicalization_inlining(
    llvm::PipelineTuningOptions &options) noexcept;

[[nodiscard]] llvm::InlineParams hip_module_priority_inline_params(
    llvm::OptimizationLevel level) noexcept;

void add_hip_module_priority_inliner(
    llvm::ModulePassManager &pipeline,
    llvm::OptimizationLevel level) noexcept;

// Marks the inner body of a generated kernel -> forwarding wrapper -> body
// chain as always-inline when every edge is a unique direct call and the
// wrapper does nothing except forward all arguments. Such a wrapper is a
// compiler representation detail, not a source-level optimization boundary.
// Returns the number of marked inner bodies.
[[nodiscard]] size_t
mark_hip_single_use_forwarded_kernel_callables_for_inlining(
    llvm::Module &module) noexcept;

}// namespace luisa::compute::hip
