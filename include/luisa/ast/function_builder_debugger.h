#pragma once

#include <luisa/ast/function_builder.h>

namespace luisa::compute::detail {

/// @brief Options controlling which runtime checks the debug generator emits.
///
/// The generator wraps every statement/expression of a function that can fail
/// at runtime with a guard: when the guard fires, a diagnostic is printed via
/// the device printer (FunctionBuilder::print_) and the thread stops early
/// (the error code is propagated up through custom callables to the root
/// kernel). See src/ast/function_builder_debugger.cpp for the complete
/// operator-by-operator analysis that drives these switches.
struct DebugKernelOptions {

    /// int/uint DIV/MOD by zero (must-check).
    bool check_int_div_mod_zero{true};
    /// float DIV by zero (must-check; the NaN/Inf result check below also
    /// fires for it, this one reports the operands of the faulty division).
    bool check_float_div_zero{true};
    /// NaN/Inf results of float ops with domain/range errors (must-check:
    /// SQRT/RSQRT/LOG*/ACOS/ASIN/ACOSH/ATANH/POW/ATAN2/EXP*/NORMALIZE/
    /// INVERSE, float ADD/SUB/MUL/DIV/MOD).
    bool check_float_nan_inf{true};
    /// SHL/SHR shift amount >= bit-width of the shifted type or negative.
    bool check_shift_amount{true};
    /// BUFFER_READ/WRITE (and volatile variants) element index bounds via
    /// CallOp::BUFFER_SIZE, byte-buffer ranges via CallOp::BYTE_BUFFER_SIZE.
    bool check_buffer_bounds{true};
    /// Bindless buffer element index bounds via CallOp::BINDLESS_BUFFER_SIZE
    /// (slot validity is a host-encoder contract and is not checked here).
    bool check_bindless_bounds{true};
    /// TEXTURE_READ/WRITE coordinates against CallOp::TEXTURE_SIZE (opt-in:
    /// writes are clamped on some backends).
    bool check_texture_bounds{false};
    /// RAY_TRACING_INSTANCE_*/SET_INSTANCE_* instance index against
    /// CallOp::ACCEL_SIZE. Requires a backend that implements ACCEL_SIZE
    /// (dx, vk and cuda in v1); disable it for backends that do not.
    bool check_accel_instance_index{true};
    /// NaN/Inf rays passed to RAY_TRACING_TRACE_*/QUERY_* (opt-in: traversal
    /// with a non-finite ray is undefined but rarely fatal).
    bool check_ray_validity{false};
    /// Give every transformed callable a trailing `reference<uint>` error
    /// out-parameter and check it at every call site, so a failure inside a
    /// callee stops the whole kernel thread.
    bool propagate_callable_errors{true};
};

/// @brief Rewrites `f` (and, transitively, every custom callable it uses) so
/// that operations that can fail at runtime are guarded.
///
/// The transform is a whole-function rebuild in the style of
/// FunctionDuplicator: expressions form a DAG owned by the source builder, so
/// hoisting guarded sub-expressions into fresh temporaries is only safe on a
/// rebuilt function. The original function is left untouched; the returned
/// builder is a new, equivalent function with guards inserted.
[[nodiscard]] LUISA_AST_API luisa::shared_ptr<const FunctionBuilder>
debug_function(const FunctionBuilder &f,
               const DebugKernelOptions &options = {}) noexcept;

/// @brief debug_function() under the LUISA_AST_DEBUG_KERNEL environment flag.
///
/// Returns `f` unchanged unless the flag is set to a truthy value ("1",
/// "true", "on", ...). Only kernels are transformed by this entry point; it is
/// the hook consulted by FunctionBuilder::define_kernel so that existing
/// programs can opt into the generator without code changes.
[[nodiscard]] LUISA_AST_API luisa::shared_ptr<const FunctionBuilder>
debug_function_if_enabled(luisa::shared_ptr<const FunctionBuilder> f) noexcept;

}// namespace luisa::compute::detail
