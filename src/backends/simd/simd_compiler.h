#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/llvm_jit.h"

namespace luisa::compute::xir {
class Function;
}// namespace luisa::compute::xir

namespace luisa::compute {
class Function;
}// namespace luisa::compute

namespace luisa::compute::simd {

struct SIMDCompiledKernel {
    std::unique_ptr<LLVMJIT> jit{};
    void *entry{nullptr};
    size_t argument_buffer_size{0u};
    uint32_t warp_width{0u};
    std::string target_triple{};
    std::vector<std::string> diagnostics{};

    [[nodiscard]] bool succeeded() const noexcept {
        return jit != nullptr && entry != nullptr && diagnostics.empty();
    }
};

// Compiles already-canonicalized XIR through Schedule IR to a host ORC entry.
// The returned function uses the packet ABI documented in
// llvm_schedule_codegen.h. Unsupported Phase-2 features are returned as
// diagnostics instead of being silently scalarized.
[[nodiscard]] SIMDCompiledKernel compile_simd_kernel(
    const xir::Function *function, uint32_t warp_width,
    std::string_view entry_name = {}, bool enable_fast_math = false);

// Translates a DSL/AST kernel to XIR, legalizes its structured control flow,
// inlines callables, promotes local SSA storage, and then invokes the packet
// compiler above. This is the front door used by the runtime backend.
[[nodiscard]] SIMDCompiledKernel compile_simd_kernel(
    const compute::Function &kernel, uint32_t warp_width,
    std::string_view entry_name = {}, bool enable_fast_math = false);

}// namespace luisa::compute::simd
