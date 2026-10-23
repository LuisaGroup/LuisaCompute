#pragma once

#include <type_traits>
#include <utility>

#include <llvm/Passes/PassBuilder.h>

namespace luisa::compute::detail {

// Upstream LLVM 24 requires a context, while downstream LLVM 24 packages
// such as ROCm can retain the older constructor. Detect the installed API
// rather than the major version, and keep the module's context lifetime.
template<typename Builder = ::llvm::PassBuilder, typename... Args>
[[nodiscard]] inline auto make_llvm_pass_builder(
    ::llvm::LLVMContext &context, Args &&...args) noexcept {
    if constexpr (std::is_constructible_v<Builder, ::llvm::LLVMContext &, Args &&...>) {
        return Builder{context, std::forward<Args>(args)...};
    } else {
        static_assert(std::is_constructible_v<Builder, Args &&...>,
                      "Unsupported LLVM PassBuilder constructor.");
        return Builder{std::forward<Args>(args)...};
    }
}

}// namespace luisa::compute::detail
