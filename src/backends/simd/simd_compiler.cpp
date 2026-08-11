#include "simd_compiler.h"

#include <memory>
#include <string>
#include <utility>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "llvm/llvm_schedule_codegen.h"
#include "schedule/xir_to_schedule.h"

namespace luisa::compute::simd {

SIMDCompiledKernel compile_simd_kernel(
    const xir::Function *function, uint32_t warp_width,
    std::string_view entry_name) {
    SIMDCompiledKernel result{
        .warp_width = warp_width,
    };
    auto schedule_result = schedule::lower_xir_to_schedule(
        function, {.logical_warp_width = warp_width});
    if (!schedule_result.succeeded()) {
        result.diagnostics.reserve(schedule_result.diagnostics.size());
        for (auto &&diagnostic : schedule_result.diagnostics) {
            result.diagnostics.emplace_back(
                std::string{schedule::to_string(diagnostic.code)} +
                ": " + diagnostic.message);
        }
        return result;
    }

    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "luisa-simd-kernel", *context);
    auto llvm_result = lower_schedule_to_llvm(
        *module, *schedule_result.function, warp_width, entry_name);
    if (!llvm_result.succeeded()) {
        result.diagnostics.emplace_back(llvm_result.error);
        return result;
    }
    result.argument_buffer_size = llvm_result.argument_buffer_size;
    auto llvm_entry_name = llvm_result.entry->getName().str();
    result.jit = std::make_unique<LLVMJIT>();
    if (!result.jit->succeeded()) {
        result.diagnostics.emplace_back(result.jit->error());
        result.jit.reset();
        return result;
    }
    result.target_triple = result.jit->target_triple();
    if (!result.jit->add_module(
            std::move(module), std::move(context))) {
        result.diagnostics.emplace_back(result.jit->error());
        result.jit.reset();
        return result;
    }
    result.entry = result.jit->lookup(llvm_entry_name);
    if (result.entry == nullptr) {
        result.diagnostics.emplace_back(result.jit->error());
        result.jit.reset();
    }
    return result;
}

}// namespace luisa::compute::simd
