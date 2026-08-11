#include "simd_compiler.h"

#include <array>
#include <memory>
#include <string>
#include <utility>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <luisa/ast/function.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/fast_math_simplify.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/translators/ast2xir.h>

#include "../common/env_flag.h"
#include "llvm/llvm_schedule_codegen.h"
#include "schedule/loop_unswitch.h"
#include "schedule/predicated_if_conversion.h"
#include "schedule/xir_to_schedule.h"

namespace luisa::compute::simd {

SIMDCompiledKernel compile_simd_kernel(
    const xir::Function *function, uint32_t warp_width,
    std::string_view entry_name, bool enable_fast_math,
    bool enable_uniform_buffer_broadcast) {
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
    auto static_block_size = std::array<uint32_t, 3u>{};
    if (function->isa<xir::KernelFunction>()) {
        auto size = static_cast<const xir::KernelFunction *>(function)
                        ->block_size();
        static_block_size = {size.x, size.y, size.z};
    }
    auto llvm_result = lower_schedule_to_llvm(
        *module, *schedule_result.function, warp_width, entry_name,
        enable_fast_math, static_block_size,
        enable_uniform_buffer_broadcast);
    if (!llvm_result.succeeded()) {
        result.diagnostics.emplace_back(llvm_result.error);
        return result;
    }
    result.argument_buffer_size = llvm_result.argument_buffer_size;
    result.uniform_buffer_broadcast_count =
        llvm_result.uniform_buffer_broadcast_count;
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

SIMDCompiledKernel compile_simd_kernel(
    const compute::Function &kernel, uint32_t warp_width,
    std::string_view entry_name, bool enable_fast_math) {
    auto *translation = xir::ast_to_xir_translate_begin({});
    auto *xir_kernel = xir::ast_to_xir_translate_add_function(
        translation, kernel);
    auto module = xir::ast_to_xir_translate_finalize(translation);
    if (module == nullptr || xir_kernel == nullptr) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back("AST to XIR translation failed");
        return result;
    }

    // Single-block callables can be folded before CFG legalization. A second
    // pass after destructuring handles multi-block callables without cloning
    // structured regions into the caller.
    static_cast<void>(xir::inline_all_pass_run_on_module(module.get()));
    static_cast<void>(xir::local_store_forward_pass_run_on_module(module.get()));
    static_cast<void>(xir::local_load_elimination_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    static_cast<void>(xir::mem2reg_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));

    auto destructure = xir::destructure_cfg_pass_run_on_module(module.get());
    if (!destructure.succeeded()) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back(
            "XIR CFG destructuring failed (errors=" +
            std::to_string(destructure.error_count) +
            ", leaked_blocks=" +
            std::to_string(destructure.leaked_block_count) + ")");
        return result;
    }
    static_cast<void>(xir::inline_all_pass_run_on_module(module.get()));
    static_cast<void>(xir::mem2reg_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    auto fast_math_info = xir::FastMathSimplifyInfo{};
    if (enable_fast_math) {
        fast_math_info =
            xir::fast_math_simplify_pass_run_on_module(
                module.get(), {.enable_fast_math = true});
        if (fast_math_info.changed()) {
            static_cast<void>(xir::dce_pass_run_on_module(module.get()));
        }
    }
    auto predication_info =
        schedule::PredicatedIfConversionInfo{};
    if (warp_width != 1u &&
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_PREDICATED_IF")) {
        predication_info =
            schedule::predicate_small_varying_diamonds(xir_kernel);
    }
    if (predication_info.changed()) {
        static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    }
    auto loop_unswitch_info = schedule::SIMDLoopUnswitchInfo{};
    if (!detail::env_flag(
            "LUISA_SIMD_DISABLE_LOOP_UNSWITCH")) {
        loop_unswitch_info =
            schedule::unswitch_invariant_varying_loop_condition(
                xir_kernel);
    }
    if (loop_unswitch_info.changed()) {
        static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    }
    auto result = compile_simd_kernel(
        xir_kernel, warp_width, entry_name, enable_fast_math,
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST"));
    result.fast_math_identity_count = fast_math_info.identity_count;
    result.fast_math_radix_pow_count = fast_math_info.radix_pow_count;
    result.predicated_diamond_count =
        predication_info.if_conversion.converted_diamond_count;
    result.predicated_instruction_count =
        predication_info.if_conversion.hoisted_inst_count;
    result.predicated_phi_count =
        predication_info.if_conversion.replaced_phi_count;
    result.factored_select_count =
        predication_info.select_factoring.factored_select_count;
    result.unswitched_loop_count =
        loop_unswitch_info.unswitch.unswitched_loop_count;
    result.unswitched_cloned_block_count =
        loop_unswitch_info.unswitch.cloned_block_count;
    result.unswitched_cloned_instruction_count =
        loop_unswitch_info.unswitch.cloned_instruction_count;
    result.unswitched_live_out_count =
        loop_unswitch_info.unswitch.merged_live_out_count;
    return result;
}

}// namespace luisa::compute::simd
