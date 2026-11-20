#include "hlsl_spirv_opt.h"

#include "../common/env_flag.h"
#include "../common/spirv_fma_rewrite.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

#ifdef LUISA_VK_HAS_SPIRV_TOOLS_OPT
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#endif

namespace lc::vk {

namespace {

#ifdef LUISA_VK_HAS_SPIRV_TOOLS_OPT

[[nodiscard]] spv_target_env spirv_target_env_for_sm(uint shader_model) noexcept {
    if (shader_model > 65u) { return SPV_ENV_VULKAN_1_3; }
    if (shader_model > 60u) { return SPV_ENV_VULKAN_1_2; }
    return SPV_ENV_VULKAN_1_0;
}

// Mirrors DXC's HLSL legalization + performance optimization pipeline while:
// 1. Interleaving CompactIds after every ID-intensive stage so large shaders
//    never exhaust SPIR-V's 4,194,303 ID bound.
// 2. Bounding ScalarReplacement to 100 elements (matching Luisa's native
//    SPIR-V compute preset).
// 3. Skipping LoopUnroll on very large modules (>250k unlegalized words)
//    where SPIRV-Tools loop analysis over hundreds of dynamic loops dominates
//    compile time and inflates IDs without runtime benefit on modern GPU LLVM
//    drivers.
[[nodiscard]] hlsl::CompileResult legalize_and_optimize_fcgl_spirv(
    hlsl::ShaderCompiler *compiler,
    hlsl::ComUniquePtr<IDxcBlob> const &fcgl_blob,
    uint shader_model,
    bool optimize,
    bool promote_fma_khr,
    vstd::string_view profile_label,
    bool profile) noexcept {
    const size_t byte_size = fcgl_blob->GetBufferSize();
    if (byte_size < 20u || (byte_size % sizeof(uint32_t)) != 0u) {
        return vstd::string{"DXC -fcgl produced an invalid SPIR-V buffer."};
    }
    std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
    std::memcpy(words.data(), fcgl_blob->GetBufferPointer(), byte_size);

    const auto target_env = spirv_target_env_for_sm(shader_model);
    spvtools::Optimizer optimizer{target_env};
    std::string opt_diagnostics;
    optimizer.SetMessageConsumer(
        [&](spv_message_level_t level, const char *, const spv_position_t &pos,
            const char *message) {
            if (level <= SPV_MSG_ERROR) {
                if (!opt_diagnostics.empty()) { opt_diagnostics.push_back('\n'); }
                opt_diagnostics += luisa::format("line {}: {}", pos.line, message);
            }
        });

    constexpr uint32_t kSroaLimit = 100u;
    const bool allow_loop_unroll = words.size() <= 250'000u;

    // Stage 1: HLSL control-flow legalization and entry-point inlining.
    optimizer.RegisterPass(spvtools::CreateWrapOpKillPass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
    optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
    optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

    // Stage 2: Private/storage-class legalization and initial load/store elim.
    optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
    optimizer.RegisterPass(spvtools::CreateFixStorageClassPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

    // Stage 3: Bounded SROA and access-chain legalization.
    optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(kSroaLimit));
    optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

    // Stage 4: Multi-store elimination and access-chain folding.
    optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateCombineAccessChainsPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

    if (optimize) {
        // Stage 5: Constant propagation, optional loop unroll, simplification.
        optimizer.RegisterPass(spvtools::CreateCCPPass());
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        if (allow_loop_unroll) {
            optimizer.RegisterPass(spvtools::CreateLoopUnrollPass(true));
        }
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        optimizer.RegisterPass(spvtools::CreateCombineAccessChainsPass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

        // Stage 6: Second SROA + SSA rewrite.
        optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(kSroaLimit));
        optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
        optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
        optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateSSARewritePass());
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

        // Stage 7: Vector/CFG/CSE cleanup.
        optimizer.RegisterPass(spvtools::CreateVectorDCEPass());
        optimizer.RegisterPass(spvtools::CreateDeadInsertElimPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateIfConversionPass());
        optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
        optimizer.RegisterPass(spvtools::CreateReduceLoadSizePass());
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
    }

    optimizer.RegisterPass(spvtools::CreateSpreadVolatileSemanticsPass());
    optimizer.RegisterPass(spvtools::CreateRemoveUnusedInterfaceVariablesPass());
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

    spvtools::OptimizerOptions opt_options;
    // Unlegalized DXC -fcgl output intentionally contains temporary Private
    // resource pointers prior to legalization; validate after optimization.
    opt_options.set_run_validator(false);

    luisa::Clock opt_clock;
    std::vector<uint32_t> optimized;
    if (!optimizer.Run(words.data(), words.size(), &optimized, opt_options) ||
        optimized.empty()) {
        return vstd::string{luisa::format(
            "SPIRV-Tools legalization/optimization failed: {}",
            opt_diagnostics.empty() ? "unknown error" : opt_diagnostics)};
    }
    const double opt_ms = opt_clock.toc();

    if (promote_fma_khr) {
        static_cast<void>(
            luisa::compute::detail::promote_glsl_fma_to_op_fma_khr(optimized));
    }

    luisa::Clock val_clock;
    spvtools::SpirvTools validator{target_env};
    std::string val_diagnostics;
    validator.SetMessageConsumer(
        [&](spv_message_level_t level, const char *, const spv_position_t &pos,
            const char *message) {
            if (level <= SPV_MSG_ERROR) {
                if (!val_diagnostics.empty()) { val_diagnostics.push_back('\n'); }
                val_diagnostics += luisa::format("line {}: {}", pos.line, message);
            }
        });
    if (!validator.Validate(optimized)) {
        return vstd::string{luisa::format(
            "SPIRV-Tools post-legalization validation failed: {}",
            val_diagnostics.empty() ? "unknown error" : val_diagnostics)};
    }
    const double val_ms = val_clock.toc();

    if (profile) {
        LUISA_INFO(
            "Vulkan HLSL-to-SPIR-V SPIRV-Tools legalize+opt '{}' "
            "(in_words={}, out_words={}, opt={:.3f} ms, val={:.3f} ms)",
            profile_label, words.size(), optimized.size(), opt_ms, val_ms);
    }

    IDxcBlobEncoding *blob_ptr{};
    auto hr = compiler->utils()->CreateBlob(
        optimized.data(),
        static_cast<UINT32>(optimized.size() * sizeof(uint32_t)),
        CP_ACP,
        &blob_ptr);
    if (hr != S_OK || blob_ptr == nullptr) {
        return vstd::string{"Failed to wrap optimized SPIR-V into IDxcBlob."};
    }
    return hlsl::ComUniquePtr<IDxcBlob>{blob_ptr};
}

[[nodiscard]] bool should_use_external_spirv_opt(
    size_t hlsl_bytes, bool optimize, bool debug) noexcept {
    if (debug) { return false; }
    if (const char *env = std::getenv("LUISA_DXC_SPIRV_EXTERNAL_OPT");
        env != nullptr && *env != '\0') {
        vstd::string_view v{env};
        if (v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF") {
            return false;
        }
        return true;
    }
    // Shaders >= 512 KiB HLSL routinely contain >100 loops and large callable
    // chains where DXC's built-in uncompacted legalization stalls or overflows
    // SPIR-V's 4,194,303 ID bound.
    constexpr size_t kLargeHlslThreshold = 512u * 1024u;
    return optimize && hlsl_bytes >= kLargeHlslThreshold;
}

#endif

}// namespace

hlsl::CompileResult compile_hlsl_compute_to_spirv(
    hlsl::ShaderCompiler *compiler,
    vstd::string_view code,
    bool optimize,
    uint shader_model,
    bool enable_unsafe_math,
    bool debug,
    vstd::string_view entry_point,
    luisa::span<const luisa::filesystem::path> include_dirs,
    hlsl::SpirvShaderCapabilities spirv_capabilities,
    vstd::string_view profile_label) noexcept {
    const bool profile =
        luisa::compute::detail::env_flag("LUISA_VULKAN_PROFILE_COMPILATION");
#ifdef LUISA_VK_HAS_SPIRV_TOOLS_OPT
    auto compile_via_fcgl = [&]() -> hlsl::CompileResult {
        luisa::Clock fcgl_clock;
        auto fcgl_result = compiler->compile_compute(
            code,
            /*optimize=*/false,
            shader_model,
            enable_unsafe_math,
            /*spirv=*/true,
            debug,
            entry_point,
            include_dirs,
            spirv_capabilities,
            /*spirv_fcgl=*/true);
        if (fcgl_result.is_type_of<vstd::string>()) {
            return fcgl_result;
        }
        if (profile) {
            LUISA_INFO(
                "Vulkan HLSL-to-SPIR-V DXC -fcgl '{}' (hlsl={} B, words={}): {:.3f} ms",
                profile_label, code.size(),
                fcgl_result.get<0>()->GetBufferSize() / sizeof(uint32_t),
                fcgl_clock.toc());
        }
        return legalize_and_optimize_fcgl_spirv(
            compiler,
            fcgl_result.get<0>(),
            shader_model,
            optimize,
            spirv_capabilities.fma,
            profile_label,
            profile);
    };

    if (should_use_external_spirv_opt(code.size(), optimize, debug)) {
        return compile_via_fcgl();
    }
#endif
    auto result = compiler->compile_compute(
        code,
        optimize,
        shader_model,
        enable_unsafe_math,
        /*spirv=*/true,
        debug,
        entry_point,
        include_dirs,
        spirv_capabilities,
        /*spirv_fcgl=*/false);
#ifdef LUISA_VK_HAS_SPIRV_TOOLS_OPT
    if (!debug && result.is_type_of<vstd::string>()) {
        auto const &err = result.get<1>();
        if (err.find("ID overflow") != vstd::string::npos) {
            LUISA_WARNING(
                "DXC SPIR-V legalization hit ID overflow ({} B HLSL); "
                "retrying via DXC -fcgl + SPIRV-Tools compact legalization.",
                code.size());
            return compile_via_fcgl();
        }
    }
#endif
    return result;
}

}// namespace lc::vk
