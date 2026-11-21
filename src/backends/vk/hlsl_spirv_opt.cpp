#include "hlsl_spirv_opt.h"

#include "../common/env_flag.h"
#include "../common/spirv_fma_rewrite.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// Marks large multi-call non-opaque functions in a DXC `-fcgl` SPIR-V module
// with `FunctionControlMask::DontInline` (0x2) before SPIRV-Tools'
// `InlineExhaustivePass`, matching the cost-bounded outlining policy of
// Luisa's native XIR-to-SPIR-V path.
//
// Only functions whose signature contains pure value types or Function-scope
// pointers to pure value types are eligible; any function taking resource,
// StorageBuffer/Uniform/Private pointer, or opaque arguments remains unmarked
// so `InlineExhaustivePass` still legalizes every resource/opaque parameter.
[[nodiscard]] size_t mark_costly_callables_dont_inline(
    std::vector<uint32_t> &words) noexcept {
    constexpr uint32_t kSpvMagic = 0x07230203u;
    constexpr uint32_t kOpEntryPoint = 15u;
    constexpr uint32_t kOpTypeVoid = 19u;
    constexpr uint32_t kOpTypeBool = 20u;
    constexpr uint32_t kOpTypeInt = 21u;
    constexpr uint32_t kOpTypeFloat = 22u;
    constexpr uint32_t kOpTypeVector = 23u;
    constexpr uint32_t kOpTypeMatrix = 24u;
    constexpr uint32_t kOpTypeArray = 28u;
    constexpr uint32_t kOpTypeStruct = 30u;
    constexpr uint32_t kOpTypePointer = 32u;
    constexpr uint32_t kOpTypeFunction = 33u;
    constexpr uint32_t kOpFunction = 54u;
    constexpr uint32_t kOpFunctionParameter = 55u;
    constexpr uint32_t kOpFunctionEnd = 56u;
    constexpr uint32_t kOpFunctionCall = 57u;
    constexpr uint32_t kOpVariable = 59u;
    constexpr uint32_t kOpLoopMerge = 246u;
    constexpr uint32_t kStorageClassFunction = 7u;
    constexpr uint32_t kFunctionControlDontInline = 0x2u;

    if (words.size() < 5u || words[0] != kSpvMagic) { return 0u; }

    struct TypeInfo {
        uint32_t opcode{0u};
        uint32_t storage_class{0u};
        std::vector<uint32_t> sub_types;
    };
    struct FuncInfo {
        size_t header_word_idx{0u};
        uint32_t func_type_id{0u};
        uint32_t inst_count{0u};
        uint32_t call_count{0u};
        bool has_loop{false};
        bool is_entry_point{false};
        bool has_body{false};
    };

    std::unordered_map<uint32_t, TypeInfo> types;
    std::unordered_map<uint32_t, FuncInfo> funcs;
    std::unordered_set<uint32_t> entry_points;

    uint32_t cur_func_id = 0u;
    for (size_t pos = 5u; pos < words.size();) {
        const uint32_t hdr = words[pos];
        const uint32_t wc = hdr >> 16u;
        const uint32_t op = hdr & 0xFFFFu;
        if (wc == 0u || pos + wc > words.size()) { return 0u; }

        switch (op) {
            case kOpEntryPoint:
                if (wc >= 3u) { entry_points.insert(words[pos + 2u]); }
                break;
            case kOpTypeVoid:
            case kOpTypeBool:
            case kOpTypeInt:
            case kOpTypeFloat:
                if (wc >= 2u) { types[words[pos + 1u]] = TypeInfo{op, 0u, {}}; }
                break;
            case kOpTypeVector:
            case kOpTypeMatrix:
            case kOpTypeArray:
                if (wc >= 3u) {
                    types[words[pos + 1u]] = TypeInfo{op, 0u, {words[pos + 2u]}};
                }
                break;
            case kOpTypeStruct:
                if (wc >= 2u) {
                    TypeInfo ti{op, 0u, {}};
                    ti.sub_types.reserve(wc - 2u);
                    for (uint32_t k = 2u; k < wc; ++k) {
                        ti.sub_types.push_back(words[pos + k]);
                    }
                    types[words[pos + 1u]] = std::move(ti);
                }
                break;
            case kOpTypePointer:
                if (wc >= 4u) {
                    types[words[pos + 1u]] =
                        TypeInfo{op, words[pos + 2u], {words[pos + 3u]}};
                }
                break;
            case kOpTypeFunction:
                if (wc >= 3u) {
                    TypeInfo ti{op, 0u, {}};
                    ti.sub_types.reserve(wc - 2u);
                    for (uint32_t k = 2u; k < wc; ++k) {
                        ti.sub_types.push_back(words[pos + k]);
                    }
                    types[words[pos + 1u]] = std::move(ti);
                }
                break;
            case kOpFunction:
                if (wc >= 5u) {
                    cur_func_id = words[pos + 2u];
                    auto &fi = funcs[cur_func_id];
                    fi.header_word_idx = pos;
                    fi.func_type_id = words[pos + 4u];
                }
                break;
            case kOpFunctionEnd:
                cur_func_id = 0u;
                break;
            case kOpFunctionParameter:
                break;
            case kOpVariable:
                if (cur_func_id != 0u) {
                    funcs[cur_func_id].has_body = true;
                    funcs[cur_func_id].inst_count++;
                }
                break;
            case kOpFunctionCall:
                if (wc >= 4u) {
                    funcs[words[pos + 3u]].call_count++;
                }
                if (cur_func_id != 0u) {
                    funcs[cur_func_id].has_body = true;
                    funcs[cur_func_id].inst_count++;
                }
                break;
            case kOpLoopMerge:
                if (cur_func_id != 0u) {
                    funcs[cur_func_id].has_body = true;
                    funcs[cur_func_id].has_loop = true;
                    funcs[cur_func_id].inst_count++;
                }
                break;
            default:
                if (cur_func_id != 0u) {
                    funcs[cur_func_id].has_body = true;
                    funcs[cur_func_id].inst_count++;
                }
                break;
        }
        pos += wc;
    }

    for (uint32_t ep : entry_points) {
        if (auto it = funcs.find(ep); it != funcs.end()) {
            it->second.is_entry_point = true;
        }
    }

    std::unordered_map<uint32_t, int8_t> value_type_memo;
    auto is_legal_value_type = [&](auto &&self, uint32_t tid,
                                   bool allow_void, uint32_t depth) -> bool {
        if (depth > 32u) { return false; }
        auto it = types.find(tid);
        if (it == types.end()) { return false; }
        const uint32_t op = it->second.opcode;
        if (op == kOpTypeVoid) { return allow_void; }
        if (auto mit = value_type_memo.find(tid); mit != value_type_memo.end()) {
            return mit->second > 0;
        }
        bool ok = false;
        switch (op) {
            case kOpTypeBool:
            case kOpTypeInt:
            case kOpTypeFloat:
                ok = true;
                break;
            case kOpTypeVector:
            case kOpTypeMatrix:
            case kOpTypeArray:
                ok = !it->second.sub_types.empty() &&
                     self(self, it->second.sub_types[0], false, depth + 1u);
                break;
            case kOpTypeStruct:
                ok = true;
                for (uint32_t mem : it->second.sub_types) {
                    if (!self(self, mem, false, depth + 1u)) {
                        ok = false;
                        break;
                    }
                }
                break;
            default:
                ok = false;
                break;
        }
        value_type_memo[tid] = ok ? 1 : 0;
        return ok;
    };

    auto has_legal_outlined_signature = [&](uint32_t func_type_id) -> bool {
        auto it = types.find(func_type_id);
        if (it == types.end() || it->second.opcode != kOpTypeFunction ||
            it->second.sub_types.empty()) {
            return false;
        }
        if (!is_legal_value_type(
                is_legal_value_type, it->second.sub_types[0], true, 0u)) {
            return false;
        }
        for (size_t k = 1u; k < it->second.sub_types.size(); ++k) {
            const uint32_t pid = it->second.sub_types[k];
            if (is_legal_value_type(is_legal_value_type, pid, false, 0u)) {
                continue;
            }
            auto pit = types.find(pid);
            if (pit != types.end() &&
                pit->second.opcode == kOpTypePointer &&
                pit->second.storage_class == kStorageClassFunction &&
                !pit->second.sub_types.empty() &&
                is_legal_value_type(
                    is_legal_value_type, pit->second.sub_types[0], false, 0u)) {
                continue;
            }
            return false;
        }
        return true;
    };

    size_t marked_count = 0u;
    for (auto &[fid, fi] : funcs) {
        if (fi.is_entry_point || !fi.has_body || fi.header_word_idx == 0u) {
            continue;
        }
        // Keep single-use helpers and small leaf helpers inlined for peak
        // runtime perf; outline multi-call loops and large multi-call helpers
        // so InlineExhaustivePass does not clone them hundreds of times.
        const bool costly_multi_call =
            fi.call_count >= 2u &&
            fi.inst_count >= 25u &&
            (fi.has_loop ||
             static_cast<uint64_t>(fi.call_count) * fi.inst_count >= 512u);
        if (!costly_multi_call) { continue; }
        if ((words[fi.header_word_idx + 3u] & 0x1u) != 0u) { continue; }
        if (!has_legal_outlined_signature(fi.func_type_id)) { continue; }
        words[fi.header_word_idx + 3u] |= kFunctionControlDontInline;
        ++marked_count;
    }
    return marked_count;
}

// Mirrors DXC's HLSL legalization + performance optimization pipeline while:
// 1. Interleaving CompactIds after every ID-intensive stage so large shaders
//    never exhaust SPIR-V's 4,194,303 ID bound.
// 2. Bounding ScalarReplacement to 100 elements (matching Luisa's native
//    SPIR-V compute preset).
// 3. Keeping costly multi-call non-opaque callables outlined during
//    InlineExhaustivePass (with automatic fallback to exhaustive inlining if
//    validation fails) and skipping LoopUnroll on very large modules (>250k
//    unlegalized words).
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
    std::vector<uint32_t> orig_words(byte_size / sizeof(uint32_t));
    std::memcpy(orig_words.data(), fcgl_blob->GetBufferPointer(), byte_size);

    const auto target_env = spirv_target_env_for_sm(shader_model);
    constexpr uint32_t kSroaLimit = 100u;

    auto run_pipeline = [&](bool allow_outlining,
                            std::vector<uint32_t> &optimized,
                            std::string &error_out,
                            size_t &outlined_funcs,
                            double &opt_ms,
                            double &val_ms) -> bool {
        std::vector<uint32_t> words = orig_words;
        outlined_funcs = 0u;
        if (allow_outlining && optimize && words.size() > 150'000u) {
            outlined_funcs = mark_costly_callables_dont_inline(words);
        }

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

        const bool allow_loop_unroll = words.size() <= 250'000u;

        // Stage 1: HLSL control-flow legalization and entry-point inlining.
        optimizer.RegisterPass(spvtools::CreateWrapOpKillPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
        optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
        if (outlined_funcs > 0u) {
            // Ensure any remaining outlined call passes legal memory objects
            // (not OpAccessChain pointers) after callers are inlined.
            optimizer.RegisterPass(spvtools::CreateFixFuncCallArgumentsPass());
        }
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

        if (outlined_funcs > 0u) {
            // Clear synthetic DontInline hints so the GPU driver's backend
            // retains full freedom over final call/inline decisions (matching
            // native XIR-to-SPIR-V emission).
            optimizer.RegisterPass(spvtools::CreateRemoveDontInlinePass());
        }
        optimizer.RegisterPass(spvtools::CreateSpreadVolatileSemanticsPass());
        optimizer.RegisterPass(spvtools::CreateRemoveUnusedInterfaceVariablesPass());
        optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

        spvtools::OptimizerOptions opt_options;
        // Unlegalized DXC -fcgl output intentionally contains temporary Private
        // resource pointers prior to legalization; validate after optimization.
        opt_options.set_run_validator(false);

        luisa::Clock opt_clock;
        optimized.clear();
        if (!optimizer.Run(words.data(), words.size(), &optimized, opt_options) ||
            optimized.empty()) {
            error_out = luisa::format(
                "SPIRV-Tools legalization/optimization failed: {}",
                opt_diagnostics.empty() ? "unknown error" : opt_diagnostics);
            return false;
        }
        opt_ms = opt_clock.toc();

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
            error_out = luisa::format(
                "SPIRV-Tools post-legalization validation failed: {}",
                val_diagnostics.empty() ? "unknown error" : val_diagnostics);
            return false;
        }
        val_ms = val_clock.toc();
        return true;
    };

    std::vector<uint32_t> optimized;
    std::string error_msg;
    size_t outlined_funcs = 0u;
    double opt_ms = 0.0;
    double val_ms = 0.0;
    bool ok = run_pipeline(true, optimized, error_msg, outlined_funcs, opt_ms, val_ms);
    if (!ok && outlined_funcs > 0u) {
        LUISA_WARNING(
            "Vulkan HLSL-to-SPIR-V outlined legalization failed ({} outlined funcs): {}; "
            "retrying with exhaustive inlining.",
            outlined_funcs, error_msg);
        error_msg.clear();
        ok = run_pipeline(false, optimized, error_msg, outlined_funcs, opt_ms, val_ms);
    }
    if (!ok) {
        return vstd::string{error_msg};
    }

    if (profile) {
        LUISA_INFO(
            "Vulkan HLSL-to-SPIR-V SPIRV-Tools legalize+opt '{}' "
            "(in_words={}, out_words={}, outlined={}, opt={:.3f} ms, val={:.3f} ms)",
            profile_label, orig_words.size(), optimized.size(),
            outlined_funcs, opt_ms, val_ms);
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
