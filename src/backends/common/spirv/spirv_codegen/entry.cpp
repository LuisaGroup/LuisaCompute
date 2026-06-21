#include "entry.h"
#include "utils.h"
#include "../../backend_print_code.h"
#include <SPIRV/disassemble.h>
#include <luisa/core/logging.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
// #include <spirv-tools/optimizer.hpp>

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace lc::spirv {

static void luisa_spirv_validate(luisa::span<const uint32_t> words, luisa::string_view stage) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    luisa::string message;
    tools.SetMessageConsumer(
        [&message](spv_message_level_t level, const char *source,
                   const spv_position_t &position, const char *text) {
            auto level_name = [level]() noexcept {
                switch (level) {
                    case SPV_MSG_FATAL: return "fatal";
                    case SPV_MSG_INTERNAL_ERROR: return "internal";
                    case SPV_MSG_ERROR: return "error";
                    case SPV_MSG_WARNING: return "warning";
                    case SPV_MSG_INFO: return "info";
                    case SPV_MSG_DEBUG: return "debug";
                }
                return "unknown";
            }();
            message.append(luisa::format("{} [{}:{}:{}]: {}\n",
                                         level_name,
                                         source == nullptr ? "" : source,
                                         position.line,
                                         position.column,
                                         text == nullptr ? "" : text));
        });
    spvtools::ValidatorOptions options;
    if (!tools.Validate(words.data(), words.size(), options)) {
        LUISA_ERROR("SPIR-V validation failed at {} stage:\n{}", stage, message);
    }
}

static void luisa_spirv_optimize(std::vector<uint32_t> &words) {
    int opt_level = 2;
    luisa::string pass_preset;
    if (auto env = std::getenv("LUISA_SPIRV_OPT_LEVEL")) {
        char *end = nullptr;
        auto val = std::strtol(env, &end, 10);
        if (end != env && *end == '\0') {
            opt_level = static_cast<int>(val);
        }
    }
    if (auto env = std::getenv("LUISA_SPIRV_OPT_PASSES")) {
        pass_preset = env;
    }
    if (opt_level == 0 && pass_preset.empty()) {
        LUISA_INFO("SPIR-V optimization skipped (LUISA_SPIRV_OPT_LEVEL=0)");
        return;
    }
    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_2);
    optimizer.SetMessageConsumer(
        [](spv_message_level_t level, const char *source,
           const spv_position_t &position, const char *message) {
            switch (level) {
                case SPV_MSG_FATAL:
                case SPV_MSG_INTERNAL_ERROR:
                    LUISA_ERROR("SPIRV-Tools [{}:{}]: {}",
                                position.line, position.column, message);
                    break;
                case SPV_MSG_ERROR:
                case SPV_MSG_WARNING:
                    LUISA_WARNING("SPIRV-Tools [{}:{}]: {}",
                                  position.line, position.column, message);
                    break;
                case SPV_MSG_INFO:
                case SPV_MSG_DEBUG:
                    LUISA_INFO("SPIRV-Tools [{}:{}]: {}",
                               position.line, position.column, message);
                    break;
            }
        });

    // Determine effective pass preset
    if (pass_preset.empty()) {
        if (opt_level == 0) pass_preset = "none";
        else if (opt_level == 1) pass_preset = "lightweight";
        else if (opt_level == 2) pass_preset = "compute";
        else pass_preset = "full";
    }

    if (pass_preset == "none") {
        LUISA_INFO("SPIR-V optimization skipped (preset=none)");
        return;
    } else if (pass_preset == "lightweight") {
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        LUISA_INFO("SPIR-V optimization preset 'lightweight' (level {})", opt_level);
    } else if (pass_preset == "compute") {
        // Compute-oriented performance passes (curated for Vulkan compute)
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
        optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
        optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
        optimizer.RegisterPass(spvtools::CreateLoopUnrollPass(true));
        optimizer.RegisterPass(spvtools::CreateCCPPass());
        optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(100));
        optimizer.RegisterPass(spvtools::CreateIfConversionPass());
        optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
        optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
        LUISA_INFO("SPIR-V optimization preset 'compute' (level {})", opt_level);
    } else if (pass_preset == "full") {
        // Full performance pass suite (original behavior)
        optimizer.RegisterPerformancePasses();
        optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
        optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
        LUISA_INFO("SPIR-V optimization preset 'full' (level {})", opt_level);
    } else {
        LUISA_WARNING("Unknown SPIR-V optimization preset '{}', falling back to compute", pass_preset);
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
        optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
        optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
        optimizer.RegisterPass(spvtools::CreateLoopUnrollPass(true));
        optimizer.RegisterPass(spvtools::CreateCCPPass());
        optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
        optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
    }
    // Fixed-point iteration: run the optimizer repeatedly until no further
    // changes (within a max iteration limit to prevent infinite loops).
    std::vector<uint32_t> optimized;
    auto before = words.size();
    for (int iter = 0; iter < 5; ++iter) {
        if (!optimizer.Run(words.data(), words.size(), &optimized)) {
            LUISA_WARNING("SPIR-V optimization failed at iteration {}, using unoptimized binary.", iter);
            break;
        }
        if (optimized.size() == words.size()) {
            // No change, stop iterating
            break;
        }
        words.assign(optimized.begin(), optimized.end());
        optimized.clear();
    }
    if (words.size() != before) {
        LUISA_INFO("SPIR-V optimized (level {}): {} -> {} words ({:.1f}%)",
                   opt_level, before, words.size(),
                   100.0 * static_cast<double>(words.size()) /
                       static_cast<double>(before));
    }
}

vstd::vector<std::pair<Variable, Usage>>
SpirvCodegenEntry::_collect_kernel_argument_usages(Function kernel, const xir::Module *module) const noexcept {
    const xir::KernelFunction *xir_kernel = nullptr;
    for (auto f : module->function_list()) {
        if (f->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
            xir_kernel = static_cast<const xir::KernelFunction *>(f);
            break;
        }
    }
    auto ast_args = kernel.arguments();
    vstd::vector<std::pair<Variable, Usage>> result;
    result.reserve(ast_args.size());
    luisa::vector<const xir::Argument *> xir_args;
    if (xir_kernel != nullptr) {
        for (auto arg : xir_kernel->arguments()) {
            xir_args.emplace_back(arg);
        }
    }
    for (auto i = 0u; i < ast_args.size(); i++) {
        auto ast_arg = ast_args[i];
        auto usage = kernel.variable_usage(ast_arg.uid());
        if (i < xir_args.size()) {
            usage = _function_argument_usage_of(xir_kernel, xir_args[i]);
        }
        result.emplace_back(ast_arg, usage);
    }
    return result;
}

SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, const ShaderOption &opt, bool use_native_float_atomics) {
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen._use_native_float_atomics = use_native_float_atomics;
    auto analysis = codegen._analyze_module_usage(xir_module.get());
    codegen._mark_atomic_buffer_types(analysis);
    codegen._analyze_function_argument_usage(xir_module.get());
    auto argument_usages = codegen._collect_kernel_argument_usages(kernel, xir_module.get());

    for (auto c : analysis.used_constants) {
        if (auto t = c->type(); t != nullptr && t->is_array()) {
            codegen._ubo_array_constants.emplace_back(c);
        }
    }

    codegen.generate_binding(kernel, argument_usages);
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    std::vector<uint32_t> words;
    codegen._builder.dump(words);
    if (luisa::compute::backend_print_code_enabled()) {
        std::ostringstream disasm;
        spv::Disassemble(disasm, words);
        LUISA_VERBOSE("=== PRE-VALIDATION SPIR-V for {} (size={}) ===\n{}", kernel.name(), words.size(), disasm.str());
    }
    luisa_spirv_validate(words, "pre-optimization");
    if (std::getenv("LUISA_DUMP_SPV")) {
        auto filename = luisa::format("/tmp/opencode/kernel_{:016x}.spv", kernel.hash());
        std::ofstream file(filename.c_str(), std::ios::binary);
        file.write(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint32_t));
    }
    luisa_spirv_optimize(words);
    luisa_spirv_validate(words, "post-optimization");
    LUISA_INFO("SPIR-V compilation successful, binary size: {} words, properties: {} binds",
               words.size(), codegen._properties.size());
    if (luisa::compute::backend_print_code_enabled()) {
        std::ostringstream disasm;
        spv::Disassemble(disasm, words);
        LUISA_INFO("=== Kernel: {} (size={}) ===\n{}", kernel.name(), words.size(), disasm.str());
    }
    auto printers = std::move(codegen).move_print_formats();
    auto props = std::move(codegen._properties);
    auto use_tex2d = codegen._use_tex2d_bindless;
    auto use_tex3d = codegen._use_tex3d_bindless;
    auto use_buffer = codegen._use_buffer_bindless;
    auto constant_ubo_data = std::move(codegen._constant_ubo_data);
    // Leak builder to avoid destructor crash
    codegen._builder_ptr.release();// NOLINT: intentional leak to avoid destructor crash
    return SpirvResult{
        std::move(words),
        std::move(props),
        std::move(argument_usages),
        std::move(printers),
        std::move(constant_ubo_data),
        use_tex2d,
        use_tex3d,
        use_buffer};
}
}// namespace lc::spirv
