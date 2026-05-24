#include "entry.h"
#include "utils.h"
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
    if (auto env = std::getenv("LUISA_SPIRV_OPT_LEVEL")) {
        char *end = nullptr;
        auto val = std::strtol(env, &end, 10);
        if (end != env && *end == '\0') {
            opt_level = static_cast<int>(val);
        }
    }
    if (opt_level == 0) {
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
    if (opt_level == 1) {
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
        LUISA_INFO("SPIR-V optimization level 1 (lightweight passes)");
    } else {
        optimizer.RegisterPerformancePasses();
        LUISA_INFO("SPIR-V optimization level 2 (performance passes)");
    }
    std::vector<uint32_t> optimized;
    if (optimizer.Run(words.data(),
                      words.size(), &optimized)) {
        auto before = words.size();
        words.assign(optimized.begin(), optimized.end());
        LUISA_INFO("SPIR-V optimized (level {}): {} -> {} words ({:.1f}%)",
                   opt_level, before, words.size(),
                   100.0 * static_cast<double>(words.size()) /
                       static_cast<double>(before));
    } else {
        LUISA_WARNING("SPIR-V optimization failed, using unoptimized binary.");
    }
}
SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, const ShaderOption &opt, bool use_native_float_atomics) {
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen._use_native_float_atomics = use_native_float_atomics;
    auto analysis = codegen._analyze_module_usage(xir_module.get());
    codegen._mark_atomic_buffer_types(analysis);

    for (auto c : analysis.used_constants) {
        if (auto t = c->type(); t != nullptr && t->is_array()) {
            codegen._ubo_array_constants.emplace_back(c);
        }
    }

    codegen.generate_binding(kernel);
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    std::vector<uint32_t> words;
    codegen._builder.dump(words);
    luisa_spirv_validate(words, "pre-optimization");
    if (std::getenv("LUISA_DUMP_SPV")) {
        auto filename = luisa::format("/tmp/opencode/kernel_{:016x}.spv", kernel.hash());
        std::ofstream file(filename.c_str(), std::ios::binary);
        file.write(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint32_t));
    }
    luisa_spirv_optimize(words);
    // luisa_spirv_validate(words, "post-optimization");
    LUISA_INFO("SPIR-V compilation successful, binary size: {} words, properties: {} binds",
               words.size(), codegen._properties.size());
    if (std::getenv("LUISA_DUMP_SOURCE")) {
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
    codegen._builder_ptr.release();  // NOLINT: intentional leak to avoid destructor crash
    return SpirvResult{
        std::move(words),
        std::move(props),
        std::move(printers),
        std::move(constant_ubo_data),
        use_tex2d,
        use_tex3d,
        use_buffer};
}
}// namespace lc::spirv
