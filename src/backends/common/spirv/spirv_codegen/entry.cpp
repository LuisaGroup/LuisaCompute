#include "entry.h"
#include "utils.h"
#include <SPIRV/disassemble.h>
#include <luisa/core/logging.h>
#include <sstream>
#include <spirv-tools/optimizer.hpp>

namespace lc::spirv {

static void luisa_spirv_optimize(std::vector<uint32_t> &words) {
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
    optimizer.RegisterPerformancePasses();
    std::vector<uint32_t> optimized;
    if (optimizer.Run(words.data(),
                      words.size(), &optimized)) {
        auto before = words.size();
        words.assign(optimized.begin(), optimized.end());
        LUISA_INFO("SPIR-V optimized: {} -> {} words ({:.1f}%)",
                   before, words.size(),
                   100.0 * static_cast<double>(words.size()) /
                       static_cast<double>(before));
    } else {
        LUISA_WARNING("SPIR-V optimization failed, using unoptimized binary.");
    }
}

SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, const ShaderOption &opt) {
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen.generate_binding(kernel);
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    std::vector<uint32_t> words;
    codegen._builder.dump(words);
    luisa_spirv_optimize(words);
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
    // Leak builder to avoid destructor crash
    codegen._builder_ptr.release();  // NOLINT: intentional leak to avoid destructor crash
    return SpirvResult{
        std::move(words),
        std::move(props),
        std::move(printers),
        use_tex2d,
        use_tex3d,
        use_buffer};
}
}// namespace lc::spirv
