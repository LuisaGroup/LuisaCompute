#include "entry.h"
#include "utils.h"
#include <SPIRV/disassemble.h>

namespace lc::spirv {
SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, ShaderOption const &opt) {
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen.generate_binding(kernel);
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    std::vector<unsigned int> words;
    codegen._builder.dump(words);
    auto printers = std::move(codegen).move_print_formats();
    auto props = std::move(codegen._properties);
    auto use_tex2d = codegen._use_tex2d_bindless;
    auto use_tex3d = codegen._use_tex3d_bindless;
    auto use_buffer = codegen._use_buffer_bindless;
    // Leak builder to avoid destructor crash
    codegen._builder_ptr.release();
    return SpirvResult{
        std::move(words),
        std::move(props),
        std::move(printers),
        use_tex2d,
        use_tex3d,
        use_buffer};
}
}// namespace lc::spirv
