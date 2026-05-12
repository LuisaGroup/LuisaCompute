#include "entry.h"
#include "utils.h"

namespace lc::spirv {
SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, ShaderOption const &opt) {
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    // Bindings: compute kernel decorate
    codegen.generate_binding(kernel);

    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    std::vector<unsigned int> words;
    codegen._builder.dump(words);
    auto printers = std::move(codegen).move_print_formats();
    return SpirvResult{
        std::move(words),
        std::move(codegen._properties),
        std::move(printers),
        codegen._use_tex2d_bindless,
        codegen._use_tex3d_bindless,
        codegen._use_buffer_bindless};
}
}// namespace lc::spirv
