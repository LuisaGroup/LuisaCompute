#include "entry.h"
#include "utils.h"
#include <cstring>

namespace lc::spirv {
SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, ShaderOption const &opt) {
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    std::vector<unsigned int> words;
    codegen._builder.dump(words);
    auto printers = std::move(codegen).move_print_formats();
    auto byte_size = words.size() * sizeof(unsigned int);
    auto ptr = new std::byte[byte_size];
    std::memcpy(ptr, words.data(), byte_size);
    luisa::BinaryBlob blob{
        ptr,
        byte_size,
        [](void *p) { delete[] static_cast<std::byte *>(p); }};
    return SpirvResult{
        std::move(blob),
        {},
        std::move(printers),
        false,
        false,
        false};
}
}// namespace lc::spirv
