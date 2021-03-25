//
// Created by Mike Smith on 2021/3/25.
//

#include <backends/metal/metal_codegen.h>

namespace luisa::compute::metal {

void MetalCodegen::emit(Function f) {
    _scratch << "[[kernel]] void kernel_" << f.uid() << "() {}\n";
}

MetalCodegen::MetalCodegen(compile::Codegen::Scratch &scratch) noexcept
    : Codegen(scratch) {}

}
