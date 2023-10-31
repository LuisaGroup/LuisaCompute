#include "metal_shader_printer.h"

namespace luisa::compute::metal {

MetalShaderPrinter::MetalShaderPrinter(
    luisa::span<const std::pair<luisa::string,
                                const Type *>>
        print_formats) noexcept {
    // TODO
}

MetalShaderPrinter::~MetalShaderPrinter() noexcept = default;

}// namespace luisa::compute::metal
