#pragma once

#include "metal_api.h"
#include "metal_shader_printer.h"

namespace luisa::compute {
class ShaderPrintFormatter;
}// namespace luisa::compute

namespace luisa::compute::metal {

class MetalCommandEncoder;

class MetalShaderPrinter {

public:
    struct Binding {
        size_t capacity;
        uint64_t address;
    };

    struct Encode {

        MTL::Buffer *buffer;
        size_t offset;
        size_t size;

        [[nodiscard]] auto binding() const noexcept {
            return Binding{size, buffer->gpuAddress() + offset};
        }
    };

    class Callback;
    using Formatter = ShaderPrintFormatter;

private:
    luisa::vector<luisa::unique_ptr<ShaderPrintFormatter>> _formatters;
    void _do_print(const void *data) const noexcept;

public:
    explicit MetalShaderPrinter(luisa::span<const std::pair<luisa::string, const Type *>> print_formats) noexcept;
    ~MetalShaderPrinter() noexcept;

public:
    [[nodiscard]] Encode encode(MetalCommandEncoder &encoder) const noexcept;
};

}// namespace luisa::compute::metal
