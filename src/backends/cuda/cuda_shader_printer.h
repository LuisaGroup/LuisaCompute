#pragma once

#include <cuda.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {
class ShaderPrintFormatter;
}// namespace luisa::compute

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAShaderPrinter {

private:
    static constexpr auto print_buffer_capacity = 1_M;// 1MB
    static constexpr auto print_buffer_content_capacity = print_buffer_capacity - sizeof(size_t);

public:
    struct Binding {
        size_t capacity;
        CUdeviceptr content;
    };

    class Callback;
    using Formatter = ShaderPrintFormatter;

private:
    luisa::vector<luisa::unique_ptr<Formatter>> _formatters;

private:
    using Log = DeviceInterface::StreamLogCallback;
    void _do_print(const void *data, const Log &log) const noexcept;

public:
    explicit CUDAShaderPrinter(luisa::vector<luisa::unique_ptr<Formatter>> &&formatters) noexcept;
    ~CUDAShaderPrinter() noexcept;

    [[nodiscard]] static luisa::unique_ptr<CUDAShaderPrinter>
    create(luisa::span<const std::pair<luisa::string, const Type *>> arg_types) noexcept;

    [[nodiscard]] static luisa::unique_ptr<CUDAShaderPrinter>
    create(luisa::span<const std::pair<luisa::string, luisa::string>> arg_types) noexcept;

public:
    [[nodiscard]] Binding encode(CUDACommandEncoder &encoder) const noexcept;
};

}// namespace luisa::compute::cuda
