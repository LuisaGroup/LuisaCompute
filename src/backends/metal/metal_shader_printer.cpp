#include "../common/shader_print_formatter.h"
#include "metal_callback_context.h"
#include "metal_command_encoder.h"
#include "metal_shader_printer.h"

namespace luisa::compute::metal {

inline constexpr auto metal_shader_printer_capacity = 1_M;
inline constexpr auto metal_shader_printer_content_capacity =
    metal_shader_printer_capacity - sizeof(size_t);

MetalShaderPrinter::MetalShaderPrinter(
    luisa::span<const std::pair<luisa::string, const Type *>> print_formats) noexcept {
    _formatters.reserve(print_formats.size());
    for (auto &&[fmt, type] : print_formats) {
        _formatters.emplace_back(luisa::make_unique<ShaderPrintFormatter>(fmt, type));
    }
}

MetalShaderPrinter::~MetalShaderPrinter() noexcept = default;

class MetalShaderPrinter::Callback : public MetalCallbackContext {

private:
    const MetalShaderPrinter *_printer;
    const void *_data;

private:
    Callback(const MetalShaderPrinter *printer,
             const void *data) noexcept
        : _printer{printer}, _data{data} {}

    [[nodiscard]] static auto _pool() noexcept {
        static Pool<Callback> pool;
        return &pool;
    }

    void recycle() noexcept override {
        _printer->_do_print(_data);
        _pool()->destroy(this);
    }

public:
    [[nodiscard]] static auto create(const MetalShaderPrinter *printer,
                                     const void *data) noexcept {
        return _pool()->create(Callback{printer, data});
    }
};

MetalShaderPrinter::Encode MetalShaderPrinter::encode(MetalCommandEncoder &encoder) const noexcept {
    Encode e{};
    encoder.with_download_buffer(
        metal_shader_printer_capacity,
        [&](MetalStageBufferPool::Allocation *buffer) noexcept {
            e = {.buffer = buffer->buffer(),
                 .offset = buffer->offset(),
                 .size = buffer->size()};
            auto data = buffer->data();
            *reinterpret_cast<size_t *>(data) = 0ul;
            encoder.add_callback(Callback::create(this, data));
        });
    return e;
}

void MetalShaderPrinter::_do_print(const void *data) const noexcept {
    struct Head {
        size_t size;
        const std::byte content[];
    };
    auto *head = reinterpret_cast<const Head *>(data);
    auto valid_size = std::min(head->size, metal_shader_printer_content_capacity);
    auto valid_content = luisa::span{head->content, valid_size};
    auto printed_size = format_shader_print(_formatters, valid_content);
    if (head->size > printed_size) {
        LUISA_WARNING("Device print overflow. {} byte(s) truncated.",
                      head->size - printed_size);
    }
}

}// namespace luisa::compute::metal
