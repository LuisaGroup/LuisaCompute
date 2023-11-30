#include "../common/shader_print_formatter.h"
#include "cuda_command_encoder.h"
#include "cuda_shader_printer.h"

namespace luisa::compute::cuda {

class CUDAShaderPrinter::Callback : public CUDACallbackContext {

private:
    using Log = DeviceInterface::StreamLogCallback;
    const CUDAShaderPrinter *_printer;
    const void *_data;
    Log _log_callback;

public:
    Callback(const CUDAShaderPrinter *printer, const void *data, Log callback) noexcept
        : _printer{printer}, _data{data}, _log_callback{std::move(callback)} {}

private:
    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<Callback> pool;
        return pool;
    }

public:
    [[nodiscard]] static auto create(const CUDAShaderPrinter *printer,
                                     const void *data,
                                     Log log_callback) noexcept {
        return _pool().create(printer, data, std::move(log_callback));
    }

public:
    void recycle() noexcept override {
        _printer->_do_print(_data, _log_callback);
        _pool().destroy(this);
    }
};

luisa::unique_ptr<CUDAShaderPrinter> CUDAShaderPrinter::create(luisa::span<const std::pair<luisa::string, const Type *>> arg_types) noexcept {
    if (arg_types.empty()) { return nullptr; }
    luisa::vector<luisa::unique_ptr<Formatter>> formatters;
    formatters.reserve(arg_types.size());
    for (auto &&[name, type] : arg_types) {
        formatters.emplace_back(luisa::make_unique<Formatter>(name, type));
    }
    return luisa::make_unique<CUDAShaderPrinter>(std::move(formatters));// TODO
}

luisa::unique_ptr<CUDAShaderPrinter> CUDAShaderPrinter::create(luisa::span<const std::pair<luisa::string, luisa::string>> arg_types) noexcept {
    luisa::vector<std::pair<luisa::string, const Type *>> types;
    types.reserve(arg_types.size());
    for (auto &&[name, type] : arg_types) {
        types.emplace_back(name, Type::from(type));
    }
    return create(types);
}

CUDAShaderPrinter::Binding CUDAShaderPrinter::encode(CUDACommandEncoder &encoder) const noexcept {
    Binding b{
        .capacity = print_buffer_content_capacity,
        .content = 0ull,
    };
    encoder.with_download_pool_no_fallback(
        print_buffer_capacity,
        [&b, &encoder, this](CUDAHostBufferPool::View *temp) noexcept {
            if (temp == nullptr) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to allocate temporary buffer for shader "
                    "printer. Printing is disabled this time.");
                return;
            }
            *reinterpret_cast<size_t *>(temp->address()) = 0ul;
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&b.content, temp->address(), 0u));
            encoder.add_callback(Callback::create(
                this, temp->address(),
                encoder.stream()->log_callback()));
        });
    return b;
}

void CUDAShaderPrinter::_do_print(const void *data, const Log &log) const noexcept {
    struct Head {
        size_t size;
        const std::byte content[];
    };
    auto *head = reinterpret_cast<const Head *>(data);
    auto valid_size = std::min(head->size, print_buffer_content_capacity);
    auto printed_size = format_shader_print(_formatters, luisa::span{head->content, valid_size}, log);
    if (head->size > printed_size) {
        LUISA_WARNING("Device print overflow. {} byte(s) truncated.",
                      head->size - printed_size);
    }
}

CUDAShaderPrinter::CUDAShaderPrinter(vector<unique_ptr<Formatter>> &&formatters) noexcept
    : _formatters{std::move(formatters)} {}

CUDAShaderPrinter::~CUDAShaderPrinter() noexcept = default;

}// namespace luisa::compute::cuda
