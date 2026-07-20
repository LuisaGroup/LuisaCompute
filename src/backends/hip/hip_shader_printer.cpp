#include <algorithm>

#include <luisa/ast/type_registry.h>
#include <luisa/core/pool.h>

#include "../common/shader_print_formatter.h"
#include "hip_check.h"
#include "hip_command_encoder.h"
#include "hip_shader_printer.h"
#include "hip_stage_buffer_pool.h"
#include "hip_stream.h"

namespace luisa::compute::hip {

class HIPShaderPrinter::Callback final : public HIPCallbackContext {

private:
    const HIPShaderPrinter *_printer;
    HIPStageBufferPool::View *_host_buffer;
    Log _log_callback;

    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<Callback> pool;
        return pool;
    }

public:
    Callback(const HIPShaderPrinter *printer,
             HIPStageBufferPool::View *host_buffer,
             Log log_callback) noexcept
        : _printer{printer},
          _host_buffer{host_buffer},
          _log_callback{std::move(log_callback)} {}

    [[nodiscard]] static auto create(
        const HIPShaderPrinter *printer,
        HIPStageBufferPool::View *host_buffer,
        Log log_callback) noexcept {
        return _pool().create(printer, host_buffer, std::move(log_callback));
    }

    [[nodiscard]] auto host_address() const noexcept {
        return _host_buffer->address();
    }

    void recycle() noexcept override {
        _printer->_do_print(host_address(), _log_callback);
        _host_buffer->recycle();
        _pool().destroy(this);
    }
};

void HIPShaderPrinter::Encode::commit(HIPCommandEncoder &encoder) noexcept {
    if (_callback == nullptr) { return; }
    auto stream = encoder.stream()->handle();
    LUISA_CHECK_HIP(hipMemcpyAsync(
        _callback->host_address(), _binding.content,
        print_buffer_capacity, hipMemcpyDeviceToHost, stream));
    LUISA_CHECK_HIP(hipFreeAsync(_binding.content, stream));
    encoder.add_callback(_callback);
    _callback = nullptr;
    _binding = {};
}

luisa::unique_ptr<HIPShaderPrinter> HIPShaderPrinter::create(
    luisa::span<const std::pair<luisa::string, luisa::string>> formats) noexcept {
    if (formats.empty()) { return nullptr; }
    luisa::vector<luisa::unique_ptr<Formatter>> formatters;
    formatters.reserve(formats.size());
    for (auto &&[format, type_description] : formats) {
        formatters.emplace_back(luisa::make_unique<Formatter>(
            format, Type::from(type_description)));
    }
    return luisa::make_unique<HIPShaderPrinter>(std::move(formatters));
}

HIPShaderPrinter::Encode HIPShaderPrinter::encode(
    HIPCommandEncoder &encoder) const noexcept {
    auto host_buffer = encoder.stream()->download_pool()->allocate(
        print_buffer_capacity);
    if (host_buffer == nullptr) {
        static thread_local bool warned = false;
        if (!warned) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to allocate a temporary HIP shader-printer buffer. "
                "Printing is disabled for this dispatch. Consecutive "
                "warnings are suppressed to avoid flooding.");
            warned = true;
        }
        return {};
    }

    void *device_buffer = nullptr;
    auto stream = encoder.stream()->handle();
    LUISA_CHECK_HIP(hipMallocAsync(
        &device_buffer, print_buffer_capacity, stream));
    LUISA_CHECK_HIP(hipMemsetAsync(
        device_buffer, 0, sizeof(size_t), stream));
    auto callback = Callback::create(
        this, host_buffer, encoder.stream()->log_callback());
    return Encode{
        Binding{print_buffer_content_capacity, device_buffer},
        callback};
}

void HIPShaderPrinter::_do_print(
    const void *data, const Log &log) const noexcept {
    struct Head {
        size_t size;
        const std::byte content[];
    };
    auto head = static_cast<const Head *>(data);
    auto valid_size = std::min(
        head->size, print_buffer_content_capacity);
    auto printed_size = format_shader_print(
        _formatters, luisa::span{head->content, valid_size}, log);
    if (head->size > printed_size) {
        LUISA_WARNING("Device print overflow. {} byte(s) truncated.",
                      head->size - printed_size);
    }
}

HIPShaderPrinter::HIPShaderPrinter(
    luisa::vector<luisa::unique_ptr<Formatter>> &&formatters) noexcept
    : _formatters{std::move(formatters)} {}

HIPShaderPrinter::~HIPShaderPrinter() noexcept = default;

}// namespace luisa::compute::hip
