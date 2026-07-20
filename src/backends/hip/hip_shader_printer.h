#pragma once

#include <utility>

#include <luisa/ast/type.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {
class ShaderPrintFormatter;
}// namespace luisa::compute

namespace luisa::compute::hip {

class HIPCommandEncoder;

class HIPShaderPrinter {

private:
    static constexpr auto print_buffer_capacity = 1_M;
    static constexpr auto print_buffer_content_capacity =
        print_buffer_capacity - sizeof(size_t);

public:
    struct Binding {
        size_t capacity;
        void *content;
    };

    class Callback;

    struct Encode {

    private:
        Binding _binding{};
        Callback *_callback{nullptr};

        friend class HIPShaderPrinter;
        Encode(Binding binding, Callback *callback) noexcept
            : _binding{binding}, _callback{callback} {}

    public:
        Encode() noexcept = default;
        Encode(Encode &&other) noexcept
            : _binding{std::exchange(other._binding, Binding{})},
              _callback{std::exchange(other._callback, nullptr)} {}
        Encode &operator=(Encode &&other) noexcept {
            if (this != &other) {
                _binding = std::exchange(other._binding, Binding{});
                _callback = std::exchange(other._callback, nullptr);
            }
            return *this;
        }
        Encode(const Encode &) = delete;
        Encode &operator=(const Encode &) = delete;
        [[nodiscard]] auto binding() const noexcept { return _binding; }
        void commit(HIPCommandEncoder &encoder) noexcept;
    };

    using Formatter = ShaderPrintFormatter;

private:
    luisa::vector<luisa::unique_ptr<Formatter>> _formatters;
    using Log = DeviceInterface::StreamLogCallback;
    void _do_print(const void *data, const Log &log) const noexcept;

public:
    explicit HIPShaderPrinter(
        luisa::vector<luisa::unique_ptr<Formatter>> &&formatters) noexcept;
    ~HIPShaderPrinter() noexcept;

    [[nodiscard]] static luisa::unique_ptr<HIPShaderPrinter>
    create(luisa::span<const std::pair<luisa::string, luisa::string>> formats) noexcept;

    [[nodiscard]] Encode encode(HIPCommandEncoder &encoder) const noexcept;
};

}// namespace luisa::compute::hip
