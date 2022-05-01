//
// Created by Mike Smith on 2022/2/13.
//

#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/printer.h>

namespace luisa::compute {

Printer::Printer(Device &device, luisa::string_view name, size_t capacity) noexcept
    : _buffer{device.create_buffer<uint>(next_pow2(capacity) + 1u)},
      _host_buffer(next_pow2(capacity) + 1u),
      _logger{std::string{name},
              luisa::detail::default_logger().sinks().cbegin(),
              luisa::detail::default_logger().sinks().cend()} {
    _logger.set_level(spdlog::level::trace);
}

Command *Printer::reset() noexcept {
    _reset_called = true;
    static const auto zero = 0u;
    return _buffer.view(_buffer.size() - 1u, 1u).copy_from(&zero);
}

std::tuple<Command *, luisa::move_only_function<void()>, Command *>
Printer::retrieve() noexcept {
    if (!_reset_called) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Printer results cannot be "
            "retrieved if never reset.");
    }
    auto print = [this] {
        auto size = std::min(
            static_cast<uint>(_buffer.size() - 1u),
            _host_buffer.back());
        auto offset = 0u;
        while (offset < size) {
            auto data = _host_buffer.data() + offset;
            auto &&item = _items[data[0u]];
            offset += item.size;
            if (offset > size) {
                LUISA_WARNING_WITH_LOCATION("Kernel log truncated.");
            } else {
                item.f(data);
            }
        }
    };
    return std::make_tuple<Command *, luisa::move_only_function<void()>, Command *>(
        _buffer.copy_to(_host_buffer.data()), print, reset());
}

}// namespace luisa::compute
