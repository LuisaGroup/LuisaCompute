//
// Created by Mike Smith on 2022/2/13.
//

#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/printer.h>

namespace luisa::compute {

Printer::Printer(Device &device, size_t capacity) noexcept
    : _buffer{device.create_buffer<uint>(next_pow2(capacity) + 1u)},
      _host_buffer(next_pow2(capacity) + 1u) {
    std::iota(_host_buffer.begin(), _host_buffer.end(), 0u);
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
            auto desc_id = _host_buffer[offset++];
            auto desc = _descriptors[desc_id];
            if (offset + desc.size() > size) { break; }
            static thread_local luisa::string item;
            item.clear();
            for (auto &&tag : desc) {
                auto record = _host_buffer[offset++];
                switch (tag) {
                    case Descriptor::Tag::INT:
                        item.append(luisa::format(
                            "{}", static_cast<int>(record)));
                        break;
                    case Descriptor::Tag::UINT:
                        item.append(luisa::format(
                            "{}", record));
                        break;
                    case Descriptor::Tag::FLOAT:
                        item.append(luisa::format(
                            "{}", luisa::bit_cast<float>(record)));
                        break;
                    case Descriptor::Tag::BOOL:
                        item.append(luisa::format(
                            "{}", static_cast<bool>(record)));
                        break;
                    case Descriptor::Tag::STRING:
                        item.append(_strings[record]);
                        break;
                }
            }
            LUISA_INFO("{}", item);
        }
    };
    return std::make_tuple<Command *, luisa::move_only_function<void()>, Command *>(
        _buffer.copy_to(_host_buffer.data()), print, reset());
}

}// namespace luisa::compute
