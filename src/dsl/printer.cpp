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

void Printer::reset(Stream &stream) noexcept {
    auto zero = 0u;
    auto size = _buffer.size() - 1u;
    stream << _buffer.view(size, 1u).copy_from(&zero);
    _reset_called = true;
}

luisa::string_view Printer::retrieve(Stream &stream) noexcept {
    if (!_reset_called) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Printer results cannot be "
            "retrieved if never reset.");
    }
    auto zero = 0u;
    stream << _buffer.copy_to(_host_buffer.data())
           << _buffer.view(_buffer.size() - 1u, 1u).copy_from(&zero)
           << synchronize();
    auto size = std::min(
        static_cast<uint>(_buffer.size() - 1u),
        _host_buffer.back());
    _scratch.clear();
    auto records = luisa::span{_host_buffer}.subspan(0u, size);
    for (auto offset = 0u; offset < size;) {
        auto desc_id = records[offset++];
        auto desc = _descriptors[desc_id];
        if (offset + desc.size() > records.size()) {
            break;
        }
        for (auto &&tag : desc) {
            auto record = records[offset++];
            switch (tag) {
                case Descriptor::Tag::INT:
                    _scratch.append(luisa::format(
                        "{}", static_cast<int>(record)));
                    break;
                case Descriptor::Tag::UINT:
                    _scratch.append(luisa::format(
                        "{}", record));
                    break;
                case Descriptor::Tag::FLOAT:
                    _scratch.append(luisa::format(
                        "{}", luisa::bit_cast<float>(record)));
                    break;
                case Descriptor::Tag::BOOL:
                    _scratch.append(luisa::format(
                        "{}", static_cast<bool>(record)));
                    break;
                case Descriptor::Tag::STRING:
                    _scratch.append(_strings[record]);
                    break;
            }
        }
        _scratch.append("\n");
    }
    return _scratch;
}

}// namespace luisa::compute
