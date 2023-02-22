#include <py/py_stream.h>
#include <runtime/command_list.h>
#include <py/ref_counter.h>

namespace luisa::compute {

PyStream::PyStream(Device &device, bool support_window) noexcept
    : _data(new Data(device, support_window)) {
}

PyStream::Data::Data(Device &device, bool support_window) noexcept
    : stream(device.create_stream(support_window ? StreamTag::GRAPHICS : StreamTag::COMPUTE)) {
}

PyStream::~PyStream() noexcept {
    if (!_data) return;
    if (!_data->buffer.empty()) [[unlikely]] {
        _data->stream << _data->buffer.commit();
    }
    _data->stream.synchronize();
}

void PyStream::add(Command *cmd) noexcept {
    _data->buffer << luisa::unique_ptr<Command>(cmd);
}

void PyStream::add(luisa::unique_ptr<Command> &&cmd) noexcept {
    _data->buffer << std::move(cmd);
}

void PyStream::execute() noexcept {
    _data->stream << _data->buffer.commit();
    _data->stream << [d = _data.get(), delegates = std::move(delegates)] {
        // LUISA_INFO("before callback {}", reinterpret_cast<size_t>(d));
        // d->readbackDisposer.clear();
        // LUISA_INFO("after clear");
        for (auto &&i : delegates) {
            i();
        }
        // LUISA_INFO("after callback");
    };
    _data->uploadDisposer.clear();
}

void PyStream::sync() noexcept {
    execute();
    _data->stream.synchronize();
}

PyStream::PyStream(PyStream &&s) noexcept
    : _data(std::move(s._data)) {}

}// namespace luisa::compute
