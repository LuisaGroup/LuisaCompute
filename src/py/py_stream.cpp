#include <py/py_stream.h>
#include <runtime/command_buffer.h>
#include <py/ref_counter.h>
namespace luisa::compute {
PyStream::PyStream(Device &device) noexcept
    : _data(new Data(device)) {
}
PyStream::Data::Data(Device &device) noexcept
    : stream(device.create_stream()),
      buffer(stream.command_buffer()) {
}
PyStream::~PyStream() noexcept {
    if (!_data) return;
    if (!_data->buffer.empty()) {
        _data->buffer.commit();
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
    _data->buffer << [d = _data.get(), delegates = std::move(delegates)] {
        d->readbackDisposer.clear();
        for (auto &&i : delegates) {
            i();
        }
    };
    _data->buffer.commit();
    _data->uploadDisposer.clear();
}
void PyStream::sync() noexcept {
    _data->buffer << [d = _data.get(), delegates = std::move(delegates)] {
        d->readbackDisposer.clear();
        for (auto &&i : delegates) {
            i();
        }
    };
    _data->buffer.synchronize();
}
PyStream::PyStream(PyStream &&s) noexcept
    : _data(std::move(s._data)) {}
}// namespace luisa::compute