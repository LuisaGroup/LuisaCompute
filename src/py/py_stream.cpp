#include "py_stream.h"
#include <luisa/runtime/command_list.h>
namespace luisa::compute {

PyStream::PyStream(Device &device, bool support_window) noexcept
    : _data{luisa::make_shared<Data>(device, support_window)} {}

PyStream::Data::Data(Device &device, bool support_window) noexcept
    : stream(device.create_stream(support_window ? StreamTag::GRAPHICS : StreamTag::COMPUTE)) {
}

PyStream::~PyStream() noexcept {
    if (!_data) return;
    sync();
}

void PyStream::add(Command *cmd) noexcept {
    _data->buffer << luisa::unique_ptr<Command>(cmd);
}

void PyStream::add(luisa::unique_ptr<Command> &&cmd) noexcept {
    _data->buffer << std::move(cmd);
}
void PyStream::Data::sync() noexcept {
    execute();
    stream << synchronize();
    // A released owner may dispose a resource and synchronize this stream again.
    // Detach the batch first so reentrant synchronization cannot release it twice.
    auto completed_readbacks = std::move(readbackDisposer);
}

void PyStream::Data::execute() noexcept {
    if (!buffer.commands().empty()) {
        for (auto &i : before_commit_tasks) {
            i.second(i.first, stream);
        }
        before_commit_tasks.clear();
    }
    if (!buffer.empty()) {
        stream << buffer.commit();
    }
    auto submitted_uploads = std::move(uploadDisposer);
}

void PyStream::execute() noexcept {
    if (!delegates.empty()) {
        _data->buffer.add_callback([delegates = std::move(delegates)] {
            for (auto &&i : delegates) { i(); }
        });
    }
    _data->execute();
}

void PyStream::sync() noexcept {
    execute();
    _data->stream.synchronize();
    // Python buffers must be released by the calling thread after all queued
    // downloads finish, including batches submitted by earlier execute() calls.
    auto completed_readbacks = std::move(_data->readbackDisposer);
}

PyStream::PyStream(PyStream &&s) noexcept
    : _data(std::move(s._data)) {}

}// namespace luisa::compute
