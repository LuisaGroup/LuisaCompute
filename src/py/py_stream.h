#pragma once

#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>

namespace luisa::compute {

class PyStream : public vstd::IOperatorNewBase {
    struct Disposer {
        void *ptr;
        vstd::func_ptr_t<void(void *ptr)> dtor;
        Disposer() noexcept {}
        Disposer(Disposer &&d) noexcept {
            ptr = d.ptr;
            d.ptr = nullptr;
            dtor = d.dtor;
        }
        ~Disposer() noexcept {
            if (!ptr) return;
            dtor(ptr);
            vengine_delete(ptr);
        }
    };

    struct Data : public vstd::IOperatorNewBase {
        Stream stream;
        CommandList buffer;
        vstd::vector<Disposer> uploadDisposer;
        // vstd::vector<Disposer> readbackDisposer;
        Data(Device &device, bool support_window) noexcept;
    };
    vstd::unique_ptr<Data> _data;

public:
    Stream &stream() const { return _data->stream; }
    PyStream(PyStream &&) noexcept;
    PyStream(PyStream const &) = delete;
    PyStream(Device &device, bool support_window) noexcept;
    ~PyStream() noexcept;
    vstd::vector<vstd::function<void()>> delegates;
    void add(Command *cmd) noexcept;
    void add(luisa::unique_ptr<Command> &&cmd) noexcept;
    template<typename T>
        requires(!std::is_reference_v<T>)
    void add_upload(T t) noexcept {
        auto &disp = _data->uploadDisposer.emplace_back();
        disp.ptr = vengine_malloc(sizeof(T));
        new (disp.ptr) T(std::move(t));
        disp.dtor = [](void *ptr) {
            vstd::destruct(reinterpret_cast<T *>(ptr));
        };
    }
    void execute() noexcept;
    void sync() noexcept;
};

}// namespace luisa::compute

