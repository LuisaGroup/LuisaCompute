#pragma once
#include <vstl/common.h>
#include <vstl/functional.h>
#include <runtime/command_buffer.h>
#include <runtime/stream.h>
#include <runtime/device.h>
namespace luisa::compute {
class PyStream : public vstd::IOperatorNewBase {
    struct Disposer {
        void *ptr;
        vstd::funcPtr_t<void(void *ptr)> dtor;
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
        CommandBuffer buffer;
        vstd::vector<Disposer> uploadDisposer;
        vstd::vector<Disposer> readbackDisposer;
        Data(Device &device) noexcept;
    };
    vstd::unique_ptr<Data> _data;

public:
    PyStream(PyStream &&) noexcept;
    PyStream(PyStream const &) = delete;
    PyStream(Device &device) noexcept;
    ~PyStream() noexcept;
    vstd::vector<vstd::function<void()>> delegates;
    void add(Command *cmd) noexcept;
    void add(luisa::unique_ptr<Command>&& cmd) noexcept;
    template<typename T>
        requires(!std::is_reference_v<T>)
    void add_upload(T t) noexcept {
        auto &disp = _data->uploadDisposer.emplace_back();
        disp.ptr = vengine_malloc(sizeof(T));
        new (disp.ptr) T(std::move(t));
        disp.dtor = [](void *ptr) { reinterpret_cast<T *>(ptr)->~T(); };
    }
    template<typename T>
        requires(!std::is_reference_v<T>)
    void add_readback(T t) noexcept {
        auto &disp = _data->readbackDisposer.emplace_back();
        disp.ptr = vengine_malloc(sizeof(T));
        new (disp.ptr) T(std::move(t));
        disp.dtor = [](void *ptr) { reinterpret_cast<T *>(ptr)->~T(); };
    }
    void execute() noexcept;
    void sync() noexcept;
};
}// namespace luisa::compute