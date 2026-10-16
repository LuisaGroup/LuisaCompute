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
        Disposer() noexcept : ptr(nullptr), dtor(nullptr) {}
        Disposer(Disposer &&d) noexcept {
            ptr = d.ptr;
            d.ptr = nullptr;
            dtor = d.dtor;
        }
        ~Disposer() noexcept {
            if (!ptr) return;
            dtor(ptr);
            vengine_free(ptr);
        }
    };
public:
    struct Data {
        Stream stream;
        luisa::unordered_map<void*, luisa::move_only_function<void(void*, Stream&)>> before_commit_tasks;
        CommandList buffer;
        vstd::vector<Disposer> uploadDisposer;
        vstd::vector<Disposer> readbackDisposer;
        Data(Device &device, bool support_window) noexcept;
        void execute() noexcept;
        void sync() noexcept;
    };
private:
    luisa::shared_ptr<Data> _data;

public:
    [[nodiscard]] auto &data() const { return _data; }
    [[nodiscard]] Stream &stream() const { return _data->stream; }
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
            std::destroy_at(static_cast<T *>(ptr));
        };
    }
    template<typename T>
        requires(!std::is_reference_v<T>)
    void add_readback(T t) noexcept {
        auto &disp = _data->readbackDisposer.emplace_back();
        disp.ptr = vengine_malloc(sizeof(T));
        new (disp.ptr) T(std::move(t));
        disp.dtor = [](void *ptr) {
            std::destroy_at(static_cast<T *>(ptr));
        };
    }
    void execute() noexcept;
    void sync() noexcept;
};

}// namespace luisa::compute
