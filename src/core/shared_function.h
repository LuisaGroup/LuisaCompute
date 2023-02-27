#pragma once
#include <core/stl/memory.h>
#include <atomic>
namespace luisa {
template<typename Func>
class SharedFunction;
template<typename Ret, typename... Args>
class SharedFunction<Ret(Args...)> {
    struct SharedFunctionBase {
        std::atomic_uint32_t ref{1};
        uint32_t align;
        void (*dtor)(void *);
    };
    SharedFunctionBase *_base;
    Ret (*_func_ptr)(void *, Args &&...);
    void _dispose() noexcept {
        if (!_base) return;
        if (--_base->ref == 0) {
            if (_base->dtor) {
                _base->dtor(_base);
            }
            luisa::detail::allocator_deallocate(_base, _base->align);
        }
        _base = nullptr;
    }

public:
    SharedFunction() noexcept : _base(nullptr) {}
    ~SharedFunction() noexcept {
        _dispose();
    }
    Ret operator()(Args... args) const noexcept {
        assert(_base);
        if constexpr (std::is_same_v<Ret, void>) {
            _func_ptr(_base, std::forward<Args>(args)...);
        } else {
            return _func_ptr(_base, std::forward<Args>(args)...);
        }
    }
    template<typename F>
        requires((!std::is_same_v<std::remove_cvref_t<F>, SharedFunction>) && (std::is_invocable_r_v<Ret, F, Args &&...>))
    SharedFunction(F &&f) noexcept {
        struct SharedFunctionDerive : public SharedFunctionBase {
            std::aligned_storage_t<sizeof(F), alignof(F)> storage;
        };
        using Func = std::remove_cvref_t<F>;
        auto derive = new (luisa::detail::allocator_allocate(sizeof(SharedFunctionDerive), alignof(SharedFunctionDerive))) SharedFunctionDerive();
        _base = derive;
        _base->align = alignof(SharedFunctionDerive);
        new (&derive->storage) Func(std::forward<F>(f));
        if constexpr (std::is_trivially_destructible_v<Func>) {
            _base->dtor = nullptr;
        } else {
            _base->dtor = [](void *ptr) noexcept -> void {
                reinterpret_cast<Func *>(&(reinterpret_cast<SharedFunctionDerive *>(ptr)->storage))->~Func();
            };
        }
        _func_ptr = [](void *ptr, Args &&...args) noexcept -> Ret {
            auto &&func = reinterpret_cast<Func &>((reinterpret_cast<SharedFunctionDerive *>(ptr)->storage));
            if constexpr (std::is_same_v<Ret, void>) {
                func(std::forward<Args>(args)...);
            } else {
                return func(std::forward<Args>(args)...);
            }
        };
    }
    template<typename F>
        requires(std::is_invocable_r_v<Ret, F, Args && ...>)
    SharedFunction &operator=(F &&f) noexcept {
        _dispose();
        new (std::launder(this)) SharedFunction(std::forward<F>(f));
        return *this;
    }
    template<typename FuncPtr_Ret, typename... FuncPtr_Args>
        requires(std::is_invocable_r_v<Ret, FuncPtr_Ret (*)(FuncPtr_Args...), Args && ...>)
    SharedFunction(FuncPtr_Ret (*func_ptr)(FuncPtr_Args...)) noexcept {
        using FuncPtr = decltype(func_ptr);
        struct SharedFunctionDerive : public SharedFunctionBase {
            FuncPtr derive_func_ptr;
        };
        auto derive = new (luisa::detail::allocator_allocate(sizeof(SharedFunctionDerive), alignof(SharedFunctionDerive))) SharedFunctionDerive();
        _base = derive;
        _base->align = alignof(SharedFunctionDerive);
        derive->derive_func_ptr = func_ptr;
        _base->dtor = nullptr;
        _func_ptr = [](void *ptr, Args &&...args) noexcept -> Ret {
            auto func = reinterpret_cast<SharedFunctionDerive *>(ptr)->derive_func_ptr;
            if constexpr (std::is_same_v<Ret, void>) {
                func(std::forward<Args>(args)...);
            } else {
                return func(std::forward<Args>(args)...);
            }
        };
    }
    template<typename FuncPtr_Ret, typename... FuncPtr_Args>
        requires(std::is_invocable_r_v<Ret, FuncPtr_Ret (*)(FuncPtr_Args...), Args && ...>)
    SharedFunction &operator=(FuncPtr_Ret (*func_ptr)(FuncPtr_Args...)) noexcept {
        _dispose();
        new (std::launder(this)) SharedFunction(func_ptr);
        return *this;
    }
    SharedFunction(SharedFunction const &x) noexcept {
        if (!x._base) {
            _base = nullptr;
            return;
        }
        _base = x._base;
        _func_ptr = x._func_ptr;
        _base->ref++;
    }
    SharedFunction(SharedFunction &&x) noexcept {
        _base = x._base;
        _func_ptr = x._func_ptr;
        x._base = nullptr;
    }
    SharedFunction &operator=(SharedFunction const &x) noexcept {
        _dispose();
        new (std::launder(this)) SharedFunction(x);
        return *this;
    }
    SharedFunction &operator=(SharedFunction &&x) noexcept {
        _dispose();
        new (std::launder(this)) SharedFunction(std::move(x));
        return *this;
    }
};
}// namespace luisa