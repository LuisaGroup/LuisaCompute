#pragma once
#include <vstl/meta_lib.h>
namespace vstd {
template<typename T>
class FuncRef;
template<typename Ret, typename... Args>
class FuncRef<Ret(Args...)> {
    void *_user_data;
    vstd::func_ptr_t<Ret(void *, Args &&...)> _func_ptr;

public:
    template<typename Lambda>
        requires(std::is_invocable_r_v<Ret, Lambda, Args...>)
    FuncRef(Lambda &lambda) {
        _user_data = &lambda;
        _func_ptr = [](void *ptr, Args &&...args) -> Ret {
            return (*reinterpret_cast<Lambda *>(ptr))(std::forward<Args>(args)...);
        };
    }
    FuncRef(vstd::func_ptr_t<Ret(Args &&...)> func_ptr) {
        _user_data = reinterpret_cast<void *>(func_ptr);
        _func_ptr = [](void *ptr, Args &&...args) -> Ret {
            return reinterpret_cast<vstd::func_ptr_t<Ret(Args && ...)>>(ptr)(std::forward<Args>(args)...);
        };
    }
    Ret operator()(Args... args) const {
        return _func_ptr(_user_data, std::forward<Args>(args)...);
    }
};
}// namespace vstd