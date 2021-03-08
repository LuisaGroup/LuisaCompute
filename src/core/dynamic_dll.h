#pragma once
//#define _WINDOWS
template<typename T>
struct funcPtr;

template<typename _Ret, typename... Args>
struct funcPtr<_Ret(Args...)> {
    using Type = _Ret (*)(Args...);
};

template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;

//////////////////// Windows Area
#ifdef _WINDOWS
#include <string_view>
#include <Windows.h>
namespace luisa::compute {
class DynamicDLL final {
private:
    template<typename T>
    struct IsFuncPtr {
        static constexpr bool value = false;
    };

    template<typename _Ret, typename... Args>
    struct IsFuncPtr<_Ret (*)(Args...)> {
        static constexpr bool value = true;
    };

public:
    DynamicDLL(char const *name);
    ~DynamicDLL();
    template<typename T>
    void GetDLLFunc(T &funcPtr, char const *name) {
        static_assert(IsFuncPtr<std::remove_cv_t<T>>::value, "DLL only support func p!");
        if (!inst) {
            funcPtr = nullptr;
            return;
        }
        auto ptr = GetProcAddress(inst, name);
        funcPtr = reinterpret_cast<T>(ptr);
    }

private:
    HINSTANCE inst;
};
//////////////////// TODO: Other Platforms
}// namespace luisa::compute
#endif
