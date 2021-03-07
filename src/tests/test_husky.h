#pragma once
#ifdef HUSKY_WINDOWS
#include <Windows.h>
#include <iostream>
#include <optional>

template<typename T>
struct funcPtr;

template<typename _Ret, typename... Args>
struct funcPtr<_Ret(Args...)> {
	using Type = _Ret (*)(Args...);
};

template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;

class DynamicDLL final {
    HINSTANCE inst;
    template<typename T>
    struct GetFuncPtrFromDll;
    template<typename Ret, typename... Args>
    struct GetFuncPtrFromDll<Ret(Args...)> {
        using FuncType = typename Ret (*)(Args...);
        static FuncType Run(HINSTANCE h, LPCSTR str) noexcept {
            auto ptr = GetProcAddress(h, str);
            if (ptr == nullptr) {
                throw 0;
            }
            return (FuncType)(ptr);
        }
    };

public:
    DynamicDLL(char const *name) {
        inst = LoadLibraryA(name);
        if (inst == nullptr) {
            std::cout << "Can not find DLL " << name;
            throw 0;
        }
    }
    ~DynamicDLL() {
        FreeLibrary(inst);
    }

    template<typename T>
    typename GetFuncPtrFromDll<T>::FuncType
    GetDLLFunc(char const *str) {
        return GetFuncPtrFromDll<T>::Run(inst, str);
    }
};
namespace luisa::compute {
class Function;
class ScopeStmt;
void RunHLSLCodeGen(Function *func) {
    static std::optional<DynamicDLL> dll;
    static funcPtr_t<void(ScopeStmt const *)> codegenFunc;
    std::cout << "Fuck!" << std::endl;
    if (!dll.has_value()) {
        dll.emplace("LC_DXBackend.dll");
        codegenFunc = dll->GetDLLFunc<void(ScopeStmt const *)>("CodegenBody");
        std::cout << "Fuck Start!" << std::endl;
    }
    std::cout << codegenFunc << std::endl;
    system("pause");
    codegenFunc(func->body());
}
}// namespace luisa::compute
#endif