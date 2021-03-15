#pragma once
#define HUSKY_WINDOWS
#ifdef HUSKY_WINDOWS
#include <iostream>
#include <optional>
#include <core/dynamic_dll.h>

#ifdef _WINDOWS

#include <Windows.h>

namespace luisa::compute {
class Function;
class Function;
void RunHLSLCodeGen(Function *func) {
    DynamicDLL dll("LC_DXBackend.dll");
    funcPtr_t<void(Function const *)> codegenFunc;

    dll.GetDLLFunc(codegenFunc, "SerializeMD5");
    std::cout << codegenFunc << std::endl;
    system("pause");
    codegenFunc(func);
}
}// namespace luisa::compute
#endif

#endif
