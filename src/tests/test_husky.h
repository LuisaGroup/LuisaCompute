#pragma once

#include <iostream>
#include <optional>
#include <core/dynamic_module.h>

#ifdef LUISA_PLATFORM_WINDOWS

namespace luisa::compute {

class Function;

void RunHLSLCodeGen(Function func) {
    DynamicModule dll{"LC_DXBackend.dll"};
    auto codegenFunc = dll.function<void(Function const &)>("SerializeMD5");
    LUISA_INFO_WITH_LOCATION("Function Pointer: {}", fmt::ptr(codegenFunc));
    codegenFunc(func);
}
}// namespace luisa::compute

#endif
