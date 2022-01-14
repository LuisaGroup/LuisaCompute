#pragma once
#include <vstl/config.h>
#include <vstl/vstring.h>
#include <initializer_list>
VENGINE_DLL_COMMON void VEngine_Log(std::string_view const &chunk);
VENGINE_DLL_COMMON void VEngine_Log(std::string_view const *chunk, size_t chunkCount);
VENGINE_DLL_COMMON void VEngine_Log(std::initializer_list<std::string_view> const &initList);
VENGINE_DLL_COMMON void VEngine_Log(char const *chunk);
VENGINE_DLL_COMMON void VEngine_Log_PureVirtual(vstd::Type tarType);
#define VENGINE_PURE_VIRTUAL                    \
    {                                           \
        VEngine_Log_PureVirtual(typeid(*this)); \
    }
#define VENGINE_PURE_VIRTUAL_RET                \
    {                                           \
        VEngine_Log_PureVirtual(typeid(*this)); \
        return {};                              \
    }

#define NOT_IMPLEMENT_EXCEPTION(T)                    \
    VEngine_Log({#T##_sv, " not implemented!\n"_sv}); \
    VENGINE_EXIT;
