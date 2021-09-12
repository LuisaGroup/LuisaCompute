#pragma once
#include <util/vstlconfig.h>
#include <util/vstring.h>
#include <initializer_list>
VENGINE_DLL_COMMON void VEngine_Log(vstd::string_view const& chunk);
VENGINE_DLL_COMMON void VEngine_Log(vstd::string_view const* chunk, size_t chunkCount);
VENGINE_DLL_COMMON void VEngine_Log(std::initializer_list<vstd::string_view> const& initList);
VENGINE_DLL_COMMON void VEngine_Log(char const* chunk);
#define NOT_IMPLEMENT_EXCEPTION(T)\
VEngine_Log({#T##_sv, " not implemented!\n"_sv});\
VSTL_ABORT();
