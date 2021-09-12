#pragma once
#include <util/vstlconfig.h>
#include <util/vstring.h>
#include <initializer_list>
LUISA_DLL void vstl_log(vstd::string_view const& chunk);
LUISA_DLL void vstl_log(vstd::string_view const* chunk, size_t chunkCount);
LUISA_DLL void vstl_log(std::initializer_list<vstd::string_view> const& initList);
LUISA_DLL void vstl_log(char const* chunk);
#define NOT_IMPLEMENT_EXCEPTION(T)\
vstl_log({#T##_sv, " not implemented!\n"_sv});\
VSTL_ABORT();
