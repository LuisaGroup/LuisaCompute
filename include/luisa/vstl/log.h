#pragma once
#include <luisa/vstl/config.h>
#include <luisa/vstl/vstring.h>
#include <luisa/core/stl/string.h>
#include <initializer_list>
LUISA_VSTL_API void vengine_log(luisa::string_view const &chunk);
LUISA_VSTL_API void vengine_log(luisa::string_view const *chunk, size_t chunkCount);
LUISA_VSTL_API void vengine_log(std::initializer_list<luisa::string_view> const &initList);
LUISA_VSTL_API void vengine_log(char const *chunk);
