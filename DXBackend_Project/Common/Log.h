#pragma once
#include "DLL.h"
#include "string_view.h"
#include <initializer_list>
 void VEngine_Log(vengine::string_view const& chunk);
 void VEngine_Log(vengine::string_view const* chunk, size_t chunkCount);
 void VEngine_Log(std::initializer_list<vengine::string_view> const& initList);