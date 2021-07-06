#pragma once
#include <VEngineConfig.h>
#include <Common/vstring.h>
#include <initializer_list>
VENGINE_DLL_COMMON void VEngine_Log(vstd::string_view const& chunk);
VENGINE_DLL_COMMON void VEngine_Log(vstd::string_view const* chunk, size_t chunkCount);
VENGINE_DLL_COMMON void VEngine_Log(std::initializer_list<vstd::string_view> const& initList);