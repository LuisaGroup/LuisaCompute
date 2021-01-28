#pragma once
#include "DLL.h"
#include "string_chunk.h"
#include <initializer_list>
 void VEngine_Log(string_chunk const& chunk);
 void VEngine_Log(string_chunk const* chunk, size_t chunkCount);
 void VEngine_Log(std::initializer_list<string_chunk> const& initList);