#pragma once
#include "vstlconfig.h"
#include <stdint.h>
enum class VEngine_AllocType : uint8_t {
	Default,
	VEngine,
	Stack
};