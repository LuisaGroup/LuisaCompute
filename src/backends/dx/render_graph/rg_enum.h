#pragma once
#include <stdint.h>
namespace luisa::compute {
enum class RGNodeState : uint32_t {
	Preparing,
	InList,
	Executed
};
}// namespace luisa::compute