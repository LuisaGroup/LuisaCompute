#pragma once
#include <stdint.h>
#include "ASTNode.h"
namespace LCShader {
struct Posture {
	NumVar v0;
	NumVar v1;
	NumOpType opType;
	bool operator==(Posture const& v) const {
		return v0 == v.v0 && v1 == v.v1 && opType == v.opType;
	}
	bool operator!=(Posture const& v) const {
		return !operator==(v);
	}
};
}// namespace LCShader