#pragma once
#include <Common/Common.h>
class IGPUAllocator {
public:
	virtual ~IGPUAllocator() {}
	virtual void Release(uint64 instanceID) = 0;
};