#pragma once
#include <VEngineConfig.h>
#include <Common/vector.h>
class JobNode;
class JobBucket;
using uint = uint32_t;
class VENGINE_DLL_COMMON JobHandle
{
	friend class JobBucket;
	friend class JobNode;
private:
	uint start;
	uint end;
public:
	JobHandle(
		uint start,
		uint end
	) : 
		start(start),
		end(end)
	{
	}
	JobHandle() : start(-1), end(-1){}
	operator bool() const noexcept {
		return start != -1;
	}
	bool operator!() const {
		return !operator bool();
	}
	uint Count() const {
		return end + 1 - start;
	}
	void Reset() { start = -1; }
};