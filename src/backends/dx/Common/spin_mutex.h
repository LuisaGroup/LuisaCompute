#pragma once
#include <VEngineConfig.h>
#include <atomic>

class VENGINE_DLL_COMMON spin_mutex final{
private:
	std::atomic_flag flag;
public:
	
	void lock() noexcept;
	spin_mutex();
	bool isLocked() const noexcept;
	void unlock() noexcept;
};
