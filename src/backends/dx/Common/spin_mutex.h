#pragma once
#include <VEngineConfig.h>
#include <atomic>
class VENGINE_DLL_COMMON spin_mutex {
	std::atomic_flag _flag;

public:
	spin_mutex() noexcept;
	void lock() noexcept;

	bool isLocked() const noexcept;
	void unlock() noexcept;
};