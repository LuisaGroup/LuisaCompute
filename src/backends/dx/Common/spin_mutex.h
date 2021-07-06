#pragma once
#include <VEngineConfig.h>
#include <atomic>
class VENGINE_DLL_COMMON spin_mutex_base {
	std::atomic_flag _flag;

public:
	void lock() noexcept;

	bool isLocked() const noexcept;
	void unlock() noexcept;
};

class spin_mutex : public spin_mutex_base {
	std::atomic_flag _flag;

public:
	spin_mutex() {
		_flag.clear(std::memory_order_acquire);
	}
};

class spin_mutex_init_lock : public spin_mutex_base {
	std::atomic_flag _flag;

public:
	spin_mutex_init_lock() {
		_flag.test_and_set(std::memory_order_release);
	}
};