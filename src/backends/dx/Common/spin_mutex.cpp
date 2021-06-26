#include <Common/spin_mutex.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define VENGINE_INTRIN_PAUSE() _mm_pause()
#elif defined(_M_X64)
#include <windows.h>
#define VENGINE_INTRIN_PAUSE() YieldProcessor()
#elif defined(__aarch64__)
#define VENGINE_INTRIN_PAUSE() asm volatile("isb"_sv)
#else
#include <mutex>
#define VENGINE_INTRIN_PAUSE() std::this_thread::yield()
#endif

spin_mutex::spin_mutex() noexcept {
	_flag.clear();
}
void spin_mutex::lock() noexcept {
	while (_flag.test_and_set(std::memory_order::acquire)) {// acquire lock
#ifdef __cpp_lib_atomic_flag_test
		while (_flag.test(std::memory_order::relaxed)) {// test lock
#endif
			VENGINE_INTRIN_PAUSE();
#ifdef __cpp_lib_atomic_flag_test
		}
#endif
	}
}

bool spin_mutex::isLocked() const noexcept {
	return _flag.test(std::memory_order::relaxed);
}
void spin_mutex::unlock() noexcept {
	_flag.clear(std::memory_order::release);
}
