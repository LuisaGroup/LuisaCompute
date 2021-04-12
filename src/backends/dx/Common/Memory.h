#pragma once
#include <VEngineConfig.h>
#include <cstdlib>
#include <stdint.h>
#include <type_traits>
#include <Common/DLL.h>
#include <Common/MetaLib.h>
namespace vengine {
VENGINE_DLL_COMMON void vengine_init_malloc();
VENGINE_DLL_COMMON void vengine_init_malloc(
	funcPtr_t<void*(size_t)> mallocFunc,
	funcPtr_t<void(void*)> freeFunc);
}// namespace vengine
namespace v_mimalloc {
struct VENGINE_DLL_COMMON Alloc {
	friend void vengine::vengine_init_malloc();
	friend void vengine::vengine_init_malloc(
		funcPtr_t<void*(size_t)> mallocFunc,
		funcPtr_t<void(void*)> freeFunc);

private:
	static funcPtr_t<void*(size_t)> mallocFunc;
	static funcPtr_t<void(void*)> freeFunc;

public:
	static funcPtr_t<void*(size_t)> GetMiMalloc() {
		return mallocFunc;
	}
	static funcPtr_t<void(void*)> GetMifree() {
		return freeFunc;
	}
	Alloc() = delete;
	Alloc(const Alloc&) = delete;
	Alloc(Alloc&&) = delete;
};
}// namespace v_mimalloc

inline void* vengine_malloc(size_t size) noexcept {
	auto malloc_func = v_mimalloc::Alloc::GetMiMalloc();
	return malloc_func(size);
}
inline void vengine_free(void* ptr) noexcept {
	auto free_func = v_mimalloc::Alloc::GetMifree();
	free_func(ptr);
}

template<typename T, typename... Args>
inline T* vengine_new(Args&&... args) noexcept {
	T* tPtr = (T*)vengine_malloc(sizeof(T));
	if constexpr (!std::is_trivially_constructible_v<T>)
		new (tPtr) T(std::forward<Args>(args)...);
	return tPtr;
}

template<typename T>
inline void vengine_delete(T* ptr) noexcept {
	if constexpr (!std::is_trivially_destructible_v<T>)
		((T*)ptr)->~T();
	vengine_free(ptr);
}
#define DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW                          \
	static void* operator new(size_t size) noexcept {                  \
		return vengine_malloc(size);                                   \
	}                                                                  \
	static void* operator new(size_t, void* place) noexcept {          \
		return place;                                                  \
	}                                                                  \
	static void operator delete(void* pdead, size_t size) noexcept {   \
		vengine_free(pdead);                                           \
	}                                                                  \
	static void* operator new[](size_t size) noexcept {                \
		return vengine_malloc(size);                                   \
	}                                                                  \
	static void operator delete[](void* pdead, size_t size) noexcept { \
		vengine_free(pdead);                                           \
	}

using OperatorNewFunctor = typename funcPtr_t<void*(size_t)>;

template<typename T>
struct DynamicObject {
	template<typename... Args>
	static constexpr T* CreateObject(
		funcPtr_t<T*(
			OperatorNewFunctor operatorNew,
			Args...)>
			createFunc,
		Args... args) {
		return createFunc(
			T::operator new,
			std::forward<Args>(args)...);
	}
};

#define KILL_COPY_CONSTRUCT(clsName)  \
	clsName(clsName const&) = delete; \
	clsName& operator=(clsName const&) = delete;

#define KILL_MOVE_CONSTRUCT(clsName) \
	clsName(clsName&&) = delete;     \
	clsName& operator=(clsName&&) = delete;
