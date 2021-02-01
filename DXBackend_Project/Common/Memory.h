#pragma once
#include <cstdlib>
#include <stdint.h>
#include <type_traits>
#include "DLL.h"
#include "MetaLib.h"
inline constexpr void* aligned_malloc(size_t size, size_t alignment) noexcept {
	if (alignment & (alignment - 1)) {
		return nullptr;
	} else {
		void* praw = malloc(sizeof(void*) + size + alignment);
		if (praw) {
			void* pbuf = reinterpret_cast<void*>(reinterpret_cast<size_t>(praw) + sizeof(void*));
			void* palignedbuf = reinterpret_cast<void*>((reinterpret_cast<size_t>(pbuf) | (alignment - 1)) + 1);
			(static_cast<void**>(palignedbuf))[-1] = praw;

			return palignedbuf;
		} else {
			return nullptr;
		}
	}
}

inline void aligned_free(void* palignedmem) noexcept {
	free(reinterpret_cast<void*>((static_cast<void**>(palignedmem))[-1]));
}
//allocFunc:: void* operator()(size_t)
template<typename AllocFunc>
inline constexpr void* aligned_malloc(size_t size, size_t alignment, const AllocFunc& allocFunc) noexcept {
	if (alignment & (alignment - 1)) {
		return nullptr;
	} else {
		void* praw = allocFunc(sizeof(void*) + size + alignment);
		if (praw) {
			void* pbuf = reinterpret_cast<void*>(reinterpret_cast<size_t>(praw) + sizeof(void*));
			void* palignedbuf = reinterpret_cast<void*>((reinterpret_cast<size_t>(pbuf) | (alignment - 1)) + 1);
			(static_cast<void**>(palignedbuf))[-1] = praw;

			return palignedbuf;
		} else {
			return nullptr;
		}
	}
}

template<typename FreeFunc>
inline constexpr void* aligned_free(void* palignedmem, const FreeFunc& func) noexcept {
	func(reinterpret_cast<void*>((static_cast<void**>(palignedmem))[-1]));
}
namespace vengine {
void vengine_init_malloc();
void vengine_init_malloc(
	funcPtr_t<void*(size_t)> mallocFunc,
	funcPtr_t<void(void*)> freeFunc);
}// namespace vengine
namespace v_mimalloc {
struct Alloc {
	friend void vengine::vengine_init_malloc();
	friend void vengine::vengine_init_malloc(
		funcPtr_t<void*(size_t)> mallocFunc,
		funcPtr_t<void(void*)> freeFunc);

private:
	static funcPtr_t<void*(size_t)> mallocFunc;
	static funcPtr_t<void(void*)> freeFunc;

public:
	static funcPtr_t<void*(size_t)> GetMalloc() {
		return mallocFunc;
	}
	static funcPtr_t<void(void*)> Getfree() {
		return freeFunc;
	}
	Alloc() = delete;
	Alloc(const Alloc&) = delete;
	Alloc(Alloc&&) = delete;
};
}// namespace v_mimalloc

inline void* vengine_malloc(uint64_t size) noexcept {
	return v_mimalloc::Alloc::GetMalloc()(size);
}
inline void vengine_free(void* ptr) noexcept {
	v_mimalloc::Alloc::Getfree()(ptr);
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

#define KILL_COPY_CONSTRUCT(clsName)             \
	clsName(clsName const&) = delete;            \
	clsName(clsName&&) = delete;                 \
	clsName& operator=(clsName const&) = delete; \
	clsName& operator=(clsName&&) = delete;