#pragma once
#include <stdint.h>
#include "Hash.h"
#include "Memory.h"
#include <type_traits>
template<typename T>
struct RunnableHash;
template<class T>
class Runnable;
template<class _Ret,
		 class... _Types>
class Runnable<_Ret(_Types...)> {
	friend class RunnableHash<_Ret(_Types...)>;
	using FunctionPtrType = typename _Ret (*)(void*, _Types&&...);

private:
	void* placePtr;
	uint64_t allocatedSize;
	FunctionPtrType funcPtr;
	void (*disposeFunc)(void*);
	void (*constructFunc)(void*, void*);
	size_t funcPtrPlaceHolder;
	void AllocateFunctor(uint64_t targetSize) noexcept {
		if (targetSize <= allocatedSize) return;
		FreeFunctor();
		allocatedSize = targetSize;
		if (targetSize <= 8) {
			placePtr = &funcPtrPlaceHolder;
		} else {
			placePtr = vengine_malloc(allocatedSize);
		}
	}
	void InitAllocateFunctor(uint64_t targetSize) noexcept {
		allocatedSize = targetSize;
		if (targetSize <= 8) {
			placePtr = &funcPtrPlaceHolder;
		} else {
			placePtr = vengine_malloc(allocatedSize);
		}
	}
	void FreeFunctor() noexcept {
		if (allocatedSize > 8) vengine_free(placePtr);
	}

public:
	bool operator==(const Runnable<_Ret(_Types...)>& obj) const noexcept {
		return funcPtr == obj.funcPtr;
	}
	bool operator!=(const Runnable<_Ret(_Types...)>& obj) const noexcept {
		return funcPtr != obj.funcPtr;
	}

	operator bool() const noexcept {
		return funcPtr;
	}

	void Dispose() noexcept {
		if (disposeFunc) disposeFunc(placePtr);
		FreeFunctor();
		placePtr = nullptr;
		disposeFunc = nullptr;
		constructFunc = nullptr;
		allocatedSize = 0;
		funcPtr = nullptr;
	}

	Runnable() noexcept
		: funcPtr(nullptr),
		  disposeFunc(nullptr),
		  constructFunc(nullptr),
		  placePtr(nullptr),
		  allocatedSize(0) {
	}

	Runnable(std::nullptr_t) noexcept
		: Runnable() {
	}

	Runnable(_Ret (*p)(_Types...)) noexcept : disposeFunc(nullptr) {
		InitAllocateFunctor(8);
		constructFunc = [](void* dest, void* source) -> void {
			*(size_t*)dest = *(size_t*)source;
		};
		memcpy(placePtr, &p, sizeof(size_t));
		funcPtr = [](void* pp, _Types&&... tt) -> _Ret {
			_Ret (*fp)(_Types...) = (_Ret(*)(_Types...))(*(void**)pp);
			return fp(std::forward<_Types>(tt)...);
		};
	}

	Runnable(const Runnable<_Ret(_Types...)>& f) noexcept {
		funcPtr = f.funcPtr;
		constructFunc = f.constructFunc;
		disposeFunc = f.disposeFunc;
		if (constructFunc) {
			InitAllocateFunctor(f.allocatedSize);
			constructFunc(placePtr, (char*)f.placePtr);
		} else {
			allocatedSize = 0;
		}
	}
	Runnable(Runnable<_Ret(_Types...)>& f) noexcept
		: Runnable(static_cast<Runnable<_Ret(_Types...)> const&>(f)) {
	}

	Runnable(Runnable<_Ret(_Types...)>&& f) noexcept {
		allocatedSize = f.allocatedSize;
		if (allocatedSize <= 8) {
			placePtr = &funcPtrPlaceHolder;
			funcPtrPlaceHolder = f.funcPtrPlaceHolder;
		} else
			placePtr = f.placePtr;
		funcPtr = f.funcPtr;
		disposeFunc = f.disposeFunc;
		constructFunc = f.constructFunc;
		f.placePtr = nullptr;
		f.disposeFunc = nullptr;
		f.constructFunc = nullptr;
		f.allocatedSize = 0;
	}

	template<typename Functor>
	Runnable(Functor&& f) noexcept {
		using PureType = typename std::remove_cvref_t<Functor>;
		InitAllocateFunctor(sizeof(PureType));
		new (placePtr) PureType{std::forward<Functor>(f)};
		constructFunc = [](void* dest, void* source) -> void {
			new (dest) PureType(std::forward<Functor>(*(PureType*)source));
		};
		funcPtr = [](void* pp, _Types&&... tt) -> _Ret {
			PureType* ff = (PureType*)pp;
			return (*ff)(std::forward<_Types>(tt)...);
		};
		disposeFunc = [](void* pp) -> void {
			PureType* ff = (PureType*)pp;
			ff->~PureType();
		};
	}

	void operator=(const Runnable<_Ret(_Types...)>& f) noexcept {
		if (&f == this) return;
		if (disposeFunc) disposeFunc(placePtr);
		AllocateFunctor(f.allocatedSize);
		funcPtr = f.funcPtr;
		constructFunc = f.constructFunc;
		disposeFunc = f.disposeFunc;
		if (constructFunc) {
			constructFunc(placePtr, (char*)f.placePtr);
		}
	}
	void operator=(Runnable<_Ret(_Types...)>& f) noexcept {
		if (&f == this) return;
		operator()(static_cast<Runnable<_Ret(_Types...)> const&>(f));
	}
	void operator=(Runnable<_Ret(_Types...)>&& f) noexcept {
		if (&f == this) return;
		if (disposeFunc) disposeFunc(placePtr);
		FreeFunctor();
		allocatedSize = f.allocatedSize;
		if (allocatedSize <= 8) {
			placePtr = &funcPtrPlaceHolder;
			funcPtrPlaceHolder = f.funcPtrPlaceHolder;
		} else
			placePtr = f.placePtr;
		funcPtr = f.funcPtr;
		disposeFunc = f.disposeFunc;
		constructFunc = f.constructFunc;
		f.placePtr = nullptr;
		f.disposeFunc = nullptr;
		f.constructFunc = nullptr;
		f.allocatedSize = 0;
	}

	void operator=(std::nullptr_t) noexcept {
		if (disposeFunc) disposeFunc(placePtr);
		Dispose();
	}

	void operator=(_Ret (*p)(_Types...)) noexcept {
		if (disposeFunc) disposeFunc(placePtr);
		disposeFunc = nullptr;
		AllocateFunctor(8);
		constructFunc = [](void* dest, void* source) -> void {
			*(size_t*)dest = *(size_t*)source;
		};
		memcpy(placePtr, &p, sizeof(size_t));
		funcPtr = [](void* pp, _Types&&... tt) -> _Ret {
			_Ret (*fp)(_Types...) = (_Ret(*)(_Types...))(*(void**)pp);
			return fp(std::forward<_Types>(tt)...);
		};
	}

	template<typename Functor>
	void operator=(Functor&& f) noexcept {
		using PureType = typename std::remove_cvref_t<Functor>;

		if (disposeFunc) disposeFunc(placePtr);
		AllocateFunctor(sizeof(PureType));
		new (placePtr) PureType{std::forward<Functor>(f)};
		constructFunc = [](void* dest, void* source) -> void {
			new (dest) PureType(std::forward<Functor>(*(PureType*)source));
		};
		funcPtr = [](void* pp, _Types&&... tt) -> _Ret {
			PureType* ff = (PureType*)pp;
			return (*ff)(std::forward<_Types>(tt)...);
		};
		disposeFunc = [](void* pp) -> void {
			PureType* ff = (PureType*)pp;
			ff->~PureType();
		};
	}

	_Ret operator()(_Types... t) const noexcept {
		return funcPtr(placePtr, std::forward<_Types>(t)...);
	}
	bool isAvaliable() const noexcept {
		return funcPtr;
	}
	~Runnable() noexcept {
		if (disposeFunc) disposeFunc(placePtr);
		FreeFunctor();
	}
};

template<typename T>
struct RunnableHash {
	size_t operator()(const Runnable<T>& runnable) const noexcept {
		vengine::hash<size_t> h;
		return h((size_t)runnable.funcPtr);
	}
};