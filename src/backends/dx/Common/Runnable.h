#pragma once
#include <VEngineConfig.h>
#include <stdint.h>
#include <Common/Hash.h>
#include <Common/Memory.h>
#include <type_traits>
template<typename T>
struct RunnableHash;
template<class T>
class Runnable;
template<class _Ret,
		 class... _Types>
class Runnable<_Ret(_Types...)> {
	friend class RunnableHash<_Ret(_Types...)>;
	using FunctionPtrType = funcPtr_t<_Ret(void*, _Types&&...)>;
	static constexpr size_t PLACEHOLDERSIZE = 24;
	using PlaceHolderType = std::aligned_storage_t<PLACEHOLDERSIZE, sizeof(size_t)>;

private:
	void* placePtr;
	FunctionPtrType funcPtr;
	funcPtr_t<void(void*)> disposeFunc;
	funcPtr_t<void(void*&, void const*, PlaceHolderType*)> constructFunc;
	funcPtr_t<void(void*&, void*)> moveFunc;
	PlaceHolderType funcPtrPlaceHolder;

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
		if (disposeFunc) {
			disposeFunc(placePtr);
			disposeFunc = nullptr;
		}
		placePtr = nullptr;
		constructFunc = nullptr;
		funcPtr = nullptr;
	}

	Runnable() noexcept
		: funcPtr(nullptr),
		  disposeFunc(nullptr),
		  constructFunc(nullptr),
		  placePtr(nullptr) {
	}

	Runnable(std::nullptr_t) noexcept
		: Runnable() {
	}

	Runnable(_Ret (*p)(_Types...)) noexcept : disposeFunc(nullptr) {
		constructFunc = [](void*& dest, void const* source, PlaceHolderType* placeHolder) -> void {
			dest = placeHolder;
			*(size_t*)dest = *(size_t const*)source;
		};
		moveFunc = [](void*& dest, void* source) -> void {
			*(size_t*)dest = *(size_t const*)source;
		};
		*(size_t*)placePtr = *(reinterpret_cast<size_t const*>(&p));
		funcPtr = [](void* pp, _Types&&... tt) -> _Ret {
			_Ret (*fp)(_Types...) = (_Ret(*)(_Types...))(*(void**)pp);
			return fp(std::forward<_Types>(tt)...);
		};
	}

	Runnable(const Runnable<_Ret(_Types...)>& f) noexcept
		: funcPtr(f.funcPtr),
		  constructFunc(f.constructFunc),
		  moveFunc(f.moveFunc),
		  disposeFunc(f.disposeFunc) {
		if (constructFunc) {
			constructFunc(placePtr, f.placePtr, &funcPtrPlaceHolder);
		}
	}
	Runnable(Runnable<_Ret(_Types...)>& f) noexcept
		: Runnable(static_cast<Runnable<_Ret(_Types...)> const&>(f)) {
	}

	Runnable(Runnable<_Ret(_Types...)>&& f) noexcept
		: funcPtr(f.funcPtr),
		  disposeFunc(f.disposeFunc),
		  moveFunc(f.moveFunc),
		  constructFunc(f.constructFunc)

	{
		if (f.placePtr == &f.funcPtrPlaceHolder) {
			placePtr = &funcPtrPlaceHolder;
			moveFunc(
				placePtr,
				f.placePtr);
			//TODO: place holder
		} else {
			placePtr = f.placePtr;
		}
		f.placePtr = nullptr;
		f.disposeFunc = nullptr;
		f.constructFunc = nullptr;
		f.moveFunc = nullptr;
	}

	template<typename Functor>
	Runnable(Functor&& f) noexcept {
		using PureType = std::remove_cvref_t<Functor>;
		constexpr bool USE_HEAP = (sizeof(PureType) > PLACEHOLDERSIZE);
		auto func = [](void*& dest, void const* source, PlaceHolderType* placeHolder) -> void {
			if constexpr (USE_HEAP) {
				dest = vengine_malloc(sizeof(PureType));
			} else {
				dest = placeHolder;
			}
			new (dest) PureType(*(PureType const*)source);
		};
		func(placePtr, &f, &funcPtrPlaceHolder);
		constructFunc = func;
		moveFunc = [](void*& dest, void* source) -> void {
			new (dest) PureType(
				std::move(*reinterpret_cast<PureType*>(source)));
		};
		funcPtr = [](void* pp, _Types&&... tt) -> _Ret {
			PureType* ff = (PureType*)pp;
			return (*ff)(std::forward<_Types>(tt)...);
		};
		disposeFunc = [](void* pp) -> void {
			PureType* ff = (PureType*)pp;
			ff->~PureType();
			if constexpr (USE_HEAP) {
				vengine_free(pp);
			}
		};
	}

	void operator=(std::nullptr_t) noexcept {
		Dispose();
	}
	template<typename Functor>
	void operator=(Functor&& f) noexcept {
		~Runnable();
		new (this) Runnable(std::forward<Functor>(f));
	}

	_Ret operator()(_Types... t) const noexcept {
		return funcPtr(placePtr, std::forward<_Types>(t)...);
	}
	bool isAvaliable() const noexcept {
		return funcPtr;
	}
	~Runnable() noexcept {
		if (disposeFunc) {
			disposeFunc(placePtr);
		}
	}
};

template<typename T>
struct RunnableHash {
	size_t operator()(const Runnable<T>& runnable) const noexcept {
		vengine::hash<size_t> h;
		return h((size_t)runnable.funcPtr);
	}
};