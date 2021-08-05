#pragma once
#include <util/vstlconfig.h>
#include <stdint.h>
#include <util/Hash.h>
#include <util/Memory.h>
#include <type_traits>
#include <new>
#include <util/VAllocator.h>

template<class T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class Runnable;
template<class Ret,
		 class... TypeArgs, VEngine_AllocType allocType>
class Runnable<Ret(TypeArgs...), allocType> {
	/////////////////////Define
	static constexpr size_t PLACEHOLDERSIZE = 32;
	using PlaceHolderType = std::aligned_storage_t<PLACEHOLDERSIZE, alignof(size_t)>;
	struct IProcessFunctor {
		virtual void Delete(void* ptr, VAllocHandle<allocType> const& allocHandle) const = 0;
		virtual void CopyConstruct(void*& dst, void const* src, PlaceHolderType* holder, VAllocHandle<allocType> const& allocHandle) const = 0;
		virtual void MoveConstruct(void*& dst, void* src, PlaceHolderType* holder) const = 0;
	};
	using ProcessorHolder = std::aligned_storage_t<sizeof(IProcessFunctor), alignof(IProcessFunctor)>;
	/////////////////////Data
	void* placePtr;
	funcPtr_t<Ret(void const*, TypeArgs&&...)> runFunc;
	ProcessorHolder logicPlaceHolder;
	PlaceHolderType funcPtrPlaceHolder;
	VAllocHandle<allocType> allocHandle;
	IProcessFunctor const* GetPtr() const noexcept {
		return reinterpret_cast<IProcessFunctor const*>(&logicPlaceHolder);
	}
	using SelfType = Runnable<Ret(TypeArgs...), allocType>;

public:
	operator bool() const noexcept {
		return placePtr;
	}

	void Dispose() noexcept {
		if (placePtr) {
			GetPtr()->Delete(placePtr, allocHandle);
			placePtr = nullptr;
		}
		runFunc = nullptr;
	}

	Runnable() noexcept
		: placePtr(nullptr),
		  runFunc(nullptr) {
	}

	Runnable(std::nullptr_t) noexcept
		: Runnable() {
	}

	Runnable(funcPtr_t<Ret(TypeArgs...)> p) noexcept {
		struct FuncPtrLogic final : public IProcessFunctor {
			void Delete(void* ptr, VAllocHandle<allocType> const& allocHandle) const override {
			}
			void CopyConstruct(void*& dest, void const* source, PlaceHolderType* placeHolder, VAllocHandle<allocType> const& allocHandle) const override {
				reinterpret_cast<size_t&>(dest) = reinterpret_cast<size_t>(source);
			}
			void MoveConstruct(void*& dst, void* src, PlaceHolderType* placeHolder) const override {
				reinterpret_cast<size_t&>(dst) = reinterpret_cast<size_t>(src);
			}
		};
		runFunc = [](void const* pp, TypeArgs&&... tt) -> Ret {
			funcPtr_t<Ret(TypeArgs...)> fp = reinterpret_cast<funcPtr_t<Ret(TypeArgs...)>>(pp);
			return fp(std::forward<TypeArgs>(tt)...);
		};
		new (&logicPlaceHolder) FuncPtrLogic();
		reinterpret_cast<size_t&>(placePtr) = reinterpret_cast<size_t>(p);
	}

	Runnable(const SelfType& f) noexcept
		: logicPlaceHolder(f.logicPlaceHolder),
		  runFunc(f.runFunc) {
		if (f.placePtr) {
			GetPtr()->CopyConstruct(placePtr, f.placePtr, &funcPtrPlaceHolder, allocHandle);
		}
	}
	Runnable(SelfType& f) noexcept
		: Runnable(static_cast<SelfType const&>(f)) {}
	Runnable(SelfType const && f) noexcept
		: Runnable(f) {}
	Runnable(SelfType&& f) noexcept
		: logicPlaceHolder(f.logicPlaceHolder),
		  runFunc(f.runFunc) {
		if (f.placePtr) {
			GetPtr()->MoveConstruct(placePtr, f.placePtr, &funcPtrPlaceHolder);
			f.placePtr = nullptr;
		}
		f.runFunc = nullptr;
	}

	template<typename Functor>
	Runnable(Functor&& f) noexcept {

		using PureType = std::remove_cvref_t<Functor>;
		constexpr bool USE_HEAP = (sizeof(PureType) > PLACEHOLDERSIZE);
		struct FuncPtrLogic final : public IProcessFunctor {
			void Delete(void* pp, VAllocHandle<allocType> const& allocHandle) const override {
				PureType* ff = (PureType*)pp;
				ff->~PureType();
				if constexpr (USE_HEAP) {
					allocHandle.Free(pp);
				}
			}
			void CopyConstruct(void*& dest, void const* source, PlaceHolderType* placeHolder, VAllocHandle<allocType> const& allocHandle) const override {
				if constexpr (USE_HEAP) {
					dest = allocHandle.Malloc(sizeof(PureType));
				} else {
					dest = placeHolder;
				}
				new (dest) PureType(*(PureType const*)source);
			}
			void MoveConstruct(void*& dest, void* source, PlaceHolderType* placeHolder) const override {
				if constexpr (USE_HEAP) {
					dest = source;
				} else {
					dest = placeHolder;
					new (dest) PureType(
						std::move(*reinterpret_cast<PureType*>(source)));
				}
			}
		};
		runFunc = [](void const* pp, TypeArgs&&... tt) -> Ret {
			PureType const* ff = reinterpret_cast<PureType const*>(pp);
			return (*ff)(std::forward<TypeArgs>(tt)...);
		};
		new (&logicPlaceHolder) FuncPtrLogic();
		if constexpr (USE_HEAP) {
			placePtr = allocHandle.Malloc(sizeof(PureType));
		} else {
			placePtr = &funcPtrPlaceHolder;
		}
		new (placePtr) PureType(std::forward<Functor>(f));
	}

	void operator=(std::nullptr_t) noexcept {
		Dispose();
	}
	template<typename Functor>
	void operator=(Functor&& f) noexcept {
		~SelfType();
		new (this) SelfType(std::forward<Functor>(f));
	}

	Ret operator()(TypeArgs... t) const noexcept {
		return runFunc(placePtr, std::forward<TypeArgs>(t)...);
	}
	~Runnable() noexcept {
		if (placePtr) {
			GetPtr()->Delete(placePtr, allocHandle);
		}
	}
};

template<typename T>
decltype(auto) MakeRunnable(T&& functor) {
	return Runnable<FuncType<std::remove_cvref_t<T>>>(functor);
}

namespace vstd {
template<class T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
using function = Runnable<T, allocType>;
}
