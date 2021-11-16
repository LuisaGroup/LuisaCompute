#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <vstl/Hash.h>
#include <vstl/Memory.h>
#include <type_traits>
#include <new>
#include <vstl/VAllocator.h>
#include <functional>
#include <vector>
namespace vstd {
template<class T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class function;
template<class Ret,
         class... TypeArgs, VEngine_AllocType allocType>
class function<Ret(TypeArgs...), allocType> {
    /////////////////////Define
    static constexpr size_t PLACEHOLDERSIZE = 32;
    using PlaceHolderType = std::aligned_storage_t<PLACEHOLDERSIZE, alignof(size_t)>;
    struct IProcessFunctor {
        virtual void Delete(void *ptr, VAllocHandle<allocType> const &allocHandle) const = 0;
        virtual void CopyConstruct(void *&dst, void const *src, PlaceHolderType *holder, VAllocHandle<allocType> const &allocHandle) const = 0;
        virtual void MoveConstruct(void *&dst, void *src, PlaceHolderType *holder) const = 0;
    };
    using ProcessorHolder = std::aligned_storage_t<sizeof(IProcessFunctor), alignof(IProcessFunctor)>;
    /////////////////////Data
    void *placePtr;
    funcPtr_t<Ret(void *, TypeArgs &&...)> runFunc;
    ProcessorHolder logicPlaceHolder;
    PlaceHolderType funcPtrPlaceHolder;
    VAllocHandle<allocType> allocHandle;
    IProcessFunctor const *GetPtr() const noexcept {
        return reinterpret_cast<IProcessFunctor const *>(&logicPlaceHolder);
    }

public:
    operator bool() const noexcept {
        return placePtr;
    }
    bool operator!() const {
        return placePtr == nullptr;
    }
    void Dispose() noexcept {
        if (placePtr) {
            GetPtr()->Delete(placePtr, allocHandle);
            placePtr = nullptr;
        }
        runFunc = nullptr;
    }
    function() noexcept
        : placePtr(nullptr),
          runFunc(nullptr) {
    }
    function(std::nullptr_t) noexcept
        : function() {
    }
    function(funcPtr_t<Ret(TypeArgs...)> p) noexcept {
        struct FuncPtrLogic final : public IProcessFunctor {
            void Delete(void *ptr, VAllocHandle<allocType> const &allocHandle) const override {
            }
            void CopyConstruct(void *&dest, void const *source, PlaceHolderType *placeHolder, VAllocHandle<allocType> const &allocHandle) const override {
                reinterpret_cast<size_t &>(dest) = reinterpret_cast<size_t>(source);
            }
            void MoveConstruct(void *&dst, void *src, PlaceHolderType *placeHolder) const override {
                reinterpret_cast<size_t &>(dst) = reinterpret_cast<size_t>(src);
            }
        };
        runFunc = [](void *pp, TypeArgs &&...tt) -> Ret {
            funcPtr_t<Ret(TypeArgs...)> fp = reinterpret_cast<funcPtr_t<Ret(TypeArgs...)>>(const_cast<void *>(pp));
            return fp(std::forward<TypeArgs>(tt)...);
        };
        new (&logicPlaceHolder) FuncPtrLogic();
        reinterpret_cast<size_t &>(placePtr) = reinterpret_cast<size_t>(p);
    }
    function(const function &f) noexcept
        : logicPlaceHolder(f.logicPlaceHolder),
          runFunc(f.runFunc) {
        if (f.placePtr) {
            GetPtr()->CopyConstruct(placePtr, f.placePtr, &funcPtrPlaceHolder, allocHandle);
        }
    }
    function(function &&f) noexcept
        : logicPlaceHolder(f.logicPlaceHolder),
          runFunc(f.runFunc) {
        if (f.placePtr) {
            GetPtr()->MoveConstruct(placePtr, f.placePtr, &funcPtrPlaceHolder);
            f.placePtr = nullptr;
        }
        f.runFunc = nullptr;
    }
    template<typename Functor>
    requires (std::is_invocable_r_v<Ret, Functor, TypeArgs...> && !std::is_same_v<std::remove_cvref_t<Functor>, function>)
    function(Functor &&f) noexcept {
        using PureType = std::remove_cvref_t<Functor>;
        constexpr bool USE_HEAP = (sizeof(PureType) > PLACEHOLDERSIZE);
        struct FuncPtrLogic final : public IProcessFunctor {
            void Delete(void *pp, VAllocHandle<allocType> const &allocHandle) const override {
                PureType *ff = (PureType *)pp;
                ff->~PureType();
                if constexpr (USE_HEAP) {
                    allocHandle.Free(pp);
                }
            }
            void CopyConstruct(void *&dest, void const *source, PlaceHolderType *placeHolder, VAllocHandle<allocType> const &allocHandle) const override {
                if constexpr (USE_HEAP) {
                    dest = allocHandle.Malloc(sizeof(PureType));
                } else {
                    dest = placeHolder;
                }
                new (dest) PureType(*(PureType const *)source);
            }
            void MoveConstruct(void *&dest, void *source, PlaceHolderType *placeHolder) const override {
                if constexpr (USE_HEAP) {
                    dest = source;
                } else {
                    dest = placeHolder;
                    new (dest) PureType(
                        std::move(*reinterpret_cast<PureType *>(source)));
                }
            }
        };
        runFunc = [](void *pp, TypeArgs &&...tt) -> Ret {
            PureType *ff = reinterpret_cast<PureType *>(pp);
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
    void operator=(Functor &&f) noexcept {
        this->~function();
        new (this) function(std::forward<Functor>(f));
    }
    Ret operator()(TypeArgs... t) const noexcept {
        return runFunc(placePtr, std::forward<TypeArgs>(t)...);
    }
    ~function() noexcept {
        if (placePtr) {
            GetPtr()->Delete(placePtr, allocHandle);
        }
    }
};
template<class Func>
class LazyEval {
private:
    Func func;

public:
    LazyEval(Func &&func)
        : func(std::move(func)) {}
    LazyEval(Func const &func)
        : func(func) {
    }
    operator std::invoke_result_t<Func>() const {
        return func();
    }
};

template<class Func>
LazyEval<std::remove_cvref_t<Func>> MakeLazyEval(Func &&func) {
    return std::forward<Func>(func);
}

template<typename T>
decltype(auto) MakeRunnable(T &&functor) {
    return function<FuncType<std::remove_cvref_t<T>>>(functor);
}
template<typename Vec, typename Func>
void push_back_func(Vec &&vec, Func &&func, size_t count) {
    std::fill_n(
        std::back_inserter(vec),
        count,
        LazyEval<std::remove_cvref_t<Func>>(std::forward<Func>(func)));
}

}// namespace vstd