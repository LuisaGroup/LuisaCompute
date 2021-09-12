#pragma once

#include <type_traits>
#include <cstdint>
#include <span>
#include <typeinfo>
#include <new>
#include <mutex>
#include <atomic>
#include <thread>

#include <vstl/config.h>
#include <vstl/Hash.h>
#include <core/basic_types.h>

using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;

namespace vstd {

LUISA_DLL void vstl_log(std::type_info const &t);
LUISA_DLL void vstl_log(char const *chunk);

template<typename T>
struct funcPtr;

template<typename _Ret, typename... Args>
struct funcPtr<_Ret(Args...)> {
    using Type = _Ret (*)(Args...);
    using FuncType = _Ret(Args...);
};

template<typename _Ret, typename... Args>
struct funcPtr<_Ret (*)(Args...)> {
    using Type = _Ret (*)(Args...);
    using FuncType = _Ret(Args...);
};

template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;
template<typename T>
using functor_t = typename funcPtr<T>::FuncType;
template<typename T, uint32_t size>
class Storage {
    alignas(T) char c[size * sizeof(T)];
};
template<typename T>
class Storage<T, 0> {};

template<typename T, bool autoDispose = false>
class StackObject;
template<typename T>
class StackObject<T, false> {
private:
    alignas(T) uint8_t storage[sizeof(T)];

public:
    using SelfType = StackObject<T, false>;
    template<typename... Args>
    inline SelfType &New(Args &&...args) &noexcept {
        new (storage) T(std::forward<Args>(args)...);
        return *this;
    }
    template<typename... Args>
    inline SelfType &&New(Args &&...args) &&noexcept {
        return std::move(New(std::forward<Args>(args)...));
    }
    template<typename... Args>
    inline SelfType &InPlaceNew(Args &&...args) &noexcept {
        new (storage) T{std::forward<Args>(args)...};
        return *this;
    }
    template<typename... Args>
    inline SelfType &&InPlaceNew(Args &&...args) &&noexcept {
        return std::move(InPlaceNew(std::forward<Args>(args)...));
    }
    inline void Delete() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>)
            (reinterpret_cast<T *>(storage))->~T();
    }
    T &operator*() &noexcept {
        return *reinterpret_cast<T *>(storage);
    }
    T &&operator*() &&noexcept {
        return std::move(*reinterpret_cast<T *>(storage));
    }
    T const &operator*() const &noexcept {
        return *reinterpret_cast<T const *>(storage);
    }
    T *operator->() noexcept {
        return reinterpret_cast<T *>(storage);
    }
    T const *operator->() const noexcept {
        return reinterpret_cast<T const *>(storage);
    }
    T *GetPtr() noexcept {
        return reinterpret_cast<T *>(storage);
    }
    T const *GetPtr() const noexcept {
        return reinterpret_cast<T const *>(storage);
    }
    operator T *() noexcept {
        return reinterpret_cast<T *>(storage);
    }
    operator T const *() const noexcept {
        return reinterpret_cast<T const *>(storage);
    }
    StackObject() noexcept {}
    StackObject(const SelfType &value) {
        if constexpr (std::is_copy_constructible_v<T>) {
            new (storage) T(*value);
        } else {
            vstl_log(typeid(T));
            VSTL_ABORT();
        }
    }
    StackObject(SelfType &&value) {
        if constexpr (std::is_move_constructible_v<T>) {
            new (storage) T(std::move(*value));
        } else {
            vstl_log(typeid(T));
            VSTL_ABORT();
        }
    }
    template<typename... Args>
    StackObject(Args &&...args) {
        new (storage) T(std::forward<Args>(args)...);
    }
    T &operator=(SelfType const &value) {
        if constexpr (std::is_copy_assignable_v<T>) {
            operator*() = *value;
        } else if constexpr (std::is_copy_constructible_v<T>) {
            Delete();
            New(*value);
        } else {
            vstl_log(typeid(T));
            VSTL_ABORT();
        }
        return **this;
    }
    T &operator=(SelfType &&value) {
        if constexpr (std::is_move_assignable_v<T>) {
            operator*() = std::move(*value);
        } else if constexpr (std::is_move_constructible_v<T>) {
            Delete();
            New(std::move(*value));
        } else {
            vstl_log(typeid(T));
            VSTL_ABORT();
        }
        return **this;
    }
    T &operator=(T const &value) {
        if constexpr (std::is_copy_assignable_v<T>) {
            operator*() = value;
        } else if constexpr (std::is_copy_constructible_v<T>) {
            Delete();
            New(value);
        } else {
            vstl_log(typeid(T));
            VSTL_ABORT();
        }
        return **this;
    }
    T &operator=(T const &&value) {
        operator=(value);
    }
    T &operator=(T &&value) {
        if constexpr (std::is_move_assignable_v<T>) {
            operator*() = std::move(value);
        } else if constexpr (std::is_move_constructible_v<T>) {
            Delete();
            New(std::move(value));
        } else {
            vstl_log(typeid(T));
            VSTL_ABORT();
        }
        return **this;
    }
};

template<typename T>
class StackObject<T, true> {
private:
    StackObject<T, false> stackObj;
    bool initialized;

public:
    using SelfType = StackObject<T, true>;
    template<typename... Args>
    inline SelfType &New(Args &&...args) &noexcept {
        if (initialized) return *this;
        initialized = true;
        stackObj.New(std::forward<Args>(args)...);
        return *this;
    }
    template<typename... Args>
    inline SelfType &InPlaceNew(Args &&...args) &noexcept {
        if (initialized) return *this;
        initialized = true;
        stackObj.InPlaceNew(std::forward<Args>(args)...);
        return *this;
    }
    template<typename... Args>
    inline SelfType &&New(Args &&...args) &&noexcept {
        return std::move(New(std::forward<Args>(args)...));
    }
    template<typename... Args>
    inline SelfType &&InPlaceNew(Args &&...args) &&noexcept {
        return std::move(InPlaceNew(std::forward<Args>(args)...));
    }
    bool hash_value() const noexcept {
        return initialized;
    }

    bool Initialized() const noexcept {
        return initialized;
    }
    operator bool() const noexcept {
        return Initialized();
    }
    operator bool() noexcept {
        return Initialized();
    }
    inline bool Delete() noexcept {
        if (!Initialized()) return false;
        initialized = false;
        stackObj.Delete();
        return true;
    }
    void reset() const noexcept {
        Delete();
    }
    T &value() &noexcept {
        return *stackObj;
    }
    T const &value() const &noexcept {
        return *stackObj;
    }
    T &&value() &&noexcept {
        return std::move(*stackObj);
    }
    template<class U>
    T value_or(U &&default_value) const & {
        if (initialized)
            return *stackObj;
        else
            return std::forward<U>(default_value);
    }
    template<class U>
    T value_or(U &&default_value) && {
        if (initialized)
            return std::move(*stackObj);
        else
            return std::forward<U>(default_value);
    }
    T &operator*() &noexcept {
        return *stackObj;
    }
    T &&operator*() &&noexcept {
        return std::move(*stackObj);
    }
    T const &operator*() const &noexcept {
        return *stackObj;
    }
    T *operator->() noexcept {
        return stackObj.operator->();
    }
    T const *operator->() const noexcept {
        return stackObj.operator->();
    }
    T *GetPtr() noexcept {
        return stackObj.GetPtr();
    }
    T const *GetPtr() const noexcept {
        return stackObj.GetPtr();
    }
    operator T *() noexcept {
        return stackObj;
    }
    operator T const *() const noexcept {
        return stackObj;
    }
    StackObject() noexcept {
        initialized = false;
    }
    template<typename... Args>
    StackObject(Args &&...args)
        : stackObj(std::forward<Args>(args)...),
          initialized(true) {
    }
    StackObject(const SelfType &value) noexcept {
        initialized = value.initialized;
        if (initialized) {
            if constexpr (std::is_copy_constructible_v<T>) {
                stackObj.New(*value);
            } else {
                vstl_log(typeid(T));
                VSTL_ABORT();
            }
        }
    }
    StackObject(SelfType &&value) noexcept {
        initialized = value.initialized;
        if (initialized) {
            if constexpr (std::is_move_constructible_v<T>) {
                stackObj.New(std::move(*value));
            } else {
                vstl_log(typeid(T));
                VSTL_ABORT();
            }
        }
    }
    ~StackObject() noexcept {
        if (Initialized())
            stackObj.Delete();
    }
    T &operator=(SelfType const &value) {
        if (!initialized) {
            if (value.initialized) {
                if constexpr (std::is_copy_constructible_v<T>) {
                    stackObj.New(*value);
                } else {
                    vstl_log(typeid(T));
                    VSTL_ABORT();
                }
                initialized = true;
            }
        } else {
            if (value.initialized) {
                stackObj = value.stackObj;
            } else {
                stackObj.Delete();
                initialized = false;
            }
        }
        return *stackObj;
    }
    T &operator=(SelfType &&value) {
        if (!initialized) {
            if (value.initialized) {
                if constexpr (std::is_move_constructible_v<T>) {
                    stackObj.New(std::move(*value));
                } else {
                    vstl_log(typeid(T));
                    VSTL_ABORT();
                }
                initialized = true;
            }
        } else {
            if (value.initialized) {
                stackObj = std::move(value.stackObj);
            } else {
                stackObj.Delete();
                initialized = false;
            }
        }
        return *stackObj;
    }
    T &operator=(T const &value) {
        if (!initialized) {
            if constexpr (std::is_copy_constructible_v<T>) {
                stackObj.New(value);
            } else {
                vstl_log(typeid(T));
                VSTL_ABORT();
            }
            initialized = true;

        } else {
            stackObj = value;
        }
        return *stackObj;
    }
    T &operator=(T &&value) {
        if (!initialized) {
            if constexpr (std::is_move_constructible_v<T>) {
                stackObj.New(std::move(value));
            } else {
                vstl_log(typeid(T));
                VSTL_ABORT();
            }
            initialized = true;
        } else {
            stackObj = std::move(value);
        }
        return *stackObj;
    }
};

template<typename T>
using optional = StackObject<T, true>;

using lockGuard = std::lock_guard<std::mutex>;

struct Type {
private:
    const std::type_info *typeEle;
    struct DefaultType {};

public:
    Type() noexcept : typeEle(&typeid(DefaultType)) {
    }
    Type(const Type &t) noexcept : typeEle(t.typeEle) {
    }
    Type(const std::type_info &info) noexcept : typeEle(&info) {
    }
    Type(std::nullptr_t) noexcept : typeEle(nullptr) {}
    bool operator==(const Type &t) const noexcept {
        return *t.typeEle == *typeEle;
    }
    bool operator!=(const Type &t) const noexcept {
        return !operator==(t);
    }
    void operator=(const Type &t) noexcept {
        typeEle = t.typeEle;
    }
    size_t HashCode() const noexcept {
        if (!typeEle) return 0;
        return typeEle->hash_code();
    }
    const std::type_info &GetType() const noexcept {
        return *typeEle;
    }
};

template<>
struct hash<Type> {
    size_t operator()(const Type &t) const noexcept {
        return t.HashCode();
    }
};
template<typename T>
struct array_meta;
template<typename T, size_t N>
struct array_meta<T[N]> {
    static constexpr size_t array_size = N;
    static constexpr size_t byte_size = N * sizeof(T);
};

template<typename T>
static constexpr size_t array_count = array_meta<T>::array_size;

template<typename T>
static constexpr size_t array_size = array_meta<T>::byte_size;
#define VENGINE_ARRAY_COUNT(arr) (vstd::array_count<decltype(arr)>)
#define VENGINE_ARRAY_SIZE(arr) (vstd::array_size<decltype(arr)>)

namespace FunctionTemplateGlobal {

template<typename T, typename... Args>
struct FunctionRetAndArgs {
    static constexpr size_t ArgsCount = sizeof...(Args);
    using RetType = T;
    inline static const Type retTypes = typeid(T);
    inline static const Type argTypes[ArgsCount] =
        {
            typeid(Args)...};
};

template<typename T>
struct memFuncPtr;

template<typename T>
struct FunctionPointerData;

template<typename _Ret, typename... Args>
struct FunctionPointerData<_Ret(Args...)> {
    using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
    static constexpr size_t ArgsCount = sizeof...(Args);
};

template<typename T>
struct FunctionType {
    using Type = typename memFuncPtr<decltype(&T::operator())>::Type;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret(Args...)> {
    using Type = FunctionType<Ret(Args...)>;
    using RetAndArgsType = typename FunctionPointerData<Ret(Args...)>::RetAndArgsType;
    using FuncType = Ret(Args...);
    using RetType = Ret;
    static constexpr size_t ArgsCount = sizeof...(Args);
    using FuncPtrType = Ret (*)(Args...);
};

template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...)> {
    using Type = FunctionType<_Ret(Args...)>;
};

template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...) const> {
    using Type = FunctionType<_Ret(Args...)>;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret (*)(Args...)> {
    using Type = FunctionType<Ret(Args...)>;
};
}// namespace FunctionTemplateGlobal

template<typename T>
using FunctionDataType = typename FunctionTemplateGlobal::FunctionType<T>::Type::RetAndArgsType;

template<typename T>
using FuncPtrType = typename FunctionTemplateGlobal::FunctionType<T>::Type::FuncPtrType;

template<typename T>
using FuncType = typename FunctionTemplateGlobal::FunctionType<T>::Type::FuncType;

template<typename T>
using FuncRetType = typename FunctionTemplateGlobal::FunctionType<T>::Type::RetType;

template<typename T>
constexpr size_t FuncArgCount = FunctionTemplateGlobal::FunctionType<T>::Type::ArgsCount;

template<typename Func, typename Target>
static constexpr bool IsFunctionTypeOf = std::is_same_v<FuncType<Func>, Target>;
template<typename A, typename B, typename C, typename... Args>
decltype(auto) select(A &&a, B &&b, C &&c, Args &&...args) {
    if (c(std::forward<Args>(args)...)) {
        return b(std::forward<Args>(args)...);
    }
    return a(std::forward<Args>(args)...);
}
struct range {
public:
    struct rangeIte {
        int64 v;
        int64 inc;
        int64 &operator++() {
            v += inc;
            return v;
        }
        int64 operator++(int) {
            auto lastV = v;
            v += inc;
            return lastV;
        }
        int64 const *operator->() const {
            return &v;
        }
        int64 const &operator*() const {
            return v;
        }
        bool operator==(rangeIte r) const {
            return r.v == v;
        }
    };
    range(int64 b, int64 e, int64 inc = 1) : b(b), e(e), inc(inc) {}
    range(int64 e) : b(0), e(e), inc(1) {}
    rangeIte begin() const {
        return {b, inc};
    }
    rangeIte end() const {
        return rangeIte{e, 0};
    }

private:
    int64 b;
    int64 e;
    int64 inc;
};
class IDisposable {
protected:
    IDisposable() = default;
    ~IDisposable() = default;

public:
    virtual void Dispose() = 0;
};
template<typename T>
class IEnumerable : public vstd::IDisposable {
public:
    virtual T GetValue() = 0;
    virtual bool End() = 0;
    virtual void GetNext() = 0;
    virtual ~IEnumerable() noexcept = default;
};
struct IteEndTag {};
template<typename T>
class Iterator {
private:
    IEnumerable<T> *ptr;

public:
    Iterator(IEnumerable<T> *ptr) : ptr(ptr) {}
    Iterator(Iterator const &) = delete;
    Iterator(Iterator &&v)
        : ptr(v.ptr) {
        v.ptr = nullptr;
    }
    ~Iterator() {
        if (ptr) ptr->Dispose();
    }
    T operator*() const {
        return ptr->GetValue();
    }
    void operator++() const {
        ptr->GetNext();
    }
    bool operator==(IteEndTag) const {
        return ptr->End();
    }
};
template<typename T>
struct ptr_range {
public:
    struct rangeIte {
        T *v{nullptr};
        int64 inc{0u};
        T *operator++() {
            v += inc;
            return v;
        }
        T *operator++(int) {
            auto lastV = v;
            v += inc;
            return lastV;
        }
        T *operator->() const {
            return v;
        }
        T &operator*() const {
            return *v;
        }
        bool operator==(rangeIte r) const {
            return r.v == v;
        }
    };

    rangeIte begin() const {
        return {b, inc};
    }
    rangeIte end() const {
        return {e};
    }
    ptr_range(T *b, T *e, int64_t inc = 1) : b(b), e(e), inc(inc) {}

private:
    T *b;
    T *e;
    int64_t inc;
};
template<typename T>
struct disposer {
    std::remove_reference_t<T> t;
    template<typename A>
    disposer(A &&a)
        : t(std::forward<A>(a)) {}
    ~disposer() {
        t();
    }
};

template<typename T>
disposer<T> create_disposer(T &&t) {
    return disposer<T>(std::forward<T>(t));
}

template<typename T>
decltype(auto) get_lvalue(T &&data) {
    return static_cast<std::remove_reference_t<T> &>(data);
}
template<typename T>
decltype(auto) get_const_lvalue(T &&data) {
    return static_cast<std::remove_reference_t<T> const &>(data);
}
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

template<typename... AA>
class variant {
public:
    static constexpr size_t argSize = sizeof...(AA);

private:
    template<size_t... szs>
    struct MaxSize;

    template<size_t v, size_t... szs>
    struct MaxSize<v, szs...> {
        static constexpr size_t MAX_SIZE = std::max(v, MaxSize<szs...>::MAX_SIZE);
    };

    template<>
    struct MaxSize<> {
        static constexpr size_t MAX_SIZE = 0u;
    };

    template<size_t idx, size_t c, typename... Args>
    struct Iterator {
        template<typename Ret, typename... Funcs>
        static void Set(funcPtr_t<Ret(void *, void *)> *funcPtr, void **funcP, Funcs &&...fs) {}

        template<typename Ret, typename... Funcs>
        static void Set_Const(funcPtr_t<Ret(void *, void const *)> *funcPtr, void **funcP, Funcs &&...fs) {}
        using Type = void;
    };

    template<size_t idx, size_t c, typename T, typename... Args>
    struct Iterator<idx, c, T, Args...> {
        template<typename Ret, typename F, typename... Funcs>
        static void Set(funcPtr_t<Ret(void *, void *)> *funcPtr, void **funcP, F &&f, Funcs &&...fs) {
            if constexpr (idx == c)
                return;
            *funcPtr = [](void *ptr, void *arg) -> Ret {
                return (*reinterpret_cast<std::remove_reference_t<F> *>(ptr))(*reinterpret_cast<std::remove_reference_t<T> *>(arg));
            };
            *funcP = &f;
            Iterator<idx + 1, c, Args...>::template Set<Ret, Funcs...>(funcPtr + 1, funcP + 1, std::forward<Funcs>(fs)...);
        }
        template<typename Ret, typename F, typename... Funcs>
        static void Set_Const(funcPtr_t<Ret(void *, void const *)> *funcPtr, void **funcP, F &&f, Funcs &&...fs) {
            if constexpr (idx == c)
                return;
            *funcPtr = [](void *ptr, void const *arg) -> Ret {
                return (*reinterpret_cast<std::remove_reference_t<F> *>(ptr))(*reinterpret_cast<std::remove_reference_t<T> const *>(arg));
            };
            *funcP = &f;
            Iterator<idx + 1, c, Args...>::template Set_Const<Ret, Funcs...>(funcPtr + 1, funcP + 1, std::forward<Funcs>(fs)...);
        }
        template<bool isTrue>
        struct Typer {
            using Type = T;
        };
        template<>
        struct Typer<false> {
            using Type = typename Iterator<idx + 1, c, Args...>::Type;
        };
        using Type = typename Typer<(idx == c)>::Type;
    };

    template<typename... Args>
    struct Constructor {
        template<typename A>
        static size_t CopyOrMoveConst(void *, size_t idx, A &&) {
            return idx;
        }
        template<typename... A>
        static size_t AnyConst(void *, size_t idx, A &&...) {
            static_assert(luisa::always_false_v<A...>, "Illegal Constructor!");
            return idx;
        }
        static void Dispose(size_t, void *) {}
        static void Copy(size_t, void *, void const *) {}
        static void Move(size_t, void *, void *) {}
    };
    template<typename B, typename... Args>
    struct Constructor<B, Args...> {
        template<typename A>
        static size_t CopyOrMoveConst(void *ptr, size_t idx, A &&a) {
            if constexpr (std::is_same_v<std::remove_cvref_t<B>, std::remove_cvref_t<A>>) {
                new (ptr) B(std::forward<A>(a));
                return idx;
            } else {
                return Constructor<Args...>::template CopyOrMoveConst<A>(ptr, idx + 1, std::forward<A>(a));
            }
        }
        template<typename... A>
        static size_t AnyConst(void *ptr, size_t idx, A &&...a) {
            if constexpr (std::is_constructible_v<B, A &&...>) {
                new (ptr) B(std::forward<A>(a)...);
                return idx;
            } else {
                return Constructor<Args...>::template AnyConst<A...>(ptr, idx + 1, std::forward<A>(a)...);
            }
        }
        static void Dispose(size_t v, void *ptr) {
            if (v == 0) {
                reinterpret_cast<B *>(ptr)->~B();
            } else {
                Constructor<Args...>::Dispose(v - 1, ptr);
            }
        }
        static void Copy(size_t v, void *ptr, void const *dstPtr) {
            if constexpr (!std::is_copy_constructible_v<B>) {
                vstl_log(typeid(B));
                VSTL_ABORT();
            } else {

                if (v == 0) {
                    new (ptr) B(*reinterpret_cast<B const *>(dstPtr));
                } else {
                    Constructor<Args...>::Copy(v - 1, ptr, dstPtr);
                }
            }
        }
        static void Move(size_t v, void *ptr, void *dstPtr) {
            if constexpr (!std::is_move_constructible_v<B>) {
                vstl_log(typeid(B));
                VSTL_ABORT();
            } else {

                if (v == 0) {
                    new (ptr) B(std::move(*reinterpret_cast<B *>(dstPtr)));
                } else {
                    Constructor<Args...>::Move(v - 1, ptr, dstPtr);
                }
            }
        }
        template<size_t v>
        static decltype(auto) Get(void *ptr) {
            if constexpr (v == 0) {
                return vstd::get_lvalue(*reinterpret_cast<B *>(ptr));
            } else {
                return Constructor<Args...>::template Get<v - 1>(ptr);
            }
        }
        template<size_t v>
        static decltype(auto) Get(void const *ptr) {
            if constexpr (v == 0) {
                return vstd::get_lvalue(*reinterpret_cast<B const *>(ptr));
            } else {
                return vstd::get_lvalue(Constructor<Args...>::template Get<v - 1>(ptr));
            }
        }
    };

    template<typename Func, typename... Funcs>
    struct VisitFuncType {
        template<typename Arg, bool value>
        struct Typer {
            using Type = Arg &&;
        };

        template<typename Arg>
        struct Typer<Arg, false> {
            using Type = Arg &;
        };

        template<typename Arg, typename... Args>
        using Type = std::invoke_result_t<Func, typename Typer<Arg, std::is_invocable_v<Arg &&>>::Type>;
    };
    std::aligned_storage_t<(MaxSize<sizeof(AA)...>::MAX_SIZE), (MaxSize<alignof(AA)...>::MAX_SIZE)> placeHolder;
    size_t switcher = 0;

    template<size_t i, typename Dest, typename... Args>
    struct IndexOfStruct {
        static constexpr size_t Index = i;
    };

    template<size_t i, typename Dest, typename T, typename... Args>
    struct IndexOfStruct<i, Dest, T, Args...> {
        static constexpr size_t Index = std::is_same_v<Dest, T> ? i : IndexOfStruct<i + 1, Dest, Args...>::Index;
    };

public:
    template<size_t i>
    using TypeOf = typename Iterator<0, i, AA...>::Type;
    template<typename TarT>
    static constexpr size_t IndexOf = IndexOfStruct<0, std::remove_cvref_t<TarT>, AA...>::Index;
    variant() {
        switcher = argSize;
    }
    template<typename T, typename... Arg>
    variant(T &&t, Arg &&...arg) {
        if constexpr (sizeof...(Arg) == 0) {
            switcher = Constructor<AA...>::template CopyOrMoveConst<T>(&placeHolder, 0, std::forward<T>(t));
            if (switcher < sizeof...(AA))
                return;
        }
        switcher = Constructor<AA...>::template AnyConst<T, Arg...>(&placeHolder, 0, std::forward<T>(t), std::forward<Arg>(arg)...);
    }

    variant(variant const &v)
        : switcher(v.switcher) {
        Constructor<AA...>::Copy(switcher, &placeHolder, &v.placeHolder);
    }
    variant(variant &&v)
        : switcher(v.switcher) {
        Constructor<AA...>::Move(switcher, &placeHolder, &v.placeHolder);
    }
    variant(variant &v)
        : variant(static_cast<variant const &>(v)) {
    }
    variant(variant const &&v)
        : variant(v) {
    }
    template<typename... Args>
    void reset(Args &&...args) {
        this->~variant();
        new (this) variant(std::forward<Args>(args)...);
    }

    bool valid() const { return switcher < argSize; }

    template<typename Func>
    void update(size_t typeIndex, Func &&setFunc) {
        this->~variant();
        if (typeIndex >= argSize) {
            switcher = argSize;
            return;
        }
        switcher = typeIndex;
        setFunc(reinterpret_cast<void *>(&placeHolder));
    }

    ~variant() {
        if (switcher >= argSize)
            return;
        Constructor<AA...>::Dispose(switcher, &placeHolder);
    }
    void dispose() {
        if (switcher >= argSize)
            return;
        Constructor<AA...>::Dispose(switcher, &placeHolder);
        switcher = argSize;
    }
    void *GetPlaceHolder() { return &placeHolder; }
    void const *GetPlaceHolder() const { return &placeHolder; }
    size_t GetType() const { return switcher; }
    size_t index() const { return switcher; }
    template<typename T>
    bool IsTypeOf() const {
        return switcher == IndexOf<T>;
    }

    template<size_t i>
    decltype(auto) get() {
#ifdef VSTL_DEBUG
        if (i != switcher) {
            vstl_log("Try get wrong variant type!\n");
            VSTL_ABORT();
        }
#endif
        return Constructor<AA...>::template Get<i>(&placeHolder);
    }
    template<size_t i>
    decltype(auto) get() const {
#ifdef VSTL_DEBUG
        if (i != switcher) {
            vstl_log("Try get wrong variant type!\n");
            VSTL_ABORT();
        }
#endif
        return Constructor<AA...>::template Get<i>(&placeHolder);
    }

    template<typename T>
    T const *try_get() const {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return nullptr;
        }
        return &Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }

    template<typename T>
    T *try_get() {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return nullptr;
        }
        return &Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
    T const &force_get() const {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
#ifdef VSTL_DEBUG
        if (tarIdx != switcher) {
            vstl_log("Try get wrong variant type!\n");
            VSTL_ABORT();
        }
#endif
        return Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }

    template<typename T>
    T &force_get() {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
#ifdef VSTL_DEBUG
        if (tarIdx != switcher) {
            vstl_log("Try get wrong variant type!\n");
            VSTL_ABORT();
        }
#endif
        return Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }
    template<typename Arg>
    variant &operator=(Arg &&arg) {
        this->~variant();
        new (this) variant(std::forward<Arg>(arg));
        return *this;
    }

    template<typename... Funcs>
    decltype(auto) visit(Funcs &&...funcs) {
        static_assert(argSize == sizeof...(Funcs), "functor size not equal!");
        using RetType = typename VisitFuncType<std::remove_cvref_t<Funcs>...>::template Type<AA...>;
        if (switcher >= argSize)
            return RetType();
        funcPtr_t<RetType(void *, void *)> ftype[argSize];
        void *funcPs[argSize];
        Iterator<0, argSize, AA...>::template Set<RetType, Funcs...>(ftype, funcPs, std::forward<Funcs>(funcs)...);
        return ftype[switcher](funcPs[switcher], &placeHolder);
    }

    template<typename... Funcs>
    decltype(auto) visit(Funcs &&...funcs) const {
        static_assert(argSize == sizeof...(Funcs), "functor size not equal!");
        using RetType = typename VisitFuncType<std::remove_cvref_t<Funcs>...>::template Type<AA...>;
        if (switcher >= argSize)
            return RetType();
        funcPtr_t<RetType(void *, void const *)> ftype[argSize];
        void *funcPs[argSize];
        Iterator<0, argSize, AA const...>::template Set_Const<RetType, Funcs...>(ftype, funcPs, std::forward<Funcs>(funcs)...);
        return ftype[switcher](funcPs[switcher], &placeHolder);
    }
    template<typename T, typename... Funcs>
    T visit_with_default(T const &defaultValue, Funcs &&...funcs) const {
        static_assert(argSize == sizeof...(Funcs), "functor size not equal!");
        if (switcher >= argSize)
            return defaultValue;
        funcPtr_t<T(void *, void const *)> ftype[argSize];
        void *funcPs[argSize];
        Iterator<0, argSize, AA const...>::template Set_Const<T, Funcs...>(ftype, funcPs, std::forward<Funcs>(funcs)...);
        return ftype[switcher](funcPs[switcher], &placeHolder);
    }
    template<typename T, typename... Funcs>
    T visit_with_default(T const &defaultValue, Funcs &&...funcs) {
        static_assert(argSize == sizeof...(Funcs), "functor size not equal!");
        if (switcher >= argSize)
            return defaultValue;
        funcPtr_t<T(void *, void *)> ftype[argSize];
        void *funcPs[argSize];
        Iterator<0, argSize, AA...>::template Set<T, Funcs...>(ftype, funcPs, std::forward<Funcs>(funcs)...);
        return ftype[switcher](funcPs[switcher], &placeHolder);
    }
};
}// namespace vstd