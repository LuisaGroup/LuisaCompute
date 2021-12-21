#pragma once
#include <vstl/config.h>
#include <type_traits>
#include <stdint.h>

#include <typeinfo>
#include <new>
#include <vstl/Hash.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <tuple>
#include <utility>
#include <vstl/AllocateType.h>
#include <vstl/Compare.h>
#include <vstl/Hash.h>
#include <core/allocator.h>
inline void *vengine_malloc(size_t size) {
    return luisa::detail::allocator_allocate(size, 0);
}
inline void vengine_free(void *ptr) {
    luisa::detail::allocator_deallocate(ptr, 0);
}
inline void *vengine_realloc(void *ptr, size_t size) {
    return luisa::detail::allocator_reallocate(ptr, size, 0);
}
VENGINE_DLL_COMMON void VEngine_Log(std::type_info const &t);
VENGINE_DLL_COMMON void VEngine_Log(char const *chunk);
namespace vstd {
template<typename T>
struct funcPtr;

template<typename Ret, typename... Args>
struct funcPtr<Ret(Args...)> {
    using Type = Ret (*)(Args...);
    using FuncType = Ret(Args...);
    using RetType = Ret;
};

template<typename Ret, typename... Args>
struct funcPtr<Ret (*)(Args...)> {
    using Type = Ret (*)(Args...);
    using FuncType = Ret(Args...);
    using RetType = Ret;
};

template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;
template<typename T>
using functor_t = typename funcPtr<T>::FuncType;
template<typename T, uint32_t size = 1>
class Storage {
    alignas(T) char c[size * sizeof(T)];
};
template<typename T>
class Storage<T, 0> {};

using lockGuard = std::lock_guard<std::mutex>;

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
        new (storage) T{std::forward<Args>(args)...};
        return *this;
    }
    template<typename... Args>
    inline SelfType &&New(Args &&...args) &&noexcept {
        return std::move(New(std::forward<Args>(args)...));
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
            VEngine_Log(typeid(T));
            VENGINE_EXIT;
        }
    }
    StackObject(SelfType &&value) {
        if constexpr (std::is_move_constructible_v<T>) {
            new (storage) T(std::move(*value));
        } else {
            VEngine_Log(typeid(T));
            VENGINE_EXIT;
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
            VEngine_Log(typeid(T));
            VENGINE_EXIT;
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
            VEngine_Log(typeid(T));
            VENGINE_EXIT;
        }
        return **this;
    }
    template<typename Arg>
        requires(std::is_assignable_v<T, Arg &&>)
    T &
    operator=(Arg &&value) {
        operator*() = std::forward<Arg>(value);
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
    inline SelfType &&New(Args &&...args) &&noexcept {
        return std::move(New(std::forward<Args>(args)...));
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
                VEngine_Log(typeid(T));
                VENGINE_EXIT;
            }
        }
    }
    StackObject(SelfType &&value) noexcept {
        initialized = value.initialized;
        if (initialized) {
            if constexpr (std::is_move_constructible_v<T>) {
                stackObj.New(std::move(*value));
            } else {
                VEngine_Log(typeid(T));
                VENGINE_EXIT;
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
                    VEngine_Log(typeid(T));
                    VENGINE_EXIT;
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
                    VEngine_Log(typeid(T));
                    VENGINE_EXIT;
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
    template<typename Arg>
        requires(std::is_assignable_v<StackObject<T, false>, Arg &&>)
    T &
    operator=(Arg &&value) {
        if (initialized) {
            return stackObj = std::forward<Arg>(value);
        } else {
            New(std::forward<Arg>(value));
            return **this;
        }
    }
};
//Declare Tuple
template<typename T>
using optional = StackObject<T, true>;

template<typename T>
using PureType_t = std::remove_pointer_t<std::remove_cvref_t<T>>;

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
template<>
struct compare<Type> {
    int32 operator()(Type const &a, Type const &b) const {
        if (a == b) return 0;
        return a.GetType().before(b.GetType()) ? -1 : 1;
    }
    int32 operator()(Type const &a, std::type_info const &b) const {
        if (a.GetType() == b) return 0;
        return a.GetType().before(b) ? -1 : 1;
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
    requires(std::is_bounded_array_v<T>)
constexpr size_t array_count(T const &t) {
    return array_meta<T>::array_size;
}
template<typename T>
    requires(std::is_bounded_array_v<T>)
constexpr size_t array_byte_size(T const &t) {
    return array_meta<T>::byte_size;
}

namespace vstl_detail {

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

template<typename Ret, typename... Args>
struct FunctionPointerData<Ret(Args...)> {
    using RetAndArgsType = FunctionRetAndArgs<Ret, Args...>;
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

template<typename Class, typename Ret, typename... Args>
struct memFuncPtr<Ret (Class::*)(Args...)> {
    using Type = FunctionType<Ret(Args...)>;
};

template<typename Class, typename Ret, typename... Args>
struct memFuncPtr<Ret (Class::*)(Args...) const> {
    using Type = FunctionType<Ret(Args...)>;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret (*)(Args...)> {
    using Type = FunctionType<Ret(Args...)>;
};
}// namespace vstl_detail

template<typename T>
using FunctionDataType = typename vstl_detail::FunctionType<T>::Type::RetAndArgsType;

template<typename T>
using FuncPtrType = typename vstl_detail::FunctionType<T>::Type::FuncPtrType;

template<typename T>
using FuncType = typename vstl_detail::FunctionType<T>::Type::FuncType;

template<typename T>
using FuncRetType = typename vstl_detail::FunctionType<T>::Type::RetType;

template<typename T>
constexpr size_t FuncArgCount = vstl_detail::FunctionType<T>::Type::ArgsCount;

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
        bool operator!=(rangeIte r) const {
            return r.v != v;
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
template<typename I, size_t sboSize = 48, size_t align = sizeof(size_t)>
class SBO {
    std::aligned_storage_t<sboSize, align> buffer;
    I *ptr;
    //void(Src, Dest)
    funcPtr_t<void *(void *, void *)> moveFunc;
    template<typename T>
    constexpr static bool IsLegalType = std::is_pointer_v<T> &&std::is_base_of_v<I, std::remove_pointer_t<T>>;

public:
    template<typename Func>
    constexpr static bool LegalCtorFunc = (std::is_invocable_v<Func, void *> && IsLegalType<FuncRetType<std::remove_cvref_t<Func>>>);
    I *operator->() const {
        return ptr;
    }
    I &operator*() const { return *ptr; }
    I *Get() const {
        return ptr;
    }

    template<typename Func>
        requires((!std::is_same_v<SBO, std::remove_cvref_t<Func>>)&&LegalCtorFunc<Func>)
    SBO(Func &&func) {
        using T = std::remove_pointer_t<FuncRetType<std::remove_cvref_t<Func>>>;
        constexpr size_t sz = sizeof(T);
        void *originPtr;
        if constexpr (sz > sboSize) {
            originPtr = vengine_malloc(sz);
        } else {
            originPtr = &buffer;
        }
        func(originPtr);
        ptr = reinterpret_cast<T *>(originPtr);
        if constexpr (std::is_move_constructible_v<T>) {
            moveFunc = [](void *src, void *dst) -> void * {
                if (dst)
                    return new (dst) T(std::move(*reinterpret_cast<T *>(src)));
                else {
                    I *ptr = reinterpret_cast<I *>(src);
                    T *offsetPtr = static_cast<T *>(ptr);
                    return offsetPtr;
                }
            };
        } else {
            moveFunc = [](void *src, void *dst) -> void * {
                if (dst)
                    VEngine_Log(typeid(T));
                else {
                    I *ptr = reinterpret_cast<I *>(0);
                    T *offsetPtr = static_cast<T *>(ptr);
                    return offsetPtr;
                }
            };
        }
    }
    SBO(SBO const &) = delete;
    SBO(SBO &&sbo)
        : moveFunc(sbo.moveFunc) {
        auto sboOriginPtr = moveFunc(sbo.ptr, nullptr);
        if (sboOriginPtr == &sbo.buffer) {
            auto originPtr = &buffer;
            moveFunc(sboOriginPtr, originPtr);
            ptr = reinterpret_cast<I *>(reinterpret_cast<size_t>(sbo.ptr) - reinterpret_cast<size_t>(sboOriginPtr) + reinterpret_cast<size_t>(originPtr));
        } else {
            ptr = sbo.ptr;
        }
        sbo.ptr = nullptr;
    }
    ~SBO() {
        if (!ptr) return;
        ptr->~I();
        auto originPtr = moveFunc(ptr, nullptr);
        if (originPtr != &buffer) {
            vengine_free(originPtr);
        }
    }
};
template<typename T>
class IEnumerable {
public:
    virtual T GetValue() = 0;
    virtual bool End() = 0;
    virtual void GetNext() = 0;
    virtual optional<size_t> Length() { return {}; }
    virtual ~IEnumerable() {}
};
struct IteEndTag {};
template<typename T>
class Iterator {
private:
    using PtrType = SBO<IEnumerable<T>>;
    PtrType ptr;

public:
    IEnumerable<T> *Get() const { return ptr; }
    template<typename Func>
        requires((!std::is_same_v<Iterator, std::remove_cvref_t<Func>>)&&PtrType::template LegalCtorFunc<Func>)
    Iterator(Func &&func) : ptr(std::forward<Func>(func)) {}
    Iterator(Iterator const &) = delete;
    Iterator(Iterator &&v)
        : ptr(v.ptr) {
        v.ptr = nullptr;
    }
    ~Iterator() {
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
    bool operator!=(IteEndTag) const {
        return !ptr->End();
    }
    operator bool() const { return operator!=({}); }
};
template<typename T>
    requires(!std::is_reference_v<T>)
struct MoveIterator {
    T *t;
    MoveIterator(T &&t) : t(&t) {}
    MoveIterator(T &t) : t(&t) {}
    MoveIterator(MoveIterator &&) = delete;
    MoveIterator(MoveIterator const &) = delete;
    decltype(auto) begin() {
        return std::move(*t).begin();
    }
    decltype(auto) end() {
        return std::move(*t).end();
    }
};
template<typename T>
struct ptr_range {
public:
    struct rangeIte {
        T *v;
        int64 inc;
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
        bool operator!=(rangeIte r) const {
            return r.v != v;
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
private:
    std::remove_reference_t<T> t;

public:
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
T *get_rvalue_ptr(T &&v) {
    static_assert(!std::is_lvalue_reference_v<T>, "only rvalue allowed!");
    return &v;
}
template<typename T>
decltype(auto) get_const_lvalue(T &&data) {
    return static_cast<std::remove_reference_t<T> const &>(data);
}
template<typename A, typename B>
decltype(auto) array_same(A &&a, B &&b) {
    auto aSize = a.size();
    auto bSize = b.size();
    if (aSize != bSize) return false;
    auto ite = a.begin();
    auto end = a.end();
    auto oIte = b.begin();
    auto oEnd = b.end();
    while (ite != end && oIte != oEnd) {
        if (*ite != *oIte) return false;
        ++ite;
        ++oIte;
    }
    return true;
}
namespace detail {
template<bool... v>
struct Any;

template<bool... vs>
struct Any<true, vs...> {
    static constexpr bool value = true;
};
template<bool... vs>
struct Any<false, vs...> {
    static constexpr bool value = Any<vs...>::value;
};
template<>
struct Any<> {
    static constexpr bool value = false;
};
template<bool... v>
static constexpr bool Any_v = Any<v...>::value;
template<size_t... size>
struct max_size {
    static constexpr auto value = static_cast<size_t>(0u);
};
template<size_t first, size_t... other>
struct max_size<first, other...> {
    static constexpr auto value = std::max(first, max_size<other...>::value);
};
template<bool isConst>
struct GetVoidType;
template<>
struct GetVoidType<true> {
    using Type = void const;
};
template<>
struct GetVoidType<false> {
    using Type = void;
};
template<typename T>
using GetVoidType_t = typename GetVoidType<std::is_const_v<std::remove_reference_t<T>>>::Type;
template<typename Ret, typename Func, typename PtrType>
constexpr static Ret FuncTable(GetVoidType_t<PtrType> *ptr, GetVoidType_t<Func> *func) {
    using PureFunc = std::remove_cvref_t<Func>;
    PureFunc *realFunc = reinterpret_cast<PureFunc *>(func);
    return (std::forward<Func>(*realFunc))(std::forward<PtrType>(*reinterpret_cast<std::remove_reference_t<PtrType> *>(ptr)));
}
template<size_t i, typename Dest, typename... Args>
struct IndexOfStruct {
    static constexpr size_t Index = i;
};

template<size_t i, typename Dest, typename T, typename... Args>
struct IndexOfStruct<i, Dest, T, Args...> {
    static constexpr size_t Index = std::is_same_v<Dest, T> ? i : IndexOfStruct<i + 1, Dest, Args...>::Index;
};
template<size_t i, typename Dest, typename... Args>
struct AssignableOfStruct {
    static constexpr size_t Index = i;
};

template<size_t i, typename Dest, typename T, typename... Args>
struct AssignableOfStruct<i, Dest, T, Args...> {
    static constexpr size_t Index = std::is_assignable_v<Dest, T> ? i : AssignableOfStruct<i + 1, Dest, Args...>::Index;
};
}// namespace detail
class Evaluable {};
template<class Func>
class LazyEval : public Evaluable {
private:
    Func func;

public:
    using EvalType = decltype(std::declval<Func>()());
    LazyEval(Func &&func)
        : func(std::move(func)) {}
    LazyEval(LazyEval const &) = delete;
    LazyEval(LazyEval &&v)
        : func(std::move(v.func)) {}
    operator decltype(auto)() {
        return func();
    }
};

template<class Func>
LazyEval<Func> MakeLazyEval(Func &&func) {
    return std::forward<Func>(func);
}
template<typename... Args>
static constexpr bool AlwaysFalse = false;
template<typename... AA>
class variant {
public:
    static constexpr size_t argSize = sizeof...(AA);
    template<typename TarT>
    static constexpr size_t IndexOf = detail::IndexOfStruct<0, std::remove_cvref_t<TarT>, AA...>::Index;
    template<typename TarT>
    static constexpr size_t AssignableOf = detail::AssignableOfStruct<0, std::remove_cvref_t<TarT>, AA...>::Index;

private:
    template<typename... Funcs>
    struct PackedFunctors {
        std::tuple<Funcs...> funcs;
        PackedFunctors(Funcs &&...funcs)
            : funcs(std::forward<Funcs>(funcs)...) {}
        template<typename T>
        decltype(auto) operator()(T &&v) {
            constexpr size_t idx = IndexOf<std::remove_cvref_t<T>>;
            return std::get<idx>(funcs)(std::forward<T>(v));
        }
    };

    template<size_t i, typename T, typename... Args>
    struct TChsStr {
        using TT = typename TChsStr<i - 1, Args...>::TT;
    };

    template<typename T, typename... Args>
    struct TChsStr<0, T, Args...> {
        using TT = T;
    };
    template<size_t i, typename... Args>
    using Types = typename TChsStr<i, Args...>::TT;

    template<typename Ret, typename Func, typename VoidPtr, typename... Types>
    class Visitor {
    public:
        static Ret Visit(
            size_t id,
            VoidPtr *ptr,
            Func &&func) {
            constexpr static auto table =
                {
                    detail::FuncTable<
                        Ret,
                        std::remove_reference_t<Func>,
                        Types>...};
            return table.begin()[id](ptr, &func);
        }
    };
    template<typename... Args>
    struct Constructor;
    template<typename B, typename... Args>
    struct Constructor<B, Args...> {
        template<typename... A>
        static size_t AnyConst(void *ptr, size_t idx, A &&...a) {
            if constexpr (std::is_constructible_v<B, A &&...>) {
                new (ptr) B(std::forward<A>(a)...);
                return idx;
            } else {
                return Constructor<Args...>::template AnyConst<A...>(ptr, idx + 1, std::forward<A>(a)...);
            }
        }

        template<size_t v>
        static decltype(auto) Get(void *ptr) {
            if constexpr (v == 0) {
                return get_lvalue(*reinterpret_cast<B *>(ptr));
            } else {
                return Constructor<Args...>::template Get<v - 1>(ptr);
            }
        }
        template<size_t v>
        static decltype(auto) Get(void const *ptr) {
            if constexpr (v == 0) {
                return get_lvalue(*reinterpret_cast<B const *>(ptr));
            } else {
                return get_lvalue(Constructor<Args...>::template Get<v - 1>(ptr));
            }
        }
    };
    using DefaultCtor = Constructor<AA...>;

    std::aligned_storage_t<(detail::max_size<sizeof(AA)...>::value), (detail::max_size<alignof(AA)...>::value)> placeHolder;
    size_t switcher = 0;
    void m_dispose() {
        if constexpr (detail::Any_v<!std::is_trivially_destructible_v<AA>...>) {
            auto disposeFunc = [&]<typename T>(T &value) {
                value.~T();
            };
            visit(disposeFunc);
        }
    }

public:
    template<size_t i>
    using TypeOf = std::tuple_element_t<i, std::tuple<AA...>>;

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

    void *GetPlaceHolder() { return &placeHolder; }
    void const *GetPlaceHolder() const { return &placeHolder; }
    size_t GetType() const { return switcher; }
    size_t index() const { return switcher; }
    template<typename T>
    bool IsTypeOf() const {
        return switcher == IndexOf<T>;
    }

    template<size_t i>
        requires(i <= argSize)
    decltype(auto) get() & {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return DefaultCtor::template Get<i>(&placeHolder);
    }
    template<size_t i>
        requires(i <= argSize)
    decltype(auto) get() && {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(DefaultCtor::template Get<i>(&placeHolder));
    }
    template<size_t i>
        requires(i <= argSize)
    decltype(auto) get() const & {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return DefaultCtor::template Get<i>(&placeHolder);
    }

    template<typename T>
        requires((IndexOf<T>) < argSize)
    T const *try_get() const & {
        static constexpr auto tarIdx = IndexOf<T>;
        if (tarIdx != switcher) {
            return nullptr;
        }
        return &DefaultCtor::template Get<tarIdx>(&placeHolder);
    }

    template<typename T>
        requires((IndexOf<T>) < argSize)
    T *try_get() & {
        static constexpr auto tarIdx = IndexOf<T>;
        if (tarIdx != switcher) {
            return nullptr;
        }
        return &DefaultCtor::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
        requires((IndexOf<T>) < argSize)
    optional<T> try_get() && {
        static constexpr auto tarIdx = IndexOf<T>;
        if (tarIdx != switcher) {
            return {};
        }
        return optional<T>(std::move(DefaultCtor::template Get<tarIdx>(&placeHolder)));
    }
    template<typename T>
        requires((IndexOf<T>) < argSize)
    T get_or(T &&value)
    const & {
        static constexpr auto tarIdx = IndexOf<T>;
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return DefaultCtor::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
        requires((IndexOf<T>) < argSize)
    T get_or(T &&value) && {
        static constexpr auto tarIdx = IndexOf<T>;
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return std::move(DefaultCtor::template Get<tarIdx>(&placeHolder));
    }
    template<typename T>
        requires((IndexOf<T>) < argSize)
    T const &force_get() const & {
        static constexpr auto tarIdx = IndexOf<T>;
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return DefaultCtor::template Get<tarIdx>(&placeHolder);
    }

    template<typename T>
        requires((IndexOf<T>) < argSize)
    T &force_get() & {
        static constexpr auto tarIdx = IndexOf<T>;
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return DefaultCtor::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
        requires((IndexOf<T>) < argSize)
    T && force_get() && {
        static constexpr auto tarIdx = IndexOf<T>;
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(DefaultCtor::template Get<tarIdx>(&placeHolder));
    }
    template<typename Func>
    void visit(Func &&func) & {
        if (switcher >= argSize) return;
        Visitor<void, Func, void, AA &...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) && {
        if (switcher >= argSize) return;
        Visitor<void, Func, void, AA...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) const & {
        if (switcher >= argSize) return;
        Visitor<void, Func, void const, AA const &...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }

    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) & {
        if (switcher >= argSize) return;
        Visitor<void, PackedFunctors<Funcs...>, void, AA &...>::Visit(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) && {
        if (switcher >= argSize) return;
        Visitor<void, PackedFunctors<Funcs...>, void, AA...>::Visit(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) const & {
        if (switcher >= argSize) return;
        Visitor<void, PackedFunctors<Funcs...>, void const, AA const &...>::Visit(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    decltype(auto) multi_visit_or(Ret &&r, Funcs &&...funcs) & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return Visitor<EvalType, PackedFunctors<Funcs...>, void, AA &...>::Visit(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return Visitor<Ret, PackedFunctors<Funcs...>, void, AA &...>::Visit(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        }
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    decltype(auto) multi_visit_or(Ret &&r, Funcs &&...funcs) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return Visitor<EvalType, PackedFunctors<Funcs...>, void, AA...>::Visit(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return Visitor<Ret, PackedFunctors<Funcs...>, void, AA...>::Visit(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        }
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    decltype(auto) multi_visit_or(Ret &&r, Funcs &&...funcs) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return Visitor<EvalType, PackedFunctors<Funcs...>, void const, AA const &...>::Visit(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return Visitor<Ret, PackedFunctors<Funcs...>, void const, AA const &...>::Visit(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        }
    }

    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return Visitor<EvalType, Func, void, AA &...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return Visitor<Ret, Func, void, AA &...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return Visitor<EvalType, Func, void, AA...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return Visitor<Ret, Func, void, AA...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return Visitor<EvalType, Func, void const, AA const &...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return Visitor<Ret, Func, void const, AA const &...>::Visit(switcher, GetPlaceHolder(), std::forward<Func>(func));
        }
    }
    void dispose() {
        m_dispose();
        switcher = argSize;
    }
    variant() {
        switcher = argSize;
    }

    template<
        typename T,
        typename... Arg>
        requires(detail::Any_v<
                 std::is_constructible_v<AA, T &&, Arg &&...>...>)
    variant(T &&t, Arg &&...arg) {
        if constexpr (sizeof...(Arg) == 0) {
            using PureT = std::remove_cvref_t<T>;
            constexpr size_t tIdx = IndexOf<PureT>;
            if constexpr (tIdx < argSize) {
                switcher = tIdx;
                new (&placeHolder) PureT(std::forward<T>(t));
            } else {
                switcher = DefaultCtor::template AnyConst<T, Arg...>(&placeHolder, 0, std::forward<T>(t), std::forward<Arg>(arg)...);
            }
        } else {
            switcher = DefaultCtor::template AnyConst<T, Arg...>(&placeHolder, 0, std::forward<T>(t), std::forward<Arg>(arg)...);
        }
    }
    variant(variant const &v)
        : switcher(v.switcher) {
        auto copyFunc = [&]<typename T>(T const &value) {
            new (GetPlaceHolder()) T(value);
        };
        v.visit(copyFunc);
    }
    variant(variant &&v)
        : switcher(v.switcher) {
        auto moveFunc = [&]<typename T>(T &value) {
            new (GetPlaceHolder()) T(std::move(value));
        };
        v.visit(moveFunc);
    }
    ~variant() {
        m_dispose();
    }
    template<typename... Args>
        requires(detail::Any_v<
                 std::is_constructible_v<AA, Args &&...>...>)
    void reset(Args &&...args) {
        this->~variant();
        new (this) variant(std::forward<Args>(args)...);
    }
    template<typename T>
        requires(detail::Any_v<
                 std::is_assignable_v<AA, T &&>...>)
    variant &
    operator=(T &&t) {
        using PureT = std::remove_cvref_t<T>;
        constexpr size_t idxOfT = IndexOf<PureT>;
        if constexpr (idxOfT < argSize) {
            if (switcher == idxOfT) {
                *reinterpret_cast<PureT *>(&placeHolder) = std::forward<T>(t);
            } else {
                dispose();
                new (&placeHolder) PureT(std::forward<T>(t));
                switcher = idxOfT;
            }
        } else {
            constexpr size_t asignOfT = AssignableOf<std::remove_cvref_t<T>>;
            static_assert(asignOfT < argSize, "illegal type");
            using CurT = TypeOf<asignOfT>;
            if (switcher == asignOfT) {
                *reinterpret_cast<CurT *>(&placeHolder) = std::forward<T>(t);
            } else {
                dispose();
                new (&placeHolder) CurT(std::forward<T>(t));
                switcher = asignOfT;
            }
        }
        return *this;
    }
    variant &operator=(variant const &a) {
        if (switcher != a.switcher) {
            this->~variant();
            new (this) variant(a);
        } else {
            auto assignFunc = [&]<typename T>(T const &v) {
                if constexpr (std::is_copy_assignable_v<T>)
                    *reinterpret_cast<T *>(&placeHolder) = v;
                else {
                    VEngine_Log(typeid(T));
                    VENGINE_EXIT;
                }
            };
            a.visit(assignFunc);
        }
        return *this;
    }
    variant &operator=(variant &&a) {
        if (switcher != a.switcher) {
            this->~variant();
            new (this) variant(std::move(a));
        } else {
            auto assignFunc = [&]<typename T>(T &v) {
                if constexpr (std::is_move_assignable_v<T>)
                    *reinterpret_cast<T *>(&placeHolder) = std::move(v);
                else {
                    VEngine_Log(typeid(T));
                    VENGINE_EXIT;
                }
            };
            a.visit(assignFunc);
        }
        return *this;
    }
};

template<typename... T>
struct hash<variant<T...>> {
    size_t operator()(variant<T...> const &v) const {
        return v.visit_or(
            size_t(0),
            [&](auto &&v) {
                const hash<std::remove_cvref_t<decltype(v)>> hs;
                return hs(v);
            });
    }
};
template<typename... T>
struct compare<variant<T...>> {
    int32 operator()(variant<T...> const &a, variant<T...> const &b) const {
        if (a.GetType() == b.GetType()) {
            return a.visit_or(
                int32(0),
                [&](auto &&v) {
                    using TT = decltype(v);
                    using PureT = std::remove_cvref_t<TT>;
                    const compare<PureT> comp;
                    return comp(v, b.template force_get<TT>());
                });
        } else
            return (a.GetType() > b.GetType()) ? 1 : -1;
    }
};
}// namespace vstd
