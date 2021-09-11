#pragma once
#include <util/vstl_config.h>
#include <type_traits>
#include <stdint.h>
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
#include <typeinfo>
#include <new>
#include <util/Hash.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <util/AllocateType.h>
namespace vstd {
VENGINE_DLL_COMMON void VEngine_Log(std::type_info const &t);
VENGINE_DLL_COMMON void VEngine_Log(char const *chunk);
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
    T &operator=(T const &value) {
        if constexpr (std::is_copy_assignable_v<T>) {
            operator*() = value;
        } else if constexpr (std::is_copy_constructible_v<T>) {
            Delete();
            New(value);
        } else {
            VEngine_Log(typeid(T));
            VENGINE_EXIT;
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
            VEngine_Log(typeid(T));
            VENGINE_EXIT;
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
    T &operator=(T const &value) {
        if (!initialized) {
            if constexpr (std::is_copy_constructible_v<T>) {
                stackObj.New(value);
            } else {
                VEngine_Log(typeid(T));
                VENGINE_EXIT;
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
                VEngine_Log(typeid(T));
                VENGINE_EXIT;
            }
            initialized = true;
        } else {
            stackObj = std::move(value);
        }
        return *stackObj;
    }
};
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
    virtual ~IEnumerable() {}
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

}// namespace vstd