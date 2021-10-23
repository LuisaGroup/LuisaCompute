#pragma once
#include <vstl/config.h>
#include <type_traits>
#include <stdint.h>
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
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
static constexpr size_t array_count = array_meta<T>::array_size;

template<typename T>
static constexpr size_t array_size = array_meta<T>::byte_size;
#define VENGINE_ARRAY_COUNT(arr) (array_count<decltype(arr)>)
#define VENGINE_ARRAY_SIZE(arr) (array_size<decltype(arr)>)

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
template<typename T>
class IEnumerable : public IDisposable {
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
    IEnumerable<T> *ptr;

public:
    IEnumerable<T> *Get() const { return ptr; }
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
    bool operator!=(IteEndTag) const {
        return !ptr->End();
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
template<typename... Args>
static constexpr bool AlwaysFalse = false;
template<typename... AA>
class variant {
public:
    static constexpr size_t argSize = sizeof...(AA);

private:
    template<size_t maxSize, size_t... szs>
    struct MaxSize;
    template<size_t maxSize, size_t v, size_t... szs>
    struct MaxSize<maxSize, v, szs...> {
        static constexpr size_t MAX_SIZE = MaxSize<(maxSize > v ? maxSize : v), szs...>::MAX_SIZE;
    };
    template<size_t maxSize>
    struct MaxSize<maxSize> {
        static constexpr size_t MAX_SIZE = maxSize;
    };

    template<size_t idx, size_t c, typename... Args>
    struct Iterator {
        using Type = void;
    };

    template<size_t idx, size_t c, typename T, typename... Args>
    struct Iterator<idx, c, T, Args...> {
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

    template<typename... Funcs>
    struct PackedFunctors {
        std::tuple<Funcs...> funcs;
        PackedFunctors(Funcs &&...funcs)
            : funcs(std::forward<Funcs>(funcs)...) {}
        template<size_t idx, typename T>
        decltype(auto) Run(T &&v) {
            return std::get<idx>(funcs)(std::forward<T>(v));
        }
    };
    template<typename Ret, typename Func, typename PtrType>
    struct Visitor {
        template<size_t begin, size_t end, typename... Args>
        static Ret Visit(
            size_t id,
            PtrType *ptr,
            Func &&f) {
            if constexpr (end - begin == 0) {
                using ArgType = Types<begin, Args...>;
                using ArgPureType = std::remove_reference_t<ArgType>;
                return f(std::forward<ArgType>(*reinterpret_cast<ArgPureType *>(ptr)));
            } else if constexpr (end - begin == 1) {
                if (id == begin) {
                    using ArgType = Types<begin, Args...>;
                    using ArgPureType = std::remove_reference_t<ArgType>;
                    return f(std::forward<ArgType>(*reinterpret_cast<ArgPureType *>(ptr)));
                } else {
                    using ArgType = Types<end, Args...>;
                    using ArgPureType = std::remove_reference_t<ArgType>;
                    return f(std::forward<ArgType>(*reinterpret_cast<ArgPureType *>(ptr)));
                }
            } else {
                constexpr size_t cut = (end + begin) / 2;
                if (id < cut) {
                    return Visitor<Ret, Func, PtrType>::template Visit<begin, cut, Args...>(id, ptr, std::forward<Func>(f));
                } else {
                    return Visitor<Ret, Func, PtrType>::template Visit<cut, end, Args...>(id, ptr, std::forward<Func>(f));
                }
            }
        }
        template<size_t begin, size_t end, typename... Args>
        static Ret MultiVisit(
            size_t id,
            PtrType *ptr,
            Func &&f) {
            if constexpr (end - begin == 0) {
                using ArgType = Types<begin, Args...>;
                using ArgPureType = std::remove_reference_t<ArgType>;
                return f.template Run<begin, ArgType>(std::forward<ArgType>(*reinterpret_cast<ArgPureType *>(ptr)));
            } else if constexpr (end - begin == 1) {
                if (id == begin) {
                    using ArgType = Types<begin, Args...>;
                    using ArgPureType = std::remove_reference_t<ArgType>;
                    return f.template Run<begin, ArgType>(std::forward<ArgType>(*reinterpret_cast<ArgPureType *>(ptr)));
                } else {
                    using ArgType = Types<end, Args...>;
                    using ArgPureType = std::remove_reference_t<ArgType>;
                    return f.template Run<end, ArgType>(std::forward<ArgType>(*reinterpret_cast<ArgPureType *>(ptr)));
                }
            } else {
                constexpr size_t cut = (end + begin) / 2;
                if (id < cut) {
                    return Visitor<Ret, Func, PtrType>::template MultiVisit<begin, cut, Args...>(id, ptr, std::forward<Func>(f));
                } else {
                    return Visitor<Ret, Func, PtrType>::template MultiVisit<cut, end, Args...>(id, ptr, std::forward<Func>(f));
                }
            }
        }
    };
    template<typename... Args>
    struct Constructor {
        template<typename A>
        static size_t CopyOrMoveConst(void *, size_t idx, A &&) {
            return idx;
        }
        template<typename... A>
        static size_t AnyConst(void *, size_t idx, A &&...) {
            static_assert(AlwaysFalse<A...>, "Illegal Constructor!");
            return idx;
        }
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

    std::aligned_storage_t<(MaxSize<0, sizeof(AA)...>::MAX_SIZE), (MaxSize<0, alignof(AA)...>::MAX_SIZE)> placeHolder;
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

    void *GetPlaceHolder() { return &placeHolder; }
    void const *GetPlaceHolder() const { return &placeHolder; }
    size_t GetType() const { return switcher; }
    size_t index() const { return switcher; }
    template<typename T>
    bool IsTypeOf() const {
        return switcher == IndexOf<T>;
    }

    template<size_t i>
    decltype(auto) get() & {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return Constructor<AA...>::template Get<i>(&placeHolder);
    }
    template<size_t i>
    decltype(auto) get() && {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(Constructor<AA...>::template Get<i>(&placeHolder));
    }
    template<size_t i>
    decltype(auto) get() const & {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return Constructor<AA...>::template Get<i>(&placeHolder);
    }

    template<typename T>
    T const *try_get() const & {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return nullptr;
        }
        return &Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }

    template<typename T>
    T *try_get() & {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return nullptr;
        }
        return &Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
    optional<T> try_get() && {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return {};
        }
        return optional<T>(std::move(Constructor<AA...>::template Get<tarIdx>(&placeHolder)));
    }
    template<typename T>
    T get_or(T &&value) const & {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
    T get_or(T &&value) && {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return std::move(Constructor<AA...>::template Get<tarIdx>(&placeHolder));
    }
    template<typename T>
    T const &force_get() const & {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }

    template<typename T>
    T &force_get() & {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return Constructor<AA...>::template Get<tarIdx>(&placeHolder);
    }
    template<typename T>
    T &&force_get() && {
        static constexpr auto tarIdx = IndexOf<T>;
        static_assert(tarIdx < argSize, "Illegal target type!");
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(Constructor<AA...>::template Get<tarIdx>(&placeHolder));
    }
    template<typename Arg>
    variant &operator=(Arg &&arg) {
        this->~variant();
        new (this) variant(std::forward<Arg>(arg));
        return *this;
    }
    template<typename Func>
    void visit(Func &&func) & {
        if (switcher >= argSize) return;
        Visitor<void, Func, void>::template Visit<0, argSize - 1, AA &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) && {
        if (switcher >= argSize) return;
        Visitor<void, Func, void>::template Visit<0, argSize - 1, AA...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) const & {
        if (switcher >= argSize) return;
        Visitor<void, Func, void const>::template Visit<0, argSize - 1, AA const &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }

    template<typename... Funcs>
    void multi_visit(Funcs &&...funcs) & {
        static_assert(sizeof...(Funcs) == argSize, "Functor size incorrect!");
        if (switcher >= argSize) return;
        Visitor<void, PackedFunctors<Funcs...>, void>::template MultiVisit<0, argSize - 1, AA &...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
    void multi_visit(Funcs &&...funcs) && {
        static_assert(sizeof...(Funcs) == argSize, "Functor size incorrect!");
        if (switcher >= argSize) return;
        Visitor<void, PackedFunctors<Funcs...>, void>::template MultiVisit<0, argSize - 1, AA...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
    void multi_visit(Funcs &&...funcs) const & {
        static_assert(sizeof...(Funcs) == argSize, "Functor size incorrect!");
        if (switcher >= argSize) return;
        Visitor<void, PackedFunctors<Funcs...>, void const>::template MultiVisit<0, argSize - 1, AA const &...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename Ret, typename... Funcs>
    std::remove_cvref_t<Ret> multi_visit_or(Ret &&r, Funcs &&...funcs) & {
        static_assert(sizeof...(Funcs) == argSize, "Functor size incorrect!");
        if (switcher >= argSize) return std::forward<Ret>(r);
        return Visitor<std::remove_cvref_t<Ret>, PackedFunctors<Funcs...>, void>::template MultiVisit<0, argSize - 1, AA &...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename Ret, typename... Funcs>
    std::remove_cvref_t<Ret> multi_visit_or(Ret &&r, Funcs &&...funcs) && {
        static_assert(sizeof...(Funcs) == argSize, "Functor size incorrect!");
        if (switcher >= argSize) return std::forward<Ret>(r);
        return Visitor<std::remove_cvref_t<Ret>, PackedFunctors<Funcs...>, void>::template MultiVisit<0, argSize - 1, AA...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename Ret, typename... Funcs>
    std::remove_cvref_t<Ret> multi_visit_or(Ret &&r, Funcs &&...funcs) const & {
        static_assert(sizeof...(Funcs) == argSize, "Functor size incorrect!");
        if (switcher >= argSize) return std::forward<Ret>(r);
        return Visitor<std::remove_cvref_t<Ret>, PackedFunctors<Funcs...>, void const>::template MultiVisit<0, argSize - 1, AA const &...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }

    template<typename Ret, typename Func>
    std::remove_cvref_t<Ret> visit_or(Ret &&r, Func &&func) & {
        if (switcher >= argSize) return std::forward<Ret>(r);
        return Visitor<std::remove_cvref_t<Ret>, Func, void>::template Visit<0, argSize - 1, AA &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Ret, typename Func>
    std::remove_cvref_t<Ret> visit_or(Ret &&r, Func &&func) && {
        if (switcher >= argSize) return std::forward<Ret>(r);
        return Visitor<std::remove_cvref_t<Ret>, Func, void>::template Visit<0, argSize - 1, AA...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Ret, typename Func>
    std::remove_cvref_t<Ret> visit_or(Ret &&r, Func &&func) const & {
        if (switcher >= argSize) return std::forward<Ret>(r);
        return Visitor<std::remove_cvref_t<Ret>, Func, void const>::template Visit<0, argSize - 1, AA const &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    void dispose() {
        auto disposeFunc = [&]<typename T>(T &value) {
            value.~T();
        };
        visit(disposeFunc);
        switcher = argSize;
    }
    variant() {
        switcher = argSize;
    }
    template<typename T, typename... Arg>
    variant(T &&t, Arg &&...arg) {
        if constexpr (sizeof...(Arg) == 0) {
            switcher = Constructor<AA...>::template CopyOrMoveConst<T>(&placeHolder, 0, std::forward<T>(t));
            if (switcher < argSize) return;
        }
        switcher = Constructor<AA...>::template AnyConst<T, Arg...>(&placeHolder, 0, std::forward<T>(t), std::forward<Arg>(arg)...);
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
    variant(variant &v)
        : variant(static_cast<variant const &>(v)) {
    }
    variant(variant const &&v)
        : variant(v) {
    }
    ~variant() {
        dispose();
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
                    return comp(v, b.force_get<TT>());
                });
        } else
            return (a.GetType() > b.GetType()) ? 1 : -1;
    }
};
}// namespace vstd