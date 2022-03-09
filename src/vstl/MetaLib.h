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
#include <core/stl.h>
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
#define VE_SUB_TEMPLATE template<typename...> \
class
namespace vstd {

template<typename T>
struct TypeOf {
    using Type = T;
};

template<typename T>
struct funcPtr;
template<typename Ret, typename... Args>
struct funcPtr<Ret(Args...)> {
    using Type = Ret (*)(Args...);
};
template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;

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
namespace detail {
class SBOInterface {
public:
    virtual void Copy(void *dstPtr, void *srcPtr) const = 0;
    virtual void Move(void *dstPtr, void *srcPtr) const = 0;
    virtual void *GetOffset(void *src) const = 0;
    virtual void *GetFallback(void *src) const = 0;
    virtual void *Malloc(void *src) const = 0;
};
}// namespace detail
template<typename I, size_t sboSize = 48, size_t align = sizeof(size_t)>
class SBO {
    std::aligned_storage_t<sboSize, align> buffer;
    I *ptr;
    //void(Src, Dest)
    using InterfaceStorage = std::aligned_storage_t<sizeof(detail::SBOInterface), alignof(detail::SBOInterface)>;
    InterfaceStorage storage;
    detail::SBOInterface const *GetInterface() const {
        return reinterpret_cast<detail::SBOInterface const *>(&storage);
    }

public:
    template<typename Func>
    static consteval bool CtorFunc() {
        if constexpr (!std::is_invocable_v<Func, void *>) {
            return false;
        } else {
            using T = std::invoke_result_t<Func, void *>;
            return std::is_pointer_v<T> && std::is_base_of_v<I, std::remove_pointer_t<T>>;
        }
    }
    I *operator->() const {
        return ptr;
    }
    I &operator*() const { return *ptr; }
    I *Get() const {
        return ptr;
    }

    template<typename Func>
        requires(CtorFunc<Func>())
    SBO(Func &&func) {
        using T = std::remove_pointer_t<std::invoke_result_t<Func, void *>>;
        constexpr size_t sz = sizeof(T);
        void *originPtr;
        if constexpr (sz > sboSize) {
            originPtr = vengine_malloc(sz);
        } else {
            originPtr = &buffer;
        }
        func(originPtr);
        ptr = reinterpret_cast<T *>(originPtr);
        class Inter : public detail::SBOInterface {
        public:
            void Copy(void *dstPtr, void *srcPtr) const override {
                if constexpr (std::is_copy_constructible_v<T>) {
                    new (dstPtr) T(*reinterpret_cast<T const *>(srcPtr));
                } else {
                    VEngine_Log(typeid(T));
                }
            }
            void Move(void *dstPtr, void *srcPtr) const override {
                if constexpr (std::is_move_constructible_v<T>) {
                    new (dstPtr) T(std::move(*reinterpret_cast<T *>(srcPtr)));
                } else {
                    VEngine_Log(typeid(T));
                }
            }
            void *GetFallback(void *src) const override {
                T *ptr = reinterpret_cast<T *>(src);
                I *offsetPtr = static_cast<I *>(ptr);
                return offsetPtr;
            }
            void *GetOffset(void *src) const override {
                I *ptr = reinterpret_cast<I *>(src);
                T *offsetPtr = static_cast<T *>(ptr);
                return offsetPtr;
            }
            void *Malloc(void *src) const override {
                void *originPtr = vengine_malloc(sz);
                Copy(originPtr, src);
                return static_cast<I *>(reinterpret_cast<T *>(originPtr));
            }
        };
        new (&storage) Inter();
    }
    SBO(SBO &&sbo)
        : storage(sbo.storage) {
        auto it = GetInterface();
        auto sboOriginPtr = it->GetOffset(sbo.ptr);
        if (sboOriginPtr == &sbo.buffer) {
            auto originPtr = &buffer;
            it->Move(originPtr, sboOriginPtr);
            ptr = reinterpret_cast<I *>(it->GetFallback(originPtr));
        } else {
            ptr = sbo.ptr;
        }
        sbo.ptr = nullptr;
    }
    SBO(SBO const &sbo)
        : storage(sbo.storage) {
        auto it = GetInterface();
        auto sboOriginPtr = it->GetOffset(sbo.ptr);
        if (sboOriginPtr == &sbo.buffer) {
            auto originPtr = &buffer;
            it->Copy(originPtr, sboOriginPtr);
            ptr = reinterpret_cast<I *>(it->GetFallback(originPtr));
        } else {
            ptr = reinterpret_cast<I *>(it->Malloc(sboOriginPtr));
        }
    }
    ~SBO() {
        if (!ptr) return;
        ptr->~I();
        auto originPtr = GetInterface()->GetOffset(ptr);
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
    IEnumerable<T> *Get() const { return ptr.Get(); }
    template<typename Func>
        requires(PtrType::template CtorFunc<Func>())
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
    void operator++(int) const {
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
template<VE_SUB_TEMPLATE map, bool reverse, typename... Tar>
struct AnyMap {
    template<typename T, typename... Args>
    static consteval bool Run() {
        if constexpr ((map<T, Tar...>::value) ^ reverse) {
            return true;
        } else if constexpr (sizeof...(Args) == 0) {
            return false;
        } else {
            return Run<Args...>();
        }
    }
};
template<size_t... size>
consteval size_t max_size() {
    auto sizes = {size...};
    size_t v = 0;
    for (auto i : sizes) {
        if (v < i) v = i;
    }
    return v;
}
template<typename T, typename... Args>
struct MapConstructible {
    static constexpr bool value = std::is_constructible_v<T, Args...>;
};

template<typename... Args>
struct MapConstructible<void, Args...> {
    static constexpr bool value = (sizeof...(Args) == 0);
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
template<typename... Args>
struct VariantConstructible {
    template<size_t i, typename T, typename... Ts>
    static consteval size_t Run() {
        if constexpr (std::is_constructible_v<T, Args...>) {
            return i;
        } else {
            if constexpr (sizeof...(Ts) == 0) {
                return i + 1;
            } else {
                return Run<i + 1, Ts...>();
            }
        }
    }
    template<size_t i, typename... Ts>
    static constexpr size_t value = Run<i, Ts...>();
};
template<typename Ret, typename Func, typename VoidPtr, typename... Types>
static Ret Visitor(
    size_t id,
    VoidPtr *ptr,
    Func &&func) {
    constexpr static auto table =
        {
            FuncTable<
                Ret,
                std::remove_reference_t<Func>,
                Types>...};
    return table.begin()[id](ptr, &func);
}
template<size_t idx, VE_SUB_TEMPLATE Judger, typename Tar, typename T, typename... Args>
static consteval size_t IndexOfFunc() {
    if constexpr (Judger<T, Tar>::value) {
        return idx;
    } else {
        if constexpr (sizeof...(Args) == 0) {
            return idx + 1;
        } else {
            return IndexOfFunc<idx + 1, Judger, Tar, Args...>();
        }
    }
}
template<size_t i, typename T, typename... Args>
static consteval decltype(auto) TypeOfFunc() {
    if constexpr (i == 0) {
        return TypeOf<T>{};
    } else if constexpr (sizeof...(Args) == 0) {
        return TypeOf<void>{};
    } else {
        return TypeOfFunc<i - 1, Args...>();
    }
}
}// namespace detail
class Evaluable {};
template<class Func>
class LazyEval : public Evaluable {
private:
    Func func;

public:
    using EvalType = decltype(std::declval<Func>()());
    template<typename TT>
    LazyEval(TT &&func)
        : func(std::forward<TT>(func)) {}
    LazyEval(LazyEval const &) = delete;
    LazyEval(LazyEval &&v)
        : func(std::move(v.func)) {}
    operator decltype(auto)() const {
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
    static constexpr size_t IndexOf =
        detail::IndexOfFunc<
            0,
            std::is_same,
            std::remove_cvref_t<TarT>, AA...>();
    template<typename TarT>
    static constexpr size_t AssignableOf =
        detail::IndexOfFunc<
            0,
            std::is_assignable,
            std::remove_cvref_t<TarT>, AA...>();
    template<size_t i>
    using TypeOf = typename decltype(detail::TypeOfFunc<i, AA...>())::Type;

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

    std::aligned_storage_t<(detail::max_size<sizeof(AA)...>()), (detail::max_size<alignof(AA)...>())> placeHolder;
    size_t switcher = 0;
    void m_dispose() {
        if constexpr (detail::AnyMap<std::is_trivially_destructible, true>::template Run<AA...>()) {
            auto disposeFunc = [&]<typename T>(T &value) {
                value.~T();
            };
            visit(disposeFunc);
        }
    }

public:
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
        return switcher == (IndexOf<T>);
    }

    template<size_t i>
        requires(i < argSize)
    decltype(auto) get() & {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *reinterpret_cast<TypeOf<i> *>(&placeHolder);
    }
    template<size_t i>
        requires(i < argSize)
    decltype(auto) get() && {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(*reinterpret_cast<TypeOf<i> *>(&placeHolder));
    }
    template<size_t i>
        requires(i < argSize)
    decltype(auto) get() const & {
#ifdef DEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *reinterpret_cast<TypeOf<i> const *>(&placeHolder);
    }

    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T const *try_get() const & {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return nullptr;
        }
        return reinterpret_cast<TypeOf<tarIdx> const *>(&placeHolder);
    }

    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T *try_get() & {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return nullptr;
        }
        return reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder);
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    optional<T> try_get() && {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return {};
        }
        return optional<T>(std::move(*reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder)));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T get_or(T &&value)
    const & {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return *reinterpret_cast<TypeOf<tarIdx> const *>(&placeHolder);
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T get_or(T &&value) && {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return std::move(*reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T const &force_get() const & {
        static constexpr auto tarIdx = (IndexOf<T>);
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *reinterpret_cast<TypeOf<tarIdx> const *>(&placeHolder);
    }

    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T &force_get() & {
        static constexpr auto tarIdx = (IndexOf<T>);
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder);
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T && force_get() && {
        static constexpr auto tarIdx = (IndexOf<T>);
#ifdef DEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(*reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder));
    }
    template<typename Func>
    void visit(Func &&func) & {
        if (switcher >= argSize) return;
        detail::Visitor<void, Func, void, AA &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) && {
        if (switcher >= argSize) return;
        detail::Visitor<void, Func, void, AA...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) const & {
        if (switcher >= argSize) return;
        detail::Visitor<void, Func, void const, AA const &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
    }

    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) & {
        if (switcher >= argSize) return;
        detail::Visitor<void, PackedFunctors<Funcs...>, void, AA &...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) && {
        if (switcher >= argSize) return;
        detail::Visitor<void, PackedFunctors<Funcs...>, void, AA...>(
            switcher,
            GetPlaceHolder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) const & {
        if (switcher >= argSize) return;
        detail::Visitor<void, PackedFunctors<Funcs...>, void const, AA const &...>(
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
            return detail::Visitor<EvalType, PackedFunctors<Funcs...>, void, AA &...>(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::Visitor<Ret, PackedFunctors<Funcs...>, void, AA &...>(
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
            return detail::Visitor<EvalType, PackedFunctors<Funcs...>, void, AA...>(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::Visitor<Ret, PackedFunctors<Funcs...>, void, AA...>(
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
            return detail::Visitor<EvalType, PackedFunctors<Funcs...>, void const, AA const &...>(
                switcher,
                GetPlaceHolder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::Visitor<Ret, PackedFunctors<Funcs...>, void const, AA const &...>(
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
            return detail::Visitor<EvalType, Func, void, AA &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::Visitor<Ret, Func, void, AA &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return detail::Visitor<EvalType, Func, void, AA...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::Visitor<Ret, Func, void, AA...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return EvalType{std::forward<Ret>(r)};
            return detail::Visitor<EvalType, Func, void const, AA const &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::Visitor<Ret, Func, void const, AA const &...>(switcher, GetPlaceHolder(), std::forward<Func>(func));
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
        requires(
            detail::AnyMap<std::is_constructible, false, T &&, Arg &&...>::
                template Run<AA...>())
    variant(T &&t, Arg &&...arg) {
        using PureT = std::remove_cvref_t<T>;
        constexpr size_t tIdx = IndexOf<PureT>;
        if constexpr ((sizeof...(Arg)) == 0 && ((tIdx) < argSize)) {
            switcher = (tIdx);
            new (&placeHolder) PureT(std::forward<T>(t));
        } else {
            constexpr size_t typeIdx = detail::VariantConstructible<T &&, Arg &&...>::template value<0, AA...>;
            switcher = typeIdx;
            new (&placeHolder) TypeOf<typeIdx>(std::forward<T>(t), std::forward<Arg>(arg)...);
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
        requires(detail::AnyMap<std::is_constructible, false, Args &&...>::
                     template Run<AA...>())
    void reset(Args &&...args) {
        this->~variant();
        new (this) variant(std::forward<Args>(args)...);
    }
    template<typename T>
        requires(detail::AnyMap<std::is_assignable, false, T>::
                     template Run<AA...>())
    variant &
    operator=(T &&t) {
        using PureT = std::remove_cvref_t<T>;
        constexpr size_t idxOfT = IndexOf<PureT>;
        if constexpr ((idxOfT) < argSize) {
            if (switcher == (idxOfT)) {
                *reinterpret_cast<PureT *>(&placeHolder) = std::forward<T>(t);
            } else {
                m_dispose();
                new (&placeHolder) PureT(std::forward<T>(t));
                switcher = (idxOfT);
            }
        } else {
            constexpr size_t asignOfT = AssignableOf<T &&>;
            static_assert((asignOfT) < argSize, "illegal type");
            using CurT = TypeOf<(asignOfT)>;
            if (switcher == (asignOfT)) {
                *reinterpret_cast<CurT *>(&placeHolder) = std::forward<T>(t);
            } else {
                m_dispose();
                new (&placeHolder) CurT(std::forward<T>(t));
                switcher = (asignOfT);
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
                    return comp(v, b.template force_get<PureT>());
                });
        } else
            return (a.GetType() > b.GetType()) ? 1 : -1;
    }
};
template<typename Vec, typename Func>
void push_back_func(Vec &&vec, Func &&func, size_t count) {
    if constexpr (std::is_invocable_v<Func>) {
        std::fill_n(
            std::back_inserter(vec),
            count,
            LazyEval<std::remove_cvref_t<Func>>(std::forward<Func>(func)));
    } else if constexpr (std::is_invocable_v<Func, size_t>) {
        size_t i = 0;
        std::fill_n(
            std::back_inserter(vec),
            count,
            MakeLazyEval([&] {
                auto d = create_disposer([&] { ++i; });
                return func(i);
            }));
    } else {
        static_assert(AlwaysFalse<Vec>, "illegal type");
    }
}
template<typename Vec>
auto erase_last(Vec &&vec) {
    auto lastIte = vec.end() - 1;
    auto disp = create_disposer([&] { vec.erase(lastIte); });
    return std::move(*lastIte);
}
#define VSTD_TRIVIAL_COMPARABLE(T)               \
    bool operator==(T const &a) const {          \
        return memcmp(this, &a, sizeof(T)) == 0; \
    }                                            \
    bool operator!=(T const &a) const {          \
        return memcmp(this, &a, sizeof(T)) != 0; \
    }                                            \
    bool operator>(T const &a) const {           \
        return memcmp(this, &a, sizeof(T)) > 0;  \
    }                                            \
    bool operator<(T const &a) const {           \
        return memcmp(this, &a, sizeof(T)) < 0;  \
    }

}// namespace vstd
