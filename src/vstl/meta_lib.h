#pragma once
#include <vstl/config.h>
#include <type_traits>
#include <stdint.h>

#include <typeinfo>
#include <new>
#include <vstl/hash.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <tuple>
#include <utility>
#include <vstl/allocate_type.h>
#include <vstl/compare.h>
#include <assert.h>
namespace luisa::detail {
LUISA_EXPORT_API void *allocator_allocate(size_t size, size_t alignment) noexcept;
LUISA_EXPORT_API void allocator_deallocate(void *p, size_t alignment) noexcept;
LUISA_EXPORT_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept;
}// namespace luisa::detail
inline void *vengine_malloc(size_t size) {
    return luisa::detail::allocator_allocate(size, 0);
}
inline void vengine_free(void *ptr) {
    luisa::detail::allocator_deallocate(ptr, 0);
}
inline void *vengine_realloc(void *ptr, size_t size) {
    return luisa::detail::allocator_reallocate(ptr, size, 0);
}
LC_VSTL_API void VEngine_Log(std::type_info const &t);
LC_VSTL_API void VEngine_Log(char const *chunk);
#define VE_SUB_TEMPLATE template<typename...> \
class
namespace vstd {
template<typename T, typename... Args>
    requires(!std::is_const_v<T> && std::is_constructible_v<T, Args && ...>)
void reset(T &v, Args &&...args) {
    v.~T();
    new (std::launder(&v)) T(std::forward<Args>(args)...);
}
template<typename T>
void destruct(T *ptr) {
    if constexpr (!std::is_void_v<T> && !std::is_trivially_destructible_v<T>)
        ptr->~T();
}
template<typename T>
struct TypeOf {
    using Type = T;
};

template<typename T>
struct func_ptr;
template<typename Ret, typename... Args>
struct func_ptr<Ret(Args...)> {
    using Type = Ret (*)(Args...);
};
template<typename T>
using func_ptr_t = typename func_ptr<T>::Type;

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
        requires(std::is_constructible_v<T, Args && ...>)
    inline SelfType &create(Args &&...args) &noexcept {
        new (storage) T(std::forward<Args>(args)...);
        return *this;
    }
    template<typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    inline SelfType &&create(Args &&...args) &&noexcept {
        return std::move(create(std::forward<Args>(args)...));
    }
    inline void destroy() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>)
            vstd::destruct(std::launder(reinterpret_cast<T *>(storage)));
    }
    T &operator*() &noexcept {
        return *std::launder(reinterpret_cast<T *>(storage));
    }
    T &&operator*() &&noexcept {
        return std::move(*std::launder(reinterpret_cast<T *>(storage)));
    }
    T const &operator*() const &noexcept {
        return *reinterpret_cast<T const *>(storage);
    }
    T *operator->() noexcept {
        return std::launder(reinterpret_cast<T *>(storage));
    }
    T const *operator->() const noexcept {
        return reinterpret_cast<T const *>(storage);
    }
    T *ptr() noexcept {
        return std::launder(reinterpret_cast<T *>(storage));
    }
    T const *ptr() const noexcept {
        return reinterpret_cast<T const *>(storage);
    }
    operator T *() noexcept {
        return std::launder(reinterpret_cast<T *>(storage));
    }
    operator T const *() const noexcept {
        return reinterpret_cast<T const *>(storage);
    }
    StackObject() noexcept {}
    StackObject(const SelfType &value) {
        if constexpr (std::is_copy_constructible_v<T>) {
            new (storage) T(*value);
        } else {
            assert(false);
            VENGINE_EXIT;
        }
    }
    StackObject(SelfType &&value) {
        if constexpr (std::is_move_constructible_v<T>) {
            new (storage) T(std::move(*value));
        } else {
            assert(false);
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
            destroy();
            create(*value);
        } else {
            assert(false);
            VENGINE_EXIT;
        }
        return **this;
    }
    T &operator=(SelfType &&value) {
        if constexpr (std::is_move_assignable_v<T>) {
            operator*() = std::move(*value);
        } else if constexpr (std::is_move_constructible_v<T>) {
            destroy();
            create(std::move(*value));
        } else {
            assert(false);
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
    bool mInitialized;

public:
    using SelfType = StackObject<T, true>;
    template<typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    inline SelfType &create(Args &&...args) &noexcept {
        if (mInitialized) return *this;
        mInitialized = true;
        stackObj.create(std::forward<Args>(args)...);
        return *this;
    }

    template<typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    inline SelfType &&create(Args &&...args) &&noexcept {
        return std::move(create(std::forward<Args>(args)...));
    }

    template<typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    inline SelfType &force_create(Args &&...args) &noexcept {
        if (mInitialized) { destroy(); }
        mInitialized = true;
        stackObj.create(std::forward<Args>(args)...);
        return *this;
    }

    template<typename... Args>
    inline SelfType &&force_create(Args &&...args) &&noexcept {
        return std::move(force_create(std::forward<Args>(args)...));
    }

    bool has_value() const noexcept {
        return mInitialized;
    }

    bool initialized() const noexcept {
        return mInitialized;
    }
    operator bool() const noexcept {
        return mInitialized;
    }
    operator bool() noexcept {
        return mInitialized;
    }
    inline bool destroy() noexcept {
        if (!mInitialized) return false;
        mInitialized = false;
        stackObj.destroy();
        return true;
    }
    void reset() const noexcept {
        destroy();
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
        if (mInitialized)
            return *stackObj;
        else
            return std::forward<U>(default_value);
    }
    template<class U>
    T value_or(U &&default_value) && {
        if (mInitialized)
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
    T *ptr() noexcept {
        return stackObj.ptr();
    }
    T const *ptr() const noexcept {
        return stackObj.ptr();
    }
    operator T *() noexcept {
        return stackObj;
    }
    operator T const *() const noexcept {
        return stackObj;
    }
    StackObject() noexcept {
        mInitialized = false;
    }
    template<typename... Args>
    StackObject(Args &&...args)
        : stackObj(std::forward<Args>(args)...),
          mInitialized(true) {
    }
    StackObject(const SelfType &value) noexcept {
        mInitialized = value.mInitialized;
        if (mInitialized) {
            if constexpr (std::is_copy_constructible_v<T>) {
                stackObj.create(*value);
            } else {
                assert(false);
                VENGINE_EXIT;
            }
        }
    }
    StackObject(SelfType &&value) noexcept {
        mInitialized = value.mInitialized;
        if (mInitialized) {
            if constexpr (std::is_move_constructible_v<T>) {
                stackObj.create(std::move(*value));
            } else {
                assert(false);
                VENGINE_EXIT;
            }
        }
    }
    ~StackObject() noexcept {
        if (mInitialized)
            stackObj.destroy();
    }
    T &operator=(SelfType const &value) {
        if (!mInitialized) {
            if (value.mInitialized) {
                if constexpr (std::is_copy_constructible_v<T>) {
                    stackObj.create(*value);
                } else {
                    assert(false);
                    VENGINE_EXIT;
                }
                mInitialized = true;
            }
        } else {
            if (value.mInitialized) {
                stackObj = value.stackObj;
            } else {
                stackObj.destroy();
                mInitialized = false;
            }
        }
        return *stackObj;
    }
    T &operator=(SelfType &&value) {
        if (!mInitialized) {
            if (value.mInitialized) {
                if constexpr (std::is_move_constructible_v<T>) {
                    stackObj.create(std::move(*value));
                } else {
                    assert(false);
                    VENGINE_EXIT;
                }
                mInitialized = true;
            }
        } else {
            if (value.mInitialized) {
                stackObj = std::move(value.stackObj);
            } else {
                stackObj.destroy();
                mInitialized = false;
            }
        }
        return *stackObj;
    }
    template<typename Arg>
        requires(std::is_assignable_v<StackObject<T, false>, Arg &&>)
    T &
    operator=(Arg &&value) {
        if (mInitialized) {
            return stackObj = std::forward<Arg>(value);
        } else {
            create(std::forward<Arg>(value));
            return **this;
        }
    }
};
//Declare Tuple
template<typename T>
using optional = StackObject<T, true>;

template<typename T>
using PureType_t = std::remove_pointer_t<std::remove_cvref_t<T>>;

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
    eastl::aligned_storage_t<sboSize, align> buffer;
    I *ptr;
    //void(Src, Dest)
    using InterfaceStorage = eastl::aligned_storage_t<sizeof(detail::SBOInterface), alignof(detail::SBOInterface)>;
    InterfaceStorage storage;
    detail::SBOInterface const *GetInterface() const {
        return reinterpret_cast<detail::SBOInterface const *>(&storage);
    }

public:
    template<typename Func>
    static constexpr bool CtorFunc() {
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
        ptr = std::launder(reinterpret_cast<T *>(originPtr));
        class Inter : public detail::SBOInterface {
        public:
            void Copy(void *dstPtr, void *srcPtr) const override {
                if constexpr (std::is_copy_constructible_v<T>) {
                    new (dstPtr) T(*reinterpret_cast<T const *>(srcPtr));
                } else {
                    assert(false);
                }
            }
            void Move(void *dstPtr, void *srcPtr) const override {
                if constexpr (std::is_move_constructible_v<T>) {
                    new (dstPtr) T(std::move(*reinterpret_cast<T *>(srcPtr)));
                } else {
                    assert(false);
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
    Iterator(Iterator &&v) : ptr(std::move(v.ptr)) {}
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
disposer<T> scope_exit(T &&t) {
    return disposer<T>(std::forward<T>(t));
}

template<typename T>
decltype(auto) get_lvalue(T &&data) {
    return static_cast<std::remove_reference_t<T> &>(data);
}
template<typename T>
T *get_rval_ptr(T &&v) {
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
    static constexpr bool Run() {
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
constexpr size_t max_size() {
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
template<typename T>
static constexpr decltype(auto) GetVoidType() {
    if constexpr (std::is_const_v<T>) {
        return TypeOf<void const>{};
    } else {
        return TypeOf<void>{};
    }
}
template<typename T>
using GetVoidType_t = typename decltype(GetVoidType<std::remove_reference_t<T>>())::Type;
template<typename Func, typename PtrType>
constexpr static void FuncTable(GetVoidType_t<PtrType> *ptr, GetVoidType_t<Func> *func) {
    if constexpr (std::is_invocable_v<Func, PtrType &&>) {
        using PureFunc = std::remove_cvref_t<Func>;
        PureFunc *realFunc = reinterpret_cast<PureFunc *>(func);
        (std::forward<Func>(*realFunc))(std::forward<PtrType>(*reinterpret_cast<std::remove_reference_t<PtrType> *>(ptr)));
    }
}
template<typename Ret, typename Eval, typename Func, typename PtrType>
constexpr static Ret FuncTableRet(GetVoidType_t<PtrType> *ptr, GetVoidType_t<Func> *func, Eval &&ret) {
    if constexpr (std::is_invocable_v<Func, PtrType &&>) {
        using PureFunc = std::remove_cvref_t<Func>;
        PureFunc *realFunc = reinterpret_cast<PureFunc *>(func);
        return (std::forward<Func>(*realFunc))(std::forward<PtrType>(*reinterpret_cast<std::remove_reference_t<PtrType> *>(ptr)));
    } else {
        return std::forward<Eval>(ret);
    }
}
template<typename... Args>
struct VariantConstructible {
    template<size_t i, typename T, typename... Ts>
    static constexpr size_t Run() {
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
template<typename Func, typename VoidPtr, typename... Types>
static void Visitor(
    size_t id,
    VoidPtr *ptr,
    Func &&func) {
    constexpr static auto table =
        {
            FuncTable<
                std::remove_reference_t<Func>,
                Types>...};
    table.begin()[id](ptr, &func);
}
template<typename Ret, typename Eval, typename Func, typename VoidPtr, typename... Types>
static Ret VisitorRet(
    size_t id,
    VoidPtr *ptr,
    Func &&func,
    Eval &&ret) {
    constexpr static auto table =
        {
            FuncTableRet<
                Ret,
                Eval,
                std::remove_reference_t<Func>,
                Types>...};
    return table.begin()[id](ptr, &func, std::forward<Eval>(ret));
}
template<size_t idx, VE_SUB_TEMPLATE Judger, typename Tar, typename T, typename... Args>
static constexpr size_t IndexOfFunc() {
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
static constexpr decltype(auto) TypeOfFunc() {
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
class v_lazy_eval : public Evaluable {
private:
    Func func;

public:
    using EvalType = decltype(std::declval<Func>()());
    v_lazy_eval(Func &&func)
        : func(std::forward<Func &&>(func)) {}
    v_lazy_eval(v_lazy_eval const &) = delete;
    v_lazy_eval(v_lazy_eval &&v)
        : func(std::move(v.func)) {}
    operator EvalType() const {
        return func();
    }
};
template<typename T>
class UndefEval : public Evaluable {
public:
    using EvalType = T;
    operator T() const {
        assert(false);
        return std::move(*reinterpret_cast<T *>(0));
    }
};

template<class Func>
v_lazy_eval<Func> lazy_eval(Func &&func) {
    return std::forward<Func>(func);
}
template<typename... Args>
static constexpr bool AlwaysFalse = false;
template<typename... Args>
static constexpr bool AlwaysTrue = true;
template<typename T>
T &decl_lvalue(T &&t) { return t; }
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

    eastl::aligned_storage_t<(detail::max_size<sizeof(AA)...>()), (detail::max_size<alignof(AA)...>())> placeHolder;
    size_t switcher = 0;
    void m_dispose() {
        if constexpr (detail::AnyMap<std::is_trivially_destructible, true>::template Run<AA...>()) {
            auto disposeFunc = [&]<typename T>(T &value) {
                vstd::destruct(&value);
            };
            visit(disposeFunc);
        }
    }

public:
    bool valid() const { return switcher < argSize; }

    template<typename... Args>
    void reset_as(size_t typeIndex, Args &&...args) {
        this->~variant();
        if (typeIndex >= argSize) {
            switcher = argSize;
            return;
        }
        switcher = typeIndex;
        auto func = [&]<typename T>(T &t) {
            constexpr bool cons = std::is_constructible_v<T, Args &&...>;
            assert(cons);
            if constexpr (cons)
                new (&t) T(std::forward<Args>(args)...);
        };
        detail::Visitor<decltype(func), void, AA &...>(typeIndex, place_holder(), std::move(func));
    }
    template<typename T, typename... Args>
        requires(
            IndexOf<T> < argSize && std::is_constructible_v<T, Args && ...>)
    void reset_as(Args &&...args) {
        this->~variant();
        switcher = IndexOf<T>;
        new (&placeHolder) T(std::forward<Args>(args)...);
    }

    void *place_holder() { return &placeHolder; }
    void const *place_holder() const { return &placeHolder; }
    size_t index() const { return switcher; }
    template<typename T>
    bool is_type_of() const {
        return switcher == (IndexOf<T>);
    }

    template<size_t i>
        requires(i < argSize) decltype(auto)
    get() & {
#ifndef NDEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *std::launder(reinterpret_cast<TypeOf<i> *>(&placeHolder));
    }
    template<size_t i>
        requires(i < argSize) decltype(auto)
    get() && {
#ifndef NDEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(*std::launder(reinterpret_cast<TypeOf<i> *>(&placeHolder)));
    }
    template<size_t i>
        requires(i < argSize) decltype(auto)
    get() const & {
#ifndef NDEBUG
        if (i != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *std::launder(reinterpret_cast<TypeOf<i> const *>(&placeHolder));
    }

    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T const *try_get() const & {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return nullptr;
        }
        return std::launder(reinterpret_cast<TypeOf<tarIdx> const *>(&placeHolder));
    }

    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T *try_get() & {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return nullptr;
        }
        return std::launder(reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    optional<T> try_get() && {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return {};
        }
        return optional<T>(std::move(*std::launder(reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder))));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T get_or(T &&value)
        const & {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return *std::launder(reinterpret_cast<TypeOf<tarIdx> const *>(&placeHolder));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T get_or(T &&value) && {
        static constexpr auto tarIdx = (IndexOf<T>);
        if (tarIdx != switcher) {
            return std::forward<T>(value);
        }
        return std::move(*std::launder(reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder)));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T const &force_get() const & {
        static constexpr auto tarIdx = (IndexOf<T>);
#ifndef NDEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *std::launder(reinterpret_cast<TypeOf<tarIdx> const *>(&placeHolder));
    }

    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T &force_get() & {
        static constexpr auto tarIdx = (IndexOf<T>);
#ifndef NDEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *std::launder(reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder));
    }
    template<typename T>
        requires(detail::AnyMap<std::is_same, false, T>::template Run<AA...>())
    T &&force_get() && {
        static constexpr auto tarIdx = (IndexOf<T>);
#ifndef NDEBUG
        if (tarIdx != switcher) {
            VEngine_Log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(*std::launder(reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder)));
    }
    template<typename Func>
    void visit(Func &&func) & {
        if (switcher >= argSize) return;
        detail::Visitor<Func, void, AA &...>(switcher, place_holder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) && {
        if (switcher >= argSize) return;
        detail::Visitor<Func, void, AA...>(switcher, place_holder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) const & {
        if (switcher >= argSize) return;
        detail::Visitor<Func, void const, AA const &...>(switcher, place_holder(), std::forward<Func>(func));
    }

    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) & {
        if (switcher >= argSize) return;
        detail::Visitor<PackedFunctors<Funcs...>, void, AA &...>(
            switcher,
            place_holder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) && {
        if (switcher >= argSize) return;
        detail::Visitor<PackedFunctors<Funcs...>, void, AA...>(
            switcher,
            place_holder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) const & {
        if (switcher >= argSize) return;
        detail::Visitor<PackedFunctors<Funcs...>, void const, AA const &...>(
            switcher,
            place_holder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize) decltype(auto)
    multi_visit_or(Ret &&r, Funcs &&...funcs) & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Ret, PackedFunctors<Funcs...>, void, AA &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...),
                std::forward<Ret>(r));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Ret, PackedFunctors<Funcs...>, void, AA &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...),
                std::forward<Ret>(r));
        }
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize) decltype(auto)
    multi_visit_or(Ret &&r, Funcs &&...funcs) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Ret, PackedFunctors<Funcs...>, void, AA...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...),
                std::forward<Ret>(r));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Ret, PackedFunctors<Funcs...>, void, AA...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...),
                std::forward<Ret>(r));
        }
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize) decltype(auto)
    multi_visit_or(Ret &&r, Funcs &&...funcs) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Ret, PackedFunctors<Funcs...>, void const, AA const &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...),
                std::forward<Ret>(r));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Ret, PackedFunctors<Funcs...>, void const, AA const &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...),
                std::forward<Ret>(r));
        }
    }

    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Ret, Func, void, AA &...>(switcher, place_holder(), std::forward<Func>(func), std::forward<Ret>(r));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Ret, Func, void, AA &...>(switcher, place_holder(), std::forward<Func>(func), std::forward<Ret>(r));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Ret, Func, void, AA...>(switcher, place_holder(), std::forward<Func>(func), std::forward<Ret>(r));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Ret, Func, void, AA...>(switcher, place_holder(), std::forward<Func>(func), std::forward<Ret>(r));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Ret, Func, void const, AA const &...>(switcher, place_holder(), std::forward<Func>(func), std::forward<Ret>(r));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Ret, Func, void const, AA const &...>(switcher, place_holder(), std::forward<Func>(func), std::forward<Ret>(r));
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
            detail::AnyMap<std::is_constructible, false, T &&, Arg && ...>::
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
            new (place_holder()) T(value);
        };
        v.visit(copyFunc);
    }
    variant(variant &&v)
        : switcher(v.switcher) {
        auto moveFunc = [&]<typename T>(T &value) {
            new (place_holder()) T(std::move(value));
        };
        v.visit(moveFunc);
    }
    ~variant() {
        m_dispose();
    }
    template<typename... Args>
        requires(detail::AnyMap<std::is_constructible, false, Args && ...>::
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
                *std::launder(reinterpret_cast<PureT *>(&placeHolder)) = std::forward<T>(t);
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
                *std::launder(reinterpret_cast<CurT *>(&placeHolder)) = std::forward<T>(t);
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
                    *std::launder(reinterpret_cast<T *>(&placeHolder)) = v;
                else {
                    assert(false);
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
                    *std::launder(reinterpret_cast<T *>(&placeHolder)) = std::move(v);
                else {
                    assert(false);
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
    using type = variant<T...>;
    size_t operator()(type const &v) const {
        return v.visit_or(
            size_t(0),
            [&](auto &&v) {
                const hash<std::remove_cvref_t<decltype(v)>> hs;
                return hs(v);
            });
    }
    template<typename V>
    size_t operator()(V const &v) const {
        return hash<V>()(v);
    }
};
template<typename... T>
struct compare<variant<T...>> {
    using type = variant<T...>;
    int32 operator()(type const &a, type const &b) const {
        if (a.index() == b.index()) {
            return a.visit_or(
                int32(0),
                [&](auto &&v) {
                    using TT = decltype(v);
                    using PureT = std::remove_cvref_t<TT>;
                    const compare<PureT> comp;
                    return comp(v, b.template force_get<PureT>());
                });
        } else
            return (a.index() > b.index()) ? 1 : -1;
    }
    template<typename V>
    int32 operator()(type const &a, V const &v) {
        constexpr size_t idx = type::template IndexOf<V>;
        if (a.index() == idx) {
            return compare<V>()(a.template get<idx>(), v);
        } else
            return (a.index() > idx) ? 1 : -1;
    }
};
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
class IOperatorNewBase {
public:
    static void *operator new(
        size_t size) noexcept {
        return vengine_malloc(size);
    }
    static void *operator new(
        size_t,
        void *place) noexcept {
        return place;
    }
    static void *operator new[](
        size_t size) noexcept {
        return vengine_malloc(size);
    }
    static void *operator new(
        size_t size, const std::nothrow_t &) noexcept {
        return vengine_malloc(size);
    }
    static void *operator new(
        size_t,
        void *place, const std::nothrow_t &) noexcept {
        return place;
    }
    static void *operator new[](
        size_t size, const std::nothrow_t &) noexcept {
        return vengine_malloc(size);
    }
    static void operator delete(
        void *pdead) noexcept {
        vengine_free(pdead);
    }
    static void operator delete[](
        void *pdead) noexcept {
        vengine_free(pdead);
    }
    static void operator delete(
        void *pdead, size_t) noexcept {
        vengine_free(pdead);
    }
    static void operator delete[](
        void *pdead, size_t) noexcept {
        vengine_free(pdead);
    }
};
}// namespace vstd
