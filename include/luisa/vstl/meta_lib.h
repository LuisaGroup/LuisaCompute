#pragma once

#include <cassert>
#include <new>
#include <typeinfo>
#include <type_traits>
#include <mutex>
#include <atomic>
#include <thread>
#include <tuple>
#include <utility>

#include <luisa/core/stl/type_traits.h>
#include <luisa/core/intrin.h>
#include <luisa/vstl/config.h>

#include <luisa/vstl/hash.h>
#include <luisa/vstl/allocate_type.h>
#include <luisa/vstl/compare.h>

inline void *vengine_malloc(size_t size) {
    return luisa::detail::allocator_allocate(size, 0);
}
inline void vengine_free(void *ptr) {
    luisa::detail::allocator_deallocate(ptr, 0);
}
inline void *vengine_realloc(void *ptr, size_t size) {
    return luisa::detail::allocator_reallocate(ptr, size, 0);
}
LUISA_VSTL_API void vengine_log(std::type_info const &t);
LUISA_VSTL_API void vengine_log(char const *chunk);
#define VE_SUB_TEMPLATE template<typename...> \
class
namespace vstd {

[[noreturn]] inline void unreachable() {
    assert(false);
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__)// MSVC
    __assume(false);
#else// GCC, Clang
    __builtin_unreachable();
#endif
}
template<size_t N, class T, class... Ts>
struct get_type {
    using type = typename get_type<N - 1, Ts...>::type;
};

template<class T, class... Ts>
struct get_type<0, T, Ts...> {
    using type = T;
};

template<size_t N, class... Ts>
using get_type_t = typename get_type<N, Ts...>::type;

template<class T, class... Ts>
struct index_of;

template<class T, class... Ts>
struct index_of<T, T, Ts...> : std::integral_constant<size_t, 0> {};

template<class T, class U, class... Ts>
struct index_of<T, U, Ts...> : std::integral_constant<size_t, 1 + index_of<T, Ts...>::value> {};

template<size_t N, int Strategy>
struct visit_strategy;

template<size_t N>
struct visit_strategy<N, 0> {
    template<class Ret, class Fn, class... Args>
    static constexpr Ret invoke(size_t idx, Fn &&fn, Args &&...args) {
        return static_cast<Fn &&>(fn).template operator()<0>(static_cast<Args &&>(args)...);
    }// namespace vstd
};
// visitor from @OEOTYAN
#ifndef LUISA_VISIT_CASE
#define LUISA_VISIT_CASE(n)                                                                        \
    case (n):                                                                                      \
        if constexpr ((n) < N) {                                                                   \
            return static_cast<Fn &&>(fn).template operator()<(n)>(static_cast<Args &&>(args)...); \
        }                                                                                          \
        unreachable();                                                                             \
        [[fallthrough]]
#endif
#define LUISA_VISIT_STAMP(stamper, n)       \
    static_assert(N > (n) / 4 && N <= (n)); \
    switch (idx) {                          \
        stamper(0, LUISA_VISIT_CASE);       \
        default:                            \
            unreachable();                  \
            break;                          \
    }

#define LUISA_STAMP4(n, x) \
    x(n);                  \
    x(n + 1);              \
    x(n + 2);              \
    x(n + 3)
#define LUISA_STAMP16(n, x) \
    LUISA_STAMP4(n, x);     \
    LUISA_STAMP4(n + 4, x); \
    LUISA_STAMP4(n + 8, x); \
    LUISA_STAMP4(n + 12, x)
#define LUISA_STAMP64(n, x)   \
    LUISA_STAMP16(n, x);      \
    LUISA_STAMP16(n + 16, x); \
    LUISA_STAMP16(n + 32, x); \
    LUISA_STAMP16(n + 48, x)
#define LUISA_STAMP256(n, x)   \
    LUISA_STAMP64(n, x);       \
    LUISA_STAMP64(n + 64, x);  \
    LUISA_STAMP64(n + 128, x); \
    LUISA_STAMP64(n + 192, x)

#define LUISA_STAMP(n, x) x(LUISA_STAMP##n, n)

template<size_t N>
struct visit_strategy<N, 1> {
    template<class Ret, class Fn, class... Args>
    static constexpr Ret invoke(size_t idx, Fn &&fn, Args &&...args) {
        LUISA_STAMP(4, LUISA_VISIT_STAMP);
        if constexpr (!std::is_void_v<Ret>) {
            unreachable();
            return *reinterpret_cast<Ret *>(114514);// make compiler happy
        }
    }
};

template<size_t N>
struct visit_strategy<N, 2> {
    template<class Ret, class Fn, class... Args>
    static constexpr Ret invoke(size_t idx, Fn &&fn, Args &&...args) {
        LUISA_STAMP(16, LUISA_VISIT_STAMP);
        if constexpr (!std::is_void_v<Ret>) {
            unreachable();
            return *reinterpret_cast<Ret *>(114514);// make compiler happy
        }
    }
};

template<size_t N>
struct visit_strategy<N, 3> {
    template<class Ret, class Fn, class... Args>
    static constexpr Ret invoke(size_t idx, Fn &&fn, Args &&...args) {
        LUISA_STAMP(64, LUISA_VISIT_STAMP);
        if constexpr (!std::is_void_v<Ret>) {
            unreachable();
            return *reinterpret_cast<Ret *>(114514);// make compiler happy
        }
    }
};

template<size_t N>
struct visit_strategy<N, 4> {
    template<class Ret, class Fn, class... Args>
    static constexpr Ret invoke(size_t idx, Fn &&fn, Args &&...args) {
        LUISA_STAMP(256, LUISA_VISIT_STAMP);
        if constexpr (!std::is_void_v<Ret>) {
            unreachable();
            return *reinterpret_cast<Ret *>(114514);// make compiler happy
        }
    }
};

template<size_t N, class Fn, class... Args>
    requires(N <= 256)
constexpr decltype(auto) visit_index(size_t index, Fn &&fn, Args &&...args) {
    constexpr int s = N == 1 ? 0 : N <= 4 ? 1 :
                               N <= 16    ? 2 :
                               N <= 64    ? 3 :
                                            4;
    using Ret = decltype((std::declval<Fn>().template operator()<0>(std::declval<Args>()...)));
    return (visit_strategy<N, s>::template invoke<Ret>(index, static_cast<Fn &&>(fn), static_cast<Args &&>(args)...));
}

template<class Ret, size_t N, class Fn, class... Args>
    requires(N <= 256)
constexpr Ret visit_index(size_t index, Fn &&fn, Args &&...args) {
    constexpr int s = N == 1 ? 0 : N <= 4 ? 1 :
                               N <= 16    ? 2 :
                               N <= 64    ? 3 :
                                            4;
    return visit_strategy<N, s>::template invoke<Ret>(index, static_cast<Fn &&>(fn), static_cast<Args &&>(args)...);
}
#undef LUISA_VISIT_CASE
#undef LUISA_VISIT_STAMP
#undef LUISA_STAMP
#undef LUISA_STAMP256
#undef LUISA_STAMP64
#undef LUISA_STAMP16
#undef LUISA_STAMP4

template<typename T, typename... Args>
    requires(luisa::is_constructible_v<T, Args && ...>)
constexpr void construct_at(T *ptr, Args &&...args) noexcept {
    std::construct_at(ptr, std::forward<Args>(args)...);
}
template<typename T, typename... Args>
    requires(!std::is_const_v<T> && luisa::is_constructible_v<T, Args && ...>)
void reset(T &v, Args &&...args) {
    std::destroy_at(std::addressof(v));
    std::construct_at(&v, std::forward<Args>(args)...);
}

template<typename T>
struct TypeOf {
    using Type = T;
};

template<typename A, typename B, bool v>
struct SelectType;
template<typename A, typename B>
struct SelectType<A, B, false> {
    using Type = B;
};
template<typename A, typename B>
struct SelectType<A, B, true> {
    using Type = A;
};
template<typename A, typename B, bool v>
using SelectType_t = typename SelectType<A, B, v>::Type;

template<typename T>
struct func_ptr;
template<typename Ret, typename... Args>
struct func_ptr<Ret(Args...)> {
    using Type = Ret (*)(Args...);
};
template<typename T>
using func_ptr_t = typename func_ptr<T>::Type;

template<typename T, uint32_t size = 1>
struct Storage {
    using Type = T;
    alignas(T) char c[size * sizeof(T)];
};
template<typename T>
struct Storage<T, 0> {
    using Type = T;
};

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
        requires(luisa::is_constructible_v<T, Args && ...>)
    inline SelfType &create(Args &&...args) & noexcept {
        new (storage) T(std::forward<Args>(args)...);
        return *this;
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    inline SelfType &&create(Args &&...args) && noexcept {
        return std::move(create(std::forward<Args>(args)...));
    }
    inline void destroy() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>)
            std::destroy_at(reinterpret_cast<T *>(storage));
    }
    T &operator*() & noexcept {
        return *std::launder(reinterpret_cast<T *>(storage));
    }
    T &&operator*() && noexcept {
        return std::move(*std::launder(reinterpret_cast<T *>(storage)));
    }
    T const &operator*() const & noexcept {
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
    StackObject(const SelfType &value)
        requires(std::is_copy_constructible_v<T>)
    {
        new (storage) T(*value);
    }
    StackObject(SelfType &&value)
        requires(std::is_move_constructible_v<T>)
    {
        new (storage) T(std::move(*value));
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    StackObject(Args &&...args) {
        new (storage) T(std::forward<Args>(args)...);
    }
    T &operator=(SelfType const &value)
        requires(std::is_copy_assignable_v<T> || std::is_copy_constructible_v<T>)
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            operator*() = *value;
        } else {
            destroy();
            create(*value);
        }
        return **this;
    }
    T &operator=(SelfType &&value)
        requires(std::is_move_assignable_v<T> || std::is_move_constructible_v<T>)
    {
        if constexpr (std::is_move_assignable_v<T>) {
            operator*() = std::move(*value);
        } else {
            destroy();
            create(std::move(*value));
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
        requires(luisa::is_constructible_v<T, Args && ...>)
    inline SelfType &create(Args &&...args) & noexcept {
        if (mInitialized) return *this;
        mInitialized = true;
        stackObj.create(std::forward<Args>(args)...);
        return *this;
    }

    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    inline SelfType &&create(Args &&...args) && noexcept {
        return std::move(create(std::forward<Args>(args)...));
    }

    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    inline SelfType &force_create(Args &&...args) & noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            if (mInitialized) { destroy(); }
        }
        mInitialized = true;
        stackObj.create(std::forward<Args>(args)...);
        return *this;
    }

    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    inline SelfType &&force_create(Args &&...args) && noexcept {
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
    void reset() noexcept {
        destroy();
    }
    T &value() & noexcept {
        return *stackObj;
    }
    T const &value() const & noexcept {
        return *stackObj;
    }
    T &&value() && noexcept {
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
    T &operator*() & noexcept {
        return *stackObj;
    }
    T &&operator*() && noexcept {
        return std::move(*stackObj);
    }
    T const &operator*() const & noexcept {
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
        requires(luisa::is_constructible_v<T, Args && ...>)
    StackObject(Args &&...args)
        : stackObj(std::forward<Args>(args)...),
          mInitialized(true) {
    }
    StackObject(const SelfType &value) noexcept
        requires(std::is_copy_constructible_v<T>)
    {
        mInitialized = value.mInitialized;
        if (mInitialized) {
            stackObj.create(*value);
        }
    }
    StackObject(SelfType &&value) noexcept
        requires(std::is_move_constructible_v<T>)
    {
        mInitialized = value.mInitialized;
        if (mInitialized) {
            stackObj.create(std::move(*value));
        }
    }
    ~StackObject() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            if (mInitialized)
                stackObj.destroy();
        }
    }
    T &operator=(SelfType const &value)
        requires(std::is_copy_constructible_v<T>)
    {
        if (!mInitialized) {
            if (value.mInitialized) {
                stackObj.create(*value);
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
    T &operator=(SelfType &&value)
        requires(std::is_move_constructible_v<T>)
    {
        if (!mInitialized) {
            if (value.mInitialized) {
                stackObj.create(std::move(*value));
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
    T &operator=(Arg &&value) {
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
    using value_type = T;
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
class IDisposable {
protected:
    IDisposable() = default;
    ~IDisposable() = default;

public:
    virtual void Dispose() = 0;
};
struct IteEndTag {};
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
    return std::addressof(v);
}
template<typename T>
decltype(auto) get_const_lvalue(T &&data) {
    return static_cast<std::remove_reference_t<T> const &>(data);
}
namespace detail {
template<typename T, typename... Args>
struct MapConstructible {
    static constexpr bool value = luisa::is_constructible_v<T, Args...>;
};

template<typename... Args>
struct MapConstructible<void, Args...> {
    static constexpr bool value = (sizeof...(Args) == 0);
};

template<size_t i, typename T, typename... Types>
constexpr auto select_type() {
    if constexpr (i == 0) {
        return vstd::TypeOf<T>{};
    } else {
        return select_type<i - 1, Types...>();
    }
}

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

template<typename... Args>
struct VariantConstructible {
    template<size_t i, typename T, typename... Ts>
    static constexpr size_t Run() {
        if constexpr (luisa::is_constructible_v<T, Args...>) {
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
    visit_index<sizeof...(Types)>(id, [&]<size_t i>() {
        using Type = typename decltype(detail::select_type<i, Types...>())::Type;
        if constexpr (std::is_const_v<VoidPtr>) {
            func(*static_cast<std::add_pointer_t<Type const>>(ptr));
        } else {
            func(*static_cast<std::add_pointer_t<Type>>(ptr));
        }
    });
}
template<typename Ret, typename Func, typename VoidPtr, typename... Types>
static Ret VisitorRet(
    size_t id,
    VoidPtr *ptr,
    Func &&func) {
    return visit_index<Ret, sizeof...(Types)>(id, [&]<size_t i>() {
        using Type = typename decltype(detail::select_type<i, Types...>())::Type;
        if constexpr (std::is_const_v<VoidPtr>) {
            return func(*static_cast<std::add_pointer_t<Type const>>(ptr));
        } else {
            return func(*static_cast<std::add_pointer_t<Type>>(ptr));
        }
    });
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
    [[noreturn]] operator T() const { abort(); }
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
        requires(i < argSize)
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
    luisa::aligned_storage_t<(detail::max_size<sizeof(AA)...>()), (detail::max_size<alignof(AA)...>())> placeHolder;
    size_t switcher = 0;
    void m_dispose() {
        if constexpr (detail::AnyMap<std::is_trivially_destructible, true>::template Run<AA...>()) {
            auto disposeFunc = [&]<typename T>(T &value) {
                std::destroy_at(std::addressof(value));
            };
            visit(disposeFunc);
        }
    }

public:
    bool valid() const { return switcher < argSize; }

    template<typename... Args>
    void reset_as(size_t typeIndex, Args &&...args) {
        LUISA_ASSUME(typeIndex < argSize);
        this->~variant();
        if (typeIndex >= argSize) {
            switcher = argSize;
            return;
        }
        switcher = typeIndex;
        auto func = [&]<typename T>(T &t) {
            constexpr bool cons = luisa::is_constructible_v<T, Args &&...>;
            LUISA_ASSUME(cons);
            if constexpr (cons)
                new (&t) T(std::forward<Args>(args)...);
        };
        detail::Visitor<decltype(func), void, AA &...>(typeIndex, place_holder(), std::move(func));
    }
    template<typename T, typename... Args>
        requires(
            IndexOf<T> < argSize && luisa::is_constructible_v<T, Args && ...>)
    void reset_as(Args &&...args) {
        this->~variant();
        switcher = IndexOf<T>;
        new (&placeHolder) T(std::forward<Args>(args)...);
    }

    void *place_holder() { return &placeHolder; }
    void const *place_holder() const { return &placeHolder; }
    size_t index() const { return switcher; }
    template<typename T>
        requires(IndexOf<T> < argSize)
    bool is_type_of() const {
        return switcher == (IndexOf<T>);
    }

    template<size_t i>
        requires(i < argSize)
    decltype(auto)
    get() & {
#ifndef NDEBUG
        if (i != switcher) {
            vengine_log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return *std::launder(reinterpret_cast<TypeOf<i> *>(&placeHolder));
    }
    template<size_t i>
        requires(i < argSize)
    decltype(auto)
    get() && {
#ifndef NDEBUG
        if (i != switcher) {
            vengine_log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(*std::launder(reinterpret_cast<TypeOf<i> *>(&placeHolder)));
    }
    template<size_t i>
        requires(i < argSize)
    decltype(auto)
    get() const & {
#ifndef NDEBUG
        if (i != switcher) {
            vengine_log("Try get wrong variant type!\n");
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
            vengine_log("Try get wrong variant type!\n");
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
            vengine_log("Try get wrong variant type!\n");
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
            vengine_log("Try get wrong variant type!\n");
            VENGINE_EXIT;
        }
#endif
        return std::move(*std::launder(reinterpret_cast<TypeOf<tarIdx> *>(&placeHolder)));
    }
    template<typename Func>
    void visit(Func &&func) & {
        if (switcher >= argSize) [[unlikely]]
            return;
        detail::Visitor<Func, void, AA &...>(switcher, place_holder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) && {
        if (switcher >= argSize) [[unlikely]]
            return;
        detail::Visitor<Func, void, AA...>(switcher, place_holder(), std::forward<Func>(func));
    }
    template<typename Func>
    void visit(Func &&func) const & {
        if (switcher >= argSize) [[unlikely]]
            return;
        detail::Visitor<Func, void const, AA const &...>(switcher, place_holder(), std::forward<Func>(func));
    }

    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) & {
        if (switcher >= argSize) [[unlikely]]
            return;
        detail::Visitor<PackedFunctors<Funcs...>, void, AA &...>(
            switcher,
            place_holder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) && {
        if (switcher >= argSize) [[unlikely]]
            return;
        detail::Visitor<PackedFunctors<Funcs...>, void, AA...>(
            switcher,
            place_holder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    void multi_visit(Funcs &&...funcs) const & {
        if (switcher >= argSize) [[unlikely]]
            return;
        detail::Visitor<PackedFunctors<Funcs...>, void const, AA const &...>(
            switcher,
            place_holder(),
            PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    decltype(auto)
    multi_visit_or(Ret &&r, Funcs &&...funcs) & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, PackedFunctors<Funcs...>, void, AA &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, PackedFunctors<Funcs...>, void, AA &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        }
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    decltype(auto)
    multi_visit_or(Ret &&r, Funcs &&...funcs) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, PackedFunctors<Funcs...>, void, AA...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, PackedFunctors<Funcs...>, void, AA...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        }
    }
    template<typename Ret, typename... Funcs>
        requires(sizeof...(Funcs) == argSize)
    decltype(auto)
    multi_visit_or(Ret &&r, Funcs &&...funcs) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, PackedFunctors<Funcs...>, void const, AA const &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, PackedFunctors<Funcs...>, void const, AA const &...>(
                switcher,
                place_holder(),
                PackedFunctors<Funcs...>(std::forward<Funcs>(funcs)...));
        }
    }

    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Func, void, AA &...>(switcher, place_holder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Func, void, AA &...>(switcher, place_holder(), std::forward<Func>(func));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) && {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Func, void, AA...>(switcher, place_holder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Func, void, AA...>(switcher, place_holder(), std::forward<Func>(func));
        }
    }
    template<typename Ret, typename Func>
    decltype(auto) visit_or(Ret &&r, Func &&func) const & {
        using RetType = std::remove_cvref_t<Ret>;
        if constexpr (std::is_base_of_v<Evaluable, RetType>) {
            using EvalType = typename RetType::EvalType;
            if (switcher >= argSize) return std::forward<Ret>(r).operator EvalType();
            return detail::VisitorRet<EvalType, Func, void const, AA const &...>(switcher, place_holder(), std::forward<Func>(func));
        } else {
            if (switcher >= argSize) return Ret{std::forward<Ret>(r)};
            return detail::VisitorRet<Ret, Func, void const, AA const &...>(switcher, place_holder(), std::forward<Func>(func));
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
        requires(!detail::AnyMap<std::is_copy_constructible, true>::template Run<AA...>())
        : switcher(v.switcher) {
        auto copyFunc = [&]<typename T>(T const &value) {
            new (place_holder()) T(value);
        };
        v.visit(copyFunc);
    }
    variant(variant &&v)
        requires(!detail::AnyMap<std::is_move_constructible, true>::template Run<AA...>())
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
    variant &operator=(variant const &a)
        requires(!detail::AnyMap<std::is_copy_assignable, true>::template Run<AA...>())
    {
        if (switcher != a.switcher) {
            this->~variant();
            new (this) variant(a);
        } else {
            auto assignFunc = [&]<typename T>(T const &v) {
                if constexpr (std::is_copy_assignable_v<T>)
                    *std::launder(reinterpret_cast<T *>(&placeHolder)) = v;
                else {
                    LUISA_ASSUME(false);
                    VENGINE_EXIT;
                }
            };
            a.visit(assignFunc);
        }
        return *this;
    }
    variant &operator=(variant &&a)
        requires(!detail::AnyMap<std::is_move_assignable, true>::template Run<AA...>())
    {
        if (switcher != a.switcher) {
            this->~variant();
            new (this) variant(std::move(a));
        } else {
            auto assignFunc = [&]<typename T>(T &v) {
                if constexpr (std::is_move_assignable_v<T>)
                    *std::launder(reinterpret_cast<T *>(&placeHolder)) = std::move(v);
                else {
                    LUISA_ASSUME(false);
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
#define VSTD_TRIVIAL_COMPARABLE(T)                    \
    bool operator==(T const &a) const {               \
        return std::memcmp(this, &a, sizeof(T)) == 0; \
    }                                                 \
    bool operator!=(T const &a) const {               \
        return std::memcmp(this, &a, sizeof(T)) != 0; \
    }                                                 \
    bool operator>(T const &a) const {                \
        return std::memcmp(this, &a, sizeof(T)) > 0;  \
    }                                                 \
    bool operator<(T const &a) const {                \
        return std::memcmp(this, &a, sizeof(T)) < 0;  \
    }
struct IOperatorNewBase {
    static void *operator new(
        size_t size) noexcept {
        return vengine_malloc(size);
    }
    // Over-aligned allocation: without this overload, `new T` for any T with
    // alignof(T) > default new alignment resolves to the class-specific plain
    // operator new above (which hides the global aligned overloads) and
    // returns under-aligned storage — release builds then fault on aligned
    // vector stores into alignas members (observed: lc::vk::Stream ctor,
    // alignof 64 from LockFreeArrayQueue's alignas(64) cache-line structs).
    static void *operator new(
        size_t size,
        std::align_val_t alignment) noexcept {
        return luisa::detail::allocator_allocate(
            size, static_cast<size_t>(alignment));
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
        size_t size,
        std::align_val_t alignment,
        const std::nothrow_t &) noexcept {
        return luisa::detail::allocator_allocate(
            size, static_cast<size_t>(alignment));
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
    // Matching deallocation for the aligned overload above (deallocation
    // ignores alignment in this allocator, so forward to the plain delete).
    static void operator delete(
        void *pdead,
        std::align_val_t) noexcept {
        vengine_free(pdead);
    }
    static void operator delete(
        void *pdead,
        size_t,
        std::align_val_t) noexcept {
        vengine_free(pdead);
    }
    static void operator delete(
        void *ptr,
        void *place) noexcept {
        // do nothing
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