//
// Credit: https://github.com/zhihaoy/nontype_functional/tree/main
//
// License:
// BSD 2-Clause License
//
// Copyright (c) 2022, Zhihao Yuan
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

namespace move_only_function_detail {
template<auto V>
struct nontype_t// freestanding
{
    explicit nontype_t() = default;
};

template<auto V>
inline constexpr nontype_t<V> nontype{};// freestanding

using std::in_place_type;
using std::in_place_type_t;
using std::initializer_list;
using std::nullptr_t;

template<class R, class F, class... Args>
    requires std::is_invocable_r_v<R, F, Args...>
constexpr R invoke_r(F &&f, Args &&...args)// freestanding
    noexcept(std::is_nothrow_invocable_r_v<R, F, Args...>) {
    if constexpr (std::is_void_v<R>)
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    else
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
}

// See also: https://www.agner.org/optimize/calling_conventions.pdf
template<class T>
inline constexpr auto _select_param_type = [] {
    if constexpr (std::is_trivially_copyable_v<T>)
        return std::type_identity<T>();
    else
        return std::add_rvalue_reference<T>();
};

template<class T>
using _param_t = typename std::invoke_result_t<decltype(_select_param_type<T>)>::type;

template<class T, class Self>
inline constexpr bool _is_not_self =
    not std::is_same_v<std::remove_cvref_t<T>, Self>;

template<class T, template<class...> class>
inline constexpr bool _looks_nullable_to_impl = std::is_member_pointer_v<T>;

template<class F, template<class...> class Self>
inline constexpr bool _looks_nullable_to_impl<F *, Self> =
    std::is_function_v<F>;

template<class... S, template<class...> class Self>
inline constexpr bool _looks_nullable_to_impl<Self<S...>, Self> = true;

template<class S, template<class...> class Self>
inline constexpr bool _looks_nullable_to =
    _looks_nullable_to_impl<std::remove_cvref_t<S>, Self>;

template<class T>
inline constexpr bool _is_not_nontype_t = true;
template<auto f>
inline constexpr bool _is_not_nontype_t<nontype_t<f>> = false;

template<class T>
struct _adapt_signature;

template<class F>
    requires std::is_function_v<F>
struct _adapt_signature<F *> {
    using type = F;
};

template<class Fp>
using _adapt_signature_t = typename _adapt_signature<Fp>::type;

template<class S>
struct _not_qualifying_this {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...)> {
    using type = R(Args...);
};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) noexcept> {
    using type = R(Args...) noexcept;
};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const> : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) volatile>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const volatile>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) &> : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const &>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) volatile &>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const volatile &>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) &&> : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const &&>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) volatile &&>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const volatile &&>
    : _not_qualifying_this<R(Args...)> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) volatile noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const volatile noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) & noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const & noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) volatile & noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const volatile & noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) && noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const && noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) volatile && noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class R, class... Args>
struct _not_qualifying_this<R(Args...) const volatile && noexcept>
    : _not_qualifying_this<R(Args...) noexcept> {};

template<class F, class T>
struct _drop_first_arg_to_invoke;

template<class T, class R, class G, class... Args>
struct _drop_first_arg_to_invoke<R (*)(G, Args...), T> {
    using type = R(Args...);
};

template<class T, class R, class G, class... Args>
struct _drop_first_arg_to_invoke<R (*)(G, Args...) noexcept, T> {
    using type = R(Args...) noexcept;
};

template<class T, class M, class G>
    requires std::is_object_v<M>
struct _drop_first_arg_to_invoke<M G::*, T> {
    using type = std::invoke_result_t<M G::*, T>();
};

template<class T, class M, class G>
    requires std::is_function_v<M>
struct _drop_first_arg_to_invoke<M G::*, T> : _not_qualifying_this<M> {};

template<class F, class T>
using _drop_first_arg_to_invoke_t = typename _drop_first_arg_to_invoke<F, T>::type;

template<class Sig>
struct _cv_fn_sig {};

template<class R, class... Args>
struct _cv_fn_sig<R(Args...)> {
    using function = R(Args...);
    template<class T>
    using cv = T;
};

template<class R, class... Args>
struct _cv_fn_sig<R(Args...) const> {
    using function = R(Args...);
    template<class T>
    using cv = T const;
};

template<class Sig>
struct _ref_quals_fn_sig : _cv_fn_sig<Sig> {
    template<class T>
    using ref = T;
};

template<class R, class... Args>
struct _ref_quals_fn_sig<R(Args...) &> : _cv_fn_sig<R(Args...)> {
    template<class T>
    using ref = T &;
};

template<class R, class... Args>
struct _ref_quals_fn_sig<R(Args...) const &> : _cv_fn_sig<R(Args...) const> {
    template<class T>
    using ref = T &;
};

template<class R, class... Args>
struct _ref_quals_fn_sig<R(Args...) &&> : _cv_fn_sig<R(Args...)> {
    template<class T>
    using ref = T &&;
};

template<class R, class... Args>
struct _ref_quals_fn_sig<R(Args...) const &&> : _cv_fn_sig<R(Args...) const> {
    template<class T>
    using ref = T &&;
};

template<bool V>
struct _noex_traits {
    static constexpr bool is_noexcept = V;
};

template<class Sig>
struct _full_fn_sig : _ref_quals_fn_sig<Sig>, _noex_traits<false> {};

template<class R, class... Args>
struct _full_fn_sig<R(Args...) noexcept> : _ref_quals_fn_sig<R(Args...)>,
                                           _noex_traits<true> {};

template<class R, class... Args>
struct _full_fn_sig<R(Args...) & noexcept> : _ref_quals_fn_sig<R(Args...) &>,
                                             _noex_traits<true> {};

template<class R, class... Args>
struct _full_fn_sig<R(Args...) && noexcept> : _ref_quals_fn_sig<R(Args...) &&>,
                                              _noex_traits<true> {};

template<class R, class... Args>
struct _full_fn_sig<R(Args...) const noexcept>
    : _ref_quals_fn_sig<R(Args...) const>, _noex_traits<true> {};

template<class R, class... Args>
struct _full_fn_sig<R(Args...) const & noexcept>
    : _ref_quals_fn_sig<R(Args...) const &>, _noex_traits<true> {};

template<class R, class... Args>
struct _full_fn_sig<R(Args...) const && noexcept>
    : _ref_quals_fn_sig<R(Args...) const &&>, _noex_traits<true> {};

constexpr inline struct
{
    constexpr auto operator()(auto &&rhs) const {
        return new auto(decltype(rhs)(rhs));
    }

    constexpr auto operator()(auto *rhs) const noexcept { return rhs; }

    template<class T>
    constexpr auto operator()(std::reference_wrapper<T> rhs) const noexcept {
        return std::addressof(rhs.get());
    }

} _take_reference;

template<class T>
constexpr auto _build_reference = [](auto &&...args) { return new T(decltype(args)(args)...); };

template<class T>
constexpr auto _build_reference<T *> = [](auto &&...args) noexcept -> T * { return {decltype(args)(args)...}; };

template<class T>
constexpr auto _build_reference<std::reference_wrapper<T>> =
    [](auto &rhs) noexcept { return std::addressof(rhs); };

struct _move_only_pointer {
    union value_type {
        void *p_ = nullptr;
        void const *cp_;
        void (*fp_)();
    } val;

    _move_only_pointer() = default;
    _move_only_pointer(_move_only_pointer const &) = delete;
    _move_only_pointer &operator=(_move_only_pointer const &) = delete;

    constexpr _move_only_pointer(_move_only_pointer &&other) noexcept
        : val(std::exchange(other.val, {})) {}

    template<class T>
        requires std::is_object_v<T>
    constexpr explicit _move_only_pointer(T *p) noexcept : val{.p_ = p} {}

    template<class T>
        requires std::is_object_v<T>
    constexpr explicit _move_only_pointer(T const *p) noexcept : val{.cp_ = p} {}

    template<class T>
        requires std::is_function_v<T>
    constexpr explicit _move_only_pointer(T *p) noexcept
        : val{.fp_ = reinterpret_cast<decltype(val.fp_)>(p)} {}

    template<class T>
        requires std::is_object_v<T>
    constexpr _move_only_pointer &operator=(T *p) noexcept {
        val.p_ = p;
        return *this;
    }

    template<class T>
        requires std::is_object_v<T>
    constexpr _move_only_pointer &operator=(T const *p) noexcept {
        val.cp_ = p;
        return *this;
    }

    template<class T>
        requires std::is_function_v<T>
    constexpr _move_only_pointer &operator=(T *p) noexcept {
        val.fp_ = reinterpret_cast<decltype(val.fp_)>(p);
        return *this;
    }

    constexpr _move_only_pointer &operator=(_move_only_pointer &&other) noexcept {
        val = std::exchange(other.val, {});
        return *this;
    }
};

template<bool noex, class R, class... Args>
struct _callable_trait {
    using handle = _move_only_pointer::value_type;

    typedef auto call_t(handle, Args...) noexcept(noex) -> R;
    typedef void destroy_t(handle) noexcept;

    struct vtable {
        call_t *call = 0;
        destroy_t *destroy = [](handle) noexcept {};
    };

    static inline constinit vtable const abstract_base;

    template<class T>
    constexpr static auto get(handle val) {
        if constexpr (std::is_const_v<T>)
            return static_cast<T *>(val.cp_);
        else if constexpr (std::is_object_v<T>)
            return static_cast<T *>(val.p_);
        else
            return reinterpret_cast<T *>(val.fp_);
    }

    // Keep the parameter pack in a named function: MSVC can mis-instantiate
    // reference arguments in the lambda of a variable-template initializer.
    template<class T, template<class> class quals>
    static auto call_target(handle this_, Args... args) noexcept(noex) -> R {
        if constexpr (std::is_lvalue_reference_v<T> or std::is_pointer_v<T>) {
            using Tp = std::remove_reference_t<std::remove_pointer_t<T>>;
            return luisa::move_only_function_detail::invoke_r<R>(*get<Tp>(this_),
                                                                 static_cast<Args>(args)...);
        } else {
            using Fp = typename quals<T>::type;
            return luisa::move_only_function_detail::invoke_r<R>(static_cast<Fp>(*get<T>(this_)),
                                                                 static_cast<Args>(args)...);
        }
    }

    // See also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=71954
    template<class T, template<class> class quals>
    static inline constinit vtable const callable_target{
        .call = &call_target<T, quals>,
        .destroy =
            [](handle this_) noexcept {
                if constexpr (not std::is_lvalue_reference_v<T> and
                              not std::is_pointer_v<T>)
                    delete get<T>(this_);
            },
    };

    template<auto f>
    static inline constinit vtable const unbound_callable_target{
        .call = [](handle, Args... args) noexcept(noex) -> R { return luisa::move_only_function_detail::invoke_r<R>(f, static_cast<Args>(args)...); },
    };

    template<auto f, class T, template<class> class quals>
    static inline constinit vtable const bound_callable_target{
        .call = [](handle this_, Args... args) noexcept(noex) -> R {
            if constexpr (std::is_pointer_v<T>) {
                using Tp = std::remove_pointer_t<T>;
                return luisa::move_only_function_detail::invoke_r<R>(f, get<Tp>(this_),
                                                                     static_cast<Args>(args)...);
            } else if constexpr (std::is_lvalue_reference_v<T>) {
                using Tp = std::remove_reference_t<T>;
                return luisa::move_only_function_detail::invoke_r<R>(f, *get<Tp>(this_),
                                                                     static_cast<Args>(args)...);
            } else {
                using Fp = typename quals<T>::type;
                return luisa::move_only_function_detail::invoke_r<R>(f, static_cast<Fp>(*get<T>(this_)),
                                                                     static_cast<Args>(args)...);
            }
        },
        .destroy =
            [](handle this_) noexcept {
                if constexpr (not std::is_lvalue_reference_v<T> and
                              not std::is_pointer_v<T>)
                    delete get<T>(this_);
            },
    };

    template<auto f, class T>
    static inline constinit vtable const boxed_callable_target{
        .call = [](handle this_, Args... args) noexcept(noex) -> R {
            return luisa::move_only_function_detail::invoke_r<R>(f, get<T>(this_),
                                                                 static_cast<Args>(args)...);
        },
        .destroy =
            [](handle this_) noexcept {
                using D = typename std::unique_ptr<T>::deleter_type;
                static_assert(std::is_trivially_default_constructible_v<D>);
                if (auto p = get<T>(this_))
                    D()(p);
            },
    };
};

template<class T, template<class...> class Primary>
inline constexpr bool _is_specialization_of = false;

template<template<class...> class Primary, class... Args>
inline constexpr bool _is_specialization_of<Primary<Args...>, Primary> = true;

template<class T, template<class...> class Primary>
inline constexpr bool _does_not_specialize =
    not _is_specialization_of<std::remove_cvref_t<T>, Primary>;

template<class S, class = typename _full_fn_sig<S>::function>
class move_only_function;

template<class S, class R, class... Args>
class move_only_function<S, R(Args...)> {
    using signature = _full_fn_sig<S>;

    template<class T>
    using cv = typename signature::template cv<T>;
    template<class T>
    using ref = typename signature::template ref<T>;

    static constexpr bool noex = signature::is_noexcept;
    static constexpr bool is_const = std::is_same_v<cv<void>, void const>;
    static constexpr bool is_lvalue_only = std::is_same_v<ref<int>, int &>;
    static constexpr bool is_rvalue_only = std::is_same_v<ref<int>, int &&>;

    template<class T>
    using cvref = ref<cv<T>>;
    template<class T>
    struct inv_quals_f
        : std::conditional<is_lvalue_only or is_rvalue_only, cvref<T>, cv<T> &> {};
    template<class T>
    using inv_quals = typename inv_quals_f<T>::type;

    template<class... T>
    static constexpr bool is_invocable_using =
        std::conditional_t<noex, std::is_nothrow_invocable_r<R, T..., Args...>,
                           std::is_invocable_r<R, T..., Args...>>::value;

    template<class VT>
    static constexpr bool is_callable_from =
        is_invocable_using<cvref<VT>> and is_invocable_using<inv_quals<VT>>;

    template<auto f, class VT>
    static constexpr bool is_callable_as_if_from =
        is_invocable_using<decltype(f), inv_quals<VT>>;

    using trait = _callable_trait<noex, R, _param_t<Args>...>;
    using vtable = typename trait::vtable;

    std::reference_wrapper<vtable const> vtbl_ = trait::abstract_base;
    _move_only_pointer obj_;

public:
    using result_type = R;

    move_only_function() = default;
    move_only_function(nullptr_t) noexcept : move_only_function() {}

    template<class F, class VT = std::decay_t<F>>
    move_only_function(F &&f) noexcept(
        std::is_nothrow_invocable_v<decltype(_take_reference), F>)
        requires _is_not_self<F, move_only_function> and
                 _does_not_specialize<F, in_place_type_t> and
                 is_callable_from<VT> and std::is_constructible_v<VT, F>
    {
        if constexpr (_looks_nullable_to<F, move_only_function>) {
            if (f == nullptr)
                return;
        }

        vtbl_ = trait::template callable_target<std::unwrap_ref_decay_t<F>,
                                                inv_quals_f>;
        obj_ = _take_reference(std::forward<F>(f));
    }

    template<auto f>
    move_only_function(nontype_t<f>) noexcept
        requires is_invocable_using<decltype(f)>
        : vtbl_(trait::template unbound_callable_target<f>) {}

    template<auto f, class T, class VT = std::decay_t<T>>
    move_only_function(nontype_t<f>, T &&x) noexcept(
        std::is_nothrow_invocable_v<decltype(_take_reference), T>)
        requires is_callable_as_if_from<f, VT> and
                     std::is_constructible_v<VT, T>
        : vtbl_(trait::template bound_callable_target<
                f, std::unwrap_ref_decay_t<T>, inv_quals_f>),
          obj_(_take_reference(std::forward<T>(x))) {}

    template<class M, class C, M C::*f, class T>
    move_only_function(nontype_t<f>, std::unique_ptr<T> &&x) noexcept
        requires std::is_base_of_v<C, T> and is_callable_as_if_from<f, T *>
        : vtbl_(trait::template boxed_callable_target<f, T>), obj_(x.release()) {}

    template<class T, class... Inits>
    explicit move_only_function(in_place_type_t<T>, Inits &&...inits) noexcept(
        std::is_nothrow_invocable_v<decltype(_build_reference<T>), Inits...>)
        requires is_callable_from<T> and std::is_constructible_v<T, Inits...>
        : vtbl_(trait::template callable_target<std::unwrap_reference_t<T>,
                                                inv_quals_f>),
          obj_(_build_reference<T>(std::forward<Inits>(inits)...)) {
        static_assert(std::is_same_v<std::decay_t<T>, T>);
    }

    template<class T, class U, class... Inits>
    explicit move_only_function(in_place_type_t<T>, initializer_list<U> ilist,
                                Inits &&...inits) noexcept(//
        std::is_nothrow_invocable_v<decltype(_build_reference<T>),
                                    decltype((ilist)), Inits...>)
        requires is_callable_from<T> and
                     std::is_constructible_v<T, decltype((ilist)), Inits...>
        : vtbl_(trait::template callable_target<std::unwrap_reference_t<T>,
                                                inv_quals_f>),
          obj_(_build_reference<T>(ilist, std::forward<Inits>(inits)...)) {
        static_assert(std::is_same_v<std::decay_t<T>, T>);
    }

    template<auto f, class T, class... Inits>
    explicit move_only_function(nontype_t<f>, in_place_type_t<T>,
                                Inits &&...inits) noexcept(//
        std::is_nothrow_invocable_v<decltype(_build_reference<T>), Inits...>)
        requires is_callable_as_if_from<f, T> and
                     std::is_constructible_v<T, Inits...>
        : vtbl_(trait::template bound_callable_target<
                f, std::unwrap_reference_t<T>, inv_quals_f>),
          obj_(_build_reference<T>(std::forward<Inits>(inits)...)) {
        static_assert(std::is_same_v<std::decay_t<T>, T>);
    }

    template<class M, class C, M C::*f, class T, class... Inits>
    explicit move_only_function(nontype_t<f> t,
                                in_place_type_t<std::unique_ptr<T>>,
                                Inits &&...inits) noexcept(//
        std::is_nothrow_constructible_v<std::unique_ptr<T>, Inits...>)
        requires std::is_base_of_v<C, T> and is_callable_as_if_from<f, T *> and
                 std::is_constructible_v<std::unique_ptr<T>,
                                         Inits...>
        : move_only_function(t,
                             std::unique_ptr<T>(std::forward<Inits>(inits)...)) {}

    template<auto f, class T, class U, class... Inits>
    explicit move_only_function(nontype_t<f>, in_place_type_t<T>,
                                initializer_list<U> ilist,
                                Inits &&...inits) noexcept(//
        std::is_nothrow_invocable_v<decltype(_build_reference<T>),
                                    decltype((ilist)), Inits...>)
        requires is_callable_as_if_from<f, T> and
                     std::is_constructible_v<T, decltype((ilist)), Inits...>
        : vtbl_(trait::template bound_callable_target<
                f, std::unwrap_reference_t<T>, inv_quals_f>),
          obj_(_build_reference<T>(ilist, std::forward<Inits>(inits)...)) {
        static_assert(std::is_same_v<std::decay_t<T>, T>);
    }

    move_only_function(move_only_function &&) = default;
    move_only_function &operator=(move_only_function &&) = default;

    void swap(move_only_function &other) noexcept {
        std::swap<move_only_function>(*this, other);
    }

    friend void swap(move_only_function &lhs, move_only_function &rhs) noexcept {
        lhs.swap(rhs);
    }

    ~move_only_function() { vtbl_.get().destroy(obj_.val); }

    explicit operator bool() const noexcept {
        return &vtbl_.get() != &trait::abstract_base;
    }

    friend bool operator==(move_only_function const &f, nullptr_t) noexcept {
        return !f;
    }

     // R operator()(Args... args) noexcept(noex)
     //     requires(!is_const and !is_lvalue_only and !is_rvalue_only)
     // {
     //     return vtbl_.get().call(obj_.val, std::forward<Args>(args)...);
     // }
     //
     // R operator()(Args... args) const noexcept(noex)
     //     requires(is_const and !is_lvalue_only and !is_rvalue_only)
     // {
     //     return vtbl_.get().call(obj_.val, std::forward<Args>(args)...);
     // }

    R operator()(Args... args) & noexcept(noex)
        requires(!is_rvalue_only)
    {
        return vtbl_.get().call(obj_.val, std::forward<Args>(args)...);
    }

    R operator()(Args... args) const & noexcept(noex)
        requires(!is_rvalue_only)
    {
        return vtbl_.get().call(obj_.val, std::forward<Args>(args)...);
    }

    R operator()(Args... args) && noexcept(noex)
        requires(!is_lvalue_only)
    {
        return vtbl_.get().call(obj_.val, std::forward<Args>(args)...);
    }

    R operator()(Args... args) const && noexcept(noex)
        requires(!is_lvalue_only)
    {
        return vtbl_.get().call(obj_.val, std::forward<Args>(args)...);
    }
};

}// namespace move_only_function_detail

using move_only_function_detail::move_only_function;
