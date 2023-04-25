#pragma once
#include <vstl/meta_lib.h>
#include <tuple>
#include <EASTL/variant.h>
namespace vstd {
namespace detail {
template<typename Func, typename PtrType>
constexpr static decltype(auto) NoArgs_FuncTable(Func *func) {
    return func->template operator()<std::remove_reference_t<PtrType>>();
}
template<typename Func, typename... Args>
static decltype(auto) VisitVariant(
    Func &&func,
    size_t idx) {
    constexpr static auto table =
        {
            NoArgs_FuncTable<
                std::remove_reference_t<Func>,
                Args>...};
    return (table.begin()[idx])(&func);
}
}// namespace detail

template<typename... Args>
class VariantVisitor {
public:
    using Type = VariantVisitor;
    template<typename T>
    static constexpr size_t IndexOf =
        detail::IndexOfFunc<
            0,
            std::is_same, std::remove_cvref_t<T>, Args...>();
    template<typename Func>
    void operator()(Func &&func, size_t idx) {
        if (idx < sizeof...(Args)) {
            detail::VisitVariant<Func, Args...>(std::forward<Func>(func), idx);
        }
    }
    template<typename T>
    static T const &get_or(luisa::variant<Args...> const &v, T &&def) {
        if (IndexOf<T> == v.index()) {
            return luisa::get<T>(v);
        }
        return std::forward<T>(def);
    }
    template<typename T>
    static T &get_or(luisa::variant<Args...> &v, T &&def) {
        if (IndexOf<T> == v.index()) {
            return luisa::get<T>(v);
        }
        return std::forward<T>(def);
    }
    template<typename T>
    static T &&get_or(luisa::variant<Args...> &&v, T &&def) {
        if (IndexOf<T> == v.index()) {
            return std::move(luisa::get<T>(v));
        }
        return std::forward<T>(def);
    }
    template<typename T>
    static T const *try_get(luisa::variant<Args...> const &v) {
        if (IndexOf<T> == v.index()) {
            return &luisa::get<T>(v);
        }
        return nullptr;
    }
    template<typename T>
    static T *try_get(luisa::variant<Args...> &v) {
        if (IndexOf<T> == v.index()) {
            return &luisa::get<T>(v);
        }
        return nullptr;
    }
    template<typename T>
    static vstd::optional<T> try_get(luisa::variant<Args...> &&v) {
        if (IndexOf<T> == v.index()) {
            return {std::move(luisa::get<T>(v))};
        }
        return {};
    }
};

template<typename... Args>
class VariantVisitor<luisa::variant<Args...>> {
public:
    using Type = VariantVisitor<Args...>;
};
template<typename... Args>
class VariantVisitor<vstd::variant<Args...>> {
public:
    using Type = VariantVisitor<vstd::variant<Args...>>;

    template<typename T>
    static constexpr size_t IndexOf = vstd::variant<Args...>::template IndexOf<T>;
    template<typename Func>
    void operator()(Func &&func, size_t idx) {
        if (idx < sizeof...(Args)) {
            detail::VisitVariant<Func, Args...>(std::forward<Func>(func), idx);
        }
    }
    template<typename T>
    static decltype(auto) get_or(vstd::variant<Args...> const &v, T &&def) {
        return v.template get_or<T>(std::forward<T>(def));
    }
    template<typename T>
    static decltype(auto) get_or(vstd::variant<Args...> &v, T &&def) {
        return v.template get_or<T>(std::forward<T>(def));
    }
    template<typename T>
    static decltype(auto) get_or(vstd::variant<Args...> &&v, T &&def) {
        return std::move(v.template get_or<T>(std::forward<T>(def)));
    }
    template<typename T>
    static decltype(auto) try_get(vstd::variant<Args...> const &v) {
        return v.template try_get<T>();
    }
    template<typename T>
    static decltype(auto) try_get(vstd::variant<Args...> &v) {
        return v.template try_get<T>();
    }
    template<typename T>
    static decltype(auto) try_get(vstd::variant<Args...> &&v) {
        return std::move(v.template try_get<T>());
    }
};
template<typename... Args>
class VariantVisitor<std::tuple<Args...>> {
public:
    using Type = VariantVisitor<Args...>;
};
template<typename... Args>
using VariantVisitor_t = typename VariantVisitor<Args...>::Type;
}// namespace vstd