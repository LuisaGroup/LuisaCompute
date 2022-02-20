#pragma once
#include <vstl/MetaLib.h>
#include <tuple>
#include <EASTL/variant.h>
namespace vstd {
namespace detail {
template<typename Func, typename PtrType>
constexpr static decltype(auto) NoArgs_FuncTable(GetVoidType_t<Func> *func) {
    using PureFunc = std::remove_cvref_t<Func>;
    PureFunc *realFunc = reinterpret_cast<PureFunc *>(func);
    return (std::forward<Func>(*realFunc)).template operator()<std::remove_reference_t<PtrType>>();
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
    static T get_or(eastl::variant<Args...> const &v, T &&def) {
        if (IndexOf<T> == v.index()) {
            return eastl::get<T>(v);
        }
        return std::forward<T>(def);
    }
    template<typename T>
    static T get_or(eastl::variant<Args...> &v, T &&def) {
        if (IndexOf<T> == v.index()) {
            return eastl::get<T>(v);
        }
        return std::forward<T>(def);
    }
    template<typename T>
    static T get_or(eastl::variant<Args...> &&v, T &&def) {
        if (IndexOf<T> == v.index()) {
            return std::move(eastl::get<T>(v));
        }
        return std::forward<T>(def);
    }
    template<typename T>
    static T const *try_get(eastl::variant<Args...> const &v) {
        if (IndexOf<T> == v.index()) {
            return &eastl::get<T>(v);
        }
        return nullptr;
    }
    template<typename T>
    static T *try_get(eastl::variant<Args...> &v) {
        if (IndexOf<T> == v.index()) {
            return &eastl::get<T>(v);
        }
        return nullptr;
    }
    template<typename T>
    static vstd::optional<T> try_get(eastl::variant<Args...> &&v) {
        if (IndexOf<T> == v.index()) {
            return {std::move(eastl::get<T>(v))};
        }
        return {};
    }
};

template<typename... Args>
class VariantVisitor<eastl::variant<Args...>> {
public:
    using Type = VariantVisitor<Args...>;
};
template<typename... Args>
class VariantVisitor<vstd::variant<Args...>> {
public:
    using Type = VariantVisitor<Args...>;
};
template<typename... Args>
class VariantVisitor<std::tuple<Args...>> {
public:
    using Type = VariantVisitor<Args...>;
};
template<typename... Args>
using VariantVisitor_t = typename VariantVisitor<Args...>::Type;
}// namespace vstd