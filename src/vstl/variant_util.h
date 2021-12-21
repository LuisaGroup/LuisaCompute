#pragma once
#include <vstl/MetaLib.h>
#include <variant>
#include <tuple>
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
    static constexpr size_t IndexOf = detail::IndexOfStruct<0, std::remove_cvref_t<T>, Args...>::Index;
    template<typename Func>
    void operator()(Func &&func, size_t idx) {
        if (idx < sizeof...(Args)) {
            detail::VisitVariant<Func, Args...>(std::forward<Func>(func), idx);
        }
    }
};

template<typename... Args>
class VariantVisitor<std::variant<Args...>> {
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