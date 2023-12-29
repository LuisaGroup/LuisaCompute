#pragma once

#include <tuple>
#include <luisa/core/concepts.h>
#include <luisa/vstl/meta_lib.h>

namespace vstd {
namespace detail {
LC_VSTL_API void one_shot_range_log_error() noexcept;
}// namespace detail
#ifndef NDEBUG
#define VSTL_ONESHOT_ITER_DECLVAR bool already_begined{false};
#define VSTL_ONESHOT_ITER_CHECK                 \
    do {                                        \
        if (already_begined) [[unlikely]]       \
            detail::one_shot_range_log_error(); \
        already_begined = true;                 \
    } while (false)
#else
#define VSTL_ONESHOT_ITER_DECLVAR
#define VSTL_ONESHOT_ITER_CHECK ((void)0)
#endif

template<typename T>
class IteRef {
    T *ptr;

public:
    IteRef(T *ptr) noexcept : ptr(ptr) {}
    decltype(auto) operator*() const noexcept {
        return (ptr->operator*());
    }
    void operator++() noexcept {
        ptr->operator++();
    }
    void operator++(int32) noexcept {
        ptr->operator++();
    }
    bool operator==(IteEndTag tag) const noexcept {
        return ptr->operator==(tag);
    }
};

template<typename T>
class IRange {
public:
    virtual ~IRange() noexcept = default;
    virtual IteRef<IRange> begin() noexcept = 0;
    IteEndTag end() const noexcept { return {}; }
    virtual bool operator==(IteEndTag) const noexcept = 0;
    virtual void operator++() noexcept = 0;
    virtual T operator*() noexcept = 0;
};
namespace detail {
template<typename Tuple, typename Func, size_t i>
constexpr static decltype(auto) SampleTupleFunc(Tuple &&t, Func &&func) noexcept {
    return (func(std::get<i>(t)));
}
template<typename Tuple, typename Func, typename Sequencer>
struct SampleTupleFuncTable;
template<typename Tuple, typename Func, size_t... i>
struct SampleTupleFuncTable<Tuple, Func, std::integer_sequence<size_t, i...>> {
    constexpr static auto table = {SampleTupleFunc<Tuple, Func, i>...};
};

template<size_t i, typename Tuple>
static decltype(auto) range_value(Tuple &&elem) noexcept {
    if constexpr (i == 0) {
        return *std::get<0>(std::forward<Tuple>(elem));
    } else {
        using Type = std::remove_cvref_t<decltype(std::get<i>(std::forward<Tuple>(elem)))>;
        if constexpr (Type::is_filter) {
            return range_value<i - 1>(std::forward<Tuple>(elem));
        } else {
            return std::get<i>(std::forward<Tuple>(elem))(range_value<i - 1>(std::forward<Tuple>(elem)));
        }
    }
}
template<typename ValueType, typename Range>
class IRangeImpl final : public IRange<ValueType> {
    Range self;
public:
    explicit IRangeImpl(Range &&self) noexcept
        : self(std::forward<Range>(self)) {}
    ~IRangeImpl() noexcept = default;
    IteRef<IRange<ValueType>> begin() noexcept override {
        self.begin();
        return {this};
    }
    bool operator==(IteEndTag i) const noexcept override {
        return self == i;
    }
    void operator++() noexcept override {
        ++self;
    }
    ValueType operator*() noexcept override {
        return *self;
    }
};
}// namespace detail
class range {
    int64 num;
    int64 b;
    int64 e;
    int64 inc;
    VSTL_ONESHOT_ITER_DECLVAR

public:
    IteRef<range> begin() noexcept {
        VSTL_ONESHOT_ITER_CHECK;
        num = b;
        return {this};
    }
    IteEndTag end() const noexcept { return {}; }
    bool operator==(IteEndTag) const noexcept {
        return num == e;
    }
    void operator++() noexcept { num += inc; }
    int64 &operator*() noexcept {
        return num;
    }

    explicit range(int64 b, int64 e, int64 inc = 1) noexcept : b(b), e(e), inc(inc) {}
    explicit range(int64 e) noexcept : b(0), e(e), inc(1) {}
    auto i_range() && noexcept {
        return detail::IRangeImpl<int64, range>{std::move(*this)};
    }
    auto i_range() & noexcept {
        return detail::IRangeImpl<int64, range &>{*this};
    }
};
template<typename T>
class ptr_range {
    T *ptr;
    T *b;
    T *e;
    int64_t inc;
    VSTL_ONESHOT_ITER_DECLVAR

public:
    ptr_range(T *b, T *e, int64_t inc = 1) noexcept : b(b), e(e), inc(inc) {}
    ptr_range(T *b, size_t e, int64_t inc = 1) noexcept : b(b), e(b + e), inc(inc) {}
    IteEndTag end() const noexcept { return {}; }
    IteRef<ptr_range> begin() noexcept {
        VSTL_ONESHOT_ITER_CHECK;
        ptr = b;
        return {this};
    }
    bool operator==(IteEndTag) const noexcept {
        return ptr == e;
    }
    void operator++() noexcept {
        ptr += inc;
    }
    T &operator*() noexcept {
        return *ptr;
    }
    using ValueType = std::remove_cvref_t<decltype(*ptr)>;
    auto i_range() && noexcept {
        return detail::IRangeImpl<ValueType, ptr_range>{std::move(*this)};
    }
    auto i_range() & noexcept {
        return detail::IRangeImpl<ValueType, ptr_range &>{*this};
    }
};
template<typename Container>
class ite_range {
    using BeginType = decltype(std::declval<Container>().begin());
    using EndType = decltype(std::declval<Container>().end());
    Container c;
    StackObject<BeginType, false> iter;
    VSTL_ONESHOT_ITER_DECLVAR

public:
    ite_range(Container &&c) noexcept : c(std::forward<Container>(c)) {}
    ~ite_range() {
        iter.destroy();
    }
    IteEndTag end() const noexcept { return {}; }
    IteRef<ite_range> begin() noexcept {
        VSTL_ONESHOT_ITER_CHECK;
        iter.create(c.begin());
        return {this};
    }
    bool operator==(IteEndTag) const noexcept {
        return *iter == c.end();
    }
    void operator++() noexcept {
        ++(*iter);
    }
    decltype(auto) operator*() noexcept {
        return (**iter);
    }
    using ValueType = std::remove_cvref_t<decltype(**iter)>;
    auto i_range() && noexcept {
        return detail::IRangeImpl<ValueType, ite_range>{std::move(*this)};
    }
    auto i_range() & noexcept {
        return detail::IRangeImpl<ValueType, ite_range &>{*this};
    }
};
template<typename... Args>
    requires(sizeof...(Args) > 1)
class range_linker {
    using ElemType = std::tuple<Args...>;
    ElemType tp;
public:
    using ValueType = std::remove_reference_t<decltype(detail::range_value<(sizeof...(Args) - 1)>(std::declval<ElemType>()))>;
private:
    StackObject<ValueType, false> _value_holder;
    bool begined{false};
    void dispose() noexcept {
        if constexpr (!std::is_trivially_destructible_v<ValueType>) {
            if (begined) {
                _value_holder.destroy();
            }
        }
    }
    template<size_t i, typename T>
    void _next(T &&last_var) noexcept {
        bool continued{false};
        auto &&self = std::get<i>(tp);
        using Type = std::remove_cvref_t<decltype(self)>;
        if constexpr (Type::is_filter) {
            while (true) {
                auto &&last_eval = [&]() -> decltype(auto) {
                    if (continued) {
                        return (detail::range_value<i - 1>(tp));
                    } else {
                        return std::forward<T>(last_var);
                    }
                }();
                continued = !(self(last_eval));
                if (continued) {
                    ++(std::get<0>(tp));
                    if (std::get<0>(tp) == IteEndTag{}) {
                        dispose();
                        begined = false;
                        return;
                    }
                } else {
                    if constexpr (i == sizeof...(Args) - 1) {
                        dispose();
                        _value_holder.create(std::move(last_eval));
                        begined = true;
                        ++(std::get<0>(tp));
                    } else {
                        _next<i + 1>(last_eval);
                    }
                    break;
                }
            };
        } else {
            if constexpr (i == sizeof...(Args) - 1) {
                dispose();
                _value_holder.create(self(std::forward<T>(last_var)));
                begined = true;
                ++(std::get<0>(tp));
            } else {
                _next<i + 1>(self(std::forward<T>(last_var)));
            }
        }
    }
public:
    explicit range_linker(Args &&...args) noexcept
        : tp(std::forward<Args>(args)...) {}
    auto begin() noexcept {
        if (begined) [[unlikely]]
            detail::one_shot_range_log_error();
        std::get<0>(tp).begin();
        if (!(std::get<0>(tp) == IteEndTag{})) {
            _next<1>(*std::get<0>(tp));
        }
        return IteRef<range_linker>{this};
    }
    range_linker(range_linker const &) = delete;
    range_linker(range_linker &&) noexcept = default;
    ~range_linker() noexcept {
        dispose();
    }
    IteEndTag end() const noexcept { return {}; }
    bool operator==(IteEndTag i) const noexcept {
        return !begined;
    }
    auto &&operator*() noexcept {
        return std::move(*_value_holder);
    }
    void operator++() noexcept {
        if (std::get<0>(tp) == IteEndTag{}) {
            dispose();
            begined = false;
            return;
        }
        _next<1>(*std::get<0>(tp));
    }
    auto i_range() && noexcept {
        return detail::IRangeImpl<ValueType, range_linker>{std::move(*this)};
    }
    auto i_range() & noexcept {
        return detail::IRangeImpl<ValueType, range_linker &>{*this};
    }
};
template<typename T>
class filter_range {
    T &&_t;
public:
    static constexpr bool is_filter = true;
    explicit filter_range(T &&t) noexcept : _t(std::forward<T>(t)) {}
    bool operator()(auto &&v) const noexcept {
        return _t(v);
    }
};
template<typename T>
class transform_range {
    T &&_t;
public:
    static constexpr bool is_filter = false;
    explicit transform_range(T &&t) noexcept : _t(std::forward<T>(t)) {}
    decltype(auto) operator()(auto &&v) const noexcept {
        return (_t(v));
    }
};
template<typename... Ts>
class tuple_range {
    std::tuple<Ts...> ites;
    size_t index;
    using Sequencer = std::make_index_sequence<sizeof...(Ts)>;
public:
    explicit tuple_range(Ts &&...args) noexcept
        : ites(std::forward<Ts>(args)...) {}
    IteRef<tuple_range> begin() noexcept {
        auto &ite = std::get<0>(ites);
        ite.begin();
        index = 0;
        InitIndex();
        return {this};
    }
    IteEndTag end() const noexcept { return {}; }
    using ValueType = std::remove_cvref_t<decltype(*std::get<0>(ites))>;
    auto operator*() noexcept -> ValueType;
    bool operator==(vstd::IteEndTag) const noexcept {
        return index == sizeof...(Ts);
    }
    void operator++() noexcept {
        auto func = [&](auto &&ite) {
            ++ite;
        };
        detail::SampleTupleFuncTable<decltype(ites) &, decltype(func) &, Sequencer>::table.begin()[index](ites, func);
        InitIndex();
    }
    auto i_range() && noexcept {
        return detail::IRangeImpl<ValueType, tuple_range>{std::move(*this)};
    }
    auto i_range() & noexcept {
        return detail::IRangeImpl<ValueType, tuple_range &>{*this};
    }
private:
    void InitIndex() noexcept;
};
template<typename... Ts>
auto tuple_range<Ts...>::operator*() noexcept -> ValueType {
    auto func = [&](auto &&ite) -> decltype(auto) {
        return (*ite);
    };
    return detail::SampleTupleFuncTable<decltype(ites) &, decltype(func) &, Sequencer>::table.begin()[index](ites, func);
}
template<typename... Ts>
void tuple_range<Ts...>::InitIndex() noexcept {
    auto func = [&](auto &&ite) {
        return ite == vstd::IteEndTag{};
    };
    auto beginFunc = [&](auto &&ite) {
        ite.begin();
    };
    using FuncType = detail::SampleTupleFuncTable<decltype(ites) &, decltype(func) &, Sequencer>;
    using BeginFuncType = detail::SampleTupleFuncTable<decltype(ites) &, decltype(beginFunc) &, Sequencer>;
    while (true) {
        if (!FuncType::table.begin()[index](ites, func)) {
            return;
        }
        ++index;
        if (index >= sizeof...(Ts)) return;
        BeginFuncType::table.begin()[index](ites, beginFunc);
    }
}
template<luisa::concepts::iterable T>
auto make_ite_range(T &&t) noexcept {
    return ite_range<T>{std::forward<T>(t)};
}
#ifndef NDEBUG
#undef VSTL_ONESHOT_ITER_DECLVAR
#undef VSTL_ONESHOT_ITER_CHECK
#endif
}// namespace vstd
