#pragma once

#include <tuple>
#include <luisa/vstl/meta_lib.h>
#ifndef NDEBUG
#include <luisa/core/logging.h>
#endif

namespace vstd {
#ifndef NDEBUG
#define VSTL_ONESHOT_ITER_DECLVAR bool already_begined{false};
#define VSTL_ONESHOT_ITER_CHECK                                                                 \
    do {                                                                                        \
        if (already_begined) [[unlikely]]                                                       \
            LUISA_ERROR("This one-shot iterator has already been used, please don't do this."); \
        already_begined = true;                                                                 \
    } while (false)
#else
#define VSTL_ONESHOT_ITER_DECLVAR
#define VSTL_ONESHOT_ITER_CHECK ((void)0)
#endif

template<typename T>
class IteRef {
    T *ptr;

public:
    IteRef(T *ptr) : ptr(ptr) {}
    decltype(auto) operator*() const {
        return (ptr->operator*());
    }
    void operator++() {
        ptr->operator++();
    }
    void operator++(int32) {
        ptr->operator++();
    }
    bool operator==(IteEndTag tag) const {
        return ptr->operator==(tag);
    }
};

template<typename T>
class IRange {
public:
    virtual ~IRange() = default;
    virtual IteRef<IRange> begin() = 0;
    IteEndTag end() const { return {}; }
    virtual bool operator==(IteEndTag) const = 0;
    virtual void operator++() = 0;
    virtual T operator*() = 0;
};
namespace detail {
template<typename Tuple, typename Func, size_t i>
constexpr static decltype(auto) SampleTupleFunc(Tuple &&t, Func &&func) {
    return (func(std::get<i>(t)));
}
template<typename Tuple, typename Func, typename Sequencer>
struct SampleTupleFuncTable;
template<typename Tuple, typename Func, size_t... i>
struct SampleTupleFuncTable<Tuple, Func, std::integer_sequence<size_t, i...>> {
    constexpr static auto table = {SampleTupleFunc<Tuple, Func, i>...};
};

template<size_t i, typename Tuple>
static decltype(auto) range_value(Tuple &&elem) {
    if constexpr (i == 0) {
        return (*std::get<0>(std::forward<Tuple>(elem)));
    } else {
        using Type = std::remove_cvref_t<decltype(std::get<i>(std::forward<Tuple>(elem)))>;
        if constexpr (Type::is_filter) {
            return (range_value<i - 1>(std::forward<Tuple>(elem)));
        } else {
            return (std::get<i>(std::forward<Tuple>(elem))(range_value<i - 1>(std::forward<Tuple>(elem))));
        }
    }
}
template<typename ValueType, typename Range>
class IRangeImpl final : public IRange<ValueType> {
    Range self;
public:
    explicit IRangeImpl(Range &&self)
        : self(std::forward<Range>(self)) {}
    ~IRangeImpl() = default;
    IteRef<IRange<ValueType>> begin() override {
        self.begin();
        return {this};
    }
    bool operator==(IteEndTag i) const override {
        return self == i;
    }
    void operator++() override {
        ++self;
    }
    ValueType operator*() override {
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
    IteRef<range> begin() {
        VSTL_ONESHOT_ITER_CHECK;
        num = b;
        return {this};
    }
    IteEndTag end() const { return {}; }
    bool operator==(IteEndTag) const {
        return num == e;
    }
    void operator++() { num += inc; }
    int64 &operator*() {
        return num;
    }

    range(int64 b, int64 e, int64 inc = 1) : b(b), e(e), inc(inc) {}
    range(int64 e) : b(0), e(e), inc(1) {}
    auto i_range() && {
        return detail::IRangeImpl<int64, range>{std::move(*this)};
    }
    auto i_range() & {
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
    ptr_range(T *b, T *e, int64_t inc = 1) : b(b), e(e), inc(inc) {}
    ptr_range(T *b, size_t e, int64_t inc = 1) : b(b), e(b + e), inc(inc) {}
    IteEndTag end() const { return {}; }
    IteRef<ptr_range> begin() {
        VSTL_ONESHOT_ITER_CHECK;
        ptr = b;
        return {this};
    }
    bool operator==(IteEndTag) const {
        return ptr == e;
    }
    void operator++() {
        ptr += inc;
    }
    T &operator*() {
        return *ptr;
    }
    using ValueType = std::remove_cvref_t<decltype(*ptr)>;
    auto i_range() && {
        return detail::IRangeImpl<ValueType, ptr_range>{std::move(*this)};
    }
    auto i_range() & {
        return detail::IRangeImpl<ValueType, ptr_range &>{*this};
    }
};
template<typename T, typename E>
    requires(std::is_trivially_destructible_v<T>)
class ite_range {
    StackObject<T, false> ptr;
    T b;
    E e;
    VSTL_ONESHOT_ITER_DECLVAR

public:
    ite_range(T &&b, E &&e) : b(std::forward<T>(b)), e(std::forward<E>(e)) {}
    IteEndTag end() const { return {}; }
    IteRef<ite_range> begin() {
        VSTL_ONESHOT_ITER_CHECK;
        ptr.create(b);
        return {this};
    }
    bool operator==(IteEndTag) const {
        return *ptr == e;
    }
    void operator++() {
        ++(*ptr);
    }
    decltype(auto) operator*() {
        return (**ptr);
    }
    using ValueType = std::remove_cvref_t<decltype(**ptr)>;
    auto i_range() && {
        return detail::IRangeImpl<ValueType, ite_range>{std::move(*this)};
    }
    auto i_range() & {
        return detail::IRangeImpl<ValueType, ite_range &>{*this};
    }
};
template<typename... Args>
    requires(sizeof...(Args) > 1)
class range_linker {
    using ElemType = std::tuple<Args...>;
    ElemType tp;

    using IterType = decltype(std::get<0>(std::declval<std::tuple<Args...>>()).begin());
public:
    using ValueType = std::remove_reference_t<decltype(detail::range_value<(sizeof...(Args) - 1)>(std::declval<ElemType>()))>;
private:
    Storage<ValueType> _value_holder;
    bool begined{false};
    void dispose() {
        if constexpr (!std::is_reference_v<ValueType> && !std::is_trivially_destructible_v<ValueType>) {
            if (begined) {
                reinterpret_cast<ValueType *>(_value_holder.c)->~ValueType();
            }
        }
    }
    template<size_t i, typename T>
    void _next(T &&last_var) {
        using RetType = decltype(detail::range_value<i>(tp));
        bool continued{false};
        auto &&self = std::get<i>(tp);
        using Type = std::remove_cvref_t<decltype(self)>;
        while (true) {
            auto &&last_eval = [&]() -> decltype(auto) {
                if (continued) {
                    return (detail::range_value<i - 1>(tp));
                } else {
                    return (last_var);
                }
            }();
            if constexpr (Type::is_filter) {
                continued = !(self(last_eval));
            } else {
                continued = false;
            }
            if (continued) {
                ++(std::get<0>(tp));
            } else {
                if constexpr (i == sizeof...(Args) - 1) {
                    dispose();
                    if constexpr (Type::is_filter) {
                        new (std::launder(_value_holder.c)) ValueType(std::move(last_eval));
                    } else {
                        new (std::launder(_value_holder.c)) ValueType(self(last_eval));
                    }
                    begined = true;
                    ++(std::get<0>(tp));
                } else {
                    if constexpr (Type::is_filter) {
                        _next<i + 1>(last_eval);
                    } else {
                        _next<i + 1>(self(last_eval));
                    }
                }
                break;
            }
        };
    }
public:
    range_linker(Args &&...args)
        : tp(std::forward<Args>(args)...) {}
    auto begin() {
        if (begined) [[unlikely]]
#ifndef NDEBUG
            LUISA_ERROR("This one-shot iterator has already been used, please don't do this.");
#else
            std::abort();
#endif
        std::get<0>(tp).begin();
        if (!(std::get<0>(tp) == IteEndTag{})) {
            _next<1>(*std::get<0>(tp));
        }
        return IteRef<range_linker>{this};
    }
    range_linker(range_linker const &) = delete;
    range_linker(range_linker &&) = default;
    ~range_linker() {
        dispose();
    }
    IteEndTag end() const { return {}; }
    bool operator==(IteEndTag i) const {
        return !begined;
    }
    auto &&operator*() {
        return reinterpret_cast<ValueType &>(_value_holder);
    }
    void operator++() {
        if (std::get<0>(tp) == IteEndTag{}) {
            if constexpr (!std::is_reference_v<ValueType> && !std::is_trivially_destructible_v<ValueType>) {
                if (begined) {
                    reinterpret_cast<ValueType &>(_value_holder).~ValueType();
                }
            }
            begined = false;
            return;
        }
        _next<1>(*std::get<0>(tp));
    }
    auto i_range() && {
        return detail::IRangeImpl<ValueType, range_linker>{std::move(*this)};
    }
    auto i_range() & {
        return detail::IRangeImpl<ValueType, range_linker &>{*this};
    }
};
template<typename T>
class filter_range {
    T &&_t;
public:
    static constexpr bool is_filter = true;
    filter_range(T &&t) : _t(std::forward<T>(t)) {}
    bool operator()(auto &&v) const {
        return _t(v);
    }
};
template<typename T>
class transform_range {
    T &&_t;
public:
    static constexpr bool is_filter = false;
    transform_range(T &&t) : _t(std::forward<T>(t)) {}
    decltype(auto) operator()(auto &&v) const {
        return (_t(v));
    }
};
template<typename... Ts>
class tuple_range {
    std::tuple<Ts...> ites;
    size_t index;
    using Sequencer = std::make_index_sequence<sizeof...(Ts)>;
public:
    tuple_range(Ts &&...args)
        : ites(std::forward<Ts>(args)...) {}
    IteRef<tuple_range> begin() {
        auto &ite = std::get<0>(ites);
        ite.begin();
        index = 0;
        InitIndex();
        return {this};
    }
    IteEndTag end() const { return {}; }
    using ValueType = std::remove_cvref_t<decltype(*std::get<0>(ites))>;
    auto operator*() -> ValueType;
    bool operator==(vstd::IteEndTag) const {
        return index == sizeof...(Ts);
    }
    void operator++() {
        auto func = [&](auto &&ite) {
            ++ite;
        };
        detail::SampleTupleFuncTable<decltype(ites) &, decltype(func) &, Sequencer>::table.begin()[index](ites, func);
        InitIndex();
    }
    auto i_range() && {
        return detail::IRangeImpl<ValueType, tuple_range>{std::move(*this)};
    }
    auto i_range() & {
        return detail::IRangeImpl<ValueType, tuple_range &>{*this};
    }
private:
    void InitIndex();
};
template<typename... Ts>
auto tuple_range<Ts...>::operator*() -> ValueType {
    auto func = [&](auto &&ite) -> decltype(auto) {
        return (*ite);
    };
    return detail::SampleTupleFuncTable<decltype(ites) &, decltype(func) &, Sequencer>::table.begin()[index](ites, func);
}
template<typename... Ts>
void tuple_range<Ts...>::InitIndex() {
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
#ifndef NDEBUG
#undef VSTL_ONESHOT_ITER_DECLVAR
#undef VSTL_ONESHOT_ITER_CHECK
#endif
}// namespace vstd
