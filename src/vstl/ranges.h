#pragma once
#include "meta_lib.h"
#include "tuple.h"
namespace vstd {
template<typename T>
class IteRef {
    T *ptr;

public:
    IteRef(T *ptr) : ptr(ptr) {}
    decltype(auto) operator*() const {
        return ptr->operator*();
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
    bool operator!=(IteEndTag tag) const { return !operator==(tag); }
};
namespace detail {
template<typename Ite, typename Builder>
class Combiner;
template<typename LeftBD, typename RightBD>
class BuilderCombiner;
template<typename Ite>
struct BuilderFlag : public IOperatorNewBase {
    static constexpr bool vstdRangeBuilder = true;
    template<typename Dst>
        requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
    decltype(auto) operator|(Dst &&dst) &;
    template<typename Dst>
        requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
    decltype(auto) operator|(Dst &&dst) &&;
};
template<typename Ite>
struct RangeFlag : public IOperatorNewBase {
    static constexpr bool vstdRange = true;
    IteEndTag end() const { return {}; }
    template<typename Dst>
        requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
    decltype(auto) operator|(Dst &&dst) &;
    template<typename Dst>
        requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
    decltype(auto) operator|(Dst &&dst) &&;
};

template<typename Ite, typename Builder>
class Combiner : public RangeFlag<Combiner<Ite, Builder>> {
    Ite ite;
    Builder builder;

public:
    Combiner(Ite &&ite, Builder &&builder)
        : ite(std::forward<Ite>(ite)), builder(std::forward<Builder>(builder)) {}
    IteRef<Combiner> begin() {
        builder.begin(ite);
        return {this};
    }
    void operator++() {
        builder.next(ite);
    }
    decltype(auto) operator*() {
        return builder.value(ite);
    }
    bool operator==(IteEndTag) const {
        return builder.is_end(ite);
    }
    bool operator!=(IteEndTag tag) const { return !operator==(tag); }
};
template<typename T, typename Ite>
struct BuilderHolder : RangeFlag<BuilderHolder<T, Ite>> {
    T &t;
    Ite &ite;
    BuilderHolder(T &t, Ite &ite)
        : t(t), ite(ite) {}
    void begin() {
        t.begin(ite);
    }
    void operator++() {
        t.next(ite);
    }
    decltype(auto) operator*() {
        return t.value(ite);
    }
    bool operator==(IteEndTag) const {
        return t.is_end(ite);
    }
};

template<typename LeftBD, typename RightBD>
class BuilderCombiner : public BuilderFlag<BuilderCombiner<LeftBD, RightBD>> {
    LeftBD left;
    RightBD right;

public:
    BuilderCombiner(LeftBD &&left, RightBD &&right)
        : left(std::forward<LeftBD>(left)),
          right(std::forward<RightBD>(right)) {}

    template<typename Ite>
    void begin(Ite &&ite) {
        right.begin(BuilderHolder<LeftBD, Ite &>(left, ite));
    }
    template<typename Ite>
    bool is_end(Ite &&ite) const {
        return right.is_end(BuilderHolder<LeftBD const, Ite &>(left, ite));
    }
    template<typename Ite>
    void next(Ite &&ite) {
        right.next(BuilderHolder<LeftBD, Ite &>(left, ite));
    }
    template<typename Ite>
    auto value(Ite &&ite) {
        return right.value(BuilderHolder<LeftBD, Ite &>(left, ite));
    }
};

template<typename Tuple, typename Func, size_t i>
constexpr static decltype(auto) SampleTupleFunc(Tuple &&t, Func &&func) {
    return func(t.template get<i>());
}
template<typename Tuple, typename Func, typename Sequencer>
struct SampleTupleFuncTable;
template<typename Tuple, typename Func, size_t... i>
struct SampleTupleFuncTable<Tuple, Func, std::integer_sequence<size_t, i...>> {
    constexpr static auto table = {SampleTupleFunc<Tuple, Func, i>...};
};
template<typename Ite>
template<typename Dst>
    requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
inline decltype(auto) RangeFlag<Ite>::operator|(Dst &&dst) & {
    return Combiner<Ite &, Dst>(static_cast<Ite &>(*this), std::forward<Dst>(dst));
}
template<typename Ite>
template<typename Dst>
    requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
inline decltype(auto) RangeFlag<Ite>::operator|(Dst &&dst) && {
    return Combiner<Ite, Dst>(static_cast<Ite &&>(*this), std::forward<Dst>(dst));
}
template<typename Ite>
template<typename Dst>
    requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
inline decltype(auto) BuilderFlag<Ite>::operator|(Dst &&dst) & {
    return BuilderCombiner<Ite, Dst>(static_cast<Ite &>(*this), std::forward<Dst>(dst));
}
template<typename Ite>
template<typename Dst>
    requires(std::remove_cvref_t<Dst>::vstdRangeBuilder)
inline decltype(auto) BuilderFlag<Ite>::operator|(Dst &&dst) && {
    return BuilderCombiner<Ite, Dst>(static_cast<Ite &&>(*this), std::forward<Dst>(dst));
}
}// namespace detail
template<typename T>
class IRange : public detail::RangeFlag<IRange<T>> {
public:
    virtual ~IRange() = default;
    virtual IteRef<IRange> begin() = 0;
    virtual bool operator==(IteEndTag) const = 0;
    virtual void operator++() = 0;
    virtual T operator*() = 0;
};
template<typename Ite>
class v_RangeImpl : public IRange<decltype(*std::declval<Ite>())> {
    using Value = decltype(*std::declval<Ite>());
    Ite ptr;

public:
    v_RangeImpl(Ite &&ptr) : ptr(std::forward<Ite>(ptr)) {}
    IteRef<IRange<Value>> begin() override {
        ptr.begin();
        return {this};
    }
    bool operator==(IteEndTag t) const override { return ptr == t; }
    void operator++() override {
        ++ptr;
    }
    Value operator*() override {
        return *ptr;
    }
    v_RangeImpl(v_RangeImpl const &) = delete;
    v_RangeImpl(v_RangeImpl &&) = default;
};
template<typename T>
class IRangePipeline : public detail::BuilderFlag<IRangePipeline<T>> {
public:
    virtual ~IRangePipeline() = default;
    virtual void begin(IRange<T> &range) = 0;
    virtual bool is_end(IRange<T> &range) const = 0;
    virtual void next(IRange<T> &range) = 0;
    virtual T value(IRange<T> &range) = 0;
};
template<typename T, typename Ite>
class v_IRangePipelineImpl : public IRangePipeline<T> {
    Ite ite;

public:
    v_IRangePipelineImpl(Ite &&ite) : ite(std::forward<Ite>(ite)) {}
    void begin(IRange<T> &range) override { ite.begin(range); }
    bool is_end(IRange<T> &range) const override { return ite.is_end(range); }
    void next(IRange<T> &range) override { ite.next(range); }
    T value(IRange<T> &range) override { return ite.value(range); }
};
class ValueRange : public detail::BuilderFlag<ValueRange> {
public:
    template<typename Ite>
    void begin(Ite &&ite) { ite.begin(); }
    template<typename Ite>
    bool is_end(Ite &&ite) const { return ite == IteEndTag{}; }
    template<typename Ite>
    void next(Ite &&ite) { ++ite; }
    template<typename Ite>
    auto value(Ite &&ite) { return *ite; }
};
template<typename FilterFunc>
class v_FilterRange : public detail::BuilderFlag<v_FilterRange<FilterFunc>> {
private:
    FilterFunc func;
    template<typename Ite>
    void GetNext(Ite &&ite) {
        while (ite != IteEndTag{}) {
            if (func(*ite)) {
                return;
            }
            ++ite;
        }
    }

public:
    template<typename Ite>
    void begin(Ite &&ite) {
        ite.begin();
        GetNext(ite);
    }
    template<typename Ite>
    bool is_end(Ite &&ite) const {
        return ite == IteEndTag{};
    }
    template<typename Ite>
    void next(Ite &&ite) {
        ++ite;
        GetNext(ite);
    }
    template<typename Ite>
    decltype(auto) value(Ite &&ite) {
        return *ite;
    }
    v_FilterRange(FilterFunc &&func)
        : func(std::forward<FilterFunc>(func)) {
    }
};
template<typename GetValue>
class v_TransformRange : public detail::BuilderFlag<v_TransformRange<GetValue>> {
    GetValue getValue;

public:
    template<typename Ite>
    void begin(Ite &&ite) {
        ite.begin();
    }
    v_TransformRange(GetValue &&getValueFunc)
        : getValue(std::forward<GetValue>(getValueFunc)) {}
    template<typename Ite>
    bool is_end(Ite &&ite) const {
        return ite == IteEndTag{};
    }
    template<typename Ite>
    void next(Ite &&ite) {
        ++ite;
    }
    template<typename Ite>
    decltype(auto) value(Ite &&ite) {
        return getValue(*ite);
    }
};
class RemoveCVRefRange : public detail::BuilderFlag<RemoveCVRefRange> {
public:
    template<typename Ite>
    void begin(Ite &&ite) { ite.begin(); }
    template<typename Ite>
    bool is_end(Ite &&ite) const { return ite == IteEndTag{}; }
    template<typename Ite>
    void next(Ite &&ite) { ++ite; }
    template<typename Ite>
    decltype(auto) value(Ite &&ite) { return static_cast<std::remove_cvref_t<decltype(*ite)>>(*ite); }
    RemoveCVRefRange() {}
};
template<typename Dst>
class StaticCastRange : public detail::BuilderFlag<StaticCastRange<Dst>> {
public:
    template<typename Ite>
    void begin(Ite &&ite) { ite.begin(); }
    template<typename Ite>
    bool is_end(Ite &&ite) const { return ite == IteEndTag{}; }
    template<typename Ite>
    void next(Ite &&ite) { ++ite; }
    template<typename Ite>
    Dst value(Ite &&ite) { return static_cast<Dst>(*ite); }
    StaticCastRange() {}
};
template<typename Dst>
class ReinterpretCastRange : public detail::BuilderFlag<ReinterpretCastRange<Dst>> {
public:
    template<typename Ite>
    void begin(Ite &&ite) { ite.begin(); }
    template<typename Ite>
    bool is_end(Ite &&ite) const { return ite == IteEndTag{}; }
    template<typename Ite>
    void next(Ite &&ite) { ++ite; }
    template<typename Ite>
    Dst value(Ite &&ite) { return reinterpret_cast<Dst>(*ite); }
    ReinterpretCastRange() {}
};
template<typename Dst>
class ConstCastRange : public detail::BuilderFlag<ConstCastRange<Dst>> {
public:
    template<typename Ite>
    void begin(Ite &&ite) { ite.begin(); }
    template<typename Ite>
    bool is_end(Ite &&ite) const { return ite == IteEndTag{}; }
    template<typename Ite>
    void next(Ite &&ite) { ++ite; }
    template<typename Ite>
    Dst value(Ite &&ite) { return const_cast<Dst>(*ite); }
    ConstCastRange() {}
};

template<typename Map>
class v_CacheEndRange : public detail::RangeFlag<v_CacheEndRange<Map>> {
public:
    using IteBegin = decltype(std::declval<Map>().begin());
    using IteEnd = decltype(std::declval<Map>().begin());

private:
    Map map;
    optional<IteBegin> ite;

public:
    v_CacheEndRange(Map &&map)
        : map(std::forward<Map>(map)) {
    }
    IteRef<v_CacheEndRange> begin() {
        ite = map.begin();
        return {this};
    }
    bool operator==(IteEndTag) const {
        return (*ite) == map.end();
    }
    void operator++() { ++(*ite); }
    decltype(auto) operator*() {
        return **ite;
    }
};
class range : public detail::RangeFlag<range> {
    int64 num;
    int64 b;
    int64 e;
    int64 inc;

public:
    IteRef<range> begin() {
        num = b;
        return {this};
    }
    bool operator==(IteEndTag) const {
        return num == e;
    }
    void operator++() { num += inc; }
    int64 &operator*() {
        return num;
    }

    range(int64 b, int64 e, int64 inc = 1) : b(b), e(e), inc(inc) {}
    range(int64 e) : b(0), e(e), inc(1) {}
};
template<typename T>
class ptr_range : public detail::RangeFlag<ptr_range<T>> {
    T *ptr;
    T *b;
    T *e;
    int64_t inc;

public:
    ptr_range(T *b, T *e, int64_t inc = 1) : b(b), e(e), inc(inc) {}
    ptr_range(T *b, size_t e, int64_t inc = 1) : b(b), e(b + e), inc(inc) {}
    IteRef<ptr_range> begin() {
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
};
template<typename T, typename E>
class ite_range : public detail::RangeFlag<ite_range<T, E>> {
    Storage<T> ptr;
    T b;
    E e;
    bool begined{false};
    T *Ptr() { return reinterpret_cast<T *>(&ptr); }
    T const *Ptr() const { return reinterpret_cast<T const *>(&ptr); }

public:
    ite_range(T &&b, E &&e) : b(std::forward<T>(b)), e(std::forward<E>(e)) {}
    ~ite_range() {
        if (begined) { destruct(Ptr()); }
    }
    IteRef<ite_range> begin() {
        if (begined) { destruct(Ptr()); }
        new (Ptr()) T(std::forward<T>(b));
        return {this};
    }
    bool operator==(IteEndTag) const {
        return (*Ptr()) == e;
    }
    void operator++() {
        ++(*Ptr());
    }
    decltype(auto) operator*() {
        return **Ptr();
    }
};
template<typename... Ts>
struct v_TupleIterator : public detail::RangeFlag<v_TupleIterator<Ts...>> {
    vstd::tuple<Ts...> ites;
    size_t index;
    using Sequencer = std::make_index_sequence<sizeof...(Ts)>;
    template<typename... TTs>
    v_TupleIterator(TTs &&...args)
        : ites(std::forward<Ts>(args)...) {}
    IteRef<v_TupleIterator> begin() {
        auto &ite = ites.template get<0>();
        ite.begin();
        index = 0;
        InitIndex();
        return {this};
    }
    decltype(auto) operator*() {
        auto func = [&](auto &&ite) {
            return *ite;
        };
        return detail::SampleTupleFuncTable<decltype(ites) &, decltype(func) &, Sequencer>::table.begin()[index](ites, func);
    }
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

private:
    void InitIndex() {
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
};

template<typename A, typename B>
struct v_PairIterator : public detail::RangeFlag<v_PairIterator<A, B>> {
    A a;
    B b;
    bool ite;
    v_PairIterator(
        A &&a,
        B &&b) : a(std::forward<A>(a)), b(std::forward<B>(b)) {}
    IteRef<v_PairIterator> begin() {
        a.begin();
        if (a == vstd::IteEndTag{}) {
            b.begin();
            ite = true;
        } else {
            ite = false;
        }
        return {this};
    }
    decltype(auto) operator*() {
        if (ite)
            return *b;
        else
            return *a;
    }
    bool operator==(vstd::IteEndTag t) const {
        return ite && b == t;
    }
    void operator++() {
        if (ite) {
            ++b;
        } else {
            ++a;
            if (a == vstd::IteEndTag{}) {
                b.begin();
                ite = true;
            }
        }
    }
};

template<typename Func>
v_FilterRange<Func> FilterRange(Func &&func) {
    return {std::forward<Func>(func)};
}
template<typename GetValue>
v_TransformRange<GetValue> TransformRange(GetValue &&func) {
    return {std::forward<GetValue>(func)};
}

template<typename Map>
    requires((AlwaysTrue<decltype(*(std::declval<Map>().begin()))>)&&(AlwaysTrue<decltype(++decl_lvalue(std::declval<Map>().begin()))>)&&std::is_same_v<bool, decltype((std::declval<Map>().begin()) == (std::declval<Map>().end()))>)
v_CacheEndRange<Map> CacheEndRange(Map &&map) {
    return {std::forward<Map>(map)};
}
template<typename Map>
    requires(std::remove_cvref_t<Map>::vstdRange)
v_RangeImpl<Map> RangeImpl(Map &&map) {
    return {std::forward<Map>(map)};
}
template<typename Map>
    requires(std::remove_cvref_t<Map>::vstdRange)
v_RangeImpl<Map>
    *NewRangeImpl(Map &&map) {
    return new v_RangeImpl<Map>{std::forward<Map>(map)};
}

template<typename Dst, typename Func>
auto IRangePipelineImpl(Func &&func) -> v_IRangePipelineImpl<Dst, Func&&> {
    return {std::forward<Func&&>(func)};
}
template<typename... Ts>
v_TupleIterator<Ts...> TupleIterator(Ts &&...ts) {
    return {std::forward<Ts>(ts)...};
}
template<typename A, typename B>
v_PairIterator<A, B> PairIterator(
    A &&a,
    B &&b) {
    return {std::forward<A>(a), std::forward<B>(b)};
}
}// namespace vstd