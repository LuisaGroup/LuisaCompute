//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <dsl/var.h>
#include <dsl/operators.h>

namespace luisa::compute {

namespace detail {

/// Build if statement
class IfStmtBuilder {

private:
    IfStmt *_stmt;

public:
    /// Construct if statement with condition expression
    explicit IfStmtBuilder(Expr<bool> condition) noexcept
        : _stmt{FunctionBuilder::current()->if_(condition.expression())} {}

    /// Add statement to false branch. f is a function.
    template<typename False>
    void else_(False &&f) &&noexcept {
        FunctionBuilder::current()->with(_stmt->false_branch(), std::forward<False>(f));
    }

    /// Add statement to true branch. t is a function. Return this.
    template<typename True>
    auto operator%(True &&t) &&noexcept {
        FunctionBuilder::current()->with(_stmt->true_branch(), std::forward<True>(t));
        return *this;
    }

    //    template<typename Body>
    //    [[nodiscard]] auto elif (Expr<bool> condition, Body &&body) &&noexcept {
    //        return FunctionBuilder::current()->with(
    //                   _stmt->false_branch(),
    //                   [condition] {
    //                       return IfStmtBuilder{condition};
    //                   }) %
    //               std::forward<Body>(body);
    //    }

    /// Add statement to false branch. Same as else_ but no return.
    template<typename False>
    void operator/(False &&f) &&noexcept {
        IfStmtBuilder{*this}.else_(std::forward<False>(f));
    }

    /// Add else-if branch. Return builder of else-if statement.
    template<typename LazyElIfCond>
    [[nodiscard]] auto operator*(LazyElIfCond &&elif_cond) &&noexcept {
        return FunctionBuilder::current()->with(_stmt->false_branch(), [&elif_cond] {
            return IfStmtBuilder{elif_cond()};
        });
    }
};

/// Build loop statement
class LoopStmtBuilder {

private:
    LoopStmt *_stmt;

public:
    LoopStmtBuilder() noexcept : _stmt{FunctionBuilder::current()->loop_()} {}

    /// Add body statement. body is a function. Return this.
    template<typename Body>
    auto operator/(Body &&body) &&noexcept {
        FunctionBuilder::current()->with(
            _stmt->body(), std::forward<Body>(body));
        return *this;
    }

    /// Add body statement. Same as operator/, but no return.
    template<typename Body>
    void operator%(Body &&body) &&noexcept {
        LoopStmtBuilder{*this} / std::forward<Body>(body);
    }
};

/// Build case statement of switch
class SwitchCaseStmtBuilder {

private:
    SwitchCaseStmt *_stmt;

public:
    /// Constrct from integral. Equals to "case c:"
    template<concepts::integral T>
    explicit SwitchCaseStmtBuilder(T c) noexcept
        : _stmt{FunctionBuilder::current()->case_(extract_expression(c))} {}

    /// Add body of case statement. Will automatically add break at the end.
    template<typename Body>
    void operator%(Body &&body) &&noexcept {
        FunctionBuilder::current()->with(_stmt->body(), [&body] {
            body();
            FunctionBuilder::current()->break_();
        });
    }
};

/// Build default statement of switch
class SwitchDefaultStmtBuilder {

private:
    SwitchDefaultStmt *_stmt;

public:
    SwitchDefaultStmtBuilder() noexcept
        : _stmt{FunctionBuilder::current()->default_()} {}

    /// Add body of default statement. Will automatically add break at the end.
    template<typename Body>
    void operator%(Body &&body) &&noexcept {
        FunctionBuilder::current()->with(_stmt->body(), [&body] {
            body();
            FunctionBuilder::current()->break_();
        });
    }
};

/// Build switch statement
class SwitchStmtBuilder {

private:
    SwitchStmt *_stmt;

public:
    /// Construct from integral expression as condition.
    template<typename T>
        requires is_integral_expr_v<T>
    explicit SwitchStmtBuilder(T &&cond) noexcept
        : _stmt{FunctionBuilder::current()->switch_(
              extract_expression(std::forward<T>(cond)))} {}

    /// Add case statement. Return this
    template<typename T, typename Body>
    auto case_(T &&case_cond, Body &&case_body) &&noexcept {
        FunctionBuilder::current()->with(_stmt->body(), [&case_cond, &case_body] {
            SwitchCaseStmtBuilder{case_cond} % std::forward<Body>(case_body);
        });
        return *this;
    }

    /// Add default statement.
    template<typename Default>
    auto default_(Default &&d) &&noexcept {
        FunctionBuilder::current()->with(_stmt->body(), [&d] {
            SwitchDefaultStmtBuilder{} % std::forward<Default>(d);
        });
    }

    /// Set body
    template<typename Body>
    void operator%(Body &&body) &&noexcept {
        FunctionBuilder::current()->with(_stmt->body(), std::forward<Body>(body));
    }
};

/// Invoke for statement body
struct ForStmtBodyInvoke {
    template<typename F>
    void operator%(F &&body) &&noexcept {
        std::invoke(std::forward<F>(body));
    }
};

/// Build for range statement
template<typename T, bool has_step>
class ForRange {

    static_assert(is_integral_v<T>);

public:
    struct ForRangeEnd {};

    /// Iterator of for range statement
    class ForRangeIter {
    private:
        Expr<T> _begin;
        Expr<T> _end;
        Expr<T> _step;
        const Expression *_var{nullptr};
        const Expression *_cond{nullptr};
        ForStmt *_stmt{nullptr};
        uint _time{0u};

    public:
        /// Construct for range iterator from begin, end and step expression
        explicit ForRangeIter(Expr<T> begin, Expr<T> end, Expr<T> step) noexcept
            : _begin{begin}, _end{end}, _step{step} {}

        /// Construct range and condition for for statement
        [[nodiscard]] auto operator*() noexcept {
            if (_time != 0u) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid RangeForIter state (with _time = {}).", _time);
            }
            auto f = FunctionBuilder::current();
            Var var{_begin};
            _var = var.expression();
            auto bool_type = Type::of<bool>();
            _cond = f->binary(bool_type, BinaryOp::LESS, _var, _end.expression());
            if constexpr (has_step) {
                // step < 0
                auto neg_step = f->binary(bool_type, BinaryOp::LESS, _step.expression(), f->literal(Type::of<T>(), T{0}));
                // ((step < 0) && !(var < end)) || (!(step < 0) && (var < end))
                _cond = f->binary(bool_type, BinaryOp::BIT_XOR, _cond, neg_step);
            }
            _stmt = f->for_(_var, _cond, _step.expression());
            f->push_scope(_stmt->body());
            return Var{std::move(var)};// to guarantee rvo
        }

        auto &operator++() noexcept {
            _time++;
            FunctionBuilder::current()->pop_scope(_stmt->body());
            return *this;
        }

        [[nodiscard]] auto operator!=(ForRangeEnd) const noexcept {
            return _time == 0u;
        }
    };

private:
    Expr<T> _begin;
    Expr<T> _end;
    Expr<T> _step;

public:
    explicit ForRange(Expr<T> begin, Expr<T> end, Expr<T> step) noexcept
        : _begin{begin}, _end{end}, _step{step} {}
    [[nodiscard]] auto begin() const noexcept { return ForRangeIter{_begin, _end, _step}; }
    [[nodiscard]] auto end() const noexcept { return ForRangeEnd{}; }
};

// FIXME: review this...
template<typename Lhs, typename Rhs, size_t... i>
inline void assign_impl(Ref<Lhs> lhs, Expr<Rhs> rhs, std::index_sequence<i...>) noexcept {
    (dsl::assign(lhs.template get<i>(), rhs.template get<i>()), ...);
}

template<typename Lhs, typename Rhs>
inline void assign_impl(Ref<Lhs> lhs, Expr<Rhs> rhs) noexcept {
    using member_tuple = struct_member_tuple_t<expr_value_t<Lhs>>;
    assign_impl(lhs, rhs, std::make_index_sequence<std::tuple_size_v<member_tuple>>{});
}

}// namespace detail

inline namespace dsl {

template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept {
    static_assert(
        std::tuple_size_v<linear_layout_t<expr_value_t<Lhs>>> ==
        std::tuple_size_v<linear_layout_t<expr_value_t<Rhs>>>);
    if constexpr (concepts::assignable<expr_value_t<Lhs>, expr_value_t<Rhs>>) {
        detail::FunctionBuilder::current()->assign(
            detail::extract_expression(std::forward<Lhs>(lhs)),
            detail::extract_expression(std::forward<Rhs>(rhs)));
    } else if (is_tuple_v<std::remove_cvref_t<Rhs>>) {
        assign(
            detail::Ref{std::forward<Lhs>(lhs)},
            compose(std::forward<Rhs>(rhs)));
    } else {
        detail::assign_impl(
            detail::Ref{std::forward<Lhs>(lhs)},
            Expr{std::forward<Rhs>(rhs)});
    }
}

// statements
/// break
inline void break_() noexcept { detail::FunctionBuilder::current()->break_(); }
/// continue
inline void continue_() noexcept { detail::FunctionBuilder::current()->continue_(); }

/// if condition then t
template<typename True>
inline auto if_(Expr<bool> condition, True &&t) noexcept {
    return detail::IfStmtBuilder{condition} % std::forward<True>(t);
}

/// while(1) body
template<typename Body>
inline void loop(Body &&body) noexcept {
    detail::LoopStmtBuilder{} % std::forward<Body>(body);
}

/// switch(expr). Return SwitchStmtBuilder
template<typename T>
inline auto switch_(T &&expr) noexcept {
    return detail::SwitchStmtBuilder{std::forward<T>(expr)};
}

/// range from 0 to end, step = 1
template<typename Te>
    requires is_integral_expr_v<Te>
[[nodiscard]] inline auto range(Te &&end) noexcept {
    using T = expr_value_t<Te>;
    Var e{std::forward<Te>(end)};
    return detail::ForRange<T, false>{static_cast<T>(0), e, static_cast<T>(1)};
}

/// range from begin to end, step = 1
template<typename Tb, typename Te>
    requires is_same_expr_v<Tb, Te> && is_integral_expr_v<Tb>
[[nodiscard]] inline auto range(Tb &&begin, Te &&end) noexcept {
    using T = expr_value_t<Tb>;
    Var e{std::forward<Te>(end)};
    return detail::ForRange<T, false>{std::forward<Tb>(begin), e, static_cast<T>(1)};
}

/// range from 0 to end, step = step
template<typename Tb, typename Te, typename Ts>
    requires is_same_expr_v<Tb, Te, Ts> && is_integral_expr_v<Tb>
[[nodiscard]] inline auto range(Tb &&begin, Te &&end, Ts &&step) noexcept {
    using T = expr_value_t<Tb>;
    Var e{std::forward<Te>(end)};
    Var s{std::forward<Ts>(step)};
    return detail::ForRange<T, true>{std::forward<Tb>(begin), e, s};
}

/// for(auto i: range(n)) body(i);
template<typename N, typename Body>
inline void loop(N &&n, Body &&body) noexcept {
    for (auto i : range(std::forward<N>(n))) {
        std::invoke(std::forward<Body>(body), std::move(i));
    }
}

/// for(auto i: range(begin, end)) body(i);
template<typename Begin, typename End, typename Body>
inline void loop(Begin &&begin, End &&end, Body &&body) noexcept {
    for (auto i : range(std::forward<Begin>(begin), std::forward<End>(end))) {
        std::invoke(std::forward<Body>(body), std::move(i));
    }
}

/// for(auto i: range(begin, end, step)) body(i);
template<typename Begin, typename End, typename Step, typename Body>
inline void loop(Begin &&begin, End &&end, Step &&step, Body &&body) noexcept {
    for (auto i : range(std::forward<Begin>(begin), std::forward<End>(end), std::forward<Step>(step))) {
        std::invoke(std::forward<Body>(body), std::move(i));
    }
}

/// Switch tag, when case t(in tags), call indexed_case(i), when default call otherwise
template<concepts::iterable AllTags, typename Tag, typename IndexedCase, typename Otherwise>
    requires concepts::invocable<IndexedCase, int> && concepts::invocable<Otherwise>
inline void match(AllTags &&tags, Tag &&tag, IndexedCase &&indexed_case, Otherwise &&otherwise) noexcept {
    auto s = switch_(std::forward<Tag>(tag));
    auto index = 0;
    for (auto &&t : std::forward<AllTags>(tags)) {
        s = std::move(s).case_(t, [&c = indexed_case, i = index] { c(i); });
        index++;
    }
    std::move(s).default_(std::forward<Otherwise>(otherwise));
}

/// Switch tag, when case t(in tags), call indexed_case(i), when default call otherwise
template<typename T, typename Tag, typename IndexedCase, typename Otherwise>
    requires concepts::invocable<IndexedCase, int> && concepts::invocable<Otherwise>
inline void match(std::initializer_list<T> all_tags, Tag &&tag, IndexedCase &&indexed_case, Otherwise &&otherwise) noexcept {
    auto s = switch_(std::forward<Tag>(tag));
    auto index = 0;
    for (auto &&t : all_tags) {
        s = std::move(s).case_(t, [&c = indexed_case, i = index] { c(i); });
        index++;
    }
    std::move(s).default_(std::forward<Otherwise>(otherwise));
}

/// Switch tag, when case t(in tags), call indexed_case(i)
template<typename AllTags, typename Tag, typename IndexedCase>
inline void match(AllTags &&tags, Tag &&tag, IndexedCase &&indexed_case) noexcept {
    match(std::forward<AllTags>(tags),
          std::forward<Tag>(tag),
          std::forward<IndexedCase>(indexed_case),
          [] {});
}

/// Switch tag, when case t(in tags), call indexed_case(i)
template<typename T, typename Tag, typename IndexedCase>
inline void match(std::initializer_list<T> tags, Tag &&tag, IndexedCase &&indexed_case) noexcept {
    match(tags, std::forward<Tag>(tag), std::forward<IndexedCase>(indexed_case), [] {});
}

/// Add comment 
template<typename S>
inline void comment(S &&s) noexcept {
    detail::FunctionBuilder::current()->comment_(luisa::string{std::forward<S>(s)});
}

template<typename T>
inline void return_(T &&t) noexcept {
    detail::FunctionBuilder::current()->return_(
        detail::extract_expression(std::forward<T>(t)));
}

inline void return_() noexcept {
    detail::FunctionBuilder::current()->return_();
}

}// namespace dsl
}// namespace luisa::compute
