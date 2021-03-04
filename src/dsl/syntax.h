//
// Created by Mike Smith on 2021/2/27.
//

#pragma once

#include <dsl/var.h>
#include <dsl/expr.h>
#include <dsl/buffer.h>
#include <dsl/func.h>
#include <dsl/constant.h>
#include <dsl/shared.h>

namespace luisa::compute::dsl {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(detail::Expr<Src> s) noexcept { return s.template cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto bitwise_cast(detail::Expr<Src> s) noexcept { return s.template bitwise_cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto reinterpret(detail::Expr<Src> s) noexcept { return s.template reinterpret<Dest>(); }

}// namespace luisa::compute::dsl

// for custom structs
#undef LUISA_STRUCT// to extend it...

#define LUISA_STRUCT_MAKE_MEMBER_EXPR(m)                                    \
private:                                                                    \
    using Type_##m = std::remove_cvref_t<decltype(std::declval<This>().m)>; \
                                                                            \
public:                                                                     \
    Expr<Type_##m> m{FunctionBuilder::current()->member(                    \
        Type::of<Type_##m>(),                                               \
        ExprBase<This>::_expression,                                        \
        _member_index(#m))};

#define LUISA_STRUCT(S, ...)                                                                                     \
    LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, __VA_ARGS__)                                                \
    namespace luisa::compute::dsl::detail {                                                                      \
    template<>                                                                                                   \
    struct Expr<S> : public ExprBase<S> {                                                                        \
    private:                                                                                                     \
        using This = S;                                                                                          \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};       \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        using ExprBase<S>::ExprBase;                                                                             \
        Expr(Expr &&another) noexcept = default;                                                                 \
        Expr(const Expr &another) noexcept = default;                                                            \
        void operator=(Expr &&rhs) noexcept { ExprBase<S>::operator=(rhs); }                                     \
        void operator=(const Expr &rhs) noexcept { ExprBase<S>::operator=(rhs); }                                \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR, __VA_ARGS__)                                                    \
    };                                                                                                           \
    }

namespace luisa::compute::dsl::detail {

struct KernelBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Kernel{std::forward<F>(def)}; }
};

struct CallableBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Callable{std::forward<F>(def)}; }
};

}// namespace luisa::compute::dsl::detail

#define LUISA_KERNEL ::luisa::compute::dsl::detail::KernelBuilder{} % [&]
#define LUISA_CALLABLE ::luisa::compute::dsl::detail::CallableBuilder{} % [&]

namespace luisa::compute::dsl {

// statement buiders
namespace detail {

class IfStmtBuilder {

private:
    ScopeStmt *_true{nullptr};
    ScopeStmt *_false{nullptr};
    bool _true_set{false};
    bool _false_set{false};

public:
    explicit IfStmtBuilder(Expr<bool> condition) noexcept
        : _true{FunctionBuilder::current()->scope()},
          _false{FunctionBuilder::current()->scope()} {
        FunctionBuilder::current()->if_(condition.expression(), _true, _false);
    }

    template<typename False>
    void else_(False &&f) noexcept {
        if (!_true_set || _false_set) { LUISA_ERROR_WITH_LOCATION("Invalid IfStmtBuilder state."); }
        _false_set = true;
        FunctionBuilder::current()->with(_false, std::forward<False>(f));
    }

    template<typename True>
    auto operator%(True &&t) noexcept {
        if (_true_set) { LUISA_ERROR_WITH_LOCATION("Invalid IfStmtBuilder state."); }
        _true_set = true;
        FunctionBuilder::current()->with(_true, std::forward<True>(t));
        return *this;
    }

    template<typename Body>
    [[nodiscard]] auto elif (Expr<bool> condition, Body &&body) noexcept {
        if (!_true_set || _false_set) { LUISA_ERROR_WITH_LOCATION("Invalid IfStmtBuilder state."); }
        _false_set = true;
        return FunctionBuilder::current()->with(_false, [condition] { return IfStmtBuilder{condition}; })
               % std::forward<Body>(body);
    }

    template<typename False>
    void operator/(False &&f) noexcept { else_(std::forward<False>(f)); }

    [[nodiscard]] auto operator/(Expr<bool> elif_cond) noexcept {
        if (!_true_set || _false_set) { LUISA_ERROR_WITH_LOCATION("Invalid IfStmtBuilder state."); }
        _false_set = true;
        return FunctionBuilder::current()->with(_false, [elif_cond] {
            return IfStmtBuilder{elif_cond};
        });
    }
};

class WhileStmtBuilder {

private:
    ScopeStmt *_body;
    bool _body_set{false};

public:
    explicit WhileStmtBuilder(Expr<bool> cond) noexcept
        : _body{FunctionBuilder::current()->scope()} {
        FunctionBuilder::current()->while_(cond.expression(), _body);
    }

    template<typename Body>
    void operator%(Body &&body) noexcept {
        if (_body_set) { LUISA_ERROR_WITH_LOCATION("Invalid WhileStmtBuilder state."); }
        _body_set = true;
        FunctionBuilder::current()->with(_body, std::forward<Body>(body));
    }
};

class SwitchCaseStmtBuilder {

private:
    ScopeStmt *_body;

public:
    template<concepts::Integral T>
    explicit SwitchCaseStmtBuilder(T c) noexcept : _body{FunctionBuilder::current()->scope()} {
        FunctionBuilder::current()->case_(extract_expression(c), _body);
    }

    template<typename Body>
    void operator%(Body &&body) noexcept {
        FunctionBuilder::current()->with(_body, std::forward<Body>(body));
    }
};

class SwitchDefaultStmtBuilder {

private:
    ScopeStmt *_body;

public:
    SwitchDefaultStmtBuilder() noexcept : _body{FunctionBuilder::current()->scope()} {
        FunctionBuilder::current()->default_(_body);
    }

    template<typename Body>
    void operator%(Body &&body) noexcept {
        FunctionBuilder::current()->with(_body, std::forward<Body>(body));
    }
};

class SwitchStmtBuilder {

private:
    ScopeStmt *_body;

public:
    template<typename T>
    explicit SwitchStmtBuilder(T &&cond) noexcept
        : _body{FunctionBuilder::current()->scope()} {
        FunctionBuilder::current()->switch_(
            extract_expression(std::forward<T>(cond)), _body);
    }

    template<typename T, typename Body>
    auto case_(T &&case_cond, Body &&case_body) noexcept {
        FunctionBuilder::current()->with(_body, [&case_cond, &case_body] {
            SwitchCaseStmtBuilder{case_cond} % std::forward<Body>(case_body);
        });
        return *this;
    }

    template<typename Default>
    auto default_(Default &&d) noexcept {
        FunctionBuilder::current()->with(_body, [&d] {
            SwitchDefaultStmtBuilder{} % std::forward<Default>(d);
        });
    }

    template<typename Body>
    void operator%(Body &&body) noexcept {
        FunctionBuilder::current()->with(_body, std::forward<Body>(body));
    }
};

}// namespace detail

// statements
inline void break_() noexcept { FunctionBuilder::current()->break_(); }
inline void continue_() noexcept { FunctionBuilder::current()->continue_(); }

template<typename True>
[[nodiscard]] inline auto if_(detail::Expr<bool> condition, True &&t) noexcept {
    return detail::IfStmtBuilder{condition} % std::forward<True>(t);
}

template<typename Body>
inline void while_(detail::Expr<bool> condition, Body &&body) noexcept {
    detail::WhileStmtBuilder{condition} % std::forward<Body>(body);
}

template<typename T>
[[nodiscard]] inline auto switch_(T &&expr) noexcept {
    return detail::SwitchStmtBuilder{std::forward<T>(expr)};
}

}// namespace luisa::compute::dsl

#define $break ::luisa::compute::dsl::break_()
#define $continue ::luisa::compute::dsl::continue_()

#define $if(...) ::luisa::compute::dsl::detail::IfStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $else / [&]() noexcept
#define $elif(...) / ::luisa::compute::dsl::detail::Expr{__VA_ARGS__} % [&]() noexcept

#define $while(...) ::luisa::compute::dsl::detail::WhileStmtBuilder{__VA_ARGS__} % [&]() noexcept

#define $switch(...) ::luisa::compute::dsl::detail::SwitchStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $case(...) ::luisa::compute::dsl::detail::SwitchCaseStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $default ::luisa::compute::dsl::detail::SwitchDefaultStmtBuilder{} % [&]() noexcept

#ifdef LUISA_MORE_SYNTAX_SUGAR

#define Break $break
#define Continue $continue

#define If(...) $if(__VA_ARGS__)
#define Else $else
#define Elif(...) $elif(__VA_ARGS__)

#define While(...) $while(__VA_ARGS__)

#define Switch(...) $switch(__VA_ARGS__)
#define Case(...) $case(__VA_ARGS__)
#define Default $default

#endif
