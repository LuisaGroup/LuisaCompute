//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <dsl/var.h>

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
