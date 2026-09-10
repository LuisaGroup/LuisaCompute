#pragma once

#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/string.h>

namespace luisa::compute::tile::bridge::tirx::detail {

// Native TVM visitors have fixed node return types. Keep a separate explicit
// diagnostic for bridge rejection, and retain the input node after failure.
// Stage callers must inspect the diagnostic before publishing or lowering any
// result. Optional candidates use their own diagnostic and provisional plans.
class Diagnostic {
private:
    luisa::string _error;

public:
    [[nodiscard]] bool failed() const noexcept { return !_error.empty(); }
    [[nodiscard]] const luisa::string &error() const noexcept { return _error; }
    void set_error(luisa::string_view error) noexcept {
        if (_error.empty()) { _error = error; }
    }
    template<typename T>
    [[nodiscard]] T reject(luisa::string_view error, T fallback) noexcept {
        set_error(error);
        return fallback;
    }
};

template<typename Base>
class DiagnosticMutator : public Base {
protected:
    Diagnostic &_diagnostic;

public:
    explicit DiagnosticMutator(Diagnostic &diagnostic) noexcept : _diagnostic{diagnostic} {}
    tvm::tirx::Stmt VisitStmt(const tvm::tirx::Stmt &statement) override {
        if (_diagnostic.failed()) { return statement; }
        // Keep a strong reference while TVM decides whether in-place mutation
        // is allowed. A failed speculative visitor must retain the input graph.
        auto original = statement;
        auto result = Base::VisitStmt(statement);
        return _diagnostic.failed() ? original : result;
    }
    tvm::Expr VisitExpr(const tvm::Expr &expression) override {
        if (_diagnostic.failed()) { return expression; }
        auto result = Base::VisitExpr(expression);
        return _diagnostic.failed() ? expression : result;
    }
};

using DiagnosticStmtMutator = DiagnosticMutator<tvm::tirx::StmtMutator>;
using DiagnosticStmtExprMutator = DiagnosticMutator<tvm::tirx::StmtExprMutator>;

}// namespace luisa::compute::tile::bridge::tirx::detail
