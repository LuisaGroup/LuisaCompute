//
// Created by Mike Smith on 2020/12/2.
//

#pragma once
#include "interface/ifunction.h"
#include <vector>
#include <fmt/format.h>

#include <core/memory.h>
#include <ast/statement.h>
#include <ast/variable.h>
#include <ast/expression.h>

namespace luisa::compute {

struct Statement;
struct Expression;

class Function : public IFunction{

public:


    struct ArgumentBinding {
        Variable variable;
        union {
            uint64_t buffer;
            uint64_t texture;
            const void *uniform;
            const void *constant;
        } binding;
    };

private:
    Arena _arena;
    ScopeStmt *_body;
    ArenaVector<ScopeStmt *> _scope_stack;
    ArenaVector<Variable> _builtin_variables;
    ArenaVector<Variable> _explicit_arguments;
    ArenaVector<ArgumentBinding> _implicit_arguments;
    uint32_t _variable_counter{0u};

    [[nodiscard]] static std::vector<Function *> &_function_stack() noexcept;
    [[nodiscard]] ScopeStmt *_current_scope() noexcept;

    static void _push(Function *func) noexcept;
    static Function *_pop() noexcept;

protected:
    [[nodiscard]] const Expression *_value(const Type *type, ValueExpr::Value value) noexcept;
    [[nodiscard]] Variable _constant(const Type *type, const void *data) noexcept;

public:
    Function() noexcept
        : _body{nullptr},
          _scope_stack{_arena},
          _builtin_variables{_arena},
          _explicit_arguments{_arena},
          _implicit_arguments{_arena} {}

    [[nodiscard]] static Function *current() noexcept;

    template<typename Def>
    void define(Def &&def) noexcept {
        if (_body != nullptr) { LUISA_ERROR_WITH_LOCATION("Multiple definition."); }
        _push(this);
        _body = scope(std::forward<Def>(def));
        if (_pop() != this) { LUISA_ERROR_WITH_LOCATION("Invalid function on stack top."); }
    }

    // variables
    [[nodiscard]] Variable local(const Type *type, std::span<const Expression *> init) noexcept;
    [[nodiscard]] Variable shared(const Type *type) noexcept;

    template<typename T, std::enable_if_t<std::disjunction_v<is_scalar<T>, is_vector<T>, is_matrix<T>>, int> = 0>
    [[nodiscard]] auto constant(std::span<T> data) noexcept {
        auto type = Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), data.size()));
        auto bytes = _arena.allocate<T>(data.size());
        std::uninitialized_copy_n(data.cbegin(), data.size(), bytes);
        return _constant(type, bytes);
    }

    // implicit arguments
    [[nodiscard]] Variable uniform(const Type *type, const void *data) noexcept;
    [[nodiscard]] Variable buffer(const Type *type, uint64_t handle) noexcept;
    [[nodiscard]] Variable texture(const Type *type, uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] Variable uniform(const Type *type) noexcept;
    [[nodiscard]] Variable buffer(const Type *type) noexcept;
    [[nodiscard]] Variable texture(const Type *type) noexcept;

    // expressions
    template<typename T, std::enable_if_t<std::disjunction_v<is_scalar<T>, is_vector<T>, is_matrix<T>>, int> = 0>
    [[nodiscard]] auto value(T value) noexcept { return _value(Type::of<T>(), value); }
    [[nodiscard]] auto value(Variable v) noexcept { return _value(v.type(), v); }
    [[nodiscard]] const Expression *unary(const Type *type, UnaryOp op, const Expression *expr) noexcept;
    [[nodiscard]] const Expression *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const Expression *member(const Type *type, const Expression *self, size_t member_index) noexcept;
    [[nodiscard]] const Expression *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const Expression *call(const Type *type /* nullptr for void */, std::string_view func, std::span<const Expression *> args) noexcept;
    [[nodiscard]] const Expression *call(const Type *type /* nullptr for void */, std::string_view func, std::initializer_list<const Expression *> args) noexcept;
    [[nodiscard]] const Expression *cast(const Type *type, CastOp op, const Expression *expr) noexcept;

    // statements
    void break_() noexcept;
    void continue_() noexcept;
    void return_(const Expression *expr = nullptr /* nullptr for void */) noexcept;

    template<typename Body>
    const Statement *scope(Body &&body) noexcept {
        auto s = _arena.create<ScopeStmt>(_arena);
        _scope_stack.emplace_back(s);
        body();
        if (_scope_stack.empty() || _scope_stack.back() != s) { LUISA_ERROR_WITH_LOCATION("Invalid scope when pop."); }
        _scope_stack.pop_back();
        return s;
    }

    void if_(const Expression *cond, const Statement *true_branch) noexcept;
    void if_(const Expression *cond, const Statement *true_branch, const Statement *false_branch) noexcept;
    void while_(const Expression *cond, const Statement *body) noexcept;
    void void_(const Expression *expr) noexcept;
    void switch_(const Expression *expr, const Statement *body) noexcept;
    void case_(const Expression *expr, const Statement *body) noexcept;
    void default_(const Statement *body) noexcept;
    void assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept;
};

}// namespace luisa::compute
