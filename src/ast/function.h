//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <vector>

#include <core/memory.h>
#include <ast/statement.h>
#include <ast/variable.h>
#include <ast/expression.h>

namespace luisa::compute {

struct Statement;
struct Expression;

class Function {

public:
    enum struct Tag {
        KERNEL,
        DEVICE,
        // TODO: Ray-tracing functions...
    };

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
    ScopeStmt _body;
    ArenaVector<ScopeStmt *> _scope_stack;
    ArenaVector<Variable> _builtin_variables;
    ArenaVector<Variable> _explicit_arguments;
    ArenaVector<ArgumentBinding> _implicit_arguments;
    uint32_t _variable_counter{0u};
    bool _defined{false};

    [[nodiscard]] static std::vector<Function *> &_function_stack() noexcept;
    [[nodiscard]] ScopeStmt *_current_scope() noexcept;
    
    static void _push(Function *func) noexcept;
    static Function *_pop() noexcept;

public:
    Function() noexcept
        : _body{ArenaVector<const Statement *>(_arena)},
          _scope_stack{_arena},
          _builtin_variables{_arena},
          _explicit_arguments{_arena},
          _implicit_arguments{_arena} { _scope_stack.emplace_back(&_body); }
    
    [[nodiscard]] static Function *current() noexcept;
    
    void define(const std::function<void()> &def) noexcept;

    // variables
    [[nodiscard]] Variable local(const Type *type, const Expression *init) noexcept;
    [[nodiscard]] Variable shared(const Type *type, const Expression *init) noexcept;
    [[nodiscard]] Variable constant(const Type *type, const Expression *init) noexcept;

    // implicit arguments
    [[nodiscard]] Variable uniform(const Type *type, const void *data) noexcept;
    [[nodiscard]] Variable buffer(const Type *type, uint64_t handle) noexcept;
    [[nodiscard]] Variable texture(const Type *type, uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] Variable uniform(const Type *type) noexcept;
    [[nodiscard]] Variable buffer(const Type *type) noexcept;
    [[nodiscard]] Variable texture(const Type *type) noexcept;

    // expressions
    [[nodiscard]] const Expression *value(ValueExpr::Value value) noexcept;
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
    const Statement *scope(const std::function<void()> &body) noexcept;
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
