//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <vector>

#include <fmt/format.h>

#include <core/memory.h>
#include <core/hash.h>
#include <core/spin_mutex.h>

#include <ast/statement.h>
#include <ast/function.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/type_registry.h>

namespace luisa::compute {

struct Statement;
struct Expression;

class FunctionBuilder {

private:
    class ScopeGuard {

    private:
        FunctionBuilder *_builder;
        ScopeStmt *_scope;

    public:
        explicit ScopeGuard(FunctionBuilder *builder, ScopeStmt *scope) noexcept
            : _builder{builder}, _scope{scope} { _builder->push_scope(_scope); }
        ~ScopeGuard() noexcept { _builder->pop_scope(_scope); }
    };

public:
    using Tag = Function::Tag;
    using ConstantBinding = Function::ConstantBinding;
    using BufferBinding = Function::BufferBinding;
    using TextureBinding = Function::TextureBinding;

private:
    ScopeStmt _body;
    const Type *_ret{nullptr};
    ArenaVector<ScopeStmt *> _scope_stack;
    ArenaVector<Variable> _builtin_variables;
    ArenaVector<Variable> _shared_variables;
    ArenaVector<ConstantBinding> _captured_constants;
    ArenaVector<BufferBinding> _captured_buffers;
    ArenaVector<TextureBinding> _captured_textures;
    ArenaVector<Variable> _arguments;
    ArenaVector<uint32_t> _used_custom_callables;
    ArenaVector<std::string_view> _used_builtin_callables;
    ArenaVector<Variable::Usage> _variable_usages;
    Tag _tag;
    uint32_t _uid;

protected:
    [[nodiscard]] static Arena &arena() noexcept;
    [[nodiscard]] static std::vector<FunctionBuilder *> &_function_stack() noexcept;
    [[nodiscard]] static spin_mutex &_function_registry_mutex() noexcept;
    [[nodiscard]] static std::vector<std::unique_ptr<FunctionBuilder>> &_function_registry() noexcept;
    [[nodiscard]] uint32_t _next_variable_uid() noexcept;

    void _append(const Statement *statement) noexcept;

    [[nodiscard]] const RefExpr *_builtin(Variable::Tag tag) noexcept;
    [[nodiscard]] const RefExpr *_texture_binding(const Type *type, uint64_t handle) noexcept;
    [[nodiscard]] const RefExpr *_ref(Variable v) noexcept;

private:
    explicit FunctionBuilder(Tag tag, uint32_t uid) noexcept;

    template<typename Def>
    static auto _define(Function::Tag tag, Def &&def) noexcept {
        auto f = create(tag);
        push(f);
        f->with(&f->_body, std::forward<Def>(def));
        pop(f);
        return Function{f};
    }

public:
    [[nodiscard]] static FunctionBuilder *current() noexcept;

    // interfaces for class Function
    [[nodiscard]] auto builtin_variables() const noexcept { return std::span{_builtin_variables.data(), _builtin_variables.size()}; }
    [[nodiscard]] auto shared_variables() const noexcept { return std::span{_shared_variables.data(), _shared_variables.size()}; }
    [[nodiscard]] auto constants() const noexcept { return std::span{_captured_constants.data(), _captured_constants.size()}; }
    [[nodiscard]] auto captured_buffers() const noexcept { return std::span{_captured_buffers.data(), _captured_buffers.size()}; }
    [[nodiscard]] auto captured_textures() const noexcept { return std::span{_captured_textures.data(), _captured_textures.size()}; }
    [[nodiscard]] auto arguments() const noexcept { return std::span{_arguments.data(), _arguments.size()}; }
    [[nodiscard]] auto custom_callables() const noexcept { return std::span{_used_custom_callables.data(), _used_custom_callables.size()}; }
    [[nodiscard]] auto builtin_callables() const noexcept { return std::span{_used_builtin_callables.data(), _used_builtin_callables.size()}; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto return_type() const noexcept { return _ret; }
    [[nodiscard]] auto variable_usage(uint32_t uid) const noexcept { return _variable_usages[uid]; }
    [[nodiscard]] static Function at(uint32_t uid) noexcept;
    [[nodiscard]] static Function callable(uint32_t uid) noexcept;
    [[nodiscard]] static Function kernel(uint32_t uid) noexcept;

    // build primitives
    template<typename Def>
    static auto define_kernel(Def &&def) noexcept {
        return _define(Function::Tag::KERNEL, std::forward<Def>(def));
    }

    template<typename Def>
    static auto define_callable(Def &&def) noexcept {
        return _define(Function::Tag::CALLABLE, std::forward<Def>(def));
    }

    [[nodiscard]] const RefExpr *thread_id() noexcept;
    [[nodiscard]] const RefExpr *block_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_id() noexcept;

    // variables
    [[nodiscard]] const RefExpr *local(const Type *type, std::span<const Expression *> init) noexcept;
    [[nodiscard]] const RefExpr *local(const Type *type, std::initializer_list<const Expression *> init) noexcept;
    [[nodiscard]] const RefExpr *shared(const Type *type) noexcept;

    [[nodiscard]] const ConstantExpr *constant(const Type *type, uint64_t hash) noexcept;
    [[nodiscard]] const RefExpr *buffer_binding(const Type *element_type, uint64_t handle, size_t offset_bytes) noexcept;
    [[nodiscard]] const RefExpr *texture_binding(const Type *type, uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] const RefExpr *uniform(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *buffer(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *texture(const Type *type) noexcept;

    // expressions
    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr::Value value) noexcept;
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expr) noexcept;
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *self, size_t member_index) noexcept;
    [[nodiscard]] const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, std::string_view func, std::initializer_list<const Expression *> args) noexcept;
    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp op, const Expression *expr) noexcept;

    // statements
    void break_() noexcept;
    void continue_() noexcept;
    void return_(const Expression *expr = nullptr /* nullptr for void */) noexcept;
    void if_(const Expression *cond, const ScopeStmt *true_branch, const ScopeStmt *false_branch) noexcept;
    void while_(const Expression *cond, const ScopeStmt *body) noexcept;
    void void_(const Expression *expr) noexcept;
    void switch_(const Expression *expr, const ScopeStmt *body) noexcept;
    void case_(const Expression *expr, const ScopeStmt *body) noexcept;
    void default_(const ScopeStmt *body) noexcept;
    void for_(const Statement *init, const Expression *condition, const Statement *update, const ScopeStmt *body) noexcept;

    void assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] ScopeStmt *scope() noexcept;

    template<typename Body>
    decltype(auto) with(ScopeStmt *s, Body &&body) noexcept {
        ScopeGuard guard{this, s};
        return body();
    }
    
    static void push(FunctionBuilder *) noexcept;
    static void pop(const FunctionBuilder *) noexcept;
    [[nodiscard]] static FunctionBuilder *create(Function::Tag) noexcept;

    void push_scope(ScopeStmt *) noexcept;
    void pop_scope(const ScopeStmt *) noexcept;
    void mark_variable_usage(uint32_t uid, Variable::Usage usage) noexcept;
};

}// namespace luisa::compute
