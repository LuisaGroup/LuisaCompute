//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <vector>

#include <util/arena.h>
#include <core/hash.h>
#include <util/spin_mutex.h>

#include <ast/statement.h>
#include <ast/function.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/constant_data.h>
#include <ast/type_registry.h>

namespace luisa::compute {

class Statement;
class Expression;

}// namespace luisa::compute

namespace luisa::compute::detail {

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
    using HeapBinding = Function::HeapBinding;
    using AccelBinding = Function::AccelBinding;

private:
    Arena *_arena;
    ScopeStmt _body;
    const Type *_ret{nullptr};
    ArenaVector<ScopeStmt *> _scope_stack;
    ArenaVector<Variable> _builtin_variables;
    ArenaVector<Variable> _shared_variables;
    ArenaVector<ConstantBinding> _captured_constants;
    ArenaVector<BufferBinding> _captured_buffers;
    ArenaVector<TextureBinding> _captured_textures;
    ArenaVector<HeapBinding> _captured_heaps;
    ArenaVector<AccelBinding> _captured_accels;
    ArenaVector<Variable> _arguments;
    ArenaVector<Function> _used_custom_callables;
    ArenaVector<Usage> _variable_usages;
    CallOpSet _used_builtin_callables;
    uint64_t _hash;
    uint3 _block_size;
    Tag _tag;
    bool _raytracing{false};

protected:
    [[nodiscard]] static std::vector<FunctionBuilder *> &_function_stack() noexcept;
    [[nodiscard]] uint32_t _next_variable_uid() noexcept;

    void _append(const Statement *statement) noexcept;

    [[nodiscard]] const RefExpr *_builtin(Variable::Tag tag) noexcept;
    [[nodiscard]] const RefExpr *_ref(Variable v) noexcept;
    void _void_expr(const Expression *expr) noexcept;
    void _compute_hash() noexcept;

private:
    template<typename Def>
    static void _define(FunctionBuilder *f, Def &&def) noexcept {
        push(f);
        f->with(&f->_body, std::forward<Def>(def));
        f->_compute_hash();
        pop(f);
    }

public:
    explicit FunctionBuilder(Arena *arena, Tag tag) noexcept;
    FunctionBuilder(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder(const FunctionBuilder &) noexcept = delete;
    FunctionBuilder &operator=(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder &operator=(const FunctionBuilder &) noexcept = delete;

    [[nodiscard]] static FunctionBuilder *current() noexcept;

    // interfaces for class Function
    [[nodiscard]] auto builtin_variables() const noexcept { return std::span{_builtin_variables.data(), _builtin_variables.size()}; }
    [[nodiscard]] auto shared_variables() const noexcept { return std::span{_shared_variables.data(), _shared_variables.size()}; }
    [[nodiscard]] auto constants() const noexcept { return std::span{_captured_constants.data(), _captured_constants.size()}; }
    [[nodiscard]] auto captured_buffers() const noexcept { return std::span{_captured_buffers.data(), _captured_buffers.size()}; }
    [[nodiscard]] auto captured_textures() const noexcept { return std::span{_captured_textures.data(), _captured_textures.size()}; }
    [[nodiscard]] auto captured_heaps() const noexcept { return std::span{_captured_heaps.data(), _captured_heaps.size()}; }
    [[nodiscard]] auto captured_accels() const noexcept { return std::span{_captured_accels.data(), _captured_accels.size()}; }
    [[nodiscard]] auto arguments() const noexcept { return std::span{_arguments.data(), _arguments.size()}; }
    [[nodiscard]] auto custom_callables() const noexcept { return std::span{_used_custom_callables.data(), _used_custom_callables.size()}; }
    [[nodiscard]] auto builtin_callables() const noexcept { return _used_builtin_callables; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto return_type() const noexcept { return _ret; }
    [[nodiscard]] auto variable_usage(uint32_t uid) const noexcept { return _variable_usages[uid]; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    [[nodiscard]] auto raytracing() const noexcept { return _raytracing; }

    // build primitives
    template<typename Def>
    static auto define_kernel(Def &&def) noexcept {
        auto arena = new Arena;
        auto f = arena->create<FunctionBuilder>(arena, Function::Tag::KERNEL);
        _define(f, [f, &def] {
            auto gid = f->dispatch_id();
            auto gs = f->dispatch_size();
            auto less = f->binary(Type::of<bool3>(), BinaryOp::LESS, gid, gs);
            auto cond = f->call(Type::of<bool>(), CallOp::ALL, {less});
            auto ret_cond = f->unary(Type::of<bool>(), UnaryOp::NOT, cond);
            auto if_body = f->scope();
            f->with(if_body, [f] { f->return_(); });
            f->if_(ret_cond, if_body, nullptr);
            def();
        });
        return std::shared_ptr<const FunctionBuilder>{f, [](FunctionBuilder *f) noexcept { delete f->_arena; }};
    }

    template<typename Def>
    static auto define_callable(Def &&def) noexcept {
        auto arena = _function_stack().empty()              // callables use
                         ? &Arena::global()                 // the global arena when defined in global scope, or
                         : _function_stack().back()->_arena;// the inherited one from parent scope if defined locally
        auto f = arena->create<FunctionBuilder>(arena, Function::Tag::CALLABLE);
        _define(f, std::forward<Def>(def));
        return std::as_const(f);
    }

    // config
    void set_block_size(uint3 size) noexcept { _block_size = size; }

    // built-in variables
    [[nodiscard]] const RefExpr *thread_id() noexcept;
    [[nodiscard]] const RefExpr *block_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_size() noexcept;

    // variables
    [[nodiscard]] const RefExpr *local(const Type *type, std::span<const Expression *> init) noexcept;
    [[nodiscard]] const RefExpr *local(const Type *type, std::initializer_list<const Expression *> init) noexcept;
    [[nodiscard]] const RefExpr *shared(const Type *type) noexcept;

    [[nodiscard]] const ConstantExpr *constant(const Type *type, ConstantData data) noexcept;
    [[nodiscard]] const RefExpr *buffer_binding(const Type *element_type, uint64_t handle, size_t offset_bytes) noexcept;
    [[nodiscard]] const RefExpr *texture_binding(const Type *type, uint64_t handle) noexcept;
    [[nodiscard]] const RefExpr *heap_binding(uint64_t handle) noexcept;
    [[nodiscard]] const RefExpr *accel_binding(uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *reference(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *buffer(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *texture(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *heap() noexcept;
    [[nodiscard]] const RefExpr *accel() noexcept;

    // expressions
    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr::Value value) noexcept;
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expr) noexcept;
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *self, size_t member_index) noexcept;
    [[nodiscard]] const MemberExpr *swizzle(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept;
    [[nodiscard]] const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp op, const Expression *expr) noexcept;
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, CallOp call_op, std::initializer_list<const Expression *> args) noexcept;
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, Function custom, std::initializer_list<const Expression *> args) noexcept;
    void call(CallOp call_op, std::initializer_list<const Expression *> args) noexcept;
    void call(Function custom, std::initializer_list<const Expression *> args) noexcept;
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, CallOp call_op, std::span<const Expression *const> args) noexcept;
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, Function custom, std::span<const Expression *const> args) noexcept;
    void call(CallOp call_op, std::span<const Expression *const> args) noexcept;
    void call(Function custom, std::span<const Expression *const> args) noexcept;

    // statements
    void break_() noexcept;
    void continue_() noexcept;
    void return_(const Expression *expr = nullptr /* nullptr for void */) noexcept;
    void if_(const Expression *cond, const ScopeStmt *true_branch, const ScopeStmt *false_branch) noexcept;
    void while_(const Expression *cond, const ScopeStmt *body) noexcept;
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

    void push_scope(ScopeStmt *) noexcept;
    void pop_scope(const ScopeStmt *) noexcept;
    void mark_variable_usage(uint32_t uid, Usage usage) noexcept;

    [[nodiscard]] decltype(auto) arena() const noexcept { return *_arena; }
    [[nodiscard]] auto function() const noexcept { return Function{this}; }
};

}// namespace luisa::compute::detail
