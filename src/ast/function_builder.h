//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <vector>

#include <core/allocator.h>
#include <core/hash.h>
#include <core/spin_mutex.h>

#include <ast/statement.h>
#include <ast/function.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/constant_data.h>
#include <ast/type_registry.h>

namespace luisa::compute {

class Statement;
class Expression;
class FuncSerializer;

}// namespace luisa::compute

namespace luisa::compute::detail {

class FunctionBuilder : public luisa::enable_shared_from_this<FunctionBuilder> {
    friend class luisa::compute::FuncSerializer;

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

    class MetaGuard {

    private:
        FunctionBuilder *_builder;
        MetaStmt *_meta;

    public:
        explicit MetaGuard(FunctionBuilder *builder, MetaStmt *meta) noexcept
            : _builder{builder}, _meta{meta} { _builder->push_meta(_meta); }
        ~MetaGuard() noexcept { _builder->pop_meta(_meta); }
    };

public:
    using Tag = Function::Tag;
    using Constant = Function::Constant;

    struct BufferBinding {
        uint64_t handle;
        size_t offset_bytes;
        BufferBinding(uint64_t handle, size_t offset_bytes) noexcept
            : handle{handle}, offset_bytes{offset_bytes} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(offset_bytes, hash64(handle, hash64("__hash_buffer_binding")));
        }
    };

    struct TextureBinding {
        uint64_t handle;
        uint32_t level;
        TextureBinding(uint64_t handle, uint32_t level) noexcept
            : handle{handle}, level{level} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(level, hash64(handle, hash64("__hash_texture_binding")));
        }
    };

    struct BindlessArrayBinding {
        uint64_t handle;
        explicit BindlessArrayBinding(uint64_t handle) noexcept
            : handle{handle} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(handle, hash64("__hash_bindless_array_binding"));
        }
    };

    struct AccelBinding {
        uint64_t handle;
        explicit AccelBinding(uint64_t handle) noexcept
            : handle{handle} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(handle, hash64("__hash_accel_binding"));
        }
    };

    using Binding = std::variant<
        std::monostate,// not bound
        BufferBinding,
        TextureBinding,
        BindlessArrayBinding,
        AccelBinding>;

private:
    MetaStmt _body;
    const Type *_ret{nullptr};
    luisa::vector<luisa::unique_ptr<Expression>> _all_expressions;
    luisa::vector<luisa::unique_ptr<Statement>> _all_statements;
    luisa::vector<MetaStmt *> _meta_stack;
    luisa::vector<ScopeStmt *> _scope_stack;
    luisa::vector<Variable> _builtin_variables;
    luisa::vector<Constant> _captured_constants;
    luisa::vector<Variable> _arguments;
    luisa::vector<Binding> _argument_bindings;
    luisa::vector<luisa::shared_ptr<const FunctionBuilder>> _used_custom_callables;
    luisa::vector<Usage> _variable_usages;
    luisa::vector<std::pair<std::byte *, size_t /* alignment */>> _temporary_data;
    CallOpSet _used_builtin_callables;
    uint64_t _hash;
    uint3 _block_size;
    Tag _tag;
    bool _using_shared_storage{false};
    bool _raytracing{false};

protected:
    [[nodiscard]] static luisa::vector<FunctionBuilder *> &_function_stack() noexcept;
    [[nodiscard]] uint32_t _next_variable_uid() noexcept;

    void _append(const Statement *statement) noexcept;

    [[nodiscard]] const RefExpr *_builtin(Variable::Tag tag) noexcept;
    [[nodiscard]] const RefExpr *_ref(Variable v) noexcept;
    void _void_expr(const Expression *expr) noexcept;
    void _compute_hash() noexcept;

    template<typename Stmt, typename... Args>
    auto _create_and_append_statement(Args &&...args) noexcept {
        auto stmt = luisa::make_unique<Stmt>(std::forward<Args>(args)...);
        auto p = stmt.get();
        _all_statements.emplace_back(std::move(stmt));
        _append(p);
        return p;
    }

    template<typename Expr, typename... Args>
    [[nodiscard]] auto _create_expression(Args &&...args) noexcept {
        auto expr = luisa::make_unique<Expr>(std::forward<Args>(args)...);
        auto p = expr.get();
        _all_expressions.emplace_back(std::move(expr));
        return p;
    }

private:
    template<typename Def>
    static auto _define(Function::Tag tag, Def &&def) noexcept {
        auto f = make_shared<FunctionBuilder>(tag);
        push(f.get());
        f->with(&f->_body, std::forward<Def>(def));
        pop(f.get());
        return luisa::const_pointer_cast<const FunctionBuilder>(f);
    }

public:
    explicit FunctionBuilder(Tag tag) noexcept;
    FunctionBuilder(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder(const FunctionBuilder &) noexcept = delete;
    FunctionBuilder &operator=(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder &operator=(const FunctionBuilder &) noexcept = delete;

    [[nodiscard]] static FunctionBuilder *current() noexcept;

    // interfaces for class Function
    [[nodiscard]] auto builtin_variables() const noexcept { return luisa::span{_builtin_variables.data(), _builtin_variables.size()}; }
    [[nodiscard]] auto constants() const noexcept { return luisa::span{_captured_constants.data(), _captured_constants.size()}; }
    [[nodiscard]] auto arguments() const noexcept { return luisa::span{_arguments.data(), _arguments.size()}; }
    [[nodiscard]] auto argument_bindings() const noexcept { return luisa::span{_argument_bindings.data(), _arguments.size()}; }
    [[nodiscard]] auto custom_callables() const noexcept { return luisa::span{_used_custom_callables.data(), _used_custom_callables.size()}; }
    [[nodiscard]] auto builtin_callables() const noexcept { return _used_builtin_callables; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto return_type() const noexcept { return _ret; }
    [[nodiscard]] auto variable_usage(uint32_t uid) const noexcept { return _variable_usages[uid]; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    [[nodiscard]] auto raytracing() const noexcept { return _raytracing; }

    // build primitives
    template<typename Def>
    static auto define_kernel(Def &&def) noexcept {
        return _define(Function::Tag::KERNEL, [&def] {
            auto f = current();
            auto gid = f->dispatch_id();
            auto gs = f->dispatch_size();
            auto less = f->binary(Type::of<bool3>(), BinaryOp::LESS, gid, gs);
            auto cond = f->call(Type::of<bool>(), CallOp::ALL, {less});
            auto if_stmt = f->if_(cond);
            f->with(if_stmt->true_branch(), [&def] { def(); });
        });
    }

    template<typename Def>
    static auto define_callable(Def &&def) noexcept {
        return _define(Function::Tag::CALLABLE, std::forward<Def>(def));
    }

    // config
    void set_block_size(uint3 size) noexcept { _block_size = size; }

    // built-in variables
    [[nodiscard]] const RefExpr *thread_id() noexcept;
    [[nodiscard]] const RefExpr *block_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_size() noexcept;

    // variables
    [[nodiscard]] const RefExpr *local(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *shared(const Type *type) noexcept;

    [[nodiscard]] const ConstantExpr *constant(const Type *type, ConstantData data) noexcept;
    [[nodiscard]] const RefExpr *buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept;
    [[nodiscard]] const RefExpr *texture_binding(const Type *type, uint64_t handle, uint32_t level) noexcept;
    [[nodiscard]] const RefExpr *bindless_array_binding(uint64_t handle) noexcept;
    [[nodiscard]] const RefExpr *accel_binding(uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *reference(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *buffer(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *texture(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *bindless_array() noexcept;
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
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, CallOp call_op, luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, Function custom, luisa::span<const Expression *const> args) noexcept;
    void call(CallOp call_op, luisa::span<const Expression *const> args) noexcept;
    void call(Function custom, luisa::span<const Expression *const> args) noexcept;

    // statements
    void break_() noexcept;
    void continue_() noexcept;
    void return_(const Expression *expr = nullptr /* nullptr for void */) noexcept;
    void comment_(luisa::string comment) noexcept;
    void assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept;

    [[nodiscard]] IfStmt *if_(const Expression *cond) noexcept;
    [[nodiscard]] LoopStmt *loop_() noexcept;
    [[nodiscard]] SwitchStmt *switch_(const Expression *expr) noexcept;
    [[nodiscard]] SwitchCaseStmt *case_(const Expression *expr) noexcept;
    [[nodiscard]] SwitchDefaultStmt *default_() noexcept;
    [[nodiscard]] ForStmt *for_(const Expression *var, const Expression *condition, const Expression *update) noexcept;
    [[nodiscard]] MetaStmt *meta(luisa::string info) noexcept;

    template<typename Body>
    decltype(auto) with(ScopeStmt *s, Body &&body) noexcept {
        ScopeGuard guard{this, s};
        return body();
    }

    template<typename T, typename... Args>
    [[nodiscard]] auto create_temporary(Args &&...args) noexcept {
        static_assert(std::is_trivially_destructible_v<T>);
        auto p = luisa::detail::allocator_allocate(sizeof(T), alignof(T));
        _temporary_data.emplace_back(std::make_pair(
            static_cast<std::byte *>(p), alignof(T)));
        return std::construct_at(
            static_cast<T *>(p),
            std::forward<Args>(args)...);
    }

    template<typename Body>
    decltype(auto) with(MetaStmt *m, Body &&body) noexcept {
        MetaGuard guard{this, m};
        return body();
    }

    static void push(FunctionBuilder *) noexcept;
    static void pop(FunctionBuilder *) noexcept;

    void push_meta(MetaStmt *meta) noexcept;
    void pop_meta(const MetaStmt *meta) noexcept;

    void push_scope(ScopeStmt *) noexcept;
    void pop_scope(const ScopeStmt *) noexcept;
    void mark_variable_usage(uint32_t uid, Usage usage) noexcept;

    [[nodiscard]] auto function() const noexcept { return Function{this}; }
};

}// namespace luisa::compute::detail
