//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <vector>

#include <core/stl.h>
#include <core/hash.h>
#include <core/spin_mutex.h>

#include <ast/statement.h>
#include <ast/function.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/constant_data.h>
#include <ast/type_registry.h>

#include <serialize/key_value_pair.h>

namespace luisa::compute {

class Statement;
class Expression;
class FuncSerializer;

}// namespace luisa::compute

namespace luisa::compute::detail {

/**
 * @brief %Function builder.
 * 
 * Build kernel or callable function
 */
class FunctionBuilder : public luisa::enable_shared_from_this<FunctionBuilder> {
    friend class luisa::compute::FuncSerializer;

private:
    /**
     * @brief Scope guard.
     * 
     * Push scope on build, pop scope on destroy.
     */
    class ScopeGuard {

    private:
        FunctionBuilder *_builder;
        ScopeStmt *_scope;

    public:
        explicit ScopeGuard(FunctionBuilder *builder, ScopeStmt *scope) noexcept
            : _builder{builder}, _scope{scope} { _builder->push_scope(_scope); }
        ~ScopeGuard() noexcept { _builder->pop_scope(_scope); }
    };

    /**
     * @brief Meta guard.
     * 
     * Push meta on build, pop meta on destroy.
     */
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

    /**
     * @brief %Buffer binding.
     * 
     * Bind buffer handle and offset.
     */
    struct BufferBinding {
        uint64_t handle;
        size_t offset_bytes;
        BufferBinding() noexcept = default;
        BufferBinding(uint64_t handle, size_t offset_bytes) noexcept
            : handle{handle}, offset_bytes{offset_bytes} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(offset_bytes, hash64(handle, hash64("__hash_buffer_binding")));
        }
        template<typename S>
        void serialize(S& s) noexcept{
            s.serialize(MAKE_NAME_PAIR(handle), MAKE_NAME_PAIR(offset_bytes));
        }
    };

    /**
     * @brief Texture binding.
     * 
     * Bind texture handle and level.
     */
    struct TextureBinding {
        uint64_t handle;
        uint32_t level;
        TextureBinding() noexcept = default;
        TextureBinding(uint64_t handle, uint32_t level) noexcept
            : handle{handle}, level{level} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(level, hash64(handle, hash64("__hash_texture_binding")));
        }
        template<typename S>
        void serialize(S& s) noexcept{
            s.serialize(MAKE_NAME_PAIR(handle), MAKE_NAME_PAIR(level));
        }
    };

    /**
     * @brief Bindless array binding.
     * 
     * Bind array handle.
     */
    struct BindlessArrayBinding {
        uint64_t handle;
        BindlessArrayBinding() noexcept = default;
        explicit BindlessArrayBinding(uint64_t handle) noexcept
            : handle{handle} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(handle, hash64("__hash_bindless_array_binding"));
        }
        template<typename S>
        void serialize(S& s) noexcept{
            s.serialize(MAKE_NAME_PAIR(handle));
        }
    };

    /**
     * @brief Acceleration structure binding.
     * 
     * Bind accel handle.
     */
    struct AccelBinding {
        uint64_t handle;
        AccelBinding() noexcept = default;
        explicit AccelBinding(uint64_t handle) noexcept
            : handle{handle} {}
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(handle, hash64("__hash_accel_binding"));
        }
        template<typename S>
        void serialize(S& s) noexcept{
            s.serialize(MAKE_NAME_PAIR(handle));
        }
    };

    using Binding = luisa::variant<
        luisa::monostate,// not bound
        BufferBinding,
        TextureBinding,
        BindlessArrayBinding,
        AccelBinding>;

private:
    MetaStmt _body;
    const Type *_ret{nullptr};
    luisa::vector<luisa::unique_ptr<Expression>> _all_expressions;
    luisa::vector<luisa::unique_ptr<Statement>> _all_statements;
    luisa::vector<MetaStmt *> _meta_stack;//
    luisa::vector<ScopeStmt *> _scope_stack;//
    luisa::vector<Variable> _builtin_variables;
    luisa::vector<Constant> _captured_constants;
    luisa::vector<Variable> _arguments;
    luisa::vector<Binding> _argument_bindings;
    luisa::vector<luisa::shared_ptr<const FunctionBuilder>> _used_custom_callables;
    luisa::vector<Usage> _variable_usages;
    luisa::vector<std::pair<std::byte *, size_t /* alignment */>> _temporary_data;//
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
    /**
     * @brief Create a function builder with given definition
     * 
     * @param tag function tag(KERNEL, CALLABLE)
     * @param def function definition
     * @return share pointer to function builder
     */
    template<typename Def>
    static auto _define(Function::Tag tag, Def &&def) noexcept {
        auto f = make_shared<FunctionBuilder>(tag);
        push(f.get());
        f->with(&f->_body, std::forward<Def>(def));
        pop(f.get());
        return luisa::const_pointer_cast<const FunctionBuilder>(f);
    }

public:
    /**
     * @brief Construct a new %Function Builder object
     * 
     * @param tag type of function(Tag::KERNEL, Tag::CALLABLE)
     */
    explicit FunctionBuilder(Tag tag = Tag::CALLABLE) noexcept;
    FunctionBuilder(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder(const FunctionBuilder &) noexcept = delete;
    FunctionBuilder &operator=(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder &operator=(const FunctionBuilder &) noexcept = delete;

    /**
     * @brief Return current function builder on function stack.
     * 
     * If stack is empty, return nullptr
     *  
     * @return FunctionBuilder* 
     */
    [[nodiscard]] static FunctionBuilder *current() noexcept;

    // interfaces for class Function
    /// Return a span of builtin variables.
    [[nodiscard]] auto builtin_variables() const noexcept { return luisa::span{_builtin_variables}; }
    /// Return a span of constants.
    [[nodiscard]] auto constants() const noexcept { return luisa::span{_captured_constants}; }
    /// Return a span of arguments.
    [[nodiscard]] auto arguments() const noexcept { return luisa::span{_arguments}; }
    /// Return a span of argument bindings.
    [[nodiscard]] auto argument_bindings() const noexcept { return luisa::span{_argument_bindings}; }
    /// Return a span of custom callables.
    [[nodiscard]] auto custom_callables() const noexcept { return luisa::span{_used_custom_callables}; }
    /// Return a CallOpSet of builtin callables.
    [[nodiscard]] auto builtin_callables() const noexcept { return _used_builtin_callables; }
    /// Return tag(KERNEL, CALLABLE).
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    /// Return pointer to body.
    [[nodiscard]] auto body() noexcept { return &_body; }
    /// Return const pointer to body.
    [[nodiscard]] auto body() const noexcept { return &_body; }
    /// Return const pointer to return type.
    [[nodiscard]] auto return_type() const noexcept { return _ret; }
    /// Return variable usage of given uid.
    [[nodiscard]] auto variable_usage(uint32_t uid) const noexcept { return _variable_usages[uid]; }
    /// Return block size in uint3.
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    /// Return hash.
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    /// Return if is raytracing.
    [[nodiscard]] auto raytracing() const noexcept { return _raytracing; }

    // build primitives
    /// Define a kernel function with given definition
    template<typename Def>
    static auto define_kernel(Def &&def) noexcept {
        return _define(Function::Tag::KERNEL, std::forward<Def>(def));
    }

    template<typename Def>
    /// Define a callable function with given definition
    static auto define_callable(Def &&def) noexcept {
        return _define(Function::Tag::CALLABLE, std::forward<Def>(def));
    }

    // config
    /// Set block size
    void set_block_size(uint3 size) noexcept { _block_size = size; }

    // built-in variables
    /// Return thread id.
    [[nodiscard]] const RefExpr *thread_id() noexcept;
    /// Return block id.
    [[nodiscard]] const RefExpr *block_id() noexcept;
    /// Return dispatch id.
    [[nodiscard]] const RefExpr *dispatch_id() noexcept;
    /// Return dispatch size.
    [[nodiscard]] const RefExpr *dispatch_size() noexcept;

    // variables
    /// Add local variable of type
    [[nodiscard]] const RefExpr *local(const Type *type) noexcept;
    /// Add shared variable of type
    [[nodiscard]] const RefExpr *shared(const Type *type) noexcept;

    /// Add constant of type and data
    [[nodiscard]] const ConstantExpr *constant(const Type *type, ConstantData data) noexcept;
    /// Add binding of buffer. Will check for already bound arguments.
    [[nodiscard]] const RefExpr *buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept;
    /// Add binding of texture. Will check for already bound arguments.
    [[nodiscard]] const RefExpr *texture_binding(const Type *type, uint64_t handle, uint32_t level) noexcept;
    /// Add binding of bidnless array. Will check for already bound arguments.
    [[nodiscard]] const RefExpr *bindless_array_binding(uint64_t handle) noexcept;
    /// Add binding of acceleration structure. Will check for already bound arguments.
    [[nodiscard]] const RefExpr *accel_binding(uint64_t handle) noexcept;

    // explicit arguments
    /// Add argument of type
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    /// Add reference argument of type
    [[nodiscard]] const RefExpr *reference(const Type *type) noexcept;
    /// Add buffer argument of type
    [[nodiscard]] const RefExpr *buffer(const Type *type) noexcept;
    /// Add texture argument of type
    [[nodiscard]] const RefExpr *texture(const Type *type) noexcept;
    /// Add bindless array argument
    [[nodiscard]] const RefExpr *bindless_array() noexcept;
    /// Add accleration structure argument
    [[nodiscard]] const RefExpr *accel() noexcept;

    // expressions
    /// Create literal expression
    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr::Value value) noexcept;
    /// Create unary expression
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expr) noexcept;
    /// Create binary expression
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    /// Create member expression
    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *self, size_t member_index) noexcept;
    /// Create swizzle expression
    [[nodiscard]] const MemberExpr *swizzle(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept;
    /// Create access expression
    [[nodiscard]] const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    /// Create cast expression
    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp op, const Expression *expr) noexcept;
    /// Create call expression
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, CallOp call_op, std::initializer_list<const Expression *> args) noexcept;
    /// Create call expression
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, Function custom, std::initializer_list<const Expression *> args) noexcept;
    /// Call function
    void call(CallOp call_op, std::initializer_list<const Expression *> args) noexcept;
    /// Call custom function
    void call(Function custom, std::initializer_list<const Expression *> args) noexcept;
    /// Create call expression
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, CallOp call_op, luisa::span<const Expression *const> args) noexcept;
    /// Create call expression
    [[nodiscard]] const CallExpr *call(const Type *type /* nullptr for void */, Function custom, luisa::span<const Expression *const> args) noexcept;
    /// Call function
    void call(CallOp call_op, luisa::span<const Expression *const> args) noexcept;
    /// Call custom function
    void call(Function custom, luisa::span<const Expression *const> args) noexcept;

    // statements
    /// Add break statement
    void break_() noexcept;
    /// Add continue statement
    void continue_() noexcept;
    /// Add return statement
    void return_(const Expression *expr = nullptr /* nullptr for void */) noexcept;
    /// Add comment statement
    void comment_(luisa::string comment) noexcept;
    /// Add assign statement
    void assign(const Expression *lhs, const Expression *rhs) noexcept;

    /// Add if statement
    [[nodiscard]] IfStmt *if_(const Expression *cond) noexcept;
    /// Add loop statement
    [[nodiscard]] LoopStmt *loop_() noexcept;
    /// Add switch statement
    [[nodiscard]] SwitchStmt *switch_(const Expression *expr) noexcept;
    /// Add case statement
    [[nodiscard]] SwitchCaseStmt *case_(const Expression *expr) noexcept;
    /// Add default statement
    [[nodiscard]] SwitchDefaultStmt *default_() noexcept;
    /// Add for statement
    [[nodiscard]] ForStmt *for_(const Expression *var, const Expression *condition, const Expression *update) noexcept;
    /// Add meta statement
    [[nodiscard]] MetaStmt *meta(luisa::string info) noexcept;

    /// Run body function in given scope s
    template<typename Body>
    decltype(auto) with(ScopeStmt *s, Body &&body) noexcept {
        ScopeGuard guard{this, s};
        return body();
    }

    /// Create temporary data of type T and construct params args
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

    /// Run body function in given meta m
    template<typename Body>
    decltype(auto) with(MetaStmt *m, Body &&body) noexcept {
        MetaGuard guard{this, m};
        return body();
    }

    /// Push a function builder in stack
    static void push(FunctionBuilder *) noexcept;
    /// Pop a function builder in stack
    static void pop(FunctionBuilder *) noexcept;

    /// Push a meta statement and push its scope
    void push_meta(MetaStmt *meta) noexcept;
    /// Pop a meta statement and pop its scope
    void pop_meta(const MetaStmt *meta) noexcept;

    /// Push a scope
    void push_scope(ScopeStmt *) noexcept;
    /// Pop a scope
    void pop_scope(const ScopeStmt *) noexcept;
    /// Mark variable uasge
    void mark_variable_usage(uint32_t uid, Usage usage) noexcept;

    template<typename S>
    void serialize(S& s){
        s.serialize(
            MAKE_NAME_PAIR(_builtin_variables),
            MAKE_NAME_PAIR(_captured_constants),
            MAKE_NAME_PAIR(_arguments),
            MAKE_NAME_PAIR(_argument_bindings),
            MAKE_NAME_PAIR(_variable_usages),
            MAKE_NAME_PAIR(_hash),
            MAKE_NAME_PAIR(_block_size),
            MAKE_NAME_PAIR(_tag),
            MAKE_NAME_PAIR(_using_shared_storage),
            MAKE_NAME_PAIR(_raytracing)
        );
    }

    /// Return a Function object constructed from this
    [[nodiscard]] auto function() const noexcept { return Function{this}; }
};

}// namespace luisa::compute::detail
