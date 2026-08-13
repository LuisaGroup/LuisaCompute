#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include <luisa/ast/function_builder.h>
#include <luisa/ast/tensor.h>

#include <type_traits>
#include <utility>

namespace luisa::compute::detail {

/**
 * @brief %Tile function builder.
 *
 * Build a tile function composed only of tile operators (the TensorStmt
 * nodes defined in <luisa/ast/tensor.h>).  Unlike %FunctionBuilder, this
 * builder does NOT support Variable / Expression / Statement; every tile
 * operator operates directly on TensorExpr / TensorStmt nodes.  R2/R3
 * payload nodes (LiteralExpr / RefExpr) are borrowed from a %FunctionBuilder
 * and are never owned by the emitted statements' builder.
 *
 * Ownership follows <luisa/ast/tensor.h>: each emitted TensorStmt owns its
 * output and input TensorExpr operands, so an operand pointer must not be
 * shared between statements unless the caller manages ownership.  The
 * temporaries returned by the value-producing operators (tile_binary,
 * tile_max, tile_rsqrt) are owned by the caller.
 */
class LUISA_AST_API TileFunctionBuilder {

public:
    /**
     * @brief A tile scope is a flat list of tile operator statements.
     */
    class TileScope {
    private:
        luisa::vector<TensorStmt *> _statements;

    public:
        void append(TensorStmt *stmt) noexcept { _statements.emplace_back(stmt); }
        [[nodiscard]] auto statements() const noexcept {
            return luisa::span<const TensorStmt *const>{_statements.data(), _statements.size()};
        }
        [[nodiscard]] auto statements() noexcept {
            return luisa::span<TensorStmt *const>{_statements.data(), _statements.size()};
        }
        [[nodiscard]] auto size() const noexcept { return _statements.size(); }
        [[nodiscard]] auto empty() const noexcept { return _statements.empty(); }
    };

    /**
     * @brief RAII-style scope guard.
     *
     * Push scope on construction, pop scope on destruction.
     */
    class ScopeGuard {

    private:
        TileFunctionBuilder *_builder;
        TileScope *_scope;

    public:
        explicit ScopeGuard(TileFunctionBuilder *builder, TileScope *scope) noexcept
            : _builder{builder}, _scope{scope} { _builder->push_scope(_scope); }
        ~ScopeGuard() noexcept { _builder->pop_scope(_scope); }
    };

    /**
     * @brief RAII-style function guard.
     *
     * Push tile function builder on construction, pop tile function builder on destruction.
     */
    class FunctionStackGuard {

    private:
        TileFunctionBuilder *_builder;

    public:
        explicit FunctionStackGuard(TileFunctionBuilder *builder) noexcept
            : _builder{builder} { push(builder); }
        ~FunctionStackGuard() noexcept { pop(_builder); }
    };

private:
    TileScope _body;
    luisa::vector<TileScope *> _scope_stack;
    luisa::vector<luisa::unique_ptr<TensorStmt>> _owned_statements;
    luisa::vector<luisa::unique_ptr<Expression>> _owned_expressions;
    mutable luisa::string _name;

protected:
    [[nodiscard]] static luisa::vector<TileFunctionBuilder *> &_function_stack() noexcept;
    void _append(TensorStmt *statement) noexcept;

    template<typename Stmt, typename... Args>
    requires std::is_base_of_v<TensorStmt, Stmt>
    auto _create_and_append_statement(Args &&...args) noexcept {
        auto stmt = luisa::make_unique<Stmt>(std::forward<Args>(args)...);
        auto p = stmt.get();
        _owned_statements.emplace_back(std::move(stmt));
        _append(p);
        return p;
    }

    /// Create an expression node and register it in this builder's expression
    /// pool (`_owned_expressions`), so the statement nodes that borrow it never
    /// need to `delete` it.  Expression nodes require a %FunctionBuilder on the
    /// builder stack (Expression::Expression reads FunctionBuilder::current()), so
    /// the node is created under a short-lived guard and is never registered in
    /// that temporary builder.
    template<typename Expr, typename... Args>
    requires std::is_base_of_v<Expression, Expr>
    [[nodiscard]] const Expr *_create_and_append_expressions(Args &&...args) noexcept {
        FunctionBuilder builder;
        FunctionBuilder::FunctionStackGuard guard{&builder};
        auto expr = luisa::make_unique<Expr>(std::forward<Args>(args)...);
        auto p = expr.get();
        _owned_expressions.emplace_back(std::move(expr));
        return p;
    }

private:
    template<typename Def>
    static auto _define(Def &&def) {
        auto f = make_shared<TileFunctionBuilder>();
        {
            FunctionStackGuard guard{f.get()};
            f->with(&f->_body, std::forward<Def>(def));
        }
        return luisa::const_pointer_cast<const TileFunctionBuilder>(f);
    }

public:
    explicit TileFunctionBuilder() noexcept;
    TileFunctionBuilder(TileFunctionBuilder &&) noexcept = delete;
    TileFunctionBuilder(const TileFunctionBuilder &) noexcept = delete;
    TileFunctionBuilder &operator=(TileFunctionBuilder &&) noexcept = delete;
    TileFunctionBuilder &operator=(const TileFunctionBuilder &) noexcept = delete;
    ~TileFunctionBuilder() noexcept;

    /**
     * @brief Return the current tile function builder on the function stack.
     *
     * If the stack is empty, abort.
     *
     * @return TileFunctionBuilder*
     */
    [[nodiscard]] static TileFunctionBuilder *current() noexcept;
    /**
     * @brief Return the current tile function builder on the function stack.
     *
     * If the stack is empty, return nullptr.
     *
     * @return TileFunctionBuilder*
     */
    [[nodiscard]] static TileFunctionBuilder *current_or_null() noexcept;
    [[nodiscard]] static luisa::span<const TileFunctionBuilder *const> stack() noexcept;

    /// Define a tile function with the given definition.
    template<typename Def>
    static auto define(Def &&def) {
        return _define(std::forward<Def>(def));
    }

    // config
    /// Set name.
    void set_name(luisa::string_view name) const noexcept;
    /// Return name.
    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }

    // body & scope
    /// Return pointer to body scope.
    [[nodiscard]] auto body() noexcept { return &_body; }
    /// Return const pointer to body scope.
    [[nodiscard]] auto body() const noexcept { return &_body; }

    /// Run body function in the given scope s.
    template<typename Body>
    decltype(auto) with(TileScope *s, Body &&body) {
        ScopeGuard guard{this, s};
        return body();
    }

    /// Push a tile scope.
    void push_scope(TileScope *scope) noexcept;
    /// Pop a tile scope.
    void pop_scope(const TileScope *scope) noexcept;
    /// Check if inside the function-level scope.
    [[nodiscard]] bool inside_function_scope() const noexcept;

    // tile operators
    /// T.empty(dims, dtype): allocate a global tensor tile. The returned
    /// TensorExpr is owned by the emitted AllocStmt.
    [[nodiscard]] TensorExpr *tile_empty(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype) noexcept;
    /// T.alloc_shared(dims, dtype): allocate a per-block shared tensor tile.
    [[nodiscard]] TensorExpr *tile_alloc_shared(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype) noexcept;
    /// T.alloc_fragment(dims, dtype): allocate a per-thread fragment tensor tile.
    [[nodiscard]] TensorExpr *tile_alloc_fragment(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype) noexcept;
    /// T.clear(t).
    void tile_clear(TensorExpr *t) noexcept;
    /// T.copy(src, dst).
    void tile_copy(TensorExpr *src, TensorExpr *dst) noexcept;
    /// T.gemm(a, b, c, trans_a, trans_b).
    void tile_gemm(TensorExpr *a, TensorExpr *b, TensorExpr *c,
                   int32_t trans_a = 0, int32_t trans_b = 0) noexcept;
    /// T.reduce_sum(x, y, dim).
    void tile_reduce_sum(TensorExpr *x, TensorExpr *y, uint32_t dim) noexcept;
    /// T.print(t, "msg").
    void tile_print(TensorExpr *t, luisa::string msg) noexcept;
    /// Tile store: lhs = rhs (op 0) or lhs *= rhs (op 1); rhs is a tensor or a
    /// scalar literal / runtime scalar ref.
    void tile_store(int32_t op, TensorExpr *lhs, TensorExpr *rhs_tensor = nullptr,
                    const LiteralExpr *rhs_literal = nullptr,
                    const RefExpr *rhs_ref = nullptr) noexcept;
    /// Whole-tile elementwise binary op (T.Parallel lowering). Returns a fresh
    /// fragment temporary tensor owned by the caller; hand it to a consuming
    /// statement (which takes ownership) or let it destruct.
    [[nodiscard]] luisa::unique_ptr<TensorExpr> tile_binary(
        BinaryOp op, TensorExpr *lhs, TensorExpr *rhs_tensor = nullptr,
        const LiteralExpr *rhs_literal = nullptr,
        const RefExpr *rhs_ref = nullptr) noexcept;
    /// T.max(a, b). Returns a fresh fragment temporary tensor owned by the caller.
    [[nodiscard]] luisa::unique_ptr<TensorExpr> tile_max(TensorExpr *a, const LiteralExpr *b) noexcept;
    /// T.rsqrt(a). Returns a fresh fragment temporary tensor owned by the caller.
    [[nodiscard]] luisa::unique_ptr<TensorExpr> tile_rsqrt(TensorExpr *a) noexcept;
    /// T.ceildiv(a, b): host-side helper, returns (a + b - 1) / b.
    [[nodiscard]] int32_t tile_ceildiv(int32_t a, int32_t b) noexcept;
    /// T.Kernel(gx, threads, bx). bx is the borrowed runtime block id.
    void tile_kernel_1d(int32_t gx, int32_t threads, const RefExpr *bx = nullptr) noexcept;
    /// T.Kernel(gx, gy, threads, bx, by). bx/by are borrowed runtime block ids.
    void tile_kernel_2d(int32_t gx, int32_t gy, int32_t threads,
                        const RefExpr *bx = nullptr, const RefExpr *by = nullptr) noexcept;
    /// T.Pipelined(count, stages, k). k is the borrowed runtime loop variable.
    void tile_pipelined(int32_t count, int32_t stages, const RefExpr *k = nullptr) noexcept;

    // stack operations
    /// Push a tile function builder in the stack.
    static void push(TileFunctionBuilder *) noexcept;
    /// Pop a tile function builder from the stack.
    static void pop(TileFunctionBuilder *) noexcept;
};

}// namespace luisa::compute::detail
