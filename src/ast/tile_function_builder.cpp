#include <luisa/ast/tile_function_builder.h>

#include <luisa/core/logging.h>

#include <utility>

namespace luisa::compute::detail {

luisa::vector<TileFunctionBuilder *> &TileFunctionBuilder::_function_stack() noexcept {
    static thread_local luisa::vector<TileFunctionBuilder *> tile_function_builder_stack;
    return tile_function_builder_stack;
}

void TileFunctionBuilder::push(TileFunctionBuilder *func) noexcept {
    _function_stack().emplace_back(func);
}

void TileFunctionBuilder::pop(TileFunctionBuilder *func) noexcept {
    if (_function_stack().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid pop on empty tile function stack.");
    }
    auto f = _function_stack().back();
    _function_stack().pop_back();
    if (func != nullptr && f != func) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid tile function on stack top.");
    }
}

TileFunctionBuilder *TileFunctionBuilder::current() noexcept {
    LUISA_ASSERT(!_function_stack().empty(), "Empty tile function stack.");
    return _function_stack().back();
}

TileFunctionBuilder *TileFunctionBuilder::current_or_null() noexcept {
    return _function_stack().empty() ?
               nullptr :
               _function_stack().back();
}

luisa::span<const TileFunctionBuilder *const> TileFunctionBuilder::stack() noexcept {
    return _function_stack();
}

TileFunctionBuilder::TileFunctionBuilder() noexcept = default;

TileFunctionBuilder::~TileFunctionBuilder() noexcept = default;

void TileFunctionBuilder::set_name(luisa::string_view name) const noexcept {
    _name.assign(name);
}

void TileFunctionBuilder::_append(TensorStmt *statement) noexcept {
    if (_scope_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Tile scope stack is empty.");
    }
    _scope_stack.back()->append(statement);
}

void TileFunctionBuilder::push_scope(TileScope *scope) noexcept {
    _scope_stack.emplace_back(scope);
}

void TileFunctionBuilder::pop_scope(const TileScope *scope) noexcept {
    if (_scope_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid pop on empty tile scope stack.");
    }
    auto s = _scope_stack.back();
    _scope_stack.pop_back();
    if (scope != nullptr && s != scope) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid tile scope on stack top.");
    }
}

bool TileFunctionBuilder::inside_function_scope() const noexcept {
    return _scope_stack.size() == 1u;
}

// --- tile operators ---------------------------------------------------------

TensorExpr *TileFunctionBuilder::tile_empty(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype) noexcept {
    auto *stmt = _create_and_append_statement<AllocStmt>(
        std::move(dims), dtype, TensorScope::Global);
    return stmt->tensor();
}

TensorExpr *TileFunctionBuilder::tile_alloc_shared(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype) noexcept {
    auto *stmt = _create_and_append_statement<AllocStmt>(
        std::move(dims), dtype, TensorScope::Shared);
    return stmt->tensor();
}

TensorExpr *TileFunctionBuilder::tile_alloc_fragment(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype) noexcept {
    auto *stmt = _create_and_append_statement<AllocStmt>(
        std::move(dims), dtype, TensorScope::Fragment);
    return stmt->tensor();
}

void TileFunctionBuilder::tile_clear(TensorExpr *t) noexcept {
    _create_and_append_statement<ClearStmt>(t);
}

void TileFunctionBuilder::tile_copy(TensorExpr *src, TensorExpr *dst) noexcept {
    _create_and_append_statement<CopyStmt>(src, dst);
}

void TileFunctionBuilder::tile_gemm(TensorExpr *a, TensorExpr *b, TensorExpr *c,
                                    int32_t trans_a, int32_t trans_b) noexcept {
    _create_and_append_statement<GemmStmt>(a, b, c, trans_a, trans_b);
}

void TileFunctionBuilder::tile_reduce_sum(TensorExpr *x, TensorExpr *y, uint32_t dim) noexcept {
    _create_and_append_statement<ReduceSumStmt>(x, y, dim);
}

void TileFunctionBuilder::tile_print(TensorExpr *t, luisa::string msg) noexcept {
    _create_and_append_statement<TilePrintStmt>(t, std::move(msg));
}

void TileFunctionBuilder::tile_store(int32_t op, TensorExpr *lhs, TensorExpr *rhs_tensor,
                                     const LiteralExpr *rhs_literal,
                                     const RefExpr *rhs_ref) noexcept {
    _create_and_append_statement<TileStoreStmt>(op, lhs, rhs_tensor, rhs_literal, rhs_ref);
}

luisa::unique_ptr<TensorExpr, std::default_delete<TensorExpr>> TileFunctionBuilder::tile_binary(
    BinaryOp op, TensorExpr *lhs, TensorExpr *rhs_tensor,
    const LiteralExpr *rhs_literal, const RefExpr *rhs_ref) noexcept {
    _create_and_append_statement<TileBinaryStmt>(op, lhs, rhs_tensor, rhs_literal, rhs_ref);
    // fresh fragment temporary with the lhs layout (elementwise result);
    // plain-`new`-allocated so a consuming statement can `delete` it.
    return TensorExprPtr{new TensorExpr{
        lhs->rank(), lhs->dtype(), TensorScope::Fragment,
        luisa::fixed_vector<int32_t, 4>{lhs->dims().begin(), lhs->dims().end()}}};
}

luisa::unique_ptr<TensorExpr, std::default_delete<TensorExpr>> TileFunctionBuilder::tile_max(TensorExpr *a, const LiteralExpr *b) noexcept {
    _create_and_append_statement<MaxStmt>(a, b);
    return TensorExprPtr{new TensorExpr{
        a->rank(), a->dtype(), TensorScope::Fragment,
        luisa::fixed_vector<int32_t, 4>{a->dims().begin(), a->dims().end()}}};
}

luisa::unique_ptr<TensorExpr, std::default_delete<TensorExpr>> TileFunctionBuilder::tile_rsqrt(TensorExpr *a) noexcept {
    _create_and_append_statement<RsqrtStmt>(a);
    return TensorExprPtr{new TensorExpr{
        a->rank(), a->dtype(), TensorScope::Fragment,
        luisa::fixed_vector<int32_t, 4>{a->dims().begin(), a->dims().end()}}};
}

int32_t TileFunctionBuilder::tile_ceildiv(int32_t a, int32_t b) noexcept {
    auto *stmt = _create_and_append_statement<CeilDivStmt>(a, b);
    return stmt->result();
}

void TileFunctionBuilder::tile_kernel_1d(int32_t gx, int32_t threads, const RefExpr *bx) noexcept {
    _create_and_append_statement<Kernel1DStmt>(gx, threads, bx);
}

void TileFunctionBuilder::tile_kernel_2d(int32_t gx, int32_t gy, int32_t threads,
                                         const RefExpr *bx, const RefExpr *by) noexcept {
    _create_and_append_statement<Kernel2DStmt>(gx, gy, threads, bx, by);
}

void TileFunctionBuilder::tile_pipelined(int32_t count, int32_t stages, const RefExpr *k) noexcept {
    _create_and_append_statement<PipelinedStmt>(count, stages, k);
}

}// namespace luisa::compute::detail
