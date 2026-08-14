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

TensorExpr *TileFunctionBuilder::tile_empty(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype,
                                             luisa::string_view name) noexcept {
    auto *stmt = _create_and_append_statement<AllocStmt>(
        std::move(dims), dtype, TensorScope::Global, nullptr, name);
    return stmt->tensor();
}

TensorExpr *TileFunctionBuilder::tile_alloc_shared(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype,
                                                   luisa::string_view name) noexcept {
    auto *stmt = _create_and_append_statement<AllocStmt>(
        std::move(dims), dtype, TensorScope::Shared, nullptr, name);
    return stmt->tensor();
}

TensorExpr *TileFunctionBuilder::tile_alloc_fragment(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype,
                                                     luisa::string_view name) noexcept {
    auto *stmt = _create_and_append_statement<AllocStmt>(
        std::move(dims), dtype, TensorScope::Fragment, nullptr, name);
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
    auto *stmt = _create_and_append_statement<TileBinaryStmt>(op, lhs, rhs_tensor, rhs_literal, rhs_ref);
    // fresh fragment temporary with the lhs layout (elementwise result);
    // plain-`new`-allocated so a consuming statement can `delete` it.
    auto temp = TensorExprPtr{new TensorExpr{
        lhs->rank(), lhs->dtype(), TensorScope::Fragment,
        luisa::fixed_vector<int32_t, 4>{lhs->dims().begin(), lhs->dims().end()}}};
    _temp_outputs[stmt] = temp.get();
    return temp;
}

luisa::unique_ptr<TensorExpr, std::default_delete<TensorExpr>> TileFunctionBuilder::tile_max(TensorExpr *a, const LiteralExpr *b) noexcept {
    auto *stmt = _create_and_append_statement<MaxStmt>(a, b);
    auto temp = TensorExprPtr{new TensorExpr{
        a->rank(), a->dtype(), TensorScope::Fragment,
        luisa::fixed_vector<int32_t, 4>{a->dims().begin(), a->dims().end()}}};
    _temp_outputs[stmt] = temp.get();
    return temp;
}

luisa::unique_ptr<TensorExpr, std::default_delete<TensorExpr>> TileFunctionBuilder::tile_rsqrt(TensorExpr *a) noexcept {
    auto *stmt = _create_and_append_statement<RsqrtStmt>(a);
    auto temp = TensorExprPtr{new TensorExpr{
        a->rank(), a->dtype(), TensorScope::Fragment,
        luisa::fixed_vector<int32_t, 4>{a->dims().begin(), a->dims().end()}}};
    _temp_outputs[stmt] = temp.get();
    return temp;
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

SIMTKernelMeta TileFunctionBuilder::compile_meta_data() const {
    // Mirror TileLang's launch-size collector (DeviceInfoCollector,
    // D:/tilelang/simd_to_simt.md §1.3): walk the whole body, find the kernel
    // launch statement, and report
    //   blockDim    = product of the threadIdx extents (T.Kernel `threads`)
    //   gridDim     = blockIdx extents (T.Kernel `gx` / `gx * gy`)
    // (D:/tilelang/simd_to_simt.md §2.3: threads -> set_block_size(...),
    // blocks -> .dispatch(gx[, gy])).
    //
    // A tile function maps to exactly ONE kernel launch (TileLang emits one
    // `__global__` per `T.Kernel`, D:/tilelang/simd_to_simt.md §1).  Zero
    // kernels cannot be dispatched; two kernels would carry two different
    // block/grid shapes that a single Shader cannot express.  Both cases log
    // an error and abort (header contract).
    //
    // "Deep" walk: every statement in the body is dispatched by TileOpKind
    // below.  The current tile IR is a flat list (nested scopes are not owned
    // by the builder), and statements that could embed sub-statements
    // (PIPELINED, LOOP_ANNOTATION, ...) trace their bodies inline into the
    // same flat list, so a single pass over body()->statements() is the
    // complete program traversal.  When the IR grows real nested scopes,
    // recurse into them from the corresponding case.
    SIMTKernelMeta meta{};
    uint32_t kernel_count = 0u;
    for (auto *stmt : body()->statements()) {
        switch (stmt->op()) {
            case TileOpKind::KERNEL_1D: {
                auto *k = static_cast<const Kernel1DStmt *>(stmt);
                if (kernel_count == 0u) {
                    meta.block_size = {static_cast<uint32_t>(k->threads()), 1u, 1u};
                    meta.dispatch_size = {static_cast<uint32_t>(k->gx()), 1u, 1u};
                }
                ++kernel_count;
                break;
            }
            case TileOpKind::KERNEL_2D: {
                auto *k = static_cast<const Kernel2DStmt *>(stmt);
                if (kernel_count == 0u) {
                    meta.block_size = {static_cast<uint32_t>(k->threads()), 1u, 1u};
                    meta.dispatch_size = {static_cast<uint32_t>(k->gx()),
                                          static_cast<uint32_t>(k->gy()), 1u};
                }
                ++kernel_count;
                break;
            }
            default:
                // All remaining TileOpKind are launch-agnostic:
                //   ALLOC / ALLOC_SPECIAL -> storage declarations; Luisa
                //     declares Shared<T> in-kernel, so there is no dynamic
                //     shared-memory launch parameter to collect here;
                //   CEILDIV -> grid extents are already materialized into
                //     gx/gy at trace time (tile_ceildiv evaluates host-side),
                //     so nothing symbolic to resolve; a runtime extent would
                //     be passed as a UInt kernel arg and used at
                //     .dispatch() time, not stored in this compile-time meta;
                //   COPY / GEMM / REDUCE / STORE / ... -> body work, not
                //     launch parameters;
                //   PIPELINED -> software-pipeline depth (stages) only, it
                //     does not change the block/grid shape.
                break;
        }
    }
    if (kernel_count == 0u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Tile function has no T.Kernel statement; cannot derive launch metadata.");
    }
    if (kernel_count > 1u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Tile function has {} T.Kernel statements; only one launch per tile "
            "function is allowed (one __global__ per T.Kernel).",
            kernel_count);
    }
    return meta;
}

}// namespace luisa::compute::detail
