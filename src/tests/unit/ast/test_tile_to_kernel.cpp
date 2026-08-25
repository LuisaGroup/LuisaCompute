// Test for tile_to_kernel — the SIMD->SIMT lowering that translates a traced
// tile function (TileFunctionBuilder, <luisa/ast/tile_function_builder.h>)
// into a REGULAR Luisa kernel (FunctionBuilder, <luisa/ast/function_builder.h>)
// as declared in <luisa/ast/tile_to_kernel.h>.
//
// This test covers:
// - translating the three example tile kernels (elementwise_add, pipelined
//   matmul, rms_norm) into FunctionBuilder instances
// - the dispatch grid equals the T.Kernel grid (KERNEL_1D -> (gx,1,1),
//   KERNEL_2D -> (gx,gy,1))
// - the kernel block size equals the T.Kernel thread count
// - one Buffer<T> argument per Global tensor of the tile function
//   (in AllocStmt order), all with Variable::Tag::BUFFER
// - the produced builder is a KERNEL-tagged FunctionBuilder with a non-empty
//   body, and wraps into a valid Function object
// - shared/fragment allocations produce shared/local array variables
//
// Pure host code: no device / backend is required.
#include "ut/ut.hpp"
#include <cstdint>
#include <functional>
#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/op.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/variable.h>
#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
#include <cerrno>
#include <unistd.h>
#include <sys/wait.h>
#endif
using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;
namespace {
// `import tilelang.language as T` -> the constexpr `T` handle.
constexpr auto T = tile::language::dsl;
using namespace tile::language;
using tile_f16 = tile::half;
using tile_f32 = tile::float32;
using tile_i32 = tile::int32;

Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 block_M = 8, block_N = 8;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_N), tile_f32{});
        auto B_shared = T.alloc_shared(T.shape(block_M, block_N), tile_f32{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});
        T.copy(A(by * block_M, bx * block_N), A_shared);
        T.copy(B(by * block_M, bx * block_N), B_shared);
        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);
        T.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

Tensor<tile_f16, 2> pipelined_matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 32;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;
    Tensor<tile_f16, 2> C = T.empty(T.shape(M, N), tile_f16{});
    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_K), tile_f16{});
        auto B_shared = T.alloc_shared(T.shape(block_K, block_N), tile_f16{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});
        T.clear(C_local);
        for (auto ko : T.Pipelined(T.ceildiv(K, block_K), num_stages)) {
            T.copy(A(by * block_M, ko * block_K), A_shared);
            T.copy(B(ko * block_K, bx * block_N), B_shared);
            T.gemm(A_shared, B_shared, C_local);
        }
    }
    return C;
}

Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;
    Tensor<tile_f32, 2> B = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(T.ceildiv(M, blk_m), threads)) {
        auto A_local = T.alloc_fragment(T.shape(blk_m, N), tile_f32{});
        auto A_pow_local = T.alloc_fragment(T.shape(blk_m, N), tile_f32{});
        auto A_powsum = T.alloc_fragment(T.shape(blk_m), tile_f32{});
        T.copy(A(T.range(bx * blk_m, (bx + 1) * blk_m), T.all()), A_local);
        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);
        T.reduce_sum(A_pow_local, A_powsum, /*dim=*/1);
        A_powsum(blk_m) = T.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);
        A_local(blk_m, N) *= A_powsum(blk_m);
        T.copy(A_local, B(T.range(bx * blk_m, (bx + 1) * blk_m), T.all()));
    }
    return B;
}

// Quantized dtype lowering: int8 global tensors lower to Buffer<byte>
// arguments (core INT8 element type); fp8 lowers to the fp8 e4m3 element type
// (zero fill is carried as a raw zero byte and cast to fp8).  As in the other
// tile kernels, global-to-global copies are not traced (global views are
// extent-less), so each copy routes through a shared/fragment intermediate.
Tensor<tile::int8, 1> int8_copy_kernel(Tensor<tile::int8, 1> A) {
    constexpr tile_i32 threads = 8;
    Tensor<tile::int8, 1> C = T.empty(T.shape(8), tile::int8{});
    for (auto bx : T.Kernel(1, threads)) {
        auto A_shared = T.alloc_shared(T.shape(8), tile::int8{});
        T.copy(A(0), A_shared);
        T.copy(A_shared, C(0));
    }
    return C;
}

Tensor<tile::fp8, 1> fp8_clear_kernel() {
    constexpr tile_i32 threads = 8;
    Tensor<tile::fp8, 1> C = T.empty(T.shape(8), tile::fp8{});
    for (auto bx : T.Kernel(1, threads)) {
        auto C_local = T.alloc_fragment(T.shape(8), tile::fp8{});
        T.clear(C_local);
        T.copy(C_local, C(0));
    }
    return C;
}

// erf has no CallOp::ERF and no portable external-function backend support,
// so the lowerer must emit a software (A&S 7.1.26) approximation built from
// ABS / EXP / SELECT / arithmetic instead of an "erf" external callable.
Tensor<tile_f32, 2> erf_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 block_M = 8, block_N = 8;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});
        auto E_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});
        T.copy(A(by * block_M, bx * block_N), A_local);
        E_local(block_M, block_N) = T.erf(A_local(block_M, block_N));
        T.copy(E_local, C(by * block_M, bx * block_N));
    }
    return C;
}

// A reduce kernel with threads = 32 (NOT a multiple of 64): used by the
// batching warp-alignment death test (batched + warp collectives require
// T.Kernel threads % 64 == 0, so this combination must abort in lower()).
Tensor<tile_f32, 2> reduce32_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 8, N = 8;
    constexpr tile_i32 blk_m = 4;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> B = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(T.ceildiv(M, blk_m), threads)) {
        auto A_local = T.alloc_fragment(T.shape(blk_m, N), tile_f32{});
        auto A_powsum = T.alloc_fragment(T.shape(blk_m), tile_f32{});
        T.copy(A(T.range(bx * blk_m, (bx + 1) * blk_m), T.all()), A_local);
        T.reduce_sum(A_local, A_powsum, /*dim=*/1);
        A_local(blk_m, N) *= A_powsum(blk_m);
        T.copy(A_local, B(T.range(bx * blk_m, (bx + 1) * blk_m), T.all()));
    }
    return B;
}

// ---------------------------------------------------------------------------
// Whole-tensor tile kernels (tensor-op fast path): Global-only operands (no
// shared/fragment staging), dispatched on a 1x1 tile grid so the full tensor
// is exactly the op's domain.  These currently ERROR in the partition path
// (op_extent_of rejects extent-less global views) and become supported via
// the full-tensor reconstruction of the TENSOR_* path.
// ---------------------------------------------------------------------------

Tensor<tile_f32, 2> global_copy(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        T.copy(A, C);
    }
    return C;
}

Tensor<tile_f32, 2> global_fill_clear() {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        T.clear(C);
        T.fill(C, 0.5f);
    }
    return C;
}

Tensor<tile_f32, 2> global_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        C(T.range(0, M), T.all()) =
            A(T.range(0, M), T.all()) + B(T.range(0, M), T.all());
    }
    return C;
}

Tensor<tile_f32, 2> global_abs(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        C(T.range(0, M), T.all()) = T.abs(A(T.range(0, M), T.all()));
    }
    return C;
}

Tensor<tile_f32, 2> global_clamp(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        T.copy(A, C);
        T.clamp(C(T.range(0, M), T.all()), 0.1f, 0.9f);
    }
    return C;
}

Tensor<tile_f32, 2> global_transpose(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 16;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 2> C = T.empty(T.shape(N, M), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        T.transpose(A(T.range(0, M), T.range(0, N)),
                    C(T.range(0, N), T.range(0, M)));
    }
    return C;
}

Tensor<tile_f32, 1> global_reduce_sum(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 1> B = T.empty(T.shape(M), tile_f32{});
    for (auto [bx, by] : T.Kernel(1, 1, threads)) {
        T.reduce_sum(A(T.range(0, M), T.range(0, N)), B, /*dim=*/1);
    }
    return B;
}

// The classic tiled GEMM pattern (shared staging + PIPELINED K-loop + register
// fragment accumulator + final C copy).  With use_tensor=true this is ONE
// MxNxK GEMM partitioned into a 2D tile grid, so it rewrites to a single
// grid-wide TENSOR_MATMUL (the shared staging and pipeline dissolve).
Tensor<tile_f32, 2> tiled_gemm(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 32;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_K), tile_f16{});
        auto B_shared = T.alloc_shared(T.shape(block_K, block_N), tile_f16{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});
        T.clear(C_local);
        for (auto ko : T.Pipelined(T.ceildiv(K, block_K), num_stages)) {
            T.copy(A(by * block_M, ko * block_K), A_shared);
            T.copy(B(ko * block_K, bx * block_N), B_shared);
            T.gemm(A_shared, B_shared, C_local);
        }
        T.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

// Same classic tiled GEMM with F32 inputs/outputs.  The whole-tensor rewrite is
// deliberately F16-only (the F32 device path is a naive scalar loop slower than
// the SIMT warp-K-split partition path), so this program must fall back.
Tensor<tile_f32, 2> tiled_gemm_f32(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 32;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;
    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});
    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_K), tile_f32{});
        auto B_shared = T.alloc_shared(T.shape(block_K, block_N), tile_f32{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});
        T.clear(C_local);
        for (auto ko : T.Pipelined(T.ceildiv(K, block_K), num_stages)) {
            T.copy(A(by * block_M, ko * block_K), A_shared);
            T.copy(B(ko * block_K, bx * block_N), B_shared);
            T.gemm(A_shared, B_shared, C_local);
        }
        T.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- structural AST walkers (batching tests) --------------------------------
// The batch machinery is visible in the emitted AST as: block_id().z /
// thread_id().z member access (batch_index mapping), dispatch_size().z inside
// a LESS comparison (batch_valid guard), and CallOp::SELECT (clamped global
// offset).  These walkers search the FunctionBuilder body for those shapes.

bool expr_walk(const Expression *e, const std::function<bool(const Expression *)> &pred) {
    if (e == nullptr) { return false; }
    if (pred(e)) { return true; }
    switch (e->tag()) {
        case Expression::Tag::UNARY:
            return expr_walk(static_cast<const UnaryExpr *>(e)->operand(), pred);
        case Expression::Tag::BINARY: {
            auto *b = static_cast<const BinaryExpr *>(e);
            return expr_walk(b->lhs(), pred) || expr_walk(b->rhs(), pred);
        }
        case Expression::Tag::MEMBER:
            return expr_walk(static_cast<const MemberExpr *>(e)->self(), pred);
        case Expression::Tag::ACCESS: {
            auto *a = static_cast<const AccessExpr *>(e);
            return expr_walk(a->range(), pred) || expr_walk(a->index(), pred);
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<const CallExpr *>(e);
            for (auto arg : c->arguments()) {
                if (expr_walk(arg, pred)) { return true; }
            }
            return false;
        }
        case Expression::Tag::CAST:
            return expr_walk(static_cast<const CastExpr *>(e)->expression(), pred);
        default:
            return false;
    }
}

bool stmt_walk(const Statement *s, const std::function<bool(const Expression *)> &expr_pred) {
    if (s == nullptr) { return false; }
    switch (s->tag()) {
        case Statement::Tag::SCOPE: {
            for (auto *c : static_cast<const ScopeStmt *>(s)->statements()) {
                if (stmt_walk(c, expr_pred)) { return true; }
            }
            return false;
        }
        case Statement::Tag::IF: {
            auto *i = static_cast<const IfStmt *>(s);
            if (expr_walk(i->condition(), expr_pred)) { return true; }
            return stmt_walk(i->true_branch(), expr_pred) ||
                   stmt_walk(i->false_branch(), expr_pred);
        }
        case Statement::Tag::FOR: {
            auto *f = static_cast<const ForStmt *>(s);
            if (expr_walk(f->variable(), expr_pred) || expr_walk(f->condition(), expr_pred) ||
                expr_walk(f->step(), expr_pred)) {
                return true;
            }
            return stmt_walk(f->body(), expr_pred);
        }
        case Statement::Tag::LOOP:
            return stmt_walk(static_cast<const LoopStmt *>(s)->body(), expr_pred);
        case Statement::Tag::ASSIGN: {
            auto *a = static_cast<const AssignStmt *>(s);
            return expr_walk(a->lhs(), expr_pred) || expr_walk(a->rhs(), expr_pred);
        }
        case Statement::Tag::EXPR:
            return expr_walk(static_cast<const ExprStmt *>(s)->expression(), expr_pred);
        default:
            return false;
    }
}

// True when `e` is a component access (x/y/z) of a builtin variable, e.g.
// thread_id().z.  The lowering emits 1-component swizzles for these.
bool is_builtin_component(const Expression *e, Variable::Tag tag, uint axis) {
    if (e == nullptr || e->tag() != Expression::Tag::MEMBER) { return false; }
    auto *m = static_cast<const MemberExpr *>(e);
    if (m->self() == nullptr || m->self()->tag() != Expression::Tag::REF) { return false; }
    if (static_cast<const RefExpr *>(m->self())->variable().tag() != tag) { return false; }
    if (m->is_swizzle()) {
        return m->swizzle_size() == 1u && m->swizzle_index(0u) == axis;
    }
    return m->member_index() == axis;
}

// True when the body contains a LESS comparison against dispatch_size().z —
// the batch_valid guard that wraps every global write / atomic.
bool stmt_contains_batch_valid_less(const Statement *s) {
    auto is_dispatch_size_z = [](const Expression *e) {
        return is_builtin_component(e, Variable::Tag::DISPATCH_SIZE, 2u);
    };
    auto is_less_with_dispatch_size_z = [&](const Expression *e) {
        if (e == nullptr || e->tag() != Expression::Tag::BINARY) { return false; }
        auto *b = static_cast<const BinaryExpr *>(e);
        if (b->op() != BinaryOp::LESS) { return false; }
        return expr_walk(b->lhs(), is_dispatch_size_z) ||
               expr_walk(b->rhs(), is_dispatch_size_z);
    };
    return stmt_walk(s, is_less_with_dispatch_size_z);
}

// Collect every TENSOR_* CallExpr in the statement tree (the tensor-op
// emitters append them as void ExprStmts to the kernel body).
void collect_tensor_calls(const Statement *s, luisa::vector<const CallExpr *> &out) {
    if (s == nullptr) { return; }
    switch (s->tag()) {
        case Statement::Tag::SCOPE: {
            for (auto *c : static_cast<const ScopeStmt *>(s)->statements()) {
                collect_tensor_calls(c, out);
            }
            return;
        }
        case Statement::Tag::EXPR: {
            auto *e = static_cast<const ExprStmt *>(s)->expression();
            if (e != nullptr && e->tag() == Expression::Tag::CALL) {
                auto *call = static_cast<const CallExpr *>(e);
                if (is_tensor_operation(call->op())) { out.emplace_back(call); }
            }
            return;
        }
        default:
            return;
    }
}

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
template<typename F>
[[nodiscard]] bool terminates_with_abort(F &&f) noexcept {
    auto pid = fork();
    if (pid < 0) { return false; }
    if (pid == 0) {
        f();
        _exit(0);
    }
    auto status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
#endif
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "elementwise_add_translates_to_regular_kernel"_test = [] {
        tile::Kernel kernel{elementwise_add};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 4u * 32u && result.dispatch_size.y == 4u);
        auto block = result.function->block_size();
        expect(block.x == 32u && block.y == 1u && block.z == 1u);
        // A, B, C -> three buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 3u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        // shared + fragment allocations exist
        expect(!result.function->shared_variables().empty());
        expect(!result.function->local_variables().empty());
        // the produced builder is a kernel with a non-empty body
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        // wraps into a valid Function object
        expect(static_cast<bool>(result.function->function()));
    };

    "pipelined_matmul_translates_to_regular_kernel"_test = [] {
        tile::Kernel kernel{pipelined_matmul};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 4u * 32u && result.dispatch_size.y == 4u);
        auto block = result.function->block_size();
        expect(block.x == 32u && block.y == 1u && block.z == 1u);
        auto args = result.function->arguments();
        expect(args.size() == 3u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "rms_norm_translates_to_1d_regular_kernel"_test = [] {
        tile::Kernel kernel{rms_norm};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 8u * 64u && result.dispatch_size.y == 1u);
        auto block = result.function->block_size();
        expect(block.x == 64u && block.y == 1u && block.z == 1u);
        // A, B -> two buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 2u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "translation_preserves_launch_metadata"_test = [] {
        // 1D kernel: dispatch is (gx, 1, 1); 2D kernel: (gx, gy, 1).
        tile::Kernel k1{rms_norm};
        auto r1 = tile_to_kernel(k1.function());
        expect(r1.dispatch_size.y == 1u);

        tile::Kernel k2{elementwise_add};
        auto r2 = tile_to_kernel(k2.function());
        expect(r2.dispatch_size.x == 4u * 32u && r2.dispatch_size.y == 4u);
    };

    "int8_tensors_lower_to_byte_buffers"_test = [] {
        tile::Kernel kernel{int8_copy_kernel};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        // T.Kernel(1, 8): 1 block of 8 threads -> dispatch.x = 1 * 8
        expect(result.dispatch_size.x == 8u && result.dispatch_size.y == 1u);
        auto block = result.function->block_size();
        expect(block.x == 8u && block.y == 1u && block.z == 1u);
        // A, C -> two buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 2u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "fp8_tensors_lower_with_zero_fill"_test = [] {
        tile::Kernel kernel{fp8_clear_kernel};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        // T.Kernel(1, 8): 1 block of 8 threads -> dispatch.x = 1 * 8
        expect(result.dispatch_size.x == 8u && result.dispatch_size.y == 1u);
        auto block = result.function->block_size();
        expect(block.x == 8u && block.y == 1u && block.z == 1u);
        // C -> one buffer argument (fp8 e4m3 element type)
        auto args = result.function->arguments();
        expect(args.size() == 1u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "erf_fast_math_lowers_in_software"_test = [] {
        tile::Kernel kernel{erf_kernel};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 4u * 32u && result.dispatch_size.y == 4u);
        auto block = result.function->block_size();
        expect(block.x == 32u && block.y == 1u && block.z == 1u);
        // A, C -> two buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 2u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
        // The software erf must not rely on an "erf" external callable.
        expect(result.function->external_callables().empty());
        // ... and must be composed of backend-supported ops.
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::EXP)) << "software erf uses exp(-x^2)";
        expect(calls.test(CallOp::ABS)) << "software erf uses |x|";
        expect(calls.test(CallOp::SELECT)) << "software erf applies sign(x)";
    };

    "batching_disabled_by_default"_test = [] {
        tile::Kernel kernel{elementwise_add};
        auto result = tile_to_kernel(kernel.function());// default config
        auto block = result.function->block_size();
        expect(block.z == 1u);
        expect(result.dispatch_size.x == 4u * 32u && result.dispatch_size.y == 4u);
        // disabled batching adds zero overhead: no SELECT batch clamp is emitted
        auto calls = result.function->direct_builtin_callables();
        expect(!calls.test(CallOp::SELECT));
    };

    "batching_enabled_selects_z_block"_test = [] {
        // erf_kernel: threads=32, no shared -> memory/IO-bound target 128 -> B_z=4
        {
            tile::Kernel kernel{erf_kernel};
            auto result = tile_to_kernel(kernel.function(),
                                         TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 64u});
            auto block = result.function->block_size();
            expect(block.z == 4u);
            expect(block.z >= 1u && block.z <= 8u && 32u * block.z <= 1024u);
        }
        // elementwise_add: threads=32, shared -> compute/LDS-bound target 256 -> B_z=8
        {
            tile::Kernel kernel{elementwise_add};
            auto result = tile_to_kernel(kernel.function(),
                                         TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 64u});
            auto block = result.function->block_size();
            expect(block.z == 8u);
            expect(block.z >= 1u && block.z <= 8u && 32u * block.z <= 1024u);
        }
    };

    "batching_slices_shared_and_offsets_global"_test = [] {
        tile::Kernel kernel{elementwise_add};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 64u});
        auto block = result.function->block_size();
        expect(block.z == 8u);// B_z
        // shared A/B tiles: 8x8 = 64 elements each -> 64 * B_z = 512 per slice set
        auto shared = result.function->shared_variables();
        expect(shared.size() == 2u);
        uint32_t shared_elements = 0u;
        for (auto v : shared) {
            expect(v.type() != nullptr && v.type()->is_array());
            if (v.type() != nullptr && v.type()->is_array()) { shared_elements += v.type()->dimension(); }
        }
        expect(shared_elements == 2u * 64u * 8u);
        // global accesses go through the clamped batch offset (CallOp::SELECT)
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::SELECT));
        expect(calls.test(CallOp::BUFFER_READ));
        expect(calls.test(CallOp::BUFFER_WRITE));
    };

    "batching_tail_block_guard"_test = [] {
        tile::Kernel kernel{elementwise_add};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 64u});
        auto block = result.function->block_size();
        expect(block.z == 8u);
        auto *body = result.function->body();
        // batch_index mapping: block_id().z and thread_id().z are used
        expect(stmt_walk(body, [](const Expression *e) {
            return is_builtin_component(e, Variable::Tag::BLOCK_ID, 2u);
        }));
        expect(stmt_walk(body, [](const Expression *e) {
            return is_builtin_component(e, Variable::Tag::THREAD_ID, 2u);
        }));
        // batch_valid guard: LESS comparison against dispatch_size().z wraps
        // the guarded global writes (idle tz threads of the tail z-block)
        expect(stmt_contains_batch_valid_less(body));
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::SELECT));
        expect(calls.test(CallOp::BUFFER_WRITE));
    };

    "batching_warp_alignment_guard"_test = [] {
        // Positive: batched elementwise kernel (no warp collectives, threads=32)
        // must NOT abort even though 32 % 64 != 0.
        {
            tile::Kernel kernel{elementwise_add};
            auto result = tile_to_kernel(kernel.function(),
                                         TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 64u});
            expect(result.function != nullptr);
        }
        // Positive (warp-collective): rms_norm uses threads=64 (a multiple of
        // 64) and REDUCE_SUM, so batching must lower without abort and select a
        // B_z > 1 z-block (compute/LDS-bound: 512-elem shared-backed fragments
        // -> target 256 -> B_z = ceil(256/64) = 4).  The per-slice warp
        // partition of _emit_tile_reduce is device-verified in
        // examples/tensor/main.cpp (batched rms_norm).
        {
            tile::Kernel kernel{rms_norm};
            auto result = tile_to_kernel(kernel.function(),
                                         TileToKernelConfig{.min_batching_size = 4u, .max_batching_size = 16u});
            expect(result.function != nullptr);
            expect(result.function->block_size().z > 1u);
        }
        // Negative: batched reduce kernel (threads=32, NOT a multiple of 64)
        // aborts via the 2.10 LUISA_ASSERT — subprocess death test (POSIX
        // only, mirroring test_xir2ast_translators.cpp).  On non-POSIX the
        // abort cannot be caught in-process; the host-side assert contract
        // covers it.
#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
        expect(terminates_with_abort([] {
            tile::Kernel kernel{reduce32_kernel};
            (void)tile_to_kernel(kernel.function(),
                                 TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 64u});
        }));
#endif
    };

    "batching_rejects_invalid_config"_test = [] {
        // min=0 / max=0 / min>max abort via LUISA_ASSERT (logging.h) and cannot
        // be caught in-process — documented.  Subprocess death tests where the
        // platform supports fork(); elsewhere rely on the assert contract.
#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
        tile::Kernel kernel{elementwise_add};
        expect(terminates_with_abort([&] {
            (void)tile_to_kernel(kernel.function(),
                                 TileToKernelConfig{.min_batching_size = 0u, .max_batching_size = 8u});
        }));
        expect(terminates_with_abort([&] {
            (void)tile_to_kernel(kernel.function(),
                                 TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 0u});
        }));
        expect(terminates_with_abort([&] {
            (void)tile_to_kernel(kernel.function(),
                                 TileToKernelConfig{.min_batching_size = 8u, .max_batching_size = 4u});
        }));
#endif
    };

    // ---- tensor-op fast path (use_tensor) -----------------------------------
    // Each whole-tensor kernel must lower to exactly ONE TENSOR_* call with
    // valid descriptor arg layouts (check_builtin_call_valid) and NO
    // per-element BUFFER_READ/BUFFER_WRITE.

    "tensor_op_global_copy"_test = [] {
        tile::Kernel kernel{global_copy};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        // Phase A + vector dispatch: the whole-tensor f32 copy processes one
        // float4 per thread, so ceil(1024/4/32)*32 = 256 threads on the flat
        // 1D grid.
        expect(result.dispatch_size.x == 32u * 8u && result.dispatch_size.y == 1u);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_COPY));
        expect(!calls.test(CallOp::BUFFER_READ));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 1u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_COPY);
        check_builtin_call_valid(tensor_calls[0]->op(), tensor_calls[0]->type(),
                                 tensor_calls[0]->arguments());
        // A, C -> two buffer arguments (unchanged contract)
        auto args = result.function->arguments();
        expect(args.size() == 2u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
    };

    "tensor_op_global_fill_clear"_test = [] {
        tile::Kernel kernel{global_fill_clear};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_FILL));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        // T.clear(C) + T.fill(C, 0.5) -> two whole-tensor fills
        expect(tensor_calls.size() == 2u);
        for (auto *c : tensor_calls) {
            expect(c->op() == CallOp::TENSOR_FILL);
            check_builtin_call_valid(c->op(), c->type(), c->arguments());
        }
    };

    "tensor_op_global_add"_test = [] {
        tile::Kernel kernel{global_add};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_ADD));
        expect(!calls.test(CallOp::BUFFER_READ));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 1u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_ADD);
        check_builtin_call_valid(tensor_calls[0]->op(), tensor_calls[0]->type(),
                                 tensor_calls[0]->arguments());
    };

    "tensor_op_global_abs"_test = [] {
        tile::Kernel kernel{global_abs};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_ABS));
        expect(!calls.test(CallOp::BUFFER_READ));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 1u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_ABS);
        check_builtin_call_valid(tensor_calls[0]->op(), tensor_calls[0]->type(),
                                 tensor_calls[0]->arguments());
    };

    "tensor_op_global_clamp"_test = [] {
        tile::Kernel kernel{global_clamp};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_COPY));
        expect(calls.test(CallOp::TENSOR_CLAMP));
        expect(!calls.test(CallOp::BUFFER_READ));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 2u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_COPY);
        expect(tensor_calls[1]->op() == CallOp::TENSOR_CLAMP);
        for (auto *c : tensor_calls) {
            check_builtin_call_valid(c->op(), c->type(), c->arguments());
        }
    };

    "tensor_op_global_transpose"_test = [] {
        tile::Kernel kernel{global_transpose};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        // Phase A: f32 transpose iterates the dst domain (512 elements) with
        // float4 vector dispatch -> ceil(512/4/32)*32 = 128 threads.
        expect(result.dispatch_size.x == 32u * 4u && result.dispatch_size.y == 1u);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_PERMUTE));
        expect(!calls.test(CallOp::BUFFER_READ));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 1u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_PERMUTE);
        check_builtin_call_valid(tensor_calls[0]->op(), tensor_calls[0]->type(),
                                 tensor_calls[0]->arguments());
    };

    "tensor_op_global_reduce_sum"_test = [] {
        tile::Kernel kernel{global_reduce_sum};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        // Phase A: reduce iterates the OUTPUT domain (32 rows -> 32 threads).
        expect(result.dispatch_size.x == 32u && result.dispatch_size.y == 1u);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_REDUCE_SUM));
        expect(!calls.test(CallOp::BUFFER_READ));
        expect(!calls.test(CallOp::BUFFER_WRITE));
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 1u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_REDUCE_SUM);
        check_builtin_call_valid(tensor_calls[0]->op(), tensor_calls[0]->type(),
                                 tensor_calls[0]->arguments());
    };

    "tensor_op_tiled_gemm_rewrites"_test = [] {
        // The classic tiled GEMM (shared staging + PIPELINED K-loop + fragment
        // accumulator + final C copy) is recognized as ONE grid-wide GEMM and
        // rewritten to a single TENSOR_MATMUL when use_tensor=true.  The
        // rewrite is end-to-end verified on the CUDA backend (F16 WMMA
        // tensor-core path, FP32 accumulator; beta=0 because the final copy
        // overwrites C_global).  The shared/fragment/pipelined staging must
        // dissolve entirely.
        tile::Kernel kernel{tiled_gemm};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(calls.test(CallOp::TENSOR_MATMUL));
        expect(calls.uses_tensor_ops());
        expect(!result.function->shared_variables().empty() == false);
        expect(!result.function->local_variables().empty() == false);
        luisa::vector<const CallExpr *> tensor_calls;
        collect_tensor_calls(result.function->body(), tensor_calls);
        expect(tensor_calls.size() == 1u);
        expect(tensor_calls[0]->op() == CallOp::TENSOR_MATMUL);
        check_builtin_call_valid(tensor_calls[0]->op(), tensor_calls[0]->type(),
                                 tensor_calls[0]->arguments());
        // 64x64x32 GEMM -> 16 16x16 tiles, one warp (32 threads) per tile.
        expect(result.dispatch_size.x == 16u * 32u);
        expect(result.dispatch_size.y == 1u);
    };

    "tensor_op_to_kernel_config_forwarding"_test = [] {
        // tile::jit(...).compile().to_kernel<Dim>(TileToKernelConfig) must
        // forward the config into tile_to_kernel — calling it WITHOUT the
        // config silently re-lowers with use_tensor=false (partition path),
        // which previously invalidated CUDA e2e verification.
        auto kernel = luisa::compute::tile::jit(global_copy).compile();
        auto typed = kernel.to_kernel<2>(TileToKernelConfig{.use_tensor = true});
        expect(typed.function() != nullptr);
        expect(typed.function()->direct_builtin_callables().test(CallOp::TENSOR_COPY));
        auto typed_default = kernel.to_kernel<2>();
        expect(typed_default.function() != nullptr);
        expect(!typed_default.function()->direct_builtin_callables().uses_tensor_ops());
    };

    "tensor_op_f32_gemm_falls_back"_test = [] {
        // The whole-tensor rewrite is deliberately F16-only: the backend's F32
        // GEMM device function is a naive scalar loop that is slower than the
        // SIMT warp-K-split partition path, so an F32 tiled GEMM with
        // use_tensor=true must keep the partition path (shared/fragment/
        // pipelined loops, zero TENSOR_*).
        tile::Kernel kernel{tiled_gemm_f32};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(!calls.uses_tensor_ops());
        expect(!result.function->shared_variables().empty());
        expect(!result.function->local_variables().empty());
    };

    "tensor_op_default_config_keeps_partition_path"_test = [] {
        // use_tensor defaults to false: tiled_gemm lowers through the existing
        // partition path (shared + fragment + pipelined loops), no TENSOR_*.
        tile::Kernel kernel{tiled_gemm};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(!calls.uses_tensor_ops());
        expect(!result.function->shared_variables().empty());
        expect(!result.function->local_variables().empty());
    };

    "tensor_op_ineligible_program_falls_back"_test = [] {
        // elementwise_add uses shared/fragment staging -> the whole program is
        // ineligible for the tensor-op path -> zero TENSOR_* calls even with
        // use_tensor=true.
        tile::Kernel kernel{elementwise_add};
        auto result = tile_to_kernel(kernel.function(),
                                     TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        auto calls = result.function->direct_builtin_callables();
        expect(!calls.uses_tensor_ops());
        expect(calls.test(CallOp::BUFFER_READ));
        expect(calls.test(CallOp::BUFFER_WRITE));
    };
}
