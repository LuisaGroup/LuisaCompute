// Test for the tile function builder in <luisa/ast/tile_function_builder.h>.
// This test covers:
// - define()/current()/current_or_null()/stack() function-stack management
// - RAII function/scope guards and with()/inside_function_scope()
// - every tile operator emitting the matching TensorStmt into the body
//   (empty, alloc_shared, alloc_fragment, clear, copy, gemm, reduce_sum,
//    print, store, binary, max, rsqrt, ceildiv, kernel_1d, kernel_2d,
//    pipelined)
// - alloc operators returning correctly laid-out TensorExpr
// - value operators returning caller-owned fragment temporaries
// - name config
//
// Pure host code: no device / backend is required.

#include "ut/ut.hpp"

#include <cstdint>

#include <luisa/ast/tile_function_builder.h>
#include <luisa/ast/function_builder.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::detail;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Expression nodes can only be constructed while a FunctionBuilder is on the
// builder stack (Expression::Expression reads FunctionBuilder::current()).
// This helper materializes R2 literal payloads under a short-lived guard; the
// tile function builder only borrows them.
template<typename F>
void with_function_builder(F &&f) {
    FunctionBuilder builder;
    FunctionBuilder::FunctionStackGuard guard{&builder};
    f();
}

const LiteralExpr *new_literal(const Type *type, LiteralExpr::Value value) {
    const LiteralExpr *out = nullptr;
    with_function_builder([&] { out = new LiteralExpr{type, std::move(value)}; });
    return out;
}

// Fresh rank-2 float global tensor; ownership transfers to the statement it
// is passed to (a TensorStmt owns its output and input TensorExpr operands).
TensorExpr *make_tile() {
    return new TensorExpr{2, TensorElementType::F32, TensorScope::Global, {16, 16}};
}

bool same_span(luisa::span<const int32_t> a, std::initializer_list<int32_t> b) {
    if (a.size() != b.size()) { return false; }
    size_t i = 0u;
    for (auto v : b) {
        if (a[i++] != v) { return false; }
    }
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "define_and_stack"_test = [] {
        expect(TileFunctionBuilder::current_or_null() == nullptr);
        expect(TileFunctionBuilder::stack().empty());
        auto f = TileFunctionBuilder::define([&] {
            auto *builder = TileFunctionBuilder::current();
            expect(builder != nullptr);
            expect(builder->inside_function_scope());
            builder->set_name("matmul_tile");
            expect(TileFunctionBuilder::stack().size() == 1u);
            expect(TileFunctionBuilder::stack().front() == builder);
        });
        expect(f != nullptr);
        expect(luisa::string_view{f->name()} == "matmul_tile");
        expect(f->body() != nullptr);
        expect(f->body()->empty());
        expect(TileFunctionBuilder::current_or_null() == nullptr);
        expect(TileFunctionBuilder::stack().empty());
    };

    "scope_guard_and_with"_test = [] {
        auto f = TileFunctionBuilder::define([&] {
            auto *builder = TileFunctionBuilder::current();
            TileFunctionBuilder::TileScope inner;
            builder->with(&inner, [&] {
                expect(!builder->inside_function_scope());
                builder->tile_clear(make_tile());
            });
            expect(builder->inside_function_scope());
            expect(inner.size() == 1u);
            expect(inner.statements().front()->op() == TileOpKind::CLEAR);
        });
        // statements appended to the inner scope are owned by the builder
        expect(f->body()->empty());
    };

    "tile_operators"_test = [] {
        auto *eps = new_literal(Type::of<float>(), 1e-12f);
        auto *scale = new_literal(Type::of<float>(), 2.0f);

        luisa::unique_ptr<TensorExpr> tmp_binary;
        luisa::unique_ptr<TensorExpr> tmp_max;
        luisa::unique_ptr<TensorExpr> tmp_rsqrt;
        int32_t ceildiv_result = 0;

        auto f = TileFunctionBuilder::define([&] {
            auto *b = TileFunctionBuilder::current();

            // alloc operators return tensors owned by their AllocStmt
            auto *A = b->tile_empty({16, 16}, TensorElementType::F32);
            auto *As = b->tile_alloc_shared({8, 8}, TensorElementType::F16);
            auto *Af = b->tile_alloc_fragment({4, 8}, TensorElementType::F32);
            expect(A != nullptr);
            expect(As != nullptr);
            expect(Af != nullptr);
            expect(A->scope() == TensorScope::Global);
            expect(As->scope() == TensorScope::Shared);
            expect(Af->scope() == TensorScope::Fragment);
            expect(A->dtype() == TensorElementType::F32);
            expect(As->dtype() == TensorElementType::F16);
            expect(Af->dtype() == TensorElementType::F32);
            expect(same_span(A->dims(), {16, 16}));
            expect(same_span(As->dims(), {8, 8}));
            expect(same_span(Af->dims(), {4, 8}));

            // each statement owns its own operands, so fresh tensors are used
            b->tile_clear(make_tile());
            b->tile_copy(make_tile(), make_tile());
            b->tile_gemm(make_tile(), make_tile(), make_tile(), 1, 0);
            b->tile_reduce_sum(make_tile(), make_tile(), 1u);
            b->tile_print(make_tile(), "hello tile");
            b->tile_store(0, make_tile(), make_tile());
            b->tile_store(1, make_tile(), nullptr, scale);
            tmp_binary = b->tile_binary(BinaryOp::ADD, make_tile(), make_tile());
            tmp_max = b->tile_max(make_tile(), eps);
            tmp_rsqrt = b->tile_rsqrt(make_tile());
            ceildiv_result = b->tile_ceildiv(17, 4);
            b->tile_kernel_1d(256, 128);
            b->tile_kernel_2d(8, 16, 32);
            b->tile_pipelined(64, 3);
        });

        auto statements = f->body()->statements();
        expect(statements.size() == 17u);

        const TileOpKind expected[] = {
            TileOpKind::ALLOC,      TileOpKind::ALLOC,      TileOpKind::ALLOC,
            TileOpKind::CLEAR,      TileOpKind::COPY,       TileOpKind::GEMM,
            TileOpKind::REDUCE_SUM, TileOpKind::PRINT,      TileOpKind::STORE,
            TileOpKind::STORE,      TileOpKind::BINARY,     TileOpKind::MAX,
            TileOpKind::RSQRT,      TileOpKind::CEILDIV,    TileOpKind::KERNEL_1D,
            TileOpKind::KERNEL_2D,  TileOpKind::PIPELINED};
        for (auto i = 0u; i < statements.size(); ++i) {
            expect(statements[i]->op() == expected[i]);
        }

        auto *alloc0 = static_cast<const AllocStmt *>(statements[0]);
        expect(alloc0->scope() == TensorScope::Global);
        expect(same_span(alloc0->dims(), {16, 16}));
        auto *alloc1 = static_cast<const AllocStmt *>(statements[1]);
        expect(alloc1->scope() == TensorScope::Shared);
        expect(same_span(alloc1->dims(), {8, 8}));
        auto *alloc2 = static_cast<const AllocStmt *>(statements[2]);
        expect(alloc2->scope() == TensorScope::Fragment);
        expect(same_span(alloc2->dims(), {4, 8}));

        auto *clear = static_cast<const ClearStmt *>(statements[3]);
        expect(clear->t() != nullptr);
        auto *copy = static_cast<const CopyStmt *>(statements[4]);
        expect(copy->src() != nullptr && copy->dst() != nullptr);

        auto *gemm = static_cast<const GemmStmt *>(statements[5]);
        expect(gemm->trans_a() == 1);
        expect(gemm->trans_b() == 0);
        expect(gemm->a() != nullptr && gemm->b() != nullptr && gemm->c() != nullptr);
        expect(gemm->c() == gemm->output());

        auto *reduce = static_cast<const ReduceSumStmt *>(statements[6]);
        expect(reduce->dim() == 1u);
        expect(reduce->x() != nullptr && reduce->y() != nullptr);

        auto *print = static_cast<const TilePrintStmt *>(statements[7]);
        expect(print->t() != nullptr);
        expect(print->msg() == "hello tile");

        auto *store0 = static_cast<const TileStoreStmt *>(statements[8]);
        expect(store0->op() == 0);
        expect(store0->lhs() != nullptr);
        expect(store0->rhs_tensor() != nullptr);
        expect(store0->rhs_literal() == nullptr);
        expect(store0->rhs_ref() == nullptr);

        auto *store1 = static_cast<const TileStoreStmt *>(statements[9]);
        expect(store1->op() == 1);
        expect(store1->rhs_tensor() == nullptr);
        expect(store1->rhs_literal() == scale);

        auto *binary = static_cast<const TileBinaryStmt *>(statements[10]);
        expect(binary->op() == BinaryOp::ADD);
        expect(binary->lhs() != nullptr);
        expect(binary->rhs_tensor() != nullptr);

        auto *max = static_cast<const MaxStmt *>(statements[11]);
        expect(max->a() != nullptr);
        expect(max->b() == eps);

        auto *rsqrt = static_cast<const RsqrtStmt *>(statements[12]);
        expect(rsqrt->a() != nullptr);

        auto *ceildiv = static_cast<const CeilDivStmt *>(statements[13]);
        expect(ceildiv->result() == 5);
        expect(ceildiv_result == 5);

        auto *k1 = static_cast<const Kernel1DStmt *>(statements[14]);
        expect(k1->gx() == 256);
        expect(k1->threads() == 128);
        expect(k1->bx() == nullptr);

        auto *k2 = static_cast<const Kernel2DStmt *>(statements[15]);
        expect(k2->gx() == 8);
        expect(k2->gy() == 16);
        expect(k2->threads() == 32);

        auto *pipe = static_cast<const PipelinedStmt *>(statements[16]);
        expect(pipe->count() == 64);
        expect(pipe->stages() == 3);

        // value operators return caller-owned fragment temporaries
        expect(tmp_binary != nullptr);
        expect(tmp_binary->scope() == TensorScope::Fragment);
        expect(same_span(tmp_binary->dims(), {16, 16}));
        expect(tmp_max != nullptr);
        expect(tmp_max->scope() == TensorScope::Fragment);
        expect(same_span(tmp_max->dims(), {16, 16}));
        expect(tmp_rsqrt != nullptr);
        expect(tmp_rsqrt->scope() == TensorScope::Fragment);
        expect(same_span(tmp_rsqrt->dims(), {16, 16}));
    };

    "kernel_control_flow"_test = [] {
        auto f = TileFunctionBuilder::define([&] {
            auto *b = TileFunctionBuilder::current();
            b->tile_kernel_1d(64, 32);
            b->tile_pipelined(16, 4);
            b->tile_kernel_2d(4, 8, 16);
        });
        auto statements = f->body()->statements();
        expect(statements.size() == 3u);
        expect(statements[0]->op() == TileOpKind::KERNEL_1D);
        expect(statements[1]->op() == TileOpKind::PIPELINED);
        expect(statements[2]->op() == TileOpKind::KERNEL_2D);
        auto *k1 = static_cast<const Kernel1DStmt *>(statements[0]);
        expect(k1->gx() == 64 && k1->threads() == 32);
        auto *pipe = static_cast<const PipelinedStmt *>(statements[1]);
        expect(pipe->count() == 16 && pipe->stages() == 4);
        auto *k2 = static_cast<const Kernel2DStmt *>(statements[2]);
        expect(k2->gx() == 4 && k2->gy() == 8 && k2->threads() == 16);
    };
}
