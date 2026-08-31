// Test for the tensor AST nodes in <luisa/ast/tensor.h>.
// This test covers:
// - TensorElementType / scope name helpers
// - TensorExpr layout construction, accessors, describe()
// - TensorStmt base members: op / output / inputs / annotations
// - serialize()/deserialize() round-trips of every tensor statement
//   (Gemm, Clear, Copy, ReduceSum, TilePrint, Alloc, TileStore, TileBinary,
//    Max, Rsqrt, CeilDiv, Kernel1D, Kernel2D, Pipelined)
// - literal (R2) and string (R2 string) payload serialization
// - malformed-input / kind-mismatch deserialization failures
//
// Pure host code: no device / backend is required.

#include "ut/ut.hpp"

#include <cstdint>
#include <cstring>

#include <luisa/ast/tensor.h>
#include <luisa/ast/function_builder.h>
#include <luisa/core/stl/variant.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Serialize `stmt`, then deserialize the same buffer into `out`.
// Returns true when deserialization succeeded, consumed the whole buffer and
// the reported serialize size matched the appended bytes.
template<typename Stmt>
bool roundtrip(Stmt *stmt, Stmt &out) {
    luisa::vector<char> buf;
    auto n = stmt->serialize(buf);
    char const *p = buf.data();
    char const *end = p + buf.size();
    if (!out.deserialize(p, end)) { return false; }
    return p == end && n == buf.size();
}

// Build a fresh tensor of rank 2, dtype float, global scope, extent 16x16.
TensorExpr *make_tile(luisa::fixed_vector<int32_t, 4> offset = {}) {
    return new TensorExpr{2, TensorElementType::F32, TensorScope::Global,
                          {16, 16}, std::move(offset), {}};
}

bool same_span(luisa::span<const int32_t> a, std::initializer_list<int32_t> b) {
    if (a.size() != b.size()) { return false; }
    size_t i = 0u;
    for (auto v : b) {
        if (a[i++] != v) { return false; }
    }
    return true;
}

// Expression nodes can only be constructed while a FunctionBuilder is on the
// stack (Expression::Expression reads FunctionBuilder::current()).  This
// helper materializes R2 literal constants under a short-lived guard.
template<typename F>
void with_builder(F &&f) {
    luisa::compute::detail::FunctionBuilder builder;
    luisa::compute::detail::FunctionBuilder::FunctionStackGuard guard{&builder};
    f();
}

const LiteralExpr *new_literal(const Type *type, LiteralExpr::Value value) {
    const LiteralExpr *out = nullptr;
    with_builder([&] { out = new LiteralExpr{type, std::move(value)}; });
    return out;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "tensor_element_type_name_and_scope_name"_test = [] {
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::F16)} == "half");
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::F32)} == "float");
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::I32)} == "int");
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::I8)} == "int8");
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::FP8)} == "fp8");
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::I4)} == "int4");
        expect(luisa::string_view{tensor_element_type_name(TensorElementType::FP4)} == "fp4");
        expect(luisa::string_view{scope_name(TensorScope::Global)} == "global");
        expect(luisa::string_view{scope_name(TensorScope::Shared)} == "shared");
        expect(luisa::string_view{scope_name(TensorScope::Fragment)} == "fragment");
    };

    "tensor_expr_all_dtypes_roundtrip"_test = [] {
        // every TensorElementType tag survives the binary layout round-trip
        for (auto e : {TensorElementType::F16, TensorElementType::F32,
                       TensorElementType::I32, TensorElementType::I8,
                       TensorElementType::FP8, TensorElementType::I4,
                       TensorElementType::FP4}) {
            TensorExpr t{2, e, TensorScope::Global, {8, 16}};
            expect(t.dtype() == e);
            luisa::vector<char> buf;
            auto n = t.serialize(buf);
            TensorExpr out;
            char const *p = buf.data();
            char const *end = p + buf.size();
            expect(out.deserialize(p, end));
            expect(p == end);
            expect(n == buf.size());
            expect(out.dtype() == e);
            expect(same_span(out.dims(), {8, 16}));
        }
    };

    "tensor_expr_layout"_test = [] {
        TensorExpr t{2, TensorElementType::F32, TensorScope::Global, {16, 16}, {8, 0}};
        expect(t.rank() == 2);
        expect(t.dtype() == TensorElementType::F32);
        expect(t.scope() == TensorScope::Global);
        expect(same_span(t.dims(), {16, 16}));
        // explicit offset preserved, extent defaults to the whole-tensor dims
        expect(same_span(t.offset(), {8, 0}));
        expect(same_span(t.extent(), {16, 16}));
        expect(t.handle() == nullptr);
        expect(luisa::string_view{t.describe()}.find("float") != luisa::string_view::npos);

        // whole-tensor view: offset defaults to zeros
        TensorExpr w{1, TensorElementType::I32, TensorScope::Shared, {32}};
        expect(same_span(w.offset(), {0}));
        expect(same_span(w.extent(), {32}));
        expect(w.scope() == TensorScope::Shared);
    };

    "tensor_expr_serialize"_test = [] {
        // a handle (R3 kernel variable) is borrowed from a FunctionBuilder and
        // must NOT be serialized: after the round-trip it is null again.
        const RefExpr *handle = nullptr;
        auto kernel = luisa::compute::detail::FunctionBuilder::define_kernel([&] {
            handle = luisa::compute::detail::FunctionBuilder::current()
                         ->local(Type::of<float>());
        });
        expect(handle != nullptr);
        expect(kernel != nullptr);

        TensorExpr t{2, TensorElementType::F32, TensorScope::Fragment, {8, 16}, {4, 8}, {8, 16}, handle};
        expect(t.handle() == handle);

        luisa::vector<char> buf;
        auto n = t.serialize(buf);
        expect(n == buf.size());
        expect(!buf.empty());

        TensorExpr out;
        char const *p = buf.data();
        char const *end = p + buf.size();
        expect(out.deserialize(p, end));
        expect(p == end);
        expect(out.rank() == 2);
        expect(out.dtype() == TensorElementType::F32);
        expect(out.scope() == TensorScope::Fragment);
        expect(same_span(out.dims(), {8, 16}));
        expect(same_span(out.offset(), {4, 8}));
        expect(same_span(out.extent(), {8, 16}));
        expect(out.handle() == nullptr);// pointer, non-serializable
    };

    "tensor_stmt_base"_test = [] {
        TensorStmt *stmt = new ClearStmt{make_tile()};
        stmt->set_annotation("coalesced_width", 32);
        stmt->set_annotation("stages", 3);
        stmt->set_annotation("coalesced_width", 64);// overwrite
        expect(stmt->op() == TileOpKind::CLEAR);
        expect(stmt->output() != nullptr);
        expect(stmt->inputs().empty());
        expect(stmt->annotations().size() == 2u);
        auto *cw = stmt->annotation("coalesced_width");
        auto *st = stmt->annotation("stages");
        auto *none = stmt->annotation("nope");
        expect(cw != nullptr && *cw == 64);
        expect(st != nullptr && *st == 3);
        expect(none == nullptr);
        delete stmt;
    };

    "stmt_gemm"_test = [] {
        auto *a = make_tile({0, 0});
        auto *b = make_tile({0, 16});
        auto *c = make_tile({0, 0});
        GemmStmt stmt{a, b, c, 1, 0};
        expect(stmt.op() == TileOpKind::GEMM);
        expect(stmt.a() == a);
        expect(stmt.b() == b);
        expect(stmt.c() == c);
        expect(stmt.trans_a() == 1);
        expect(stmt.trans_b() == 0);
        expect(stmt.output() == c);
        expect(stmt.inputs().size() == 2u);

        GemmStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.trans_a() == 1);
        expect(out.trans_b() == 0);
        expect(out.c() != nullptr && out.c() != c);
        expect(out.a() != nullptr && out.b() != nullptr);
        expect(same_span(out.a()->dims(), {16, 16}));
        expect(same_span(out.b()->offset(), {0, 16}));
        expect(out.a()->dtype() == TensorElementType::F32);
    };

    "stmt_clear"_test = [] {
        auto *t = make_tile();
        ClearStmt stmt{t};
        expect(stmt.t() == t);
        expect(stmt.output() == t);
        ClearStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.t() != nullptr);
        expect(same_span(out.t()->dims(), {16, 16}));
    };

    "stmt_copy"_test = [] {
        auto *src = make_tile({0, 0});
        auto *dst = make_tile({16, 0});
        CopyStmt stmt{src, dst};
        expect(stmt.src() == src);
        expect(stmt.dst() == dst);
        CopyStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.src() != nullptr && out.dst() != nullptr);
        expect(same_span(out.src()->offset(), {0, 0}));
        expect(same_span(out.dst()->offset(), {16, 0}));
    };

    "stmt_reduce_sum"_test = [] {
        auto *x = make_tile();
        auto *y = make_tile();
        ReduceSumStmt stmt{x, y, 1u};
        expect(stmt.x() == x);
        expect(stmt.y() == y);
        expect(stmt.dim() == 1u);

        ReduceSumStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.x() != nullptr && out.y() != nullptr);
        expect(out.dim() == 1u);
    };

    "stmt_print"_test = [] {
        auto *t = make_tile();
        TilePrintStmt stmt{t, "hello tile"};
        expect(stmt.t() == t);
        expect(stmt.msg() == "hello tile");
        expect(stmt.output() == nullptr);

        TilePrintStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.t() != nullptr);
        expect(out.msg() == "hello tile");
    };

    "stmt_alloc"_test = [] {
        AllocStmt stmt{{16, 16}, TensorElementType::F16, TensorScope::Shared};
        expect(stmt.rank() == 2);
        expect(stmt.dtype() == TensorElementType::F16);
        expect(stmt.scope() == TensorScope::Shared);
        expect(same_span(stmt.dims(), {16, 16}));
        expect(stmt.tensor() == stmt.output());

        AllocStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.rank() == 2);
        expect(out.dtype() == TensorElementType::F16);
        expect(out.scope() == TensorScope::Shared);
        expect(same_span(out.dims(), {16, 16}));
    };

    "stmt_tile_store_tensor_rhs"_test = [] {
        auto *lhs = make_tile();
        auto *rhs = make_tile();
        TileStoreStmt stmt{0, lhs, rhs};// lhs = rhs
        expect(stmt.lhs() == lhs);
        expect(stmt.rhs_tensor() == rhs);
        expect(stmt.rhs_literal() == nullptr);
        expect(stmt.op() == 0);

        TileStoreStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == 0);
        expect(out.lhs() != nullptr);
        expect(out.rhs_tensor() != nullptr);
        expect(out.rhs_literal() == nullptr);
        expect(same_span(out.lhs()->dims(), {16, 16}));
    };

    "stmt_tile_store_literal_rhs"_test = [] {
        auto *lhs = make_tile();
        auto *v = new_literal(Type::of<float>(), 2.0f);
        TileStoreStmt stmt{1, lhs, nullptr, v};// lhs *= 2.0f (row-broadcast)
        expect(stmt.op() == 1);
        expect(stmt.rhs_tensor() == nullptr);
        expect(stmt.rhs_literal() == v);
        expect(stmt.inputs().empty());

        TileStoreStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == 1);
        expect(out.rhs_tensor() == nullptr);
        expect(out.rhs_literal() != nullptr && out.rhs_literal() != v);
        expect(std::abs(luisa::get<float>(out.rhs_literal()->value().to_variant()) - 2.0f) < 1e-6f);
    };

    "stmt_tile_binary"_test = [] {
        auto *lhs = make_tile();
        auto *rhs = make_tile();
        TileBinaryStmt stmt{BinaryOp::MUL, lhs, rhs};
        expect(stmt.op() == BinaryOp::MUL);
        expect(stmt.lhs() == lhs);
        expect(stmt.rhs_tensor() == rhs);
        expect(stmt.rhs_literal() == nullptr);
        expect(stmt.output() == nullptr);
        expect(stmt.inputs().size() == 2u);

        TileBinaryStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == BinaryOp::MUL);
        expect(out.lhs() != nullptr);
        expect(out.rhs_tensor() != nullptr);
        expect(out.rhs_literal() == nullptr);

        // scalar-constant rhs: A + 2.0f
        auto *lhs2 = make_tile();
        auto *v = new_literal(Type::of<float>(), 2.0f);
        TileBinaryStmt stmt2{BinaryOp::ADD, lhs2, nullptr, v};
        TileBinaryStmt out2;
        expect(roundtrip(&stmt2, out2));
        expect(out2.op() == BinaryOp::ADD);
        expect(out2.rhs_tensor() == nullptr);
        expect(out2.rhs_literal() != nullptr);
        expect(std::abs(luisa::get<float>(out2.rhs_literal()->value().to_variant()) - 2.0f) < 1e-6f);
    };

    "stmt_max"_test = [] {
        auto *a = make_tile();
        auto *b = new_literal(Type::of<float>(), 1e-12f);
        MaxStmt stmt{a, b};
        expect(stmt.a() == a);
        expect(stmt.b() == b);
        expect(stmt.output() == nullptr);

        MaxStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.a() != nullptr);
        expect(out.b() != nullptr && out.b() != b);
        expect(std::abs(luisa::get<float>(out.b()->value().to_variant()) - 1e-12f) < 1e-20f);
    };

    "stmt_rsqrt"_test = [] {
        auto *a = make_tile();
        RsqrtStmt stmt{a};
        expect(stmt.a() == a);
        RsqrtStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.a() != nullptr);
        expect(same_span(out.a()->extent(), {16, 16}));
    };

    "stmt_ceildiv"_test = [] {
        CeilDivStmt stmt{17, 4};
        expect(stmt.a() == 17);
        expect(stmt.b() == 4);
        expect(stmt.result() == 5);// (17 + 4 - 1) / 4
        expect(stmt.inputs().empty());
        expect(stmt.output() == nullptr);

        CeilDivStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.a() == 17);
        expect(out.b() == 4);
        expect(out.result() == 5);
    };

    "stmt_kernel1d"_test = [] {
        const RefExpr *bx = nullptr;
        auto kernel = luisa::compute::detail::FunctionBuilder::define_kernel([&] {
            bx = luisa::compute::detail::FunctionBuilder::current()->block_id();
        });
        expect(bx != nullptr);
        expect(kernel != nullptr);
        Kernel1DStmt stmt{256, 128, bx};
        expect(stmt.gx() == 256);
        expect(stmt.threads() == 128);
        expect(stmt.bx() == bx);

        Kernel1DStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.gx() == 256);
        expect(out.threads() == 128);
        expect(out.bx() == nullptr);// R3 pointer, non-serializable
    };

    "stmt_kernel2d"_test = [] {
        Kernel2DStmt stmt{8, 16, 32};
        expect(stmt.gx() == 8);
        expect(stmt.gy() == 16);
        expect(stmt.threads() == 32);

        Kernel2DStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.gx() == 8);
        expect(out.gy() == 16);
        expect(out.threads() == 32);
    };

    "stmt_pipelined"_test = [] {
        PipelinedStmt stmt{64, 3};
        expect(stmt.count() == 64);
        expect(stmt.stages() == 3);

        PipelinedStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.count() == 64);
        expect(out.stages() == 3);
    };

    "serialize_literal_values"_test = [] {
        // scalar uint dim (R1 host-side value)
        {
            ReduceSumStmt stmt{make_tile(), make_tile(), 7u};
            ReduceSumStmt out;
            expect(roundtrip(&stmt, out));
            expect(out.dim() == 7u);
        }
        // scalar half
        {
            auto *b = new_literal(Type::of<half>(), half{0.5f});
            MaxStmt stmt{make_tile(), b};
            MaxStmt out;
            expect(roundtrip(&stmt, out));
            auto hv = luisa::get<half>(out.b()->value().to_variant());
            expect(std::abs(static_cast<float>(hv) - 0.5f) < 1e-6f);
            expect(out.b()->type() == Type::of<half>());
        }
        // vector float2
        {
            auto *v = new_literal(Type::of<float2>(), float2{1.0f, 2.0f});
            TileStoreStmt stmt{0, make_tile(), nullptr, v};
            TileStoreStmt out;
            expect(roundtrip(&stmt, out));
            auto vv = luisa::get<float2>(out.rhs_literal()->value().to_variant());
            expect(std::abs(vv.x - 1.0f) < 1e-6f);
            expect(std::abs(vv.y - 2.0f) < 1e-6f);
            expect(out.rhs_literal()->type() == Type::of<float2>());
        }
        // uint scalar
        {
            auto *b = new_literal(Type::of<uint>(), 42u);
            MaxStmt stmt{make_tile(), b};
            MaxStmt out;
            expect(roundtrip(&stmt, out));
            expect(luisa::get<uint>(out.b()->value().to_variant()) == 42u);
        }
    };

    "deserialize_kind_mismatch"_test = [] {
        // a Gemm buffer must not deserialize into a Clear statement
        auto *a = make_tile();
        auto *b = make_tile();
        auto *c = make_tile();
        GemmStmt stmt{a, b, c};
        luisa::vector<char> buf;
        stmt.serialize(buf);
        char const *p = buf.data();
        char const *end = p + buf.size();
        ClearStmt wrong;
        expect(!wrong.deserialize(p, end));

        // and vice versa: a Clear buffer must not deserialize into a Copy
        ClearStmt clear{make_tile()};
        luisa::vector<char> buf2;
        clear.serialize(buf2);
        char const *p2 = buf2.data();
        CopyStmt wrong2;
        expect(!wrong2.deserialize(p2, p2 + buf2.size()));
    };

    "deserialize_truncated"_test = [] {
        auto *a = make_tile();
        auto *b = make_tile();
        auto *c = make_tile();
        GemmStmt stmt{a, b, c, 1, 1};
        luisa::vector<char> buf;
        stmt.serialize(buf);
        expect(!buf.empty());
        auto half = buf.size() / 2u;
        char const *p = buf.data();
        GemmStmt out;
        expect(!out.deserialize(p, p + half));// cut mid-stream
        expect(!out.deserialize(p, p + 0));   // empty stream
    };

    "deserialize_malformed_scope"_test = [] {
        // rank=1, dtype=F32, scope=7 (invalid), dims/offset/extent empty
        luisa::vector<char> buf;
        auto put32 = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i) { buf.push_back(static_cast<char>((v >> (8 * i)) & 0xffu)); }
        };
        put32(1);                                                // rank
        put32(static_cast<uint32_t>(TensorElementType::F32));    // dtype
        put32(7);                                                // invalid scope
        put32(0);                                                // dims count
        put32(0);                                                // offset count
        put32(0);                                                // extent count
        char const *p = buf.data();
        TensorExpr out;
        expect(!out.deserialize(p, p + buf.size()));
    };

    "deserialize_malformed_dtype"_test = [] {
        // rank=1, dtype=999 (invalid tag), scope=Global, dims/offset/extent empty
        luisa::vector<char> buf;
        auto put32 = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i) { buf.push_back(static_cast<char>((v >> (8 * i)) & 0xffu)); }
        };
        put32(1);                                                // rank
        put32(static_cast<uint32_t>(TensorElementType::FP4) + 1);// invalid dtype
        put32(static_cast<uint32_t>(TensorScope::Global));       // scope
        put32(0);                                                // dims count
        put32(0);                                                // offset count
        put32(0);                                                // extent count
        char const *p = buf.data();
        TensorExpr out;
        expect(!out.deserialize(p, p + buf.size()));
    };

    "deserialize_malformed_literal"_test = [] {
        // Max buffer with a literal whose variant index is out of range.
        luisa::vector<char> buf;
        auto put32 = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i) { buf.push_back(static_cast<char>((v >> (8 * i)) & 0xffu)); }
        };
        put32(static_cast<uint32_t>(TileOpKind::MAX));// op
        put32(0);                                     // no output
        put32(0);                                     // no inputs
        put32(0);                                     // no annotations
        put32(3);                                     // literal dtype "int"
        buf.push_back('i');
        buf.push_back('n');
        buf.push_back('t');
        buf.push_back(static_cast<char>(0xE7));// variant index 999 (invalid)
        buf.push_back(static_cast<char>(0x03));
        char const *p = buf.data();
        MaxStmt out;
        expect(!out.deserialize(p, p + buf.size()));
    };

    "serialize_size_is_compact"_test = [] {
        // The layout of one rank-2 float tensor: rank(4) + dtype(4) + scope(4)
        // + 3 vectors (each 4 + 2*4, int32 dims).  Only statically meaningful
        // members are stored — no padding, no pointers.
        TensorExpr t{2, TensorElementType::F32, TensorScope::Global, {16, 16}};
        luisa::vector<char> buf;
        auto n = t.serialize(buf);
        expect(n == 4u + 4u + 4u + 3u * (4u + 8u));
        expect(n == buf.size());
    };
    // ----------------------------------------------------------------------
    // gap-analysis statements (section 5b of <luisa/ast/tensor.h>)
    // ----------------------------------------------------------------------

    "stmt_gemm_knobs"_test = [] {
        auto *a = make_tile();
        auto *b = make_tile();
        auto *c = make_tile();
        GemmStmt stmt{a, b, c, GemmWarpPolicy::FullRow, 1, 2};
        expect(stmt.policy() == GemmWarpPolicy::FullRow);
        expect(stmt.clear_accum() == 1);
        expect(stmt.k_pack() == 2);
        expect(stmt.mbar() == nullptr);
        expect(stmt.a() == a && stmt.b() == b && stmt.c() == c);

        GemmStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.policy() == GemmWarpPolicy::FullRow);
        expect(out.clear_accum() == 1);
        expect(out.k_pack() == 2);
        expect(out.mbar() == nullptr);

        // mbarrier is an optional third input tensor; each statement owns its
        // own operands, so fresh tensors are used here
        GemmStmt stmt2{make_tile(), make_tile(), make_tile(),
                       GemmWarpPolicy::Square, 0, 1, make_tile()};
        expect(stmt2.mbar() != nullptr);
        expect(stmt2.inputs().size() == 3u);
        GemmStmt out2;
        expect(roundtrip(&stmt2, out2));
        expect(out2.mbar() != nullptr);
        expect(out2.inputs().size() == 3u);
    };

    "stmt_fill"_test = [] {
        auto *v = new_literal(Type::of<float>(), 0.5f);
        FillStmt stmt{make_tile(), v};
        expect(stmt.buf() != nullptr);
        expect(stmt.value_literal() == v);
        expect(stmt.value_ref() == nullptr);

        FillStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.buf() != nullptr);
        expect(out.value_literal() != nullptr && out.value_literal() != v);
        expect(std::abs(luisa::get<float>(out.value_literal()->value().to_variant()) - 0.5f) < 1e-6f);
        expect(out.value_ref() == nullptr);

        // R3 runtime scalar form is stored but not serialized
        const RefExpr *ref = nullptr;
        auto kernel = luisa::compute::detail::FunctionBuilder::define_kernel([&] {
            ref = luisa::compute::detail::FunctionBuilder::current()->local(Type::of<float>());
        });
        expect(ref != nullptr && kernel != nullptr);
        FillStmt ref_stmt{make_tile(), ref};
        expect(ref_stmt.value_ref() == ref);
        FillStmt ref_out;
        expect(roundtrip(&ref_stmt, ref_out));
        expect(ref_out.value_ref() == nullptr);// R3 pointer, non-serializable
    };

    "stmt_transpose"_test = [] {
        auto *src = make_tile();
        auto *dst = make_tile();
        TransposeStmt stmt{src, dst};
        expect(stmt.src() == src);
        expect(stmt.dst() == dst);
        expect(stmt.dst() == stmt.output());
        TransposeStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.src() != nullptr && out.dst() != nullptr);
        expect(same_span(out.src()->dims(), {16, 16}));
    };

    "stmt_im2col"_test = [] {
        Im2ColStmt stmt{make_tile(), make_tile(), nullptr, nullptr, 3, 1, 2, 1, 1};
        expect(stmt.img() != nullptr);
        expect(stmt.col() != nullptr);
        expect(stmt.kernel() == 3);
        expect(stmt.stride() == 1);
        expect(stmt.dilation() == 2);
        expect(stmt.pad() == 1);
        expect(stmt.eviction_policy() == 1);
        Im2ColStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.img() != nullptr && out.col() != nullptr);
        expect(out.kernel() == 3 && out.stride() == 1);
        expect(out.dilation() == 2 && out.pad() == 1);
        expect(out.eviction_policy() == 1);
    };

    "stmt_async_copy"_test = [] {
        AsyncCopyStmt stmt{make_tile(), make_tile(), 64};
        expect(stmt.src() != nullptr && stmt.dst() != nullptr);
        expect(stmt.coalesced_width() == 64);
        AsyncCopyStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.src() != nullptr && out.dst() != nullptr);
        expect(out.coalesced_width() == 64);
    };

    "stmt_copy_cluster"_test = [] {
        CopyClusterStmt stmt{make_tile(), make_tile(), 1, 0x3, 16};
        expect(stmt.src() != nullptr && stmt.dst() != nullptr);
        expect(stmt.dst_block() == 1);
        expect(stmt.cluster_mask() == 0x3);
        expect(stmt.coalesced_width() == 16);
        CopyClusterStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.dst_block() == 1);
        expect(out.cluster_mask() == 0x3);
        expect(out.coalesced_width() == 16);
    };

    "stmt_tma_copy"_test = [] {
        // with a barrier mbarrier tensor (inputs[1])
        TmaCopyStmt stmt{make_tile(), make_tile(), make_tile(), 128, 2};
        expect(stmt.src() != nullptr && stmt.dst() != nullptr);
        expect(stmt.barrier() != nullptr);
        expect(stmt.leader_scope_threads() == 128);
        expect(stmt.eviction_policy() == 2);
        TmaCopyStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.src() != nullptr && out.dst() != nullptr);
        expect(out.barrier() != nullptr);
        expect(out.leader_scope_threads() == 128);
        expect(out.eviction_policy() == 2);

        // store direction: no barrier
        TmaCopyStmt store{make_tile(), make_tile()};
        expect(store.barrier() == nullptr);
        TmaCopyStmt store_out;
        expect(roundtrip(&store, store_out));
        expect(store_out.barrier() == nullptr);
    };

    "stmt_tma_gather4_scatter4"_test = [] {
        TmaGather4Stmt g{make_tile(), make_tile(), nullptr,
                         luisa::fixed_vector<int32_t, 4>{0, 8, 16, 24},
                         make_tile(), 1};
        expect(g.src() != nullptr && g.dst() != nullptr);
        expect(g.barrier() != nullptr);
        expect(same_span(g.rows(), {0, 8, 16, 24}));
        expect(g.eviction_policy() == 1);
        TmaGather4Stmt g_out;
        expect(roundtrip(&g, g_out));
        expect(g_out.barrier() != nullptr);
        expect(same_span(g_out.rows(), {0, 8, 16, 24}));
        expect(g_out.eviction_policy() == 1);

        TmaScatter4Stmt s{make_tile(), make_tile(), nullptr,
                          luisa::fixed_vector<int32_t, 4>{3, 7, 11, 15}, 0};
        expect(s.src() != nullptr && s.dst() != nullptr);
        expect(same_span(s.rows(), {3, 7, 11, 15}));
        TmaScatter4Stmt s_out;
        expect(roundtrip(&s, s_out));
        expect(same_span(s_out.rows(), {3, 7, 11, 15}));
    };

    "stmt_reshape_view"_test = [] {
        ReshapeStmt stmt{make_tile(), {256}};
        expect(stmt.src() != nullptr);
        expect(stmt.dst() != nullptr);
        expect(stmt.dst()->rank() == 1);
        expect(same_span(stmt.dst()->dims(), {256}));
        expect(stmt.dst()->dtype() == TensorElementType::F32);
        ReshapeStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.src() != nullptr && out.dst() != nullptr);
        expect(same_span(out.dst()->dims(), {256}));

        // view with a new dtype; dims default to the source dims
        ViewStmt vw{make_tile(), TensorElementType::I32};
        expect(vw.src() != nullptr && vw.dst() != nullptr);
        expect(vw.dst()->dtype() == TensorElementType::I32);
        expect(same_span(vw.dst()->dims(), {16, 16}));
        ViewStmt vw_out;
        expect(roundtrip(&vw, vw_out));
        expect(vw_out.dst()->dtype() == TensorElementType::I32);
        expect(same_span(vw_out.dst()->dims(), {16, 16}));

        // quantized dtype views (int8 / fp8) also round-trip their tag
        ViewStmt vw8{make_tile(), TensorElementType::I8};
        expect(vw8.dst()->dtype() == TensorElementType::I8);
        ViewStmt vw8_out;
        expect(roundtrip(&vw8, vw8_out));
        expect(vw8_out.dst()->dtype() == TensorElementType::I8);
        ViewStmt vwf8{make_tile(), TensorElementType::FP8, {4, 8}};
        expect(vwf8.dst()->dtype() == TensorElementType::FP8);
        ViewStmt vwf8_out;
        expect(roundtrip(&vwf8, vwf8_out));
        expect(vwf8_out.dst()->dtype() == TensorElementType::FP8);
        expect(same_span(vwf8_out.dst()->dims(), {4, 8}));

        // view with an explicit shape
        ViewStmt vw2{make_tile(), TensorElementType::F32, {8, 32}};
        expect(same_span(vw2.dst()->dims(), {8, 32}));
    };

    "stmt_reduce"_test = [] {
        ReduceStmt stmt{TileReduceOp::ABS_MAX, make_tile(), make_tile(), 1u, 1, 4, 1};
        expect(stmt.buf() != nullptr && stmt.out() != nullptr);
        expect(stmt.op() == TileReduceOp::ABS_MAX);
        expect(stmt.dim() == 1u);
        expect(stmt.clear() == 1);
        expect(stmt.batch() == 4);
        expect(stmt.nan_propagate() == 1);
        ReduceStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileReduceOp::ABS_MAX);
        expect(out.dim() == 1u && out.clear() == 1);
        expect(out.batch() == 4 && out.nan_propagate() == 1);

        // every reduce op tag round-trips
        ReduceStmt ops[]{ReduceStmt{TileReduceOp::SUM, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::MAX, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::MIN, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::ABS_SUM, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::ABS_MAX, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::BIT_AND, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::BIT_OR, make_tile(), make_tile(), 0u},
                         ReduceStmt{TileReduceOp::BIT_XOR, make_tile(), make_tile(), 0u}};
        for (auto &op : ops) {
            ReduceStmt r_out;
            expect(roundtrip(&op, r_out));
            expect(r_out.op() == op.op());
        }
    };

    "stmt_finalize_reducer"_test = [] {
        FinalizeReducerStmt stmt{make_tile(), 8};
        expect(stmt.reducer() != nullptr);
        expect(stmt.batch() == 8);
        FinalizeReducerStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.reducer() != nullptr);
        expect(out.batch() == 8);
    };

    "stmt_warp_reduce"_test = [] {
        WarpReduceStmt stmt{TileWarpReduceOp::BIT_AND, make_tile()};
        expect(stmt.op() == TileWarpReduceOp::BIT_AND);
        expect(stmt.value() != nullptr);
        WarpReduceStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileWarpReduceOp::BIT_AND);
        expect(out.value() != nullptr);
    };

    "stmt_cumsum_cummax"_test = [] {
        CumSumStmt cs{make_tile(), make_tile(), 1u, 1};
        expect(cs.src() != nullptr && cs.dst() != nullptr);
        expect(cs.dim() == 1u && cs.reverse() == 1);
        CumSumStmt cs_out;
        expect(roundtrip(&cs, cs_out));
        expect(cs_out.dim() == 1u && cs_out.reverse() == 1);

        CumMaxStmt cm{make_tile(), make_tile(), 0u, 0};
        expect(cm.dim() == 0u && cm.reverse() == 0);
        CumMaxStmt cm_out;
        expect(roundtrip(&cm, cm_out));
        expect(cm_out.dim() == 0u && cm_out.reverse() == 0);
    };

    "stmt_wgmma_gemm"_test = [] {
        WgmmaGemmStmt stmt{make_tile(), make_tile(), make_tile(), 1, 0,
                           GemmWarpPolicy::FullCol, 1};
        expect(stmt.a() != nullptr && stmt.b() != nullptr && stmt.c() != nullptr);
        expect(stmt.trans_a() == 1 && stmt.trans_b() == 0);
        expect(stmt.policy() == GemmWarpPolicy::FullCol);
        expect(stmt.clear_accum() == 1);
        WgmmaGemmStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.trans_a() == 1 && out.trans_b() == 0);
        expect(out.policy() == GemmWarpPolicy::FullCol);
        expect(out.clear_accum() == 1);
    };

    "stmt_tcgen05_gemm"_test = [] {
        Tcgen05GemmStmt stmt{make_tile(), make_tile(), make_tile(), 0, 1,
                             GemmWarpPolicy::Square, 0, make_tile()};
        expect(stmt.a() != nullptr && stmt.b() != nullptr && stmt.c() != nullptr);
        expect(stmt.mbar() != nullptr);
        expect(stmt.trans_a() == 0 && stmt.trans_b() == 1);
        Tcgen05GemmStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.mbar() != nullptr);
        expect(out.trans_a() == 0 && out.trans_b() == 1);
    };

    "stmt_tcgen05_blockscaled"_test = [] {
        Tcgen05GemmBlockscaledStmt stmt{make_tile(), make_tile(), make_tile(),
                                        make_tile(), make_tile(), 0, 0, 1, 32, 2, 4};
        expect(stmt.a() != nullptr && stmt.b() != nullptr && stmt.c() != nullptr);
        expect(stmt.sfa() != nullptr && stmt.sfb() != nullptr);
        expect(stmt.clear_accum() == 1);
        expect(stmt.k_start() == 32);
        expect(stmt.sf_a_granularity_k() == 2);
        expect(stmt.sf_b_granularity_k() == 4);
        Tcgen05GemmBlockscaledStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.sfa() != nullptr && out.sfb() != nullptr);
        expect(out.clear_accum() == 1 && out.k_start() == 32);
        expect(out.sf_a_granularity_k() == 2 && out.sf_b_granularity_k() == 4);
    };

    "stmt_gemm_sp_family"_test = [] {
        GemmSpStmt stmt{make_tile(), make_tile(), make_tile(), make_tile(),
                        1, 0, 1, GemmWarpPolicy::FullRow, 1};
        expect(stmt.a() != nullptr && stmt.e() != nullptr);
        expect(stmt.b() != nullptr && stmt.c() != nullptr);
        expect(stmt.trans_a() == 1 && stmt.trans_e() == 0 && stmt.trans_b() == 1);
        expect(stmt.policy() == GemmWarpPolicy::FullRow);
        expect(stmt.clear_accum() == 1);
        GemmSpStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.a() != nullptr && out.e() != nullptr);
        expect(out.trans_a() == 1 && out.trans_e() == 0 && out.trans_b() == 1);
        expect(out.policy() == GemmWarpPolicy::FullRow);
        expect(out.clear_accum() == 1);

        WgmmaGemmSpStmt wg{make_tile(), make_tile(), make_tile(), make_tile()};
        expect(wg.a() != nullptr && wg.e() != nullptr);
        expect(wg.b() != nullptr && wg.c() != nullptr);
        WgmmaGemmSpStmt wg_out;
        expect(roundtrip(&wg, wg_out));
        expect(wg_out.a() != nullptr && wg_out.e() != nullptr);

        Tcgen05GemmSpStmt tc{make_tile(), make_tile(), make_tile(), make_tile(),
                             0, 1, 0};
        expect(tc.trans_e() == 1);
        Tcgen05GemmSpStmt tc_out;
        expect(roundtrip(&tc, tc_out));
        expect(tc_out.trans_e() == 1);
    };

    "stmt_atomic"_test = [] {
        auto *v = new_literal(Type::of<float>(), 3.0f);
        // tensor value form
        AtomicStmt stmt{TileAtomicOp::ADD, make_tile(), make_tile(), nullptr, nullptr,
                        TileMemoryOrder::ACQ_REL, 1, 1};
        expect(stmt.dst() != nullptr);
        expect(stmt.value_tensor() != nullptr);
        expect(stmt.value_literal() == nullptr);
        expect(stmt.op() == TileAtomicOp::ADD);
        expect(stmt.memory_order() == TileMemoryOrder::ACQ_REL);
        expect(stmt.return_prev() == 1);
        expect(stmt.use_tma() == 1);
        AtomicStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.value_tensor() != nullptr);
        expect(out.op() == TileAtomicOp::ADD);
        expect(out.memory_order() == TileMemoryOrder::ACQ_REL);
        expect(out.return_prev() == 1 && out.use_tma() == 1);

        // scalar literal value form
        AtomicStmt lit{TileAtomicOp::MAX, make_tile(), nullptr, v};
        expect(lit.value_tensor() == nullptr);
        expect(lit.value_literal() == v);
        AtomicStmt lit_out;
        expect(roundtrip(&lit, lit_out));
        expect(lit_out.value_tensor() == nullptr);
        expect(lit_out.value_literal() != nullptr && lit_out.value_literal() != v);
        expect(std::abs(luisa::get<float>(lit_out.value_literal()->value().to_variant()) - 3.0f) < 1e-6f);
    };

    "stmt_clamp"_test = [] {
        auto *lo = new_literal(Type::of<float>(), 0.0f);
        auto *hi = new_literal(Type::of<float>(), 1.0f);
        ClampStmt stmt{make_tile(), lo, hi};
        expect(stmt.dst() != nullptr);
        expect(stmt.lo_literal() == lo && stmt.hi_literal() == hi);
        expect(stmt.lo_ref() == nullptr && stmt.hi_ref() == nullptr);
        ClampStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.dst() != nullptr);
        expect(out.lo_literal() != nullptr && out.hi_literal() != nullptr);
        expect(out.lo_ref() == nullptr && out.hi_ref() == nullptr);
        expect(std::abs(luisa::get<float>(out.lo_literal()->value().to_variant())) < 1e-6f);
        expect(std::abs(luisa::get<float>(out.hi_literal()->value().to_variant()) - 1.0f) < 1e-6f);
    };

    "stmt_dp4a"_test = [] {
        Dp4aStmt stmt{make_tile(), make_tile(), make_tile()};
        expect(stmt.a() != nullptr && stmt.b() != nullptr && stmt.c() != nullptr);
        expect(stmt.c() == stmt.output());
        Dp4aStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.a() != nullptr && out.b() != nullptr && out.c() != nullptr);
    };

    "stmt_loop_break"_test = [] {
        LoopBreakStmt stmt;
        expect(stmt.op() == TileOpKind::LOOP_BREAK);
        expect(stmt.output() == nullptr);
        expect(stmt.inputs().empty());
        LoopBreakStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileOpKind::LOOP_BREAK);
    };

    "stmt_any_all_of"_test = [] {
        AnyOfStmt any{make_tile()};
        expect(any.buf() != nullptr);
        expect(any.output() == nullptr);
        AnyOfStmt any_out;
        expect(roundtrip(&any, any_out));
        expect(any_out.buf() != nullptr);

        AllOfStmt all{make_tile()};
        expect(all.buf() != nullptr);
        AllOfStmt all_out;
        expect(roundtrip(&all, all_out));
        expect(all_out.buf() != nullptr);
    };

    "stmt_sync"_test = [] {
        SyncStmt stmt{TileSyncOp::WARP, 0xFFFFu, -1, 0};
        expect(stmt.op() == TileSyncOp::WARP);
        expect(stmt.mask() == 0xFFFFu);
        expect(stmt.barrier_id() == -1);
        expect(stmt.arrive_count() == 0);
        SyncStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileSyncOp::WARP);
        expect(out.mask() == 0xFFFFu);
        expect(out.barrier_id() == -1);

        SyncStmt named{TileSyncOp::THREADS, 0, 3, 128};
        expect(named.barrier_id() == 3 && named.arrive_count() == 128);
        SyncStmt named_out;
        expect(roundtrip(&named, named_out));
        expect(named_out.barrier_id() == 3 && named_out.arrive_count() == 128);
    };

    "stmt_barrier"_test = [] {
        BarrierStmt arrive{TileBarrierOp::ARRIVE, make_tile()};
        expect(arrive.op() == TileBarrierOp::ARRIVE);
        expect(arrive.mbarrier() != nullptr);
        BarrierStmt arrive_out;
        expect(roundtrip(&arrive, arrive_out));
        expect(arrive_out.mbarrier() != nullptr);

        BarrierStmt wait{TileBarrierOp::WAIT, make_tile(), 1};
        expect(wait.parity() == 1);
        BarrierStmt wait_out;
        expect(roundtrip(&wait, wait_out));
        expect(wait_out.parity() == 1);

        BarrierStmt named{TileBarrierOp::NAMED_ARRIVE, nullptr, 0, 5, 256};
        expect(named.mbarrier() == nullptr);
        expect(named.barrier_id() == 5 && named.thread_count() == 256);
        BarrierStmt named_out;
        expect(roundtrip(&named, named_out));
        expect(named_out.mbarrier() == nullptr);
        expect(named_out.barrier_id() == 5 && named_out.thread_count() == 256);
    };

    "stmt_mbarrier"_test = [] {
        MBarrierStmt stmt{TileMBarrierOp::ARRIVE_EXPECT_TX, make_tile(), 4096, 0, -1};
        expect(stmt.op() == TileMBarrierOp::ARRIVE_EXPECT_TX);
        expect(stmt.mbarrier() != nullptr);
        expect(stmt.tx() == 4096);
        expect(stmt.parity() == 0);
        expect(stmt.cta_id() == -1);
        MBarrierStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileMBarrierOp::ARRIVE_EXPECT_TX);
        expect(out.mbarrier() != nullptr);
        expect(out.tx() == 4096);

        MBarrierStmt wait{TileMBarrierOp::WAIT_PARITY, make_tile(), 0, 1};
        expect(wait.parity() == 1);
        MBarrierStmt wait_out;
        expect(roundtrip(&wait, wait_out));
        expect(wait_out.parity() == 1);
    };

    "stmt_warp_vote"_test = [] {
        auto *pred = new_literal(Type::of<uint>(), 1u);
        WarpVoteStmt stmt{TileWarpVoteOp::BALLOT_SYNC, nullptr, pred};
        expect(stmt.op() == TileWarpVoteOp::BALLOT_SYNC);
        expect(stmt.mask() == nullptr);
        expect(stmt.pred_literal() == pred);
        expect(stmt.pred_ref() == nullptr);
        WarpVoteStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileWarpVoteOp::BALLOT_SYNC);
        expect(out.mask() == nullptr);
        expect(out.pred_literal() != nullptr && out.pred_literal() != pred);
        expect(out.pred_ref() == nullptr);

        WarpVoteStmt act{TileWarpVoteOp::ACTIVEMASK};
        expect(act.pred_literal() == nullptr && act.mask() == nullptr);
        WarpVoteStmt act_out;
        expect(roundtrip(&act, act_out));
        expect(act_out.op() == TileWarpVoteOp::ACTIVEMASK);
    };

    "stmt_shuffle"_test = [] {
        auto *v = new_literal(Type::of<float>(), 1.5f);
        ShuffleStmt stmt{TileShuffleOp::XOR, v, nullptr, nullptr, nullptr, 32, 4};
        expect(stmt.op() == TileShuffleOp::XOR);
        expect(stmt.value_literal() == v);
        expect(stmt.width() == 32);
        expect(stmt.delta() == 4);
        expect(stmt.thread_extent() == 0);
        expect(stmt.mask() == nullptr && stmt.src_lane() == nullptr);
        ShuffleStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileShuffleOp::XOR);
        expect(out.value_literal() != nullptr && out.value_literal() != v);
        expect(out.width() == 32 && out.delta() == 4);

        ShuffleStmt elect{TileShuffleOp::ELECT, nullptr, nullptr, nullptr, nullptr, 0, 0, 64};
        expect(elect.thread_extent() == 64);
        ShuffleStmt elect_out;
        expect(roundtrip(&elect, elect_out));
        expect(elect_out.thread_extent() == 64);
    };

    "stmt_sync_threads_vote"_test = [] {
        auto *pred = new_literal(Type::of<int>(), 1);
        SyncThreadsVoteStmt stmt{TileSyncThreadsVoteOp::AND, pred};
        expect(stmt.op() == TileSyncThreadsVoteOp::AND);
        expect(stmt.pred_literal() == pred);
        expect(stmt.pred_ref() == nullptr);
        SyncThreadsVoteStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileSyncThreadsVoteOp::AND);
        expect(out.pred_literal() != nullptr && out.pred_literal() != pred);
        expect(out.pred_ref() == nullptr);
    };

    "stmt_fast_rcp"_test = [] {
        FastRcpStmt stmt{make_tile()};
        expect(stmt.a() != nullptr);
        FastRcpStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.a() != nullptr);
        expect(same_span(out.a()->extent(), {16, 16}));
    };

    "stmt_ieee_math"_test = [] {
        // unary
        IeeeMathStmt u{TileIeeeOp::FSQRT, make_tile(), nullptr, nullptr, 1};
        expect(u.op() == TileIeeeOp::FSQRT);
        expect(u.a() != nullptr && u.b() == nullptr && u.c() == nullptr);
        expect(u.rounding_mode() == 1);
        IeeeMathStmt u_out;
        expect(roundtrip(&u, u_out));
        expect(u_out.op() == TileIeeeOp::FSQRT && u_out.rounding_mode() == 1);
        expect(u_out.a() != nullptr && u_out.b() == nullptr);

        // binary
        IeeeMathStmt b{TileIeeeOp::FDIV, make_tile(), make_tile()};
        expect(b.a() != nullptr && b.b() != nullptr);
        IeeeMathStmt b_out;
        expect(roundtrip(&b, b_out));
        expect(b_out.a() != nullptr && b_out.b() != nullptr && b_out.c() == nullptr);

        // ternary (fmaf)
        IeeeMathStmt t{TileIeeeOp::FMAF, make_tile(), make_tile(), make_tile(), 2};
        expect(t.c() != nullptr);
        IeeeMathStmt t_out;
        expect(roundtrip(&t, t_out));
        expect(t_out.op() == TileIeeeOp::FMAF && t_out.c() != nullptr);
        expect(t_out.rounding_mode() == 2);
    };

    "stmt_packed_math"_test = [] {
        PackedMathStmt fma{TilePackedOp::FMA2, make_tile(), make_tile(), make_tile()};
        expect(fma.op() == TilePackedOp::FMA2);
        expect(fma.a() != nullptr && fma.b() != nullptr && fma.c() != nullptr);
        PackedMathStmt fma_out;
        expect(roundtrip(&fma, fma_out));
        expect(fma_out.op() == TilePackedOp::FMA2 && fma_out.c() != nullptr);

        PackedMathStmt abs{TilePackedOp::ABS2, make_tile()};
        expect(abs.b() == nullptr && abs.c() == nullptr);
        PackedMathStmt abs_out;
        expect(roundtrip(&abs, abs_out));
        expect(abs_out.op() == TilePackedOp::ABS2);
        expect(abs_out.a() != nullptr && abs_out.b() == nullptr);
    };

    "stmt_fast_math"_test = [] {
        FastMathStmt stmt{TileFastMathOp::EXP10, make_tile()};
        expect(stmt.op() == TileFastMathOp::EXP10);
        expect(stmt.a() != nullptr);
        FastMathStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.op() == TileFastMathOp::EXP10);
        expect(out.a() != nullptr);
    };

    "stmt_alloc_special"_test = [] {
        auto *init = new_literal(Type::of<float>(), 1.0f);
        AllocSpecialStmt stmt{TileAllocKind::VAR, make_tile(), 0, 0, 0, init};
        expect(stmt.kind() == TileAllocKind::VAR);
        expect(stmt.tensor() != nullptr);
        expect(stmt.init() == init);
        AllocSpecialStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.kind() == TileAllocKind::VAR);
        expect(out.tensor() != nullptr);
        expect(out.init() != nullptr && out.init() != init);

        AllocSpecialStmt bar{TileAllocKind::CLUSTER_BARRIER, make_tile(), 128};
        expect(bar.count() == 128);
        AllocSpecialStmt bar_out;
        expect(roundtrip(&bar, bar_out));
        expect(bar_out.kind() == TileAllocKind::CLUSTER_BARRIER);
        expect(bar_out.count() == 128);

        AllocSpecialStmt red{TileAllocKind::REDUCER, make_tile(), 0, 1};
        expect(red.reducer_op() == 1);
        AllocSpecialStmt red_out;
        expect(roundtrip(&red, red_out));
        expect(red_out.reducer_op() == 1);

        AllocSpecialStmt desc{TileAllocKind::DESCRIPTOR, make_tile(), 0, 0, 2};
        expect(desc.desc_kind() == 2);
        AllocSpecialStmt desc_out;
        expect(roundtrip(&desc, desc_out));
        expect(desc_out.desc_kind() == 2);
    };

    "stmt_loop_annotation"_test = [] {
        LoopAnnotationStmt stmt{TileLoopAnnotKind::UNROLL, 4, 0};
        expect(stmt.kind() == TileLoopAnnotKind::UNROLL);
        expect(stmt.extent() == 4);
        expect(stmt.coalesced_width() == 0);
        LoopAnnotationStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.kind() == TileLoopAnnotKind::UNROLL);
        expect(out.extent() == 4);

        LoopAnnotationStmt par{TileLoopAnnotKind::PARALLEL, 128, 32};
        expect(par.coalesced_width() == 32);
        LoopAnnotationStmt par_out;
        expect(roundtrip(&par, par_out));
        expect(par_out.kind() == TileLoopAnnotKind::PARALLEL);
        expect(par_out.coalesced_width() == 32);
    };

    "stmt_annotate"_test = [] {
        AnnotateStmt stmt{TileAnnotKind::L2_HIT_RATIO, make_tile(), 0, 0, 1, 0, 0.75f};
        expect(stmt.kind() == TileAnnotKind::L2_HIT_RATIO);
        expect(stmt.tensor() != nullptr);
        expect(std::abs(stmt.hit_ratio() - 0.75f) < 1e-6f);
        AnnotateStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.kind() == TileAnnotKind::L2_HIT_RATIO);
        expect(out.tensor() != nullptr);
        expect(std::abs(out.hit_ratio() - 0.75f) < 1e-6f);

        AnnotateStmt sw{TileAnnotKind::USE_SWIZZLE, nullptr, 128, 1, 1};
        expect(sw.panel_size() == 128 && sw.order() == 1 && sw.enable() == 1);
        AnnotateStmt sw_out;
        expect(roundtrip(&sw, sw_out));
        expect(sw_out.panel_size() == 128 && sw_out.order() == 1);

        auto *safe = new_literal(Type::of<float>(), -1.0f);
        AnnotateStmt sv{TileAnnotKind::SAFE_VALUE, make_tile(), 0, 0, 1, 0, 0.0f, safe};
        expect(sv.safe_value() == safe);
        AnnotateStmt sv_out;
        expect(roundtrip(&sv, sv_out));
        expect(sv_out.safe_value() != nullptr && sv_out.safe_value() != safe);

        AnnotateStmt mb{TileAnnotKind::MIN_BLOCKS_PER_SM, nullptr, 0, 0, 1, 2};
        expect(mb.value() == 2);
        AnnotateStmt mb_out;
        expect(roundtrip(&mb, mb_out));
        expect(mb_out.value() == 2);
    };

    "stmt_dynamic_symbolic"_test = [] {
        DynamicStmt dyn{"M", TensorElementType::I32};
        expect(dyn.name() == "M");
        expect(dyn.dtype() == TensorElementType::I32);
        expect(dyn.op() == TileOpKind::DYNAMIC);
        DynamicStmt dyn_out;
        expect(roundtrip(&dyn, dyn_out));
        expect(dyn_out.name() == "M");
        expect(dyn_out.dtype() == TensorElementType::I32);

        SymbolicStmt sym{"N", TensorElementType::F32};
        expect(sym.name() == "N");
        expect(sym.dtype() == TensorElementType::F32);
        SymbolicStmt sym_out;
        expect(roundtrip(&sym, sym_out));
        expect(sym_out.name() == "N");
        expect(sym_out.dtype() == TensorElementType::F32);
    };

    "stmt_inline_meta_class"_test = [] {
        InlineStmt stmt{"import numpy as np"};
        expect(stmt.message() == "import numpy as np");
        InlineStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.message() == "import numpy as np");

        MetaClassStmt meta{"MyMeta"};
        expect(meta.message() == "MyMeta");
        MetaClassStmt meta_out;
        expect(roundtrip(&meta, meta_out));
        expect(meta_out.message() == "MyMeta");
    };

    "stmt_access_ptr"_test = [] {
        AccessPtrStmt stmt{make_tile(), 2, 8, 32, 1};
        expect(stmt.base() != nullptr);
        expect(stmt.access_type() == 2);
        expect(stmt.offset() == 8);
        expect(stmt.extent() == 32);
        expect(stmt.ignore_last_ndim() == 1);
        AccessPtrStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.base() != nullptr);
        expect(out.access_type() == 2 && out.offset() == 8);
        expect(out.extent() == 32 && out.ignore_last_ndim() == 1);
    };

    "stmt_index_to_coordinates"_test = [] {
        auto *idx = new_literal(Type::of<int>(), 42);
        IndexToCoordinatesStmt stmt{idx, {4, 8}};
        expect(stmt.index_literal() == idx);
        expect(stmt.index_ref() == nullptr);
        expect(same_span(stmt.shape(), {4, 8}));
        IndexToCoordinatesStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.index_literal() != nullptr && out.index_literal() != idx);
        expect(out.index_ref() == nullptr);
        expect(same_span(out.shape(), {4, 8}));
    };

}
