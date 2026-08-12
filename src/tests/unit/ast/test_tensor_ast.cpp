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
TensorExpr *make_tile(luisa::vector<int64_t> offset = {}) {
    return new TensorExpr{2, TensorElementType::F32, TensorScope::Global,
                          {16, 16}, std::move(offset), {}};
}

bool same_span(luisa::span<const int64_t> a, std::initializer_list<int64_t> b) {
    if (a.size() != b.size()) { return false; }
    size_t i = 0u;
    for (auto v : b) {
        if (a[i++] != v) { return false; }
    }
    return true;
}

// Expression nodes can only be constructed while a FunctionBuilder is on the
// stack (Expression::Expression reads FunctionBuilder::current()).  These
// helpers materialize R2 literal / string constants under a short-lived guard.
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

const StringIDExpr *new_sid(luisa::string s) {
    const StringIDExpr *out = nullptr;
    with_builder([&] { out = new StringIDExpr{std::move(s)}; });
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
        expect(luisa::string_view{scope_name(TensorScope::Global)} == "global");
        expect(luisa::string_view{scope_name(TensorScope::Shared)} == "shared");
        expect(luisa::string_view{scope_name(TensorScope::Fragment)} == "fragment");
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
        auto *dim = new_literal(Type::of<int>(), 1);
        ReduceSumStmt stmt{x, y, dim};
        expect(stmt.x() == x);
        expect(stmt.y() == y);
        expect(stmt.dim() == dim);

        ReduceSumStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.x() != nullptr && out.y() != nullptr);
        expect(out.dim() != nullptr && out.dim() != dim);
        expect(out.dim()->type() == Type::of<int>());
        expect(luisa::get<int>(out.dim()->value().to_variant()) == 1);
    };

    "stmt_print"_test = [] {
        auto *t = make_tile();
        auto *msg = new_sid("hello tile");
        TilePrintStmt stmt{t, msg};
        expect(stmt.t() == t);
        expect(stmt.msg() == msg);
        expect(stmt.output() == nullptr);

        TilePrintStmt out;
        expect(roundtrip(&stmt, out));
        expect(out.t() != nullptr);
        expect(out.msg() != nullptr && out.msg() != msg);
        expect(out.msg()->data() == "hello tile");
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
        // scalar int dim
        {
            auto *dim = new_literal(Type::of<int>(), -7);
            ReduceSumStmt stmt{make_tile(), make_tile(), dim};
            ReduceSumStmt out;
            expect(roundtrip(&stmt, out));
            expect(luisa::get<int>(out.dim()->value().to_variant()) == -7);
            expect(out.dim()->type() == Type::of<int>());
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
        put32(static_cast<uint32_t>(TensorElementType::I32) + 1);// invalid dtype
        put32(static_cast<uint32_t>(TensorScope::Global));       // scope
        put32(0);                                                // dims count
        put32(0);                                                // offset count
        put32(0);                                                // extent count
        char const *p = buf.data();
        TensorExpr out;
        expect(!out.deserialize(p, p + buf.size()));
    };

    "deserialize_malformed_literal"_test = [] {
        // ReduceSum buffer with a literal whose variant index is out of range.
        luisa::vector<char> buf;
        auto put32 = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i) { buf.push_back(static_cast<char>((v >> (8 * i)) & 0xffu)); }
        };
        put32(static_cast<uint32_t>(TileOpKind::REDUCE_SUM));// op
        put32(0);                                            // no output
        put32(0);                                            // no inputs
        put32(0);                                            // no annotations
        put32(3);                                            // literal dtype "int"
        buf.push_back('i');
        buf.push_back('n');
        buf.push_back('t');
        buf.push_back(static_cast<char>(0xE7));// variant index 999 (invalid)
        buf.push_back(static_cast<char>(0x03));
        char const *p = buf.data();
        ReduceSumStmt out;
        expect(!out.deserialize(p, p + buf.size()));
    };

    "serialize_size_is_compact"_test = [] {
        // The layout of one rank-2 float tensor: rank(4) + dtype(4) + scope(4)
        // + 3 vectors (each 4 + 2*8).  Only statically meaningful members are
        // stored — no padding, no pointers.
        TensorExpr t{2, TensorElementType::F32, TensorScope::Global, {16, 16}};
        luisa::vector<char> buf;
        auto n = t.serialize(buf);
        expect(n == 4u + 4u + 4u + 3u * (4u + 16u));
        expect(n == buf.size());
    };
}
