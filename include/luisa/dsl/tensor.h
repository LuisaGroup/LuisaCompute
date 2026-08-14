// =============================================================================
// tensor.h — TileLang-style tile / tensor DSL (header-only)
// =============================================================================
// A compile-time interface for the "pure tile / tensor" DSL described in
// `D:/tilelang/dsl_report/tilelang_cpp_tile_style.cpp`.  It makes that
// pseudo-code *valid C++*: every TileLang construct is spelled with real C++
// syntax.  While a tile::Kernel (or tile::jit(...).compile()) is being traced,
// every tile op below emits the matching TensorStmt into the active
// luisa::compute::detail::TileFunctionBuilder, so the kernel's
// function()->body()->statements() contains the real tile IR.  Outside a
// Kernel the same ops fall back to host-side logging only (pure stub mode) —
// there is no device execution in either mode.
//
// Mapping (pseudo-code -> this header):
// namespace T = luisa::compute::tile::language; -> constexpr auto T = luisa::compute::tile::language::dsl;
//                                                    (a C++ namespace is only
//                                                    reachable with `::`, so the
//                                                    `T.*` dot syntax is exposed
//                                                    through a constexpr handle)
// using f16 = luisa::compute::tile::half; -> dtype handle (also a scalar)
// using f32 = luisa::compute::tile::float32;
// using i32 = luisa::compute::tile::int32;
//   Tensor<f16, 2> A;                            -> typed rank-N tensor
//   T.empty({M, N}, f32)                         -> T.empty(T.shape(M, N), f32{})
//   T.alloc_shared({BM, BK}, f16)                -> T.alloc_shared(T.shape(BM, BK), f16{})
//   T.alloc_fragment({blk_m}, f32)               -> T.alloc_fragment(T.shape(blk_m), f32{})
//   for (auto [bx, by] : T.Kernel(gx, gy, t))    -> iterates one representative block
//   for (auto k : T.Pipelined(n, stages))        -> iterates one representative step
//   A[by * BM, bx * BN]                          -> A(by * BM, bx * BN)  (multi-arg
//                                                    `operator[]` is C++23-only,
//                                                    so indexing uses `operator()`)
//   A_shared[BM, BN]                             -> A_shared(BM, BN)  (whole local tile)
//   A[bx * blk_m : (bx + 1) * blk_m, :]          -> A(T.range(bx * blk_m, (bx + 1) * blk_m), T.all())
//   C_local[BM, BN] = A + B                      -> tile-store (logged)
//   T.copy / T.clear / T.gemm / T.reduce_sum     -> logged tile ops
//   T.max / T.rsqrt / T.ceildiv                  -> tile / scalar helpers
//   T.print(tile, "msg")                         -> logged
// luisa::compute::tile::jit(f).compile() -> traces `f` and logs "kernel.compile";
//   shape/tile parameters (M, N, K, block_*, threads, num_stages) live in the
//   kernel function itself (baked constants / function params), not in compile()
//   (mirrors tilelang, where @tilelang.jit functions carry their own config)
// luisa::compute::tile::testing::assert_close(a, b, ...) -> logged
//
// Pseudo kernel:
//   luisa::compute::tile::Kernel k{elementwise_add}; -> the tile-DSL analogue
//     of `luisa::compute::Kernel` in <luisa/dsl/func.h>: it takes a lambda or
//     prim function (e.g. `elementwise_add`) and *traces* the tile program into
//     a `luisa::compute::detail::TileFunctionBuilder`
//     (<luisa/ast/tile_function_builder.h>).  Every tile op below (T.empty,
//     T.alloc_shared, T.copy, T.gemm, tile-store, ...) emits the matching
//     TensorStmt into the active builder, so `k.function()->body()->statements()`
//     contains the real tile IR.  Outside a Kernel the same ops keep logging
//     only (pure host-side stub trace).
//
// All logs go through the LuisaCompute core logger (lc_core), e.g. LUISA_INFO.
// =============================================================================

#pragma once

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>// luisa::format
#include <luisa/core/stl/memory.h>// luisa::unique_ptr / shared_ptr
#include <luisa/core/stl/string.h>// luisa::string
#include <luisa/ast/tile_function_builder.h>// TileFunctionBuilder / TensorExpr / TensorStmt

#include <array>
#include <cstdint>
#include <functional>// std::invoke
#include <tuple>
#include <type_traits>
#include <utility>

// Marks a function as a TileLang kernel factory (the C++ spelling of
// `@T.prim_func`).  The stub performs no TIR lowering, so it is a no-op.
#define TILELANG_PRIM_FUNC

namespace luisa::compute::tile {

// The AST tile-function builder this DSL traces into when a pseudo kernel is
// being defined (see tile::Kernel below).  Alias for luisa::compute::detail::
// TileFunctionBuilder, which lives in <luisa/ast/tile_function_builder.h>.
using TileFunctionBuilder = ::luisa::compute::detail::TileFunctionBuilder;

namespace language {

// Memory scope of a tensor: global (kernel argument / result), shared
// (per-block on-chip memory) or fragment (per-thread registers).  Declared
// before the detail helpers below because they map it to the AST scope.
enum class Scope : uint8_t { Global, Shared, Fragment };

}// namespace language

// ---------------------------------------------------------------------------
// Scalar dtype handles
// ---------------------------------------------------------------------------
// A dtype is used both as a template argument (`Tensor<f16, 2>`) and as a
// scalar value (`f32(0.0f)`, `f32(N)`), so the pseudo-code expressions such as
// `f32(N) + 1e-12f` and `constexpr i32 M = 512;` are valid C++.
struct half {
    float value = 0.0f;
    constexpr half() noexcept = default;
    constexpr half(float v) noexcept : value(v) {}
    constexpr operator float() const noexcept { return value; }
};

struct float32 {
    float value = 0.0f;
    constexpr float32() noexcept = default;
    constexpr float32(float v) noexcept : value(v) {}
    constexpr operator float() const noexcept { return value; }
};

struct int32 {
    int value = 0;
    constexpr int32() noexcept = default;
    constexpr int32(int v) noexcept : value(v) {}
    constexpr operator int() const noexcept { return value; }
};

namespace detail {

inline int next_tensor_id() noexcept {
    static int id = 0;
    return id++;
}

template<typename T>
inline const char *dtype_name() noexcept { return "scalar"; }
template<>
inline const char *dtype_name<half>() noexcept { return "f16"; }
template<>
inline const char *dtype_name<float32>() noexcept { return "f32"; }
template<>
inline const char *dtype_name<int32>() noexcept { return "i32"; }

// DSL dtype handle -> AST TensorElementType tag (R1, <luisa/ast/tensor.h>).
template<typename T>
struct tensor_element_type {
    static constexpr TensorElementType value = TensorElementType::F32;
};
template<>
struct tensor_element_type<half> {
    static constexpr TensorElementType value = TensorElementType::F16;
};
template<>
struct tensor_element_type<float32> {
    static constexpr TensorElementType value = TensorElementType::F32;
};
template<>
struct tensor_element_type<int32> {
    static constexpr TensorElementType value = TensorElementType::I32;
};
template<typename T>
inline constexpr auto tensor_element_type_v = tensor_element_type<T>::value;

// DSL memory scope -> AST TensorScope tag.
inline TensorScope to_ast_scope(language::Scope s) noexcept {
    switch (s) {
        case language::Scope::Global: return TensorScope::Global;
        case language::Scope::Shared: return TensorScope::Shared;
        case language::Scope::Fragment: return TensorScope::Fragment;
    }
    return TensorScope::Global;
}

// std::array<int, R> -> the fixed_vector<int32_t, 4> consumed by the AST.
template<size_t R>
inline luisa::fixed_vector<int32_t, 4> to_fixed_vector(const std::array<int, R> &a) {
    return luisa::fixed_vector<int32_t, 4>{a.begin(), a.end()};
}

// Human-readable TileOpKind name (used when dumping a traced kernel body).
inline const char *tile_op_name(TileOpKind kind) noexcept {
    switch (kind) {
        case TileOpKind::ALLOC: return "alloc";
        case TileOpKind::CLEAR: return "clear";
        case TileOpKind::COPY: return "copy";
        case TileOpKind::GEMM: return "gemm";
        case TileOpKind::REDUCE_SUM: return "reduce_sum";
        case TileOpKind::PRINT: return "print";
        case TileOpKind::STORE: return "store";
        case TileOpKind::BINARY: return "binary";
        case TileOpKind::MAX: return "max";
        case TileOpKind::RSQRT: return "rsqrt";
        case TileOpKind::CEILDIV: return "ceildiv";
        case TileOpKind::KERNEL_1D: return "kernel_1d";
        case TileOpKind::KERNEL_2D: return "kernel_2d";
        case TileOpKind::PIPELINED: return "pipelined";
        default: return "op";
    }
}

template<size_t R>
inline luisa::string join_ints(const std::array<int, R> &a, const char *sep = ",") {
    luisa::string s;
    for (size_t i = 0; i < R; ++i) {
        if (i != 0) { s += sep; }
        s += luisa::format("{}", a[i]);
    }
    return s;
}

}// namespace detail

namespace language {

inline const char *scope_name(Scope s) noexcept {
    switch (s) {
        case Scope::Global: return "global";
        case Scope::Shared: return "shared";
        case Scope::Fragment: return "fragment";
    }
    return "?";
}

// Shape<Rank> is a compile-time-ranked extent list; `T.shape(...)` builds it.
// (`T.empty({M, N}, f32)` is not valid C++ because a braced list cannot
// deduce the rank, so shapes are spelled `T.shape(M, N)`.)
template<size_t R>
struct Shape {
    std::array<int, R> dims{};
};

template<typename... Ints>
constexpr auto shape(Ints... dims) -> Shape<sizeof...(Ints)> {
    return Shape<sizeof...(Ints)>{{static_cast<int>(dims)...}};
}

// Slice: `A[T.range(begin, end), T.all()]` is the C++ spelling of the
// pseudo-code slice `A[begin : end, :]`.
struct Slice {
    int begin = 0;
    int end = -1;// exclusive; -1 means "to the end"
    bool is_all = false;
};

inline constexpr Slice all() noexcept { return {0, -1, true}; }
inline constexpr Slice range(int begin, int end) noexcept { return {begin, end, false}; }

// A tile region: a (possibly derived) view of a tensor.  Whole-tile
// expressions and tile-to-tile stores are spelled as operations on these.
//
// While a tile::Kernel is being traced, a TileExpr also carries the AST
// operand (`ast`) that the tile IR operates on: `owned` holds the operand
// when this TileExpr created it (fresh view clones and value temporaries),
// otherwise `ast` is borrowed.  Passing a TileExpr to a consuming statement
// (tile_copy / tile_store / tile_binary / ...) transfers that ownership via
// take().  Like the AST itself, an operand pointer must not be handed to two
// statements.
template<size_t Rank>
class TileExpr {
public:
    static constexpr size_t rank = Rank;

    luisa::string name;
    Scope scope = Scope::Global;
    std::array<int, Rank> offset{};// anchor in the base tensor
    std::array<int, Rank> extent{};// tile extents (0 = unknown)

    // AST operand (non-template, <luisa/ast/tensor.h>); null in pure
    // host-stub mode.  `owned` is non-null when this TileExpr created the
    // operand and must transfer it to a consuming statement.  Operands are
    // plain-`new`-allocated and freed with plain `delete` (TensorStmt frees
    // its operands with `delete`; eastl::make_unique does NOT match).
    TensorExpr *ast = nullptr;
    mutable TileFunctionBuilder::TensorExprPtr owned;

    TileExpr() = default;
    TileExpr(luisa::string n, Scope sc,
             std::array<int, Rank> off,
             std::array<int, Rank> ext) noexcept
        : name(std::move(n)), scope(sc),
          offset(std::move(off)), extent(std::move(ext)) {}
    TileExpr(TileExpr &&) noexcept = default;
    TileExpr(const TileExpr &) = delete;// operands are move-only (unique_ptr)

    /// Hand the AST operand to a consuming statement (which takes ownership):
    /// releases `owned` when this TileExpr created the operand, otherwise
    /// clones the borrowed `ast`.  Returns nullptr in pure host-stub mode.
    [[nodiscard]] TensorExpr *take() const noexcept {
        if (owned) { return owned.release(); }
        return ast != nullptr ? new TensorExpr(*ast) : nullptr;
    }

    [[nodiscard]] luisa::string describe() const {
        if (extent != std::array<int, Rank>{}) {
            return luisa::format("{}[{}]", name, detail::join_ints(extent));
        }
        return luisa::format("{}@({})", name, detail::join_ints(offset));
    }

    // C_local[BM, BN] = <tile expr>;  A_powsum[blk_m] = T.rsqrt(...)
    // Every assignment form (rvalue / lvalue / cross-rank) lowers to
    // TileStoreStmt (op 0) while a kernel is traced.
    TileExpr &operator=(TileExpr &&rhs) noexcept { return _store(0, rhs); }
    TileExpr &operator=(const TileExpr &rhs) noexcept { return _store(0, rhs); }
    template<size_t R2>
    TileExpr &operator=(const TileExpr<R2> &rhs) noexcept {
        return _store(0, rhs);
    }

    // A_local[blk_m, N] *= A_powsum[blk_m];  (row-broadcast scale)
    // Lowers to TileStoreStmt (op 1) while a kernel is traced.
    TileExpr &operator*=(TileExpr &&rhs) noexcept { return _store(1, rhs); }
    TileExpr &operator*=(const TileExpr &rhs) noexcept { return _store(1, rhs); }
    template<size_t R2>
    TileExpr &operator*=(const TileExpr<R2> &rhs) noexcept {
        return _store(1, rhs);
    }

private:
    template<size_t R2>
    TileExpr &_store(int op, const TileExpr<R2> &rhs) noexcept {
        LUISA_INFO("[tensor-dsl] tile-store: {} {} {}", describe(),
                   op == 0 ? "=" : "*=", rhs.describe());
        if (auto *builder = TileFunctionBuilder::current_or_null()) {
            builder->tile_store(op, take(), rhs.take());
        }
        return *this;
    }
};

// A typed, rank-N, shape-carrying tensor.  The stub tracks its memory scope
// (global / shared / fragment) and logs every allocation.
template<typename DType, size_t Rank>
class Tensor {
public:
    static constexpr size_t rank = Rank;
    using dtype = DType;

private:
    std::array<int, Rank> _dims{};
    Scope _scope = Scope::Global;
    luisa::string _name;
    // Borrowed AST operand (global / shared / fragment tensor), owned by the
    // AllocStmt emitted into the active TileFunctionBuilder; null in pure
    // host-stub mode.
    TensorExpr *_ast = nullptr;

    [[nodiscard]] luisa::string make_default_name() const {
        return luisa::format("{}<{},{}>#{}", scope_name(_scope), detail::dtype_name<DType>(),
                             Rank, detail::next_tensor_id());
    }

    [[nodiscard]] const char *factory_prefix() const noexcept {
        switch (_scope) {
            case Scope::Shared: return "T.alloc_shared";
            case Scope::Fragment: return "T.alloc_fragment";
            default: return "tensor";
        }
    }

public:
    Tensor() {
        _name = make_default_name();
    }

    explicit Tensor(const Shape<Rank> &s, Scope scope = Scope::Global, luisa::string name = {})
        : _dims(s.dims), _scope(scope), _name(std::move(name)) {
        if (_name.empty()) { _name = make_default_name(); }
        LUISA_INFO("[tensor-dsl] {}: {}({})", factory_prefix(), _name, detail::join_ints(_dims));
    }

    template<typename... Ints>
        requires(sizeof...(Ints) == Rank)
    explicit Tensor(Ints... dims)
        : _dims{static_cast<int>(dims)...}, _scope(Scope::Global) {
        _name = make_default_name();
        LUISA_INFO("[tensor-dsl] tensor: {}({})", _name, detail::join_ints(_dims));
    }

    [[nodiscard]] const std::array<int, Rank> &dims() const noexcept { return _dims; }
    [[nodiscard]] Scope scope() const noexcept { return _scope; }
    [[nodiscard]] const luisa::string &name() const noexcept { return _name; }

    /// Attach the AST operand of this tensor (borrowed; the AllocStmt emitted
    /// by the active TileFunctionBuilder owns it).  Internal DSL use.
    void set_ast(TensorExpr *ast) noexcept { _ast = ast; }
    /// The borrowed AST operand, or nullptr in pure host-stub mode.
    [[nodiscard]] TensorExpr *ast() const noexcept { return _ast; }
    /// A fresh deep copy of the AST operand for a consuming statement (which
    /// takes ownership), or nullptr in pure host-stub mode.
    [[nodiscard]] TensorExpr *clone_ast() const noexcept {
        return _ast != nullptr ? new TensorExpr(*_ast) : nullptr;
    }

    [[nodiscard]] luisa::string describe() const {
        return luisa::format("{}({})", _name, detail::join_ints(_dims));
    }

    // Tile indexing is spelled with `operator()`: the pseudo-code `A[i, j]`
    // is not valid C++20 (a multi-parameter `operator[]` is C++23-only), so
    // the stub exposes `A(i, j)` (like Eigen) instead.
    //
    // A(i, j, ...) on a global tensor anchors the tile at (i, j); on a
    // shared / fragment tensor it denotes the whole local tile of that extent.
    template<typename... Ints>
        requires(sizeof...(Ints) == Rank &&
                 (std::is_convertible_v<Ints, int> && ...))
    [[nodiscard]] TileExpr<Rank> operator()(Ints... idx) const {
        std::array<int, Rank> values{static_cast<int>(idx)...};
        std::array<int, Rank> off{};
        std::array<int, Rank> ext{};
        if (_scope == Scope::Global) {
            off = values;// anchor in the global tensor
        } else {
            ext = values;// whole local tile
        }
        TileExpr<Rank> e(_name, _scope, off, ext);
        if (auto *builder = TileFunctionBuilder::current_or_null()) {
            // Fresh AST operand for the tile view; `e` owns it (plain new /
            // plain delete, matching TensorStmt operand ownership) until it
            // is handed to a consuming statement (tile_copy / tile_store / ...).
            auto texpr = TileFunctionBuilder::TensorExprPtr{new TensorExpr{
                static_cast<int32_t>(Rank), detail::tensor_element_type_v<DType>,
                detail::to_ast_scope(_scope), detail::to_fixed_vector(_dims),
                detail::to_fixed_vector(off), detail::to_fixed_vector(ext),
                nullptr, e.name}};
            e.ast = texpr.get();
            e.owned = std::move(texpr);
        }
        return e;
    }

    // A(T.range(begin, end), T.all()) — rank-2 row slice.
    [[nodiscard]] TileExpr<Rank> operator()(const Slice &s0, const Slice &s1) const
        requires(Rank == 2) {
        std::array<int, Rank> off{};
        std::array<int, Rank> ext{};
        off[0] = s0.is_all ? 0 : s0.begin;
        off[1] = s1.is_all ? 0 : s1.begin;
        ext[0] = s0.is_all ? _dims[0] : s0.end - s0.begin;
        ext[1] = s1.is_all ? _dims[1] : s1.end - s1.begin;
        TileExpr<Rank> e(_name, _scope, off, ext);
        if (auto *builder = TileFunctionBuilder::current_or_null()) {
            auto texpr = TileFunctionBuilder::TensorExprPtr{new TensorExpr{
                static_cast<int32_t>(Rank), detail::tensor_element_type_v<DType>,
                detail::to_ast_scope(_scope), detail::to_fixed_vector(_dims),
                detail::to_fixed_vector(off), detail::to_fixed_vector(ext),
                nullptr, e.name}};
            e.ast = texpr.get();
            e.owned = std::move(texpr);
        }
        return e;
    }
};

}// namespace language

namespace detail {

template<typename T>
struct tile_rank;
template<typename DType, size_t R>
struct tile_rank<language::Tensor<DType, R>> : std::integral_constant<size_t, R> {};
template<size_t R>
struct tile_rank<language::TileExpr<R>> : std::integral_constant<size_t, R> {};

template<typename T>
inline constexpr size_t tile_rank_v = tile_rank<std::remove_cvref_t<T>>::value;

template<typename DType, size_t R>
inline luisa::string describe(const language::Tensor<DType, R> &t) { return t.describe(); }
template<size_t R>
inline luisa::string describe(const language::TileExpr<R> &t) { return t.describe(); }

// ---- AST operand extraction ------------------------------------------------
// TileExpr and Tensor are the two operand kinds a tile op can consume.
template<typename T>
struct is_tile_expr : std::false_type {};
template<size_t R>
struct is_tile_expr<language::TileExpr<R>> : std::true_type {};
template<typename T>
inline constexpr bool is_tile_expr_v = is_tile_expr<std::remove_cvref_t<T>>::value;

template<typename T>
struct is_tile_tensor : std::false_type {};
template<typename DType, size_t R>
struct is_tile_tensor<language::Tensor<DType, R>> : std::true_type {};
template<typename T>
inline constexpr bool is_tile_tensor_v = is_tile_tensor<std::remove_cvref_t<T>>::value;

// Hand a fresh, statement-owned AST operand for `t` to a consuming tile
// statement: TileExpr hands over its owned view / value temporary, Tensor
// hands over a fresh clone of its borrowed operand.  Returns nullptr in pure
// host-stub mode (no TileFunctionBuilder active).
template<typename T>
inline TensorExpr *extract_operand(T &t) noexcept {
    if constexpr (is_tile_expr_v<T>) {
        return t.take();
    } else {
        return t.clone_ast();
    }
}

// DSL op spelling -> AST BinaryOp tag.
inline BinaryOp binary_op_from(const char *op) noexcept {
    switch (op[0]) {
        case '+': return BinaryOp::ADD;
        case '-': return BinaryOp::SUB;
        case '*': return BinaryOp::MUL;
        case '/': return BinaryOp::DIV;
        default: return BinaryOp::ADD;
    }
}

// Whole-tile elementwise binary ops.  While a kernel is traced they emit a
// TileBinaryStmt and wrap the returned fragment temporary in the result
// TileExpr (owned by the caller, i.e. the TileExpr).
template<size_t R>
inline language::TileExpr<R> binary_op(const char *op,
                                       const language::TileExpr<R> &a,
                                       const language::TileExpr<R> &b) {
    LUISA_INFO("[tensor-dsl] tile-op: {} {} {}", describe(a), op, describe(b));
    language::TileExpr<R> e;
    e.name = luisa::format("expr({})", op);
    e.scope = a.scope;
    e.offset = a.offset;
    e.extent = a.extent;
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        auto tmp = builder->tile_binary(binary_op_from(op), a.take(), b.take());
        e.ast = tmp.get();
        e.owned = std::move(tmp);
    }
    return e;
}

// Whole-tile scalar binary ops (e.g. `A_powsum(blk_m) / f32(N) + 1e-12f`):
// rhs is an R2 literal embedded in the emitted TileBinaryStmt.
template<size_t R>
inline language::TileExpr<R> scalar_op(const char *op,
                                       const language::TileExpr<R> &a,
                                       float b) {
    LUISA_INFO("[tensor-dsl] tile-op: {} {} {}", describe(a), op, b);
    language::TileExpr<R> e;
    e.name = luisa::format("expr({})", op);
    e.scope = a.scope;
    e.offset = a.offset;
    e.extent = a.extent;
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        auto lit = builder->create_literal(Type::of<float>(), b);
        auto tmp = builder->tile_binary(binary_op_from(op), a.take(), nullptr, lit);
        e.ast = tmp.get();
        e.owned = std::move(tmp);
    }
    return e;
}

// ---- callable traits (function pointers and lambdas/function objects) ------
template<typename F>
struct fn_traits : fn_traits<decltype(&std::remove_cvref_t<F>::operator())> {};
template<typename Ret, typename... Args>
struct fn_traits<Ret (*)(Args...)> {
    using return_type = Ret;
    using arg_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};
template<typename Ret, typename... Args>
struct fn_traits<Ret (*)(Args...) noexcept> {
    using return_type = Ret;
    using arg_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};
#define LUISA_TILE_FN_TRAITS_MEMBER(QUAL)                                     \
    template<typename C, typename Ret, typename... Args>                      \
    struct fn_traits<Ret (C::*)(Args...) QUAL> {                              \
        using return_type = Ret;                                              \
        using arg_tuple = std::tuple<Args...>;                                \
        static constexpr size_t arity = sizeof...(Args);                      \
    };
LUISA_TILE_FN_TRAITS_MEMBER()
LUISA_TILE_FN_TRAITS_MEMBER(const)
LUISA_TILE_FN_TRAITS_MEMBER(noexcept)
LUISA_TILE_FN_TRAITS_MEMBER(const noexcept)
#undef LUISA_TILE_FN_TRAITS_MEMBER

// ---- kernel argument creation ----------------------------------------------
// Create a default kernel argument of type T.  Tile tensors become global
// tensors (AllocStmt, TensorScope::Global) in the active TileFunctionBuilder;
// other argument types are default-constructed (stub: only tensors matter).
template<typename T>
T make_kernel_arg() {
    if constexpr (is_tile_tensor_v<T>) {
        auto *builder = TileFunctionBuilder::current();
        T t;
        std::array<int, T::rank> zeros{};
        t.set_ast(builder->tile_empty(detail::to_fixed_vector(zeros),
                                      tensor_element_type_v<typename T::dtype>,
                                      t.name()));
        return t;
    } else {
        return T{};
    }
}

// One-line summary of the statements of a traced kernel body (op names).
inline luisa::string describe(const TileFunctionBuilder &builder) {
    luisa::string s;
    for (auto *stmt : builder.body()->statements()) {
        if (!s.empty()) { s += ", "; }
        s += tile_op_name(stmt->op());
    }
    return s;
}

}// namespace detail

namespace language {

// Whole-tile elementwise operators (the C++ spelling of `T.Parallel` loops).
template<size_t R>
inline TileExpr<R> operator+(const TileExpr<R> &a, const TileExpr<R> &b) {
    return detail::binary_op("+", a, b);
}
template<size_t R>
inline TileExpr<R> operator+(const TileExpr<R> &a, float b) {
    return detail::scalar_op("+", a, b);
}
template<size_t R>
inline TileExpr<R> operator*(const TileExpr<R> &a, const TileExpr<R> &b) {
    return detail::binary_op("*", a, b);
}
template<size_t R>
inline TileExpr<R> operator/(const TileExpr<R> &a, float b) {
    return detail::scalar_op("/", a, b);
}

template<size_t R>
inline TileExpr<R> max(const TileExpr<R> &a, float b) {
    LUISA_INFO("[tensor-dsl] tile-op: max({}, {})", detail::describe(a), b);
    TileExpr<R> e;
    e.name = "expr(max)";
    e.scope = a.scope;
    e.offset = a.offset;
    e.extent = a.extent;
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        auto lit = builder->create_literal(Type::of<float>(), b);
        auto tmp = builder->tile_max(a.take(), lit);
        e.ast = tmp.get();
        e.owned = std::move(tmp);
    }
    return e;
}

template<size_t R>
inline TileExpr<R> rsqrt(const TileExpr<R> &a) {
    LUISA_INFO("[tensor-dsl] tile-op: rsqrt({})", detail::describe(a));
    TileExpr<R> e;
    e.name = "expr(rsqrt)";
    e.scope = a.scope;
    e.offset = a.offset;
    e.extent = a.extent;
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        auto tmp = builder->tile_rsqrt(a.take());
        e.ast = tmp.get();
        e.owned = std::move(tmp);
    }
    return e;
}

inline int ceildiv(int a, int b) noexcept {
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        return builder->tile_ceildiv(a, b);
    }
    return (a + b - 1) / b;
}

// ---- index binders ----------------------------------------------------------
// `T.Kernel(gx, gy, threads)` is the C++ spelling of `with T.Kernel(...) as
// (bx, by)`.  The stub iterates a single representative block: a real lowering
// would iterate all gx*gy blocks, but the tile program itself is identical for
// every block, so one trace is enough.
struct Kernel2D {
    int gx = 1;
    int gy = 1;
    int threads = 1;

    struct Iterator {
        const Kernel2D *kernel = nullptr;
        int i = 0;
        [[nodiscard]] std::tuple<int, int> operator*() const {
            return {i % kernel->gx, i / kernel->gx};// (bx, by)
        }
        Iterator &operator++() noexcept { ++i; return *this; }
        [[nodiscard]] bool operator!=(const Iterator &other) const noexcept {
            return i != other.i;
        }
    };

    [[nodiscard]] Iterator begin() const noexcept {
        LUISA_INFO("[tensor-dsl] T.Kernel: grid=({},{}), threads={} [stub: tracing one representative block]",
                   gx, gy, threads);
        if (auto *builder = TileFunctionBuilder::current_or_null()) {
            builder->tile_kernel_2d(gx, gy, threads);
        }
        return {this, 0};
    }
    [[nodiscard]] Iterator end() const noexcept { return {this, 1}; }
};

struct Kernel1D {
    int gx = 1;
    int threads = 1;

    struct Iterator {
        const Kernel1D *kernel = nullptr;
        int i = 0;
        [[nodiscard]] int operator*() const noexcept { return i; }
        Iterator &operator++() noexcept { ++i; return *this; }
        [[nodiscard]] bool operator!=(const Iterator &other) const noexcept {
            return i != other.i;
        }
    };

    [[nodiscard]] Iterator begin() const noexcept {
        LUISA_INFO("[tensor-dsl] T.Kernel: grid=({}), threads={} [stub: tracing one representative block]",
                   gx, threads);
        if (auto *builder = TileFunctionBuilder::current_or_null()) {
            builder->tile_kernel_1d(gx, threads);
        }
        return {this, 0};
    }
    [[nodiscard]] Iterator end() const noexcept { return {this, 1}; }
};

inline Kernel1D Kernel(int gx, int threads) { return {gx, threads}; }
inline Kernel2D Kernel(int gx, int gy, int threads) { return {gx, gy, threads}; }

// `for (auto k : T.Pipelined(n, stages))` — the software-pipelined K loop.
// The stub iterates a single representative step: like Kernel1D/Kernel2D, the
// tile program inside the loop is identical for every K step (a real lowering
// would unroll/pipeline `count` iterations), so one trace is enough.  This is
// the tile analogue of the `ForRange` binder in <luisa/dsl/stmt.h>, which also
// emits its loop construct once in begin() and iterates one representative
// trip (F1: end() must yield after the first step, not after `count`).
struct PipelinedRange {
    int count = 0;
    int stages = 1;

    struct Iterator {
        int i = 0;
        [[nodiscard]] int operator*() const noexcept { return i; }
        Iterator &operator++() noexcept { ++i; return *this; }
        [[nodiscard]] bool operator!=(const Iterator &other) const noexcept {
            return i != other.i;
        }
    };

    [[nodiscard]] Iterator begin() const noexcept {
        LUISA_INFO("[tensor-dsl] T.Pipelined: {} iterations x {} stages [stub: tracing one representative step]",
                   count, stages);
        if (auto *builder = TileFunctionBuilder::current_or_null()) {
            builder->tile_pipelined(count, stages);
        }
        return {0};
    }
    [[nodiscard]] Iterator end() const noexcept { return {1}; }
};

inline PipelinedRange Pipelined(int count, int stages) { return {count, stages}; }

// ---- tile allocation & ops -------------------------------------------------
template<typename DType, size_t R>
inline Tensor<DType, R> empty(const Shape<R> &dims, DType) {
    Tensor<DType, R> t(dims, Scope::Global);
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        // T.empty: global tensor (kernel argument / result buffer).
        t.set_ast(builder->tile_empty(detail::to_fixed_vector(dims.dims),
                                      detail::tensor_element_type_v<DType>,
                                      t.name()));
    }
    return t;
}

template<typename DType, size_t R>
inline Tensor<DType, R> alloc_shared(const Shape<R> &dims, DType) {
    Tensor<DType, R> t(dims, Scope::Shared);
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        t.set_ast(builder->tile_alloc_shared(detail::to_fixed_vector(dims.dims),
                                             detail::tensor_element_type_v<DType>,
                                             t.name()));
    }
    return t;
}

template<typename DType, size_t R>
inline Tensor<DType, R> alloc_fragment(const Shape<R> &dims, DType) {
    Tensor<DType, R> t(dims, Scope::Fragment);
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        t.set_ast(builder->tile_alloc_fragment(detail::to_fixed_vector(dims.dims),
                                               detail::tensor_element_type_v<DType>,
                                               t.name()));
    }
    return t;
}

template<typename Src, typename Dst>
inline void copy(const Src &src, const Dst &dst) {
    static_assert(detail::tile_rank_v<Src> == detail::tile_rank_v<Dst>,
                  "T.copy requires source and destination tiles of equal rank");
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        builder->tile_copy(detail::extract_operand(src), detail::extract_operand(dst));
    }
    LUISA_INFO("[tensor-dsl] T.copy: {} -> {}", detail::describe(src), detail::describe(dst));
}

template<typename T>
inline void clear(const T &t) {
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        builder->tile_clear(detail::extract_operand(t));
    }
    LUISA_INFO("[tensor-dsl] T.clear: {}", detail::describe(t));
}

template<typename A, typename B, typename C>
inline void gemm(const A &a, const B &b, const C &c) {
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        builder->tile_gemm(detail::extract_operand(a), detail::extract_operand(b),
                           detail::extract_operand(c));
    }
    LUISA_INFO("[tensor-dsl] T.gemm: {} x {} -> {}", detail::describe(a),
               detail::describe(b), detail::describe(c));
}

template<typename X, typename Y>
inline void reduce_sum(const X &x, const Y &y, int dim) {
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        builder->tile_reduce_sum(detail::extract_operand(x), detail::extract_operand(y),
                                 static_cast<uint32_t>(dim));
    }
    LUISA_INFO("[tensor-dsl] T.reduce_sum: {} -> {} (dim={})",
               detail::describe(x), detail::describe(y), dim);
}

template<typename T>
inline void print(const T &t, const char *msg) {
    if (auto *builder = TileFunctionBuilder::current_or_null()) {
        builder->tile_print(detail::extract_operand(t), msg);
    }
    LUISA_INFO("[tensor-dsl] T.print: {} {}", msg, detail::describe(t));
}

// ---------------------------------------------------------------------------
// The `T` handle — mirrors `import tilelang.language as T`.
//
// A C++ namespace can only be addressed with `::`, never with `.`, so the
// pseudo-code `T.empty(...)` / `T.copy(...)` dot syntax is exposed through a
// constexpr handle object.  The user file writes:
//
//   constexpr auto T = luisa::compute::tile::language::dsl;
//
// and then uses exactly the TileLang spelling `T.empty(...)`, `T.Kernel(...)`,
// `T.copy(...)`, `T.gemm(...)`, `T.ceildiv(...)`, ...
// ---------------------------------------------------------------------------
struct dsl_t {

    template<typename... Ints>
    constexpr auto shape(Ints... dims) const {
        return luisa::compute::tile::language::shape(dims...);
    }

    template<typename DType, size_t R>
    auto empty(const Shape<R> &dims, DType d) const {
        return luisa::compute::tile::language::empty(dims, d);
    }

    template<typename DType, size_t R>
    auto alloc_shared(const Shape<R> &dims, DType d) const {
        return luisa::compute::tile::language::alloc_shared(dims, d);
    }

    template<typename DType, size_t R>
    auto alloc_fragment(const Shape<R> &dims, DType d) const {
        return luisa::compute::tile::language::alloc_fragment(dims, d);
    }

    template<typename Src, typename Dst>
    void copy(const Src &src, const Dst &dst) const {
        luisa::compute::tile::language::copy(src, dst);
    }

    template<typename DType, size_t R>
    void clear(const Tensor<DType, R> &t) const {
        luisa::compute::tile::language::clear(t);
    }

    template<typename A, typename B, typename C>
    void gemm(const A &a, const B &b, const C &c) const {
        luisa::compute::tile::language::gemm(a, b, c);
    }

    template<typename X, typename Y>
    void reduce_sum(const X &x, const Y &y, int dim) const {
        luisa::compute::tile::language::reduce_sum(x, y, dim);
    }

    template<typename T>
    void print(const T &t, const char *msg) const {
        luisa::compute::tile::language::print(t, msg);
    }

    template<size_t R>
    auto max(const TileExpr<R> &a, float b) const {
        return luisa::compute::tile::language::max(a, b);
    }

    template<size_t R>
    auto rsqrt(const TileExpr<R> &a) const {
        return luisa::compute::tile::language::rsqrt(a);
    }

    int ceildiv(int a, int b) const noexcept {
        return luisa::compute::tile::language::ceildiv(a, b);
    }

    constexpr Kernel1D Kernel(int gx, int threads) const {
        return {gx, threads};
    }
    constexpr Kernel2D Kernel(int gx, int gy, int threads) const {
        return {gx, gy, threads};
    }

    constexpr PipelinedRange Pipelined(int count, int stages) const {
        return {count, stages};
    }

    constexpr Slice all() const noexcept {
        return luisa::compute::tile::language::all();
    }
    constexpr Slice range(int begin, int end) const noexcept {
        return luisa::compute::tile::language::range(begin, end);
    }
};

inline constexpr dsl_t dsl{};

}// namespace language

// ---------------------------------------------------------------------------
// tile::Kernel — the pseudo kernel entry point (the tile-DSL analogue of
// luisa::compute::Kernel in <luisa/dsl/func.h>).
//
// Constructing a Kernel traces the given lambda or prim function (e.g.
// `elementwise_add` in examples/compute/tensor_stub.cpp) into a
// luisa::compute::detail::TileFunctionBuilder: the function is invoked once
// on the host with freshly allocated global argument tensors, and every tile
// op it performs (T.empty, T.alloc_shared, T.copy, tile-store, T.gemm, ...)
// emits the matching TensorStmt into the builder.  The resulting IR is
// available through function() / describe().
// ---------------------------------------------------------------------------
template<typename F>
class Kernel {

private:
    using fn_type = std::remove_cvref_t<F>;
    using traits = detail::fn_traits<fn_type>;
    luisa::shared_ptr<const TileFunctionBuilder> _builder;

public:
    /// Trace `def` (a lambda or a prim function like elementwise_add) into a
    /// TileFunctionBuilder, mirroring how func.h's Kernel executes its lambda
    /// once on the host to record the AST.
    explicit Kernel(F def) {
        _builder = TileFunctionBuilder::define([&def] {
            []<size_t... i>(auto &&f, std::index_sequence<i...>) {
                using arg_tuple = typename traits::arg_tuple;
                auto args = std::make_tuple(
                    detail::make_kernel_arg<std::tuple_element_t<i, arg_tuple>>()...);
                // The return value is the kernel's result tensor (or void);
                // a real lowering would turn it into an output argument.
                [[maybe_unused]] auto result =
                    std::invoke(std::forward<decltype(f)>(f), std::get<i>(args)...);
            }(def, std::make_index_sequence<traits::arity>{});
        });
    }

    /// The traced tile IR (a TileFunctionBuilder holding the emitted
    /// TensorStmt nodes), or nullptr if tracing failed.
    [[nodiscard]] auto function() const noexcept { return _builder; }

    /// One-line summary of the traced statement list, e.g.
    /// "alloc, ceildiv, kernel_2d, alloc, alloc, copy, binary, store".
    [[nodiscard]] luisa::string describe() const {
        return _builder ? detail::describe(*_builder) : luisa::string{};
    }
};

// ---------------------------------------------------------------------------
// Host side: luisa::compute::tile::jit(kernel).compile()  (mirrors @tilelang.jit)
//
// Like tilelang's `jit.compile`, this takes NO shape / tile parameters: the
// kernel function itself carries its configuration (baked constexpr sizes, or
// plain function parameters with defaults).  compile() only traces the prim
// function into a TileFunctionBuilder and wraps it in a CompiledKernel.
// ---------------------------------------------------------------------------
template<typename Ret>
class CompiledKernel;

template<typename F>
class jit {
    F _fn;

public:
    explicit jit(F f) : _fn(std::move(f)) {}

    // No shape/tile arguments (M, N, K, block_M, block_N, block_K, threads,
    // num_stages, ...): those are part of the kernel function, exactly like
    // tilelang where `compile()` is generic and the @tilelang.jit function
    // owns its compile-time constants.  compile() traces `_fn` (using the
    // defaults baked into it) and keeps the traced IR for introspection.
    auto compile() const {
        using traits = detail::fn_traits<std::remove_cvref_t<F>>;
        LUISA_INFO("[tensor-dsl] kernel.compile: {} (config baked into the kernel function)",
                   "prim_function");
        // Trace the prim function into a TileFunctionBuilder exactly like a
        // real DSL would at compile time.
        Kernel<std::remove_cvref_t<F>> kernel{_fn};
        // Enforce the one-launch-per-function rule at compile time: a tile
        // function maps to exactly one T.Kernel (TileLang emits one
        // `__global__` per `T.Kernel`).  Deriving the SIMT launch metadata
        // validates the traced body — it logs an error and aborts when the
        // body contains zero or more than one T.Kernel.
        [[maybe_unused]] auto meta = kernel.function()->compile_meta_data();
        return CompiledKernel<typename traits::return_type>{kernel.function()};
    }
};

// A compiled kernel: callable (`matmul_kernel(A, B)`) and introspectable
// (`get_kernel_source()`, `function()`).  The stub logs and returns a default
// tensor; the traced TileFunctionBuilder is kept for introspection.
template<typename Ret>
class CompiledKernel {
public:
    luisa::string name = "compiled_kernel";
    luisa::shared_ptr<const TileFunctionBuilder> builder;

    CompiledKernel() = default;
    explicit CompiledKernel(luisa::string n) : name(std::move(n)) {}
    explicit CompiledKernel(luisa::shared_ptr<const TileFunctionBuilder> b,
                            luisa::string n = "compiled_kernel")
        : name(std::move(n)), builder(std::move(b)) {}

    /// The traced tile IR (TileFunctionBuilder) produced by tile::Kernel.
    [[nodiscard]] auto function() const noexcept { return builder; }

    /// One-line summary of the traced statement list (see tile::Kernel::describe).
    [[nodiscard]] luisa::string describe() const {
        return builder ? detail::describe(*builder) : luisa::string{};
    }

    template<typename... Args>
    Ret operator()(Args &&.../*args*/) const {
        LUISA_INFO("[tensor-dsl] kernel.run: {} (stub: no device execution)", name);
        return Ret{};// stub: no real computation
    }

    [[nodiscard]] luisa::string get_kernel_source() const {
        LUISA_INFO("[tensor-dsl] kernel.get_kernel_source: stub (no kernel source generated)");
        luisa::string src = "// tensor-dsl stub: traced tile IR\n";
        if (builder) { src += "// " + detail::describe(*builder) + "\n"; }
        return src;
    }
};

namespace testing {

template<typename A, typename B>
inline void assert_close(const A &a, const B &b, float rtol, float atol) {
    LUISA_INFO("[tensor-dsl] testing::assert_close: {} vs {} (rtol={}, atol={})",
               detail::describe(a), detail::describe(b), rtol, atol);
}

}// namespace testing

inline void print(const luisa::string &s) {
    LUISA_INFO("[tensor-dsl] luisa::compute::tile::print: {}", s);
}

/*
TODO
implement a `tile_to_kernel` dsl version
*/

}// namespace luisa::compute::tile
