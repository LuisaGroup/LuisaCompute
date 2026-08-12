// =============================================================================
// tensor.h — TileLang-style tile / tensor DSL (header-only STUB)
// =============================================================================
// A compile-time stub for the "pure tile / tensor" DSL described in
// `D:/tilelang/dsl_report/tilelang_cpp_tile_style.cpp`.  It makes that
// pseudo-code *valid C++*: every TileLang construct is spelled with real C++
// syntax, and every construct lowers to a host-side log line (there is no
// device execution — the stub only traces the tile program).
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
// luisa::compute::tile::jit(f).compile(M, N, ...) -> traces `f` and logs "kernel.compile"
// luisa::compute::tile::testing::assert_close(a, b, ...) -> logged
//
// All logs go through the LuisaCompute core logger (lc_core), e.g. LUISA_INFO.
// =============================================================================

#pragma once

#include <luisa/core/logging.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

// Marks a function as a TileLang kernel factory (the C++ spelling of
// `@T.prim_func`).  The stub performs no TIR lowering, so it is a no-op.
#define TILELANG_PRIM_FUNC

namespace luisa::compute::tile {

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

template<size_t R>
inline std::string join_ints(const std::array<int, R> &a, const char *sep = ",") {
    std::string s;
    for (size_t i = 0; i < R; ++i) {
        if (i != 0) { s += sep; }
        s += std::to_string(a[i]);
    }
    return s;
}

}// namespace detail

namespace language {

// Memory scope of a tensor: global (kernel argument / result), shared
// (per-block on-chip memory) or fragment (per-thread registers).
enum class Scope : uint8_t { Global, Shared, Fragment };

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
template<size_t Rank>
class TileExpr {
public:
    static constexpr size_t rank = Rank;

    std::string name;
    Scope scope = Scope::Global;
    std::array<int, Rank> offset{};// anchor in the base tensor
    std::array<int, Rank> extent{};// tile extents (0 = unknown)

    TileExpr() = default;
    TileExpr(std::string n, Scope sc,
             std::array<int, Rank> off,
             std::array<int, Rank> ext) noexcept
        : name(std::move(n)), scope(sc),
          offset(std::move(off)), extent(std::move(ext)) {}

    [[nodiscard]] std::string describe() const {
        std::string s = name;
        if (extent != std::array<int, Rank>{}) {
            s += "[" + detail::join_ints(extent) + "]";
        } else {
            s += "@(" + detail::join_ints(offset) + ")";
        }
        return s;
    }

    // C_local[BM, BN] = <tile expr>;  A_powsum[blk_m] = T.rsqrt(...)
    template<size_t R2>
    TileExpr &operator=(const TileExpr<R2> &rhs) noexcept {
        LUISA_INFO("[tensor-dsl] tile-store: {} = {}", describe(), rhs.describe());
        return *this;
    }

    // A_local[blk_m, N] *= A_powsum[blk_m];  (row-broadcast scale)
    template<size_t R2>
    TileExpr &operator*=(const TileExpr<R2> &rhs) noexcept {
        LUISA_INFO("[tensor-dsl] tile-store: {} *= {}", describe(), rhs.describe());
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
    std::string _name;

    [[nodiscard]] std::string make_default_name() const {
        return std::string(scope_name(_scope)) + "<" + detail::dtype_name<DType>() + "," +
               std::to_string(Rank) + ">#" + std::to_string(detail::next_tensor_id());
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

    explicit Tensor(const Shape<Rank> &s, Scope scope = Scope::Global, std::string name = {})
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
    [[nodiscard]] const std::string &name() const noexcept { return _name; }

    [[nodiscard]] std::string describe() const {
        return _name + "(" + detail::join_ints(_dims) + ")";
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
        return TileExpr<Rank>(_name, _scope, off, ext);
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
        return TileExpr<Rank>(_name, _scope, off, ext);
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
inline std::string describe(const language::Tensor<DType, R> &t) { return t.describe(); }
template<size_t R>
inline std::string describe(const language::TileExpr<R> &t) { return t.describe(); }

template<size_t R>
inline language::TileExpr<R> binary_op(const char *op,
                                       const language::TileExpr<R> &a,
                                       const language::TileExpr<R> &b) {
    LUISA_INFO("[tensor-dsl] tile-op: {} {} {}", describe(a), op, describe(b));
    language::TileExpr<R> e = a;
    e.name = std::string("expr(") + op + ")";
    return e;
}

template<size_t R>
inline language::TileExpr<R> scalar_op(const char *op,
                                       const language::TileExpr<R> &a,
                                       float b) {
    LUISA_INFO("[tensor-dsl] tile-op: {} {} {}", describe(a), op, b);
    language::TileExpr<R> e = a;
    e.name = std::string("expr(") + op + ")";
    return e;
}

template<typename F>
struct fn_traits;
template<typename Ret, typename... Args>
struct fn_traits<Ret (*)(Args...)> {
    using return_type = Ret;
    using arg_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

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
    return detail::scalar_op("max", a, b);
}

template<size_t R>
inline TileExpr<R> rsqrt(const TileExpr<R> &a) {
    LUISA_INFO("[tensor-dsl] tile-op: rsqrt({})", detail::describe(a));
    TileExpr<R> e = a;
    e.name = "expr(rsqrt)";
    return e;
}

inline int ceildiv(int a, int b) noexcept { return (a + b - 1) / b; }

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
        return {this, 0};
    }
    [[nodiscard]] Iterator end() const noexcept { return {this, 1}; }
};

inline Kernel1D Kernel(int gx, int threads) { return {gx, threads}; }
inline Kernel2D Kernel(int gx, int gy, int threads) { return {gx, gy, threads}; }

// `for (auto k : T.Pipelined(n, stages))` — the software-pipelined K loop.
// The stub iterates a single representative step.
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
        LUISA_INFO("[tensor-dsl] T.Pipelined: {} iterations x {} stages [stub: tracing iteration 0]",
                   count, stages);
        return {0};
    }
    [[nodiscard]] Iterator end() const noexcept { return {count}; }
};

inline PipelinedRange Pipelined(int count, int stages) { return {count, stages}; }

// ---- tile allocation & ops -------------------------------------------------
template<typename DType, size_t R>
inline Tensor<DType, R> empty(const Shape<R> &dims, DType) {
    return Tensor<DType, R>(dims, Scope::Global);
}

template<typename DType, size_t R>
inline Tensor<DType, R> alloc_shared(const Shape<R> &dims, DType) {
    return Tensor<DType, R>(dims, Scope::Shared);
}

template<typename DType, size_t R>
inline Tensor<DType, R> alloc_fragment(const Shape<R> &dims, DType) {
    return Tensor<DType, R>(dims, Scope::Fragment);
}

template<typename Src, typename Dst>
inline void copy(const Src &src, const Dst &dst) {
    static_assert(detail::tile_rank_v<Src> == detail::tile_rank_v<Dst>,
                  "T.copy requires source and destination tiles of equal rank");
    LUISA_INFO("[tensor-dsl] T.copy: {} -> {}", detail::describe(src), detail::describe(dst));
}

template<typename DType, size_t R>
inline void clear(const Tensor<DType, R> &t) {
    LUISA_INFO("[tensor-dsl] T.clear: {}", detail::describe(t));
}

template<typename A, typename B, typename C>
inline void gemm(const A &a, const B &b, const C &c) {
    LUISA_INFO("[tensor-dsl] T.gemm: {} x {} -> {}", detail::describe(a),
               detail::describe(b), detail::describe(c));
}

template<typename X, typename Y>
inline void reduce_sum(const X &x, const Y &y, int dim) {
    LUISA_INFO("[tensor-dsl] T.reduce_sum: {} -> {} (dim={})",
               detail::describe(x), detail::describe(y), dim);
}

template<typename T>
inline void print(const T &t, const char *msg) {
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
// Host side: luisa::compute::tile::jit(kernel).compile(...)  (mirrors @tilelang.jit)
// ---------------------------------------------------------------------------
template<typename Ret>
class CompiledKernel;

template<typename F>
class jit {
    F _fn;

public:
    explicit jit(F f) : _fn(std::move(f)) {}

    template<typename... ConfigArgs>
    auto compile(ConfigArgs &&...config_args) const {
        using traits = detail::fn_traits<F>;
        std::string cfg;
        ((cfg += std::to_string(static_cast<long long>(config_args)), cfg += " "), ...);
        LUISA_INFO("[tensor-dsl] kernel.compile: {} ({} compile-time args: {})",
                   "prim_function", sizeof...(ConfigArgs), cfg);
        // Trace the kernel body exactly like a real DSL would at compile time:
        // invoke the prim function with default-constructed input tensors.
        auto inputs = typename traits::arg_tuple{};
        [[maybe_unused]] auto result = std::apply(_fn, inputs);
        return CompiledKernel<typename traits::return_type>{};
    }
};

// A compiled kernel: callable (`matmul_kernel(A, B)`) and introspectable
// (`get_kernel_source()`).  The stub logs and returns a default tensor.
template<typename Ret>
class CompiledKernel {
public:
    std::string name = "compiled_kernel";

    CompiledKernel() = default;
    explicit CompiledKernel(std::string n) : name(std::move(n)) {}

    template<typename... Args>
    Ret operator()(Args &&.../*args*/) const {
        LUISA_INFO("[tensor-dsl] kernel.run: {} (stub: no device execution)", name);
        return Ret{};// stub: no real computation
    }

    [[nodiscard]] std::string get_kernel_source() const {
        LUISA_INFO("[tensor-dsl] kernel.get_kernel_source: stub (no kernel source generated)");
        return "// tensor-dsl stub: no kernel source generated\n";
    }
};

namespace testing {

template<typename A, typename B>
inline void assert_close(const A &a, const B &b, float rtol, float atol) {
    LUISA_INFO("[tensor-dsl] testing::assert_close: {} vs {} (rtol={}, atol={})",
               detail::describe(a), detail::describe(b), rtol, atol);
}

}// namespace testing

inline void print(const std::string &s) {
    LUISA_INFO("[tensor-dsl] luisa::compute::tile::print: {}", s);
}

}// namespace luisa::compute::tile
