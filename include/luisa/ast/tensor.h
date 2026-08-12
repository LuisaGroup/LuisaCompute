#pragma once
/*
Storage rules:
  R1  C++ host / compile-time value    -> store the C++ variable itself,
                                          e.g. `int32_t rank = 2;`
                                          (a non-type template parameter /
                                          host compile-time constant; no AST
                                          node is allocated for it).

  R2  Host-side constant, fixed at the -> store `const LiteralExpr *`
      compiling stage                  (include/luisa/ast/expression.h):
                                          LiteralExpr(const Type *type,
                                                      detail::LiteralValue v)
                                          - Value = detail::LiteralValue, a
                                            variant over basic_types (bool,
                                            float, int, uint, short, ushort,
                                            slong, ulong, half, double and
                                            their vector/matrix forms)
                                          - Tag::LITERAL, accessor value()
                                          - string literals are NOT in the
                                            variant: use `const StringIDExpr *`
                                            (Tag::STRING_ID, accessor data()).

  R3  Runtime variable truly in kernel  -> store `const RefExpr *`
                                          (include/luisa/ast/expression.h):
                                          RefExpr(Variable v)
                                          - holds a Variable { type, uid,
                                            tag } (include/luisa/ast/
                                            variable.h); Variable::Tag:
                                            LOCAL, SHARED, REFERENCE, BUFFER,
                                            TEXTURE, BINDLESS_ARRAY, ACCEL,
                                            THREAD_ID, BLOCK_ID, DISPATCH_ID,
                                            DISPATCH_SIZE, KERNEL_ID,
                                            WARP_LANE_COUNT, WARP_LANE_ID, ...
                                          - Tag::REF, accessor variable().

Shorthand used below:
  i32     = int32_t                       (R1: host-side value)
  ten     = TensorExpr *                  (tensor operand, section 4)
  lit     = const LiteralExpr *           (R2: host-side constant fixed at
                                            the compiling stage)
  ref     = const RefExpr *               (R3: runtime kernel variable)
  sid     = const StringIDExpr *          (R2 string: static const string)

Reference implementation: include/luisa/dsl/tensor.h (tracing stub).
Namespace: luisa::compute::tile (language = luisa::compute::tile::language).
Dot-syntax handle: constexpr auto T = luisa::compute::tile::language::dsl;

=============================================================================
0. Shared storage legend
=============================================================================
kind  stored as                  used when
----  -------------------------  -------------------------------------------
i32   int32_t                    member is a host-side value fixed on the
                                 host during the compiling stage (rank,
                                 extents, scope, grid, stages...)
ten   TensorExpr *               member is a tensor operand (section 4)
lit   const LiteralExpr *        member is a constant value embedded in the
                                 kernel IR (0.0f, 1e-12f, dim=1, block 0)
sid   const StringIDExpr *       member is a constant string in the kernel
                                 (print message / format tag)
ref   const RefExpr *            member is a real kernel variable (tensor,
                                 block id, loop index, tile base)

=============================================================================
1. Scalar / dtype nodes (host-side type tags, R1)
=============================================================================
Node    Member  Kind  C++ member                        Note
------  ------  ----  --------------------------------  ----------------------
half    -       -     (type tag only)                   f16; scalar value
float32 -       -     (type tag only)                   f32; scalar value
int32   -       -     (type tag only)                   i32; scalar value
  -> in the DSL a dtype is used as a template argument (`Tensor<f16, 2>`);
     in the AST the dtype is stored as an ordinary R1 member of TensorExpr
     (e.g. TensorElementType), NOT as a template parameter (F1); its
     *values* (`f32(0.0f)`, `f32(N)`) are R2 `lit`.

=============================================================================
2. Memory scope (R1)
=============================================================================
Node         Member  Kind  C++ member      Note
-----------  ------  ----  --------------  ------------------------------
Scope        -       -     (enum)          Global=0, Shared=1, Fragment=2
scope_name   s       i32   0|1|2           kept as host-side value

=============================================================================
3. Shape / slice nodes
=============================================================================
Node                  Member     Kind  C++ member                Note
--------------------  ---------  ----  -------------------------  ---------------
Shape<R>              dims       i32   std::array<int32_t, R>     R1: extents are
                                                                    host constexpr
shape(Ints...)        (builder)  -     -> Shape<sizeof...(Ints)>  R1
Slice                 begin      i32   int32_t                    R1
                      end        i32   int32_t                    exclusive; -1
                                                                  = to end
                      is_all     i32   int32_t                    R1 (0|1)
all()                 (builder)  -     Slice{0, -1, 1}            R1
range(b,e)            (builder)  -     Slice{b, e, 0}             R1
*/
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/type.h>
#include <luisa/ast/variable.h>

#include <cstdint>
#include <utility>

namespace luisa::compute {

// =============================================================================
// Tensor AST nodes — concrete implementation of the design above.
//
// Every tensor operation (Gemm, Clear, Copy, ...) is a Statement-like node
// (`TensorStmt`), never an Expression (F2).  `TensorExpr` is the independent,
// NON-TEMPLATE tensor operand that carries the layout (F1).
//
// Ownership:
//   - a TensorStmt OWNS its `output` and `inputs` TensorExpr operands;
//   - a statement also OWNS any LiteralExpr / StringIDExpr members it was
//     constructed with;
//   - RefExpr members (`handle`, runtime scalars, kernel binders) are borrowed
//     (created by a FunctionBuilder) and are never deleted by the statement.
//
// serialization contract:
//   `size_t serialize(luisa::vector<char>&)` appends a compact binary encoding
//   of the STATICALLY MEANINGFUL members and returns the number of bytes
//   appended.  `bool deserialize(char const*&, char const*)` reads the same
//   encoding back (advancing the cursor) and returns false on any failure.
//   Pointers are non-serializable (e.g. `RefExpr*`), so they are skipped;
//   the *values* carried by LiteralExpr / StringIDExpr (host-side constants
//   fixed at the compiling stage) ARE serialized because they do not change
//   between runs.
// =============================================================================

/// Memory scope of a tensor (R1, stored as a host-side value).
enum struct TensorScope : uint32_t {
    Global = 0,  // kernel argument / result buffer
    Shared = 1,  // per-block on-chip memory
    Fragment = 2 // per-thread registers
};

/// Element dtype tag of a tensor (host-side, R1).  TensorExpr stores the
/// element dtype directly as this tag; only scalar element types are
/// supported (F16 / F32 / I32).
enum struct TensorElementType : uint32_t {
    F16 = 0, // half
    F32 = 1, // float
    I32 = 2  // int32_t
};

/// Discriminator of every tensor statement (the `op` metadata member).
/// Maps 1:1 to the TensorStmt sub-classes implemented in this header.
enum struct TileOpKind : uint32_t {
    ALLOC,       // T.empty / T.alloc_shared / T.alloc_fragment
    CLEAR,       // T.clear
    COPY,        // T.copy
    GEMM,        // T.gemm
    REDUCE_SUM,  // T.reduce_sum
    PRINT,       // T.print
    STORE,       // tile-store: lhs = rhs / lhs *= rhs
    BINARY,      // whole-tile elementwise binary op (T.Parallel lowering)
    MAX,         // T.max(a, b)
    RSQRT,       // T.rsqrt(a)
    CEILDIV,     // T.ceildiv(a, b)
    KERNEL_1D,   // T.Kernel(gx, threads)
    KERNEL_2D,   // T.Kernel(gx, gy, threads)
    PIPELINED    // T.Pipelined(count, stages)
};

[[nodiscard]] LUISA_AST_API const char *scope_name(TensorScope scope) noexcept;
[[nodiscard]] LUISA_AST_API const char *tensor_element_type_name(TensorElementType e) noexcept;

// ---------------------------------------------------------------------------
// 4. TensorExpr — the independent, non-template tensor node (F1)
// ---------------------------------------------------------------------------
class LUISA_AST_API TensorExpr {

private:
    int32_t _rank{}; // R1: host-side rank
    TensorElementType _dtype{TensorElementType::F32}; // R1: element type (scalar only)
    TensorScope _scope{TensorScope::Global};// R1: Global / Shared / Fragment
    luisa::vector<int64_t> _dims;          // R1: M, N, K ... (host)
    luisa::vector<int64_t> _offset;        // R1: tile anchor (host); runtime anchor is R3 `ref`
    luisa::vector<int64_t> _extent;        // R1: tile size BM, BN (host)
    const RefExpr *_handle{nullptr};       // R3: kernel-side variable (BUFFER/SHARED/LOCAL)

public:
    TensorExpr() noexcept = default;

    /// Construct a tensor with the given layout.  When `extent` is empty it
    /// defaults to the whole-tensor extent (`dims`); when `offset` is empty it
    /// defaults to zeros.  `handle` is borrowed and not owned.
    TensorExpr(int32_t rank,
               TensorElementType dtype,
               TensorScope scope,
               luisa::vector<int64_t> dims,
               luisa::vector<int64_t> offset = {},
               luisa::vector<int64_t> extent = {},
               const RefExpr *handle = nullptr) noexcept;

    [[nodiscard]] auto rank() const noexcept { return _rank; }
    [[nodiscard]] auto dtype() const noexcept { return _dtype; }
    [[nodiscard]] auto scope() const noexcept { return _scope; }
    [[nodiscard]] auto dims() const noexcept { return luisa::span<const int64_t>{_dims.data(), _dims.size()}; }
    [[nodiscard]] auto offset() const noexcept { return luisa::span<const int64_t>{_offset.data(), _offset.size()}; }
    [[nodiscard]] auto extent() const noexcept { return luisa::span<const int64_t>{_extent.data(), _extent.size()}; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }

    /// Human readable description, e.g. "A(16,16)@(0,0)".
    [[nodiscard]] luisa::string describe() const;

    /// Append the statically meaningful layout (rank, dtype, scope, dims,
    /// offset, extent) to `output_buffer`; `handle` is a pointer and is not
    /// serialized.  Returns the number of bytes appended.
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer);

    /// Read the layout back, advancing `input_ptr`.  The handle is left null
    /// (pointers are non-serializable).  Returns false on malformed input.
    bool deserialize(char const *&input_ptr, char const *end_ptr);
};

// ---------------------------------------------------------------------------
// 5. Tensor op nodes — STATEMENTS only, no Expression (F2)
// ---------------------------------------------------------------------------
class LUISA_AST_API TensorStmt {

private:
    TileOpKind _op{TileOpKind::ALLOC};               // meta: op discriminator
    TensorExpr *_output{nullptr};                    // result tensor (owned; may be null)
    luisa::vector<TensorExpr *> _inputs;             // input argument tensors (owned)
    luisa::vector<std::pair<luisa::string, int64_t>> _annotations; // host-side meta

protected:
    void _clear_owned() noexcept;

public:
    explicit TensorStmt(TileOpKind op) noexcept;
    TensorStmt(TileOpKind op,
               TensorExpr *output,
               luisa::vector<TensorExpr *> inputs) noexcept;
    virtual ~TensorStmt();

    [[nodiscard]] auto op() const noexcept { return _op; }
    /// Result tensor; null when the operation has no return value.
    [[nodiscard]] auto output() const noexcept { return _output; }
    /// Input argument tensors (READ operands; the output may also be read+write).
    [[nodiscard]] auto inputs() const noexcept {
        return luisa::span<const TensorExpr *const>{_inputs.data(), _inputs.size()};
    }
    /// Host-side annotations (coalesced_width, stages, ...).
    [[nodiscard]] auto annotations() const noexcept {
        return luisa::span<const std::pair<luisa::string, int64_t>>{_annotations.data(), _annotations.size()};
    }
    void set_annotation(luisa::string key, int64_t value) noexcept;
    [[nodiscard]] const int64_t *annotation(luisa::string_view key) const noexcept;

    /// Append op + output + inputs + annotations.  Returns bytes appended.
    [[nodiscard]] virtual size_t serialize(luisa::vector<char> &output_buffer);
    /// Read the common members back.  Returns false on malformed input or a
    /// TileOpKind mismatch.
    virtual bool deserialize(char const *&input_ptr, char const *end_ptr);
};

// --- Gemm: T.gemm(a, b, c) --------------------------------------------------
// c (accum, READ+WRITE) is the result tensor; a, b are the READ operands.
class LUISA_AST_API GemmStmt final : public TensorStmt {
    int32_t _trans_a{}; // R1: 0|1
    int32_t _trans_b{}; // R1: 0|1

public:
    GemmStmt() noexcept : TensorStmt{TileOpKind::GEMM} {}
    GemmStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c,
             int32_t trans_a = 0, int32_t trans_b = 0) noexcept
        : TensorStmt{TileOpKind::GEMM, c, {a, b}}, _trans_a{trans_a}, _trans_b{trans_b} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Clear: T.clear(t) ------------------------------------------------------
class LUISA_AST_API ClearStmt final : public TensorStmt {
public:
    ClearStmt() noexcept : TensorStmt{TileOpKind::CLEAR} {}
    explicit ClearStmt(TensorExpr *t) noexcept
        : TensorStmt{TileOpKind::CLEAR, t, {}} {}
    [[nodiscard]] auto t() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Copy: T.copy(src, dst) -------------------------------------------------
class LUISA_AST_API CopyStmt final : public TensorStmt {
public:
    CopyStmt() noexcept : TensorStmt{TileOpKind::COPY} {}
    CopyStmt(TensorExpr *src, TensorExpr *dst) noexcept
        : TensorStmt{TileOpKind::COPY, dst, {src}} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- ReduceSum: T.reduce_sum(x, y, dim) -------------------------------------
class LUISA_AST_API ReduceSumStmt final : public TensorStmt {
    const LiteralExpr *_dim{nullptr}; // R2: e.g. LiteralExpr(Type::of<int>(), 1)

public:
    ReduceSumStmt() noexcept : TensorStmt{TileOpKind::REDUCE_SUM} {}
    ReduceSumStmt(TensorExpr *x, TensorExpr *y, const LiteralExpr *dim) noexcept
        : TensorStmt{TileOpKind::REDUCE_SUM, y, {x}}, _dim{dim} {}
    ~ReduceSumStmt() override { delete _dim; }
    [[nodiscard]] auto x() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto y() const noexcept { return output(); }
    [[nodiscard]] auto dim() const noexcept { return _dim; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TilePrint: T.print(t, "msg") --------------------------------------------
class LUISA_AST_API TilePrintStmt final : public TensorStmt {
    const StringIDExpr *_msg{nullptr}; // R2 string: StringIDExpr("msg")

public:
    TilePrintStmt() noexcept : TensorStmt{TileOpKind::PRINT} {}
    TilePrintStmt(TensorExpr *t, const StringIDExpr *msg) noexcept
        : TensorStmt{TileOpKind::PRINT, nullptr, {t}}, _msg{msg} {}
    ~TilePrintStmt() override { delete _msg; }
    [[nodiscard]] auto t() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto msg() const noexcept { return _msg; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Alloc: T.empty / T.alloc_shared / T.alloc_fragment ---------------------
// The statement owns the created TensorExpr (the output); its layout
// (dims / dtype / scope) is the alloc information.  The R3 kernel-side
// variable is kept in the TensorExpr::handle.
class LUISA_AST_API AllocStmt final : public TensorStmt {
public:
    AllocStmt() noexcept : TensorStmt{TileOpKind::ALLOC} {}
    AllocStmt(luisa::vector<int64_t> dims, TensorElementType dtype, TensorScope scope,
              const RefExpr *handle = nullptr) noexcept;
    [[nodiscard]] auto tensor() const noexcept { return output(); }
    [[nodiscard]] auto rank() const noexcept { return output() == nullptr ? 0 : output()->rank(); }
    [[nodiscard]] auto dims() const noexcept {
        return output() == nullptr ? luisa::span<const int64_t>{} : output()->dims();
    }
    [[nodiscard]] auto dtype() const noexcept {
        return output() == nullptr ? TensorElementType::F32 : output()->dtype();
    }
    [[nodiscard]] auto scope() const noexcept {
        return output() == nullptr ? TensorScope::Global : output()->scope();
    }
    [[nodiscard]] auto handle() const noexcept { return output() == nullptr ? nullptr : output()->handle(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TileStore: C_local[BM, BN] = expr ; A_local[...] *= ... ----------------
// rhs is a tensor (stored in base inputs), or a scalar literal (R2), or a
// runtime scalar RefExpr (R3, not serializable).
class LUISA_AST_API TileStoreStmt final : public TensorStmt {
    int32_t _op{};                        // R1: 0 = `=`, 1 = `*=` row-broadcast
    const LiteralExpr *_rhs_literal{nullptr}; // R2 (owned)
    const RefExpr *_rhs_ref{nullptr};     // R3 (borrowed, non-serializable)

public:
    TileStoreStmt() noexcept : TensorStmt{TileOpKind::STORE} {}
    TileStoreStmt(int32_t op, TensorExpr *lhs, TensorExpr *rhs_tensor = nullptr,
                  const LiteralExpr *rhs_literal = nullptr,
                  const RefExpr *rhs_ref = nullptr) noexcept;
    ~TileStoreStmt() override { delete _rhs_literal; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto lhs() const noexcept { return output(); }
    [[nodiscard]] auto rhs_tensor() const noexcept {
        return inputs().size() > 0 ? inputs()[0] : nullptr;
    }
    [[nodiscard]] auto rhs_literal() const noexcept { return _rhs_literal; }
    [[nodiscard]] auto rhs_ref() const noexcept { return _rhs_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TileBinary: whole-tile elementwise A+B, A*B, A/2.0f ---------------------
// lhs/rhs are READ operands; the result is a temporary tile (no output).
class LUISA_AST_API TileBinaryStmt final : public TensorStmt {
    BinaryOp _op{BinaryOp::ADD};              // R1: BinaryOp as int32_t
    const LiteralExpr *_rhs_literal{nullptr}; // R2 (owned)
    const RefExpr *_rhs_ref{nullptr};         // R3 (borrowed, non-serializable)

public:
    TileBinaryStmt() noexcept : TensorStmt{TileOpKind::BINARY} {}
    TileBinaryStmt(BinaryOp op, TensorExpr *lhs, TensorExpr *rhs_tensor = nullptr,
                   const LiteralExpr *rhs_literal = nullptr,
                   const RefExpr *rhs_ref = nullptr) noexcept;
    ~TileBinaryStmt() override { delete _rhs_literal; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto lhs() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto rhs_tensor() const noexcept {
        return inputs().size() > 1 ? inputs()[1] : nullptr;
    }
    [[nodiscard]] auto rhs_literal() const noexcept { return _rhs_literal; }
    [[nodiscard]] auto rhs_ref() const noexcept { return _rhs_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Max: T.max(a, b) -------------------------------------------------------
class LUISA_AST_API MaxStmt final : public TensorStmt {
    const LiteralExpr *_b{nullptr}; // R2: e.g. LiteralExpr(Type::of<float>(), 1e-12f)

public:
    MaxStmt() noexcept : TensorStmt{TileOpKind::MAX} {}
    MaxStmt(TensorExpr *a, const LiteralExpr *b) noexcept
        : TensorStmt{TileOpKind::MAX, nullptr, {a}}, _b{b} {}
    ~MaxStmt() override { delete _b; }
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return _b; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Rsqrt: T.rsqrt(a) ------------------------------------------------------
class LUISA_AST_API RsqrtStmt final : public TensorStmt {
public:
    RsqrtStmt() noexcept : TensorStmt{TileOpKind::RSQRT} {}
    explicit RsqrtStmt(TensorExpr *a) noexcept
        : TensorStmt{TileOpKind::RSQRT, nullptr, {a}} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- CeilDiv: T.ceildiv(a, b) (host-side helper, no tensors) ----------------
class LUISA_AST_API CeilDivStmt final : public TensorStmt {
    int32_t _a{}; // R1
    int32_t _b{}; // R1; result (a + b - 1) / b

public:
    CeilDivStmt() noexcept : TensorStmt{TileOpKind::CEILDIV} {}
    CeilDivStmt(int32_t a, int32_t b) noexcept
        : TensorStmt{TileOpKind::CEILDIV}, _a{a}, _b{b} {}
    [[nodiscard]] auto a() const noexcept { return _a; }
    [[nodiscard]] auto b() const noexcept { return _b; }
    [[nodiscard]] auto result() const noexcept { return (_a + _b - 1) / _b; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// ---------------------------------------------------------------------------
// 6. Control-flow / index binder statements
// ---------------------------------------------------------------------------

/// T.Kernel(gx, threads) — grid loop is host-side; bx is the yielded builtin
/// Variable{Tag::BLOCK_ID, uid=0} (R3, not serializable).
class LUISA_AST_API Kernel1DStmt final : public TensorStmt {
    int32_t _gx{};         // R1
    int32_t _threads{};    // R1
    const RefExpr *_bx{nullptr}; // R3 (borrowed)

public:
    Kernel1DStmt() noexcept : TensorStmt{TileOpKind::KERNEL_1D} {}
    Kernel1DStmt(int32_t gx, int32_t threads, const RefExpr *bx = nullptr) noexcept
        : TensorStmt{TileOpKind::KERNEL_1D}, _gx{gx}, _threads{threads}, _bx{bx} {}
    [[nodiscard]] auto gx() const noexcept { return _gx; }
    [[nodiscard]] auto threads() const noexcept { return _threads; }
    [[nodiscard]] auto bx() const noexcept { return _bx; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

/// T.Kernel(gx, gy, threads).
class LUISA_AST_API Kernel2DStmt final : public TensorStmt {
    int32_t _gx{};         // R1
    int32_t _gy{};         // R1
    int32_t _threads{};    // R1
    const RefExpr *_bx{nullptr}; // R3 (borrowed)
    const RefExpr *_by{nullptr}; // R3 (borrowed)

public:
    Kernel2DStmt() noexcept : TensorStmt{TileOpKind::KERNEL_2D} {}
    Kernel2DStmt(int32_t gx, int32_t gy, int32_t threads,
                 const RefExpr *bx = nullptr, const RefExpr *by = nullptr) noexcept
        : TensorStmt{TileOpKind::KERNEL_2D}, _gx{gx}, _gy{gy}, _threads{threads},
          _bx{bx}, _by{by} {}
    [[nodiscard]] auto gx() const noexcept { return _gx; }
    [[nodiscard]] auto gy() const noexcept { return _gy; }
    [[nodiscard]] auto threads() const noexcept { return _threads; }
    [[nodiscard]] auto bx() const noexcept { return _bx; }
    [[nodiscard]] auto by() const noexcept { return _by; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

/// T.Pipelined(count, stages) — software pipeline metadata (R1); k is the
/// yielded runtime loop variable (R3, not serializable).
class LUISA_AST_API PipelinedStmt final : public TensorStmt {
    int32_t _count{};      // R1
    int32_t _stages{};     // R1
    const RefExpr *_k{nullptr}; // R3 (borrowed)

public:
    PipelinedStmt() noexcept : TensorStmt{TileOpKind::PIPELINED} {}
    PipelinedStmt(int32_t count, int32_t stages, const RefExpr *k = nullptr) noexcept
        : TensorStmt{TileOpKind::PIPELINED}, _count{count}, _stages{stages}, _k{k} {}
    [[nodiscard]] auto count() const noexcept { return _count; }
    [[nodiscard]] auto stages() const noexcept { return _stages; }
    [[nodiscard]] auto k() const noexcept { return _k; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

}// namespace luisa::compute

