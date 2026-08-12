/*
=============================================================================
tensor.h — AST-node member design for the TileLang-style tile / tensor DSL
=============================================================================
This header is a DESIGN REPORT ONLY (comments, no code).  It describes the
AST nodes of the tile / tensor DSL (include/luisa/dsl/tensor.h) and assigns a
concrete C++ member type to each node member.

Three key facts about the tensor AST:

  F1  `TensorExpr` is an independent, NON-TEMPLATE type.  It is not a
      template parameterized by <DType, Rank>; it has the same structure as
      the DSL handle in include/luisa/dsl/tensor.h and stores its rank /
      dtype / scope / dims as ordinary members (see section 4).

  F2  For the tensor part there are NO Expression nodes — only Statement
      nodes.  Every tensor operation (Gemm, Clear, Copy, ReduceSum, Print,
      Alloc, TileStore, ...) is a Statement, exactly like the statements in
      include/luisa/ast/statement.h; no tensor operation yields an
      Expression result.

  F3  One tensor Statement may take any mix of inputs:
        - multiple TensorExpr   (tensor operands, section 4)
        - RefExpr               (runtime kernel variable, R3)
        - C++ host variable     (host-side value, R1)
        - LiteralExpr           (host-side constant fixed at the compiling
                                 stage, R2)

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

=============================================================================
4. TensorExpr — the independent, non-template tensor node (F1)
=============================================================================
TensorExpr  (the tensor itself: T.empty / T.alloc_shared /
             T.alloc_fragment / A(i, j) / A(range, all))
  Member    Kind  C++ member                          Note
  --------  ----  ----------------------------------  ------------------------
  rank      i32   int32_t                             R1: host-side value (not
                                                        a template parameter)
  dtype     i32   TensorElementType / int32_t         R1: f16/f32/i32 (host)
  scope     i32   int32_t                             R1: Global/Shared/Fragment
  dims      i32   std::vector<int64_t> / span         R1: M, N, K ... (host)
  offset    i32   std::vector<int64_t> / span         R1: tile anchor (host);
                                                        a runtime anchor is
                                                        instead R3 `ref`
  extent    i32   std::vector<int64_t> / span         R1: tile size BM, BN
  handle    ref   const RefExpr *                     R3: the kernel-side
                                                        variable created by
                                                        the alloc; Variable
                                                        Tag = BUFFER (Global),
                                                        SHARED (Shared),
                                                        LOCAL (Fragment)
  Accessors: dims() / scope() / name() / describe()   (stub logging only)

  - `TensorExpr` is NOT a template: rank / dtype / scope / dims are ordinary
    R1 members, so ONE class represents every `Tensor<DType, R>` and
    `TileExpr<R>` of the DSL.  It has the same structure as
    include/luisa/dsl/tensor.h (`Tensor` / `TileExpr`).
  - TensorExpr is a tensor OPERAND of the tensor Statements (F3); it is not
    an Expression and never appears in an expression tree (F2).

=============================================================================
5. Tensor op nodes — STATEMENTS only, no Expression (F2)
=============================================================================
Every tensor operation is a Statement: it is stored in the kernel body like
the statements of include/luisa/ast/statement.h and accepts a StmtVisitor.
There is NO tensor Expression node — a tensor op never yields a value, it
only mutates tensors (Gemm, Clear, Copy, ...).

Each statement's inputs (F3) are drawn from:
  ten   TensorExpr *            a tensor operand (section 4)
  ref   const RefExpr *         a runtime kernel variable (R3)
  i32   int32_t / host value    a C++ host variable (R1, fixed on the host
                                side during the compiling stage)
  lit   const LiteralExpr *     a host-side constant fixed at the compiling
                                stage (R2)
  sid   const StringIDExpr *    a constant string (R2 string)

Gemm  — T.gemm(a, b, c)
  Member   Kind  C++ member         Note
  -------  ----  -----------------  -------------------------------------
  a        ten   TensorExpr *       A tile   (READ; Usage::READ)
  b        ten   TensorExpr *       B tile   (READ; Usage::READ)
  c        ten   TensorExpr *       C accum  (READ+WRITE; Usage::WRITE)
  trans_a  i32   int32_t            R1: host-side constexpr 0|1
  trans_b  i32   int32_t            R1: host-side constexpr 0|1
  Log: "T.gemm: A x B -> C"
  (a Statement — GemmStmt; no Expression result)

Clear  — T.clear(t)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  t       ten   TensorExpr *     tensor to zero (Usage::WRITE)
  Log: "T.clear: t"
  (a Statement — ClearStmt)

Copy  — T.copy(src, dst)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  src     ten   TensorExpr *     source tile (Usage::READ)
  dst     ten   TensorExpr *     dest tile   (Usage::WRITE)
  Log: "T.copy: src -> dst"
  (a Statement — CopyStmt)

ReduceSum  — T.reduce_sum(x, y, dim)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  x       ten   TensorExpr *     input  (Usage::READ)
  y       ten   TensorExpr *     output (Usage::WRITE)
  dim     lit   const LiteralExpr *  R2: e.g. LiteralExpr(Type::of<int>(), 1)
  Log: "T.reduce_sum: x -> y (dim=d)"
  (a Statement — ReduceSumStmt)

Print  — T.print(t, "msg")
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  t       ten   TensorExpr *     value to print (Usage::READ)
  msg     sid   const StringIDExpr *  R2 string: StringIDExpr("msg")
  Log: "T.print: msg t"
  (a Statement — PrintStmt)

Alloc  — T.empty / T.alloc_shared / T.alloc_fragment (see TensorExpr in sec. 4)
  Member  Kind  C++ member                           Note
  ------  ----  -----------------------------------  -----------------------
  dims    i32   const Shape<R> & (std::array<int32_t,R>)  R1
  dtype   i32   TensorElementType / int32_t          R1
  scope   i32   int32_t (Global|Shared|Fragment)     R1: chosen by factory
  Log: "tensor: name(dims)" / "T.alloc_shared: ..." / "T.alloc_fragment: ..."
  (a Statement — AllocStmt; it creates a TensorExpr handle, not an Expr)

TileStore  — C_local[BM, BN] = expr ; A_local[...] *= ...
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  op      i32   int32_t          R1: store kind (0 = `=`, 1 = `*=` row-broadcast)
  lhs     ten   TensorExpr *     destination tile (Usage::WRITE)
  rhs     ten   TensorExpr *     source tile (Usage::READ); a scalar
                                constant operand is instead R2 `lit`, and a
                                runtime scalar is R3 `ref`
  Log: "tile-store: lhs = rhs" / "tile-store: lhs *= rhs"
  (a Statement — TileStoreStmt)

TileBinary  — whole-tile elementwise: A+B, A*B, A/2.0f (T.Parallel lowering)
  Member  Kind  C++ member        Note
  ------  ----  ----------------  -----------------------------------------
  op      i32   int32_t           R1: BinaryOp as int32_t (+, *, /)
  lhs     ten   TensorExpr *      left tile operand  (Usage::READ)
  rhs     ten   TensorExpr *      right tile operand (Usage::READ);
                                  a scalar constant operand (A + 2.0f) is
                                  instead R2 `lit`, and a runtime scalar is
                                  R3 `ref`
  Log: "tile-op: a op b", result named "expr(op)"
  (a Statement — TileBinaryStmt)

Max  — T.max(a, b)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  a       ten   TensorExpr *     tile operand (Usage::READ)
  b       lit   const LiteralExpr *  R2: e.g. LiteralExpr(Type::of<float>(), 1e-12f)
  Log: "tile-op: a max b"
  (a Statement — MaxStmt)

Rsqrt  — T.rsqrt(a)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  a       ten   TensorExpr *     tile operand (Usage::READ)
  Log: "tile-op: rsqrt(a)"
  (a Statement — RsqrtStmt)

CeilDiv  — T.ceildiv(a, b)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  a       i32   int32_t          R1: host-side helper
  b       i32   int32_t          R1: host-side value; result (a+b-1)/b
                                 (when applied to a runtime value, both
                                 become R3 `ref`)
  (a Statement — CeilDivStmt)

=============================================================================
6. Control-flow / index binder statements
=============================================================================
Kernel2D  — for (auto [bx, by] : T.Kernel(gx, gy, threads))
  Member    Kind  C++ member       Note
  -------   ----  ---------------  ----------------------------------------
  gx        i32   int32_t          R1: host-side constexpr grid X
  gy        i32   int32_t          R1: host-side constexpr grid Y
  threads   i32   int32_t          R1: host-side constexpr block size
  bx        ref   const RefExpr *  R3: yielded runtime builtin
                                   Variable{Tag::BLOCK_ID, uid=0}
  by        ref   const RefExpr *  R3: Variable{Tag::BLOCK_ID, uid=1}
  Log: "T.Kernel: grid=(gx,gy), threads=t [stub: tracing one representative block]"

Kernel1D  — for (auto bx : T.Kernel(gx, threads))
  Member    Kind  C++ member       Note
  -------   ----  ---------------  ----------------------------------------
  gx        i32   int32_t          R1: host-side constexpr grid
  threads   i32   int32_t          R1: host-side constexpr block size
  bx        ref   const RefExpr *  R3: Variable{Tag::BLOCK_ID, uid=0}
  Log: "T.Kernel: grid=(gx), threads=t [stub: tracing one representative block]"

Pipelined  — for (auto k : T.Pipelined(count, stages))
  Member    Kind  C++ member       Note
  -------   ----  ---------------  --------------------------------
  count     i32   int32_t          R1: host-side constexpr trip count
                                   (e.g. ceildiv(N, BK) folded on host)
  stages    i32   int32_t          R1: host-side constexpr pipeline depth
                                   (e.g. 3); software-pipeline must know it
                                   at compile time
  k         ref   const RefExpr *  R3: yielded runtime loop variable
                                   Variable{Tag::LOCAL, uid=...}
  Log: "T.Pipelined: count iterations x stages [stub: tracing iteration 0]"

=============================================================================
7. Compilation / host nodes (no kernel AST members; host-side only)
=============================================================================
Node             Member     Kind  C++ member         Note
---------------  ---------  ----  -----------------  -----------------------
dsl_t (T handle) (forward)  -     (methods only)     mirrors T.empty / T.copy
                                                     / T.gemm / T.Kernel ...
jit              fn         -     (host callable)    wraps the prim function;
                config     i32   int32_t...          R1: compile(M, N, ...)
                                                     fold into a template
                                                     config; then TRACES the
                                                     kernel body via
                                                     std::apply(_fn, inputs)
CompiledKernel   name       sid   std::string        host-side kernel name
                operator()  -     (host dispatch)    logs "kernel.run"
                get_kernel_source()  -               stub source string
testing::assert_close
                 a, b       ten   TensorExpr *       R1/R3: tiles compared
                 rtol       lit   const LiteralExpr* R2: float tolerance
                 atol       lit   const LiteralExpr* R2: float tolerance
print(std::string) s        sid   std::string        host-side helper

=============================================================================
Summary: node -> members by storage rule
=============================================================================
Node        R1 (int32_t / host)            R2 (LiteralExpr)      R3 (RefExpr)
----------  -----------------------------  --------------------  ----------------
TensorExpr  rank, dtype, scope, dims,      -                     handle
            offset, extent
Gemm        trans_a, trans_b               -                     a, b, c (ten)
Clear       -                              -                     t (ten)
Copy        -                              -                     src, dst (ten)
ReduceSum   -                              dim                   x, y (ten)
Print       -                              (msg: StringIDExpr)   t (ten)
Alloc       rank, scope, dims[R]           -                     handle
TileStore   op                             rhs (scalar const)    lhs, rhs (ten)
TileBinary  op                             rhs (scalar const)    lhs, rhs (ten)
Max         -                              b                     a (ten)
Rsqrt       -                              -                     a (ten)
CeilDiv     a, b                           -                     -
Kernel1D/2D gx, gy, threads                -                     bx, by
Pipelined   count, stages                  -                     k
assert_close -                             rtol, atol            a, b (ten)

---------------------------------------------------------------------------
NOTE: this is a design only.  The real include/luisa/dsl/tensor.h stub has
no visitor dispatch and no Usage marking; every op logs and returns.  A
concrete implementation of this design would give each tensor op a Tag and
an accept(StmtVisitor&) — tensor ops are Statements only, there is NO tensor
Expression (F2) — and would mark every TensorExpr / RefExpr operand with
Usage::READ / Usage::WRITE exactly like include/luisa/ast/statement.h.
Each Statement may combine multiple TensorExpr, RefExpr (runtime variable),
C++ host variable, and LiteralExpr inputs (F3), with host-side values and
literals fixed at the compiling stage.
=============================================================================
*/
#pragma once
