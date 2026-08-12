/*
=============================================================================
tensor.h — AST-node member design for the TileLang-style tile / tensor DSL
=============================================================================
This header is a DESIGN REPORT ONLY (comments, no code).  It lists every
AST node of the tile / tensor DSL (include/luisa/dsl/tensor.h) and assigns a
concrete C++ member type to each node member, following exactly three rules:

  R1  Template constexpr value          -> store the C++ variable itself,
                                          e.g. `int32_t rank = 2;`
                                          (a non-type template parameter /
                                          host compile-time constant; no AST
                                          node is allocated for it).

  R2  Static const value at runtime     -> store `const LiteralExpr *`
                                          (include/luisa/ast/expression.h):
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
  i32     = int32_t                       (R1: template constexpr)
  lit     = const LiteralExpr *           (R2: static const at runtime)
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
i32   int32_t                    member parameterizes the node at C++ compile
                                 time (rank, extents, scope, grid, stages...)
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
  -> a dtype used as a template argument stays a C++ template TYPE parameter
     (`Tensor<f16, 2>`); its *values* (`f32(0.0f)`, `f32(N)`) are R2 `lit`.

=============================================================================
2. Memory scope (R1)
=============================================================================
Node         Member  Kind  C++ member      Note
-----------  ------  ----  --------------  ------------------------------
Scope        -       -     (enum)          Global=0, Shared=1, Fragment=2
scope_name   s       i32   0|1|2           kept as template constexpr

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
4. Tile expression / tensor nodes
=============================================================================
Tensor<DType, R>  (an allocation: T.empty / T.alloc_shared /
                   T.alloc_fragment)
  Member    Kind  C++ member                          Note
  --------  ----  ----------------------------------  ------------------------
  rank      i32   int32_t                             R1: template constexpr
  dtype     i32   (template type param DType)         R1: f16/f32/i32
  scope     i32   int32_t                             R1: Global/Shared/Fragment
  dims      i32   std::array<int32_t, R>              R1: M, N, K ...
  handle    ref   const RefExpr *                     R3: the kernel-side
                                                        variable created by
                                                        the alloc; Variable
                                                        Tag = BUFFER (Global),
                                                        SHARED (Shared),
                                                        LOCAL (Fragment)
  Accessors: dims() / scope() / name() / describe()   (stub logging only)

TileExpr<R>  (a view: A(i, j), A(range, all), or the result of A+B, max, ...)
  Member    Kind  C++ member                          Note
  --------  ----  ----------------------------------  ------------------------
  rank      i32   int32_t                             R1: template constexpr
  scope     i32   int32_t                             R1
  base      ref   const RefExpr *                     R3: base tensor variable
  offset    ref   std::array<const RefExpr *, R>      R3: tile anchor, e.g.
                                                        by*BM (runtime); a
                                                        constant anchor (0)
                                                        is instead R2 `lit`
  extent    i32   std::array<int32_t, R>              R1: tile size BM, BN
  (store ops: operator= / operator*= are TileStore, see section 5)

=============================================================================
5. Tile op nodes (the "builtins")
=============================================================================
Gemm  — T.gemm(a, b, c)
  Member   Kind  C++ member         Note
  -------  ----  -----------------  -------------------------------------
  a        ref   const RefExpr *    R3: A tile   (READ; Usage::READ)
  b        ref   const RefExpr *    R3: B tile   (READ; Usage::READ)
  c        ref   const RefExpr *    R3: C accum  (READ+WRITE; Usage::WRITE)
  trans_a  i32   int32_t            R1: template constexpr 0|1
  trans_b  i32   int32_t            R1: template constexpr 0|1
  Log: "T.gemm: A x B -> C"

Clear  — T.clear(t)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  t       ref   const RefExpr *  R3: tensor to zero (Usage::WRITE)
  Log: "T.clear: t"

Copy  — T.copy(src, dst)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  src     ref   const RefExpr *  R3: source tile (Usage::READ)
  dst     ref   const RefExpr *  R3: dest tile   (Usage::WRITE)
  Log: "T.copy: src -> dst"

ReduceSum  — T.reduce_sum(x, y, dim)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  x       ref   const RefExpr *  R3: input  (Usage::READ)
  y       ref   const RefExpr *  R3: output (Usage::WRITE)
  dim     lit   const LiteralExpr *  R2: e.g. LiteralExpr(Type::of<int>(), 1)
  Log: "T.reduce_sum: x -> y (dim=d)"

Print  — T.print(t, "msg")
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  t       ref   const RefExpr *  R3: value to print (Usage::READ)
  msg     sid   const StringIDExpr *  R2 string: StringIDExpr("msg")
  Log: "T.print: msg t"

Alloc  — T.empty / T.alloc_shared / T.alloc_fragment (see Tensor in sec. 4)
  Member  Kind  C++ member                           Note
  ------  ----  -----------------------------------  -----------------------
  dims    i32   const Shape<R> & (std::array<int32_t,R>)  R1
  dtype   i32   (template type param DType)          R1
  scope   i32   int32_t (Global|Shared|Fragment)     R1: chosen by factory
  Log: "tensor: name(dims)" / "T.alloc_shared: ..." / "T.alloc_fragment: ..."

TileStore  — C_local[BM, BN] = expr ; A_local[...] *= ...
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  op      i32   int32_t          R1: store kind (0 = `=`, 1 = `*=` row-broadcast)
  lhs     ref   const RefExpr *  R3: destination tile (Usage::WRITE)
  rhs     ref   const RefExpr *  R3: source tile (Usage::READ); a scalar
                                  constant operand is instead R2 `lit`
  Log: "tile-store: lhs = rhs" / "tile-store: lhs *= rhs"

TileBinary  — whole-tile elementwise: A+B, A*B, A/2.0f (T.Parallel lowering)
  Member  Kind  C++ member        Note
  ------  ----  ----------------  -----------------------------------------
  op      i32   int32_t           R1: BinaryOp as int32_t (+, *, /)
  lhs     ref   const RefExpr *   R3: left tile operand  (Usage::READ)
  rhs     ref   const RefExpr *   R3: right tile operand (Usage::READ);
                                   a scalar constant operand (A + 2.0f) is
                                   instead R2 `lit`
  Log: "tile-op: a op b", result named "expr(op)"

Max  — T.max(a, b)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  a       ref   const RefExpr *  R3: tile operand (Usage::READ)
  b       lit   const LiteralExpr *  R2: e.g. LiteralExpr(Type::of<float>(), 1e-12f)
  Log: "tile-op: a max b"

Rsqrt  — T.rsqrt(a)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  a       ref   const RefExpr *  R3: tile operand (Usage::READ)
  Log: "tile-op: rsqrt(a)"

CeilDiv  — T.ceildiv(a, b)
  Member  Kind  C++ member       Note
  ------  ----  ---------------  -----------------------------------------
  a       i32   int32_t          R1: template constexpr (host helper)
  b       i32   int32_t          R1: template constexpr; result (a+b-1)/b
                                 (when applied to a runtime value, both
                                 become R3 `ref`)

=============================================================================
6. Control-flow / index binder nodes
=============================================================================
Kernel2D  — for (auto [bx, by] : T.Kernel(gx, gy, threads))
  Member    Kind  C++ member       Note
  -------   ----  ---------------  ----------------------------------------
  gx        i32   int32_t          R1: template constexpr grid X
  gy        i32   int32_t          R1: template constexpr grid Y
  threads   i32   int32_t          R1: template constexpr block size
  bx        ref   const RefExpr *  R3: yielded runtime builtin
                                   Variable{Tag::BLOCK_ID, uid=0}
  by        ref   const RefExpr *  R3: Variable{Tag::BLOCK_ID, uid=1}
  Log: "T.Kernel: grid=(gx,gy), threads=t [stub: tracing one representative block]"

Kernel1D  — for (auto bx : T.Kernel(gx, threads))
  Member    Kind  C++ member       Note
  -------   ----  ---------------  ----------------------------------------
  gx        i32   int32_t          R1: template constexpr grid
  threads   i32   int32_t          R1: template constexpr block size
  bx        ref   const RefExpr *  R3: Variable{Tag::BLOCK_ID, uid=0}
  Log: "T.Kernel: grid=(gx), threads=t [stub: tracing one representative block]"

Pipelined  — for (auto k : T.Pipelined(count, stages))
  Member    Kind  C++ member       Note
  -------   ----  ---------------  ----------------------------------------
  count     i32   int32_t          R1: template constexpr trip count
                                   (e.g. ceildiv(N, BK) folded on host)
  stages    i32   int32_t          R1: template constexpr pipeline depth
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
                 a, b       ref   const RefExpr *    R3 (tiles compared)
                 rtol       lit   const LiteralExpr* R2: float tolerance
                 atol       lit   const LiteralExpr* R2: float tolerance
print(std::string) s        sid   std::string        host-side helper

=============================================================================
Summary: node -> members by storage rule
=============================================================================
Node        R1 (int32_t / template)      R2 (LiteralExpr)      R3 (RefExpr)
----------  ---------------------------  --------------------  ----------------
Gemm        trans_a, trans_b             -                     a, b, c
Clear       -                            -                     t
Copy        -                            -                     src, dst
ReduceSum   -                            dim                   x, y
Print       -                            (msg: StringIDExpr)   t
Alloc        rank, scope, dims[R]        -                     handle
TileExpr     rank, scope, extent[R]      offset[i] (const 0)   base, offset[i]
TileStore    op                          rhs (scalar const)    lhs, rhs
TileBinary   op                          rhs (scalar const)    lhs, rhs
Max          -                            b                    a
Rsqrt        -                            -                     a
CeilDiv      a, b                        -                     -
Kernel1D/2D  gx, gy, threads             -                     bx, by
Pipelined    count, stages               -                     k
assert_close -                            rtol, atol            a, b

---------------------------------------------------------------------------
NOTE: this is a design only.  The real include/luisa/dsl/tensor.h stub has
no visitor dispatch and no Usage marking; every op logs and returns.  A
concrete implementation of this design would give each node a Tag, an
accept(StmtVisitor&)/accept(ExprVisitor&), and mark R3 operands with
Usage::READ / Usage::WRITE exactly like include/luisa/ast/statement.h.
=============================================================================
*/
#pragma once
