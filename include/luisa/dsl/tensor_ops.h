//
// DSL surface for the runtime tensor operators (plan.md §2).
//
// One helper per tensor `CallOp`. Each helper emits a side-effecting
// `CallExpr` (result type void) into the current kernel trace, following the
// `device_clock()` precedent in `builtin.h`. The ops are *cooperative*:
// every thread of the enclosing dispatch executes the same op, and the
// backend maps the op's iteration domain onto the dispatch (grid-stride for
// element-wise/copy, one thread per output element for F32 GEMM, one warp per
// 16x16 tile for the F16 tensor-core GEMM). The caller must dispatch the
// kernel with a size covering the op's iteration domain (see each helper's
// doc comment); extra threads are no-ops.
//
// AST argument layouts (each descriptor expands to five arguments:
// dtype:uint, rank:uint, extents:uint4, strides:uint4, offset:uint;
// each address is a uint64 device address expression):
//
// TENSOR_COPY / TENSOR_CAST : [dst_desc, dst_addr, src_desc, src_addr, count]
// TENSOR_FILL : [dst_desc, dst_addr, value_bits:uint, count]
// TENSOR_PERMUTE : [dst_desc, dst_addr, src_desc, src_addr, perm:uint4]
// TENSOR_CONCAT : [dst_desc, dst_addr, dim:uint, num_src:uint,
// src0_desc, src0_addr, ..., src7_desc, src7_addr]
// TENSOR_PAD : [dst_desc, dst_addr, src_desc, src_addr, pad:uint4]
// unary (TENSOR_NEG ...) : [out_desc, out_addr, in_desc, in_addr, count]
// binary (TENSOR_ADD ...) : [out_desc, out_addr, a_desc, a_addr, b_desc, b_addr, count]
// TENSOR_CLAMP : [out_desc, out_addr, in_desc, in_addr, lo_bits:uint, hi_bits:uint, count]
// TENSOR_FMA : [out_desc, out_addr, a_desc, a_addr, b_desc, b_addr, c_desc, c_addr, count]
// TENSOR_MATMUL : [c_desc, c_addr, a_desc, a_addr, b_desc, b_addr,
// compute_dtype:uint, trans_a:uint, trans_b:uint,
// alpha:float, beta:float, epilogue:uint]
// TENSOR_BATCH_MATMUL : [c_desc, c_addr, a_desc, a_addr, b_desc, b_addr,
// compute_dtype:uint, trans_a:uint, trans_b:uint,
// alpha:float, beta:float, epilogue:uint, batch:uint]
// TENSOR_CONTRACT : [c_desc, c_addr, a_desc, a_addr, b_desc, b_addr,
// mode_a:uint4, mode_b:uint4, mode_c:uint4, compute_dtype:uint]
// TENSOR_REDUCE_* : [out_desc, out_addr, in_desc, in_addr, num_dims:uint, dims:uint4]
// TENSOR_CUMSUM : [out_desc, out_addr, in_desc, in_addr, dim:uint]
//

#pragma once

#include <luisa/core/stl/memory.h>// luisa::bit_cast
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/dsl/expr.h>
#include <luisa/dsl/var.h>
#include <luisa/tensor/tensor_descriptor.h>

namespace luisa::compute {

/**
 * @brief A tensor operand inside the DSL trace: the host-side layout
 * descriptor plus the expression yielding the 64-bit device address of the
 * backing storage (from `BUFFER_ADDRESS`-style resource access).
 */
struct TensorOperand {
    TensorDescriptor desc;
    const Expression *address{nullptr};// uint64 device address expression
};

namespace detail {

[[nodiscard]] inline const Expression *tensor_literal(uint32_t v) noexcept {
    return FunctionBuilder::current()->literal(Type::of<uint32_t>(), v);
}

[[nodiscard]] inline const Expression *tensor_literal(float v) noexcept {
    return FunctionBuilder::current()->literal(Type::of<float>(), v);
}

[[nodiscard]] inline const Expression *tensor_literal(uint4 v) noexcept {
    return FunctionBuilder::current()->literal(Type::of<uint4>(), v);
}

[[nodiscard]] inline const Expression *tensor_literal(uint64_t v) noexcept {
    return FunctionBuilder::current()->literal(Type::of<uint64_t>(), v);
}

/// Push the five descriptor arguments followed by the address argument.
inline void tensor_push_operand(luisa::vector<const Expression *> &args,
                                const TensorOperand &operand) noexcept {
    auto &d = operand.desc;
    LUISA_ASSERT(operand.address != nullptr,
                 "Tensor operand has no device address expression.");
    args.emplace_back(tensor_literal(luisa::to_underlying(d.dtype)));
    args.emplace_back(tensor_literal(d.rank));
    args.emplace_back(tensor_literal(uint4{d.extents[0], d.extents[1], d.extents[2], d.extents[3]}));
    args.emplace_back(tensor_literal(uint4{d.strides[0], d.strides[1], d.strides[2], d.strides[3]}));
    args.emplace_back(tensor_literal(d.storage_offset));
    args.emplace_back(operand.address);
}

inline void tensor_emit(CallOp op, luisa::span<const Expression *const> args) noexcept {
    FunctionBuilder::current()->call(Type::of<void>(), op, args);
}

/// Validate the element-wise shape/dtype rules of plan.md §A.3.
inline void tensor_check_elementwise(const TensorDescriptor &out,
                                     const TensorDescriptor &a,
                                     const char *op_name) noexcept {
    LUISA_ASSERT(out.dtype == a.dtype && out.numel() == a.numel(),
                 "{}: tensor shape/dtype mismatch.", op_name);
}

inline void tensor_check_elementwise(const TensorDescriptor &out,
                                     const TensorDescriptor &a,
                                     const TensorDescriptor &b,
                                     const char *op_name) noexcept {
    LUISA_ASSERT(a.dtype == b.dtype && a.dtype == out.dtype &&
                     a.numel() == b.numel() && a.numel() == out.numel(),
                 "{}: tensor shape/dtype mismatch.", op_name);
}

}// namespace detail

/// Create a tensor operand from a byte-buffer view: binds the buffer into the
/// current function and takes its device address.
[[nodiscard]] inline TensorOperand tensor_operand(const TensorDescriptor &desc,
                                                  const ByteBufferView &storage) noexcept {
    auto fb = detail::FunctionBuilder::current();
    auto binding = fb->buffer_binding(Type::of<ByteBuffer>(), storage.handle(),
                                      storage.offset_bytes(), storage.size_bytes());
    auto addr = fb->call(Type::of<uint64_t>(), CallOp::BUFFER_ADDRESS, {binding});
    return TensorOperand{desc, addr};
}

/// Convenience overload taking a typed buffer view.
template<typename T>
[[nodiscard]] inline TensorOperand tensor_operand(const TensorDescriptor &desc,
                                                  const BufferView<T> &storage) noexcept {
    return tensor_operand(desc, ByteBufferView{storage});
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

/// dst = src (element-wise copy honoring strides; dtypes must match).
/// Iteration domain: dst.numel() threads.
inline void tensor_copy(const TensorOperand &dst, const TensorOperand &src) noexcept {
    detail::tensor_check_elementwise(dst.desc, src.desc, "tensor_copy");
    luisa::vector<const Expression *> args;
    args.reserve(13u);
    detail::tensor_push_operand(args, dst);
    detail::tensor_push_operand(args, src);
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(dst.desc.numel())));
    detail::tensor_emit(CallOp::TENSOR_COPY, args);
}

/// dst = cast<dst.dtype>(src) (element-wise copy with dtype conversion).
/// Iteration domain: dst.numel() threads.
inline void tensor_cast(const TensorOperand &dst, const TensorOperand &src) noexcept {
    LUISA_ASSERT(dst.desc.numel() == src.desc.numel(),
                 "tensor_cast: tensor shape mismatch.");
    luisa::vector<const Expression *> args;
    args.reserve(13u);
    detail::tensor_push_operand(args, dst);
    detail::tensor_push_operand(args, src);
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(dst.desc.numel())));
    detail::tensor_emit(CallOp::TENSOR_CAST, args);
}

namespace detail {
[[nodiscard]] inline uint32_t tensor_fill_bits(TensorElementType dtype, double value) noexcept {
    switch (dtype) {
        case TensorElementType::F16: {
            auto h = half{static_cast<float>(value)};
            return static_cast<uint32_t>(luisa::bit_cast<uint16_t>(h));
        }
        case TensorElementType::F32:
            return luisa::bit_cast<uint32_t>(static_cast<float>(value));
        case TensorElementType::I32:
            return luisa::bit_cast<uint32_t>(static_cast<int32_t>(value));
        default:
            LUISA_ERROR_WITH_LOCATION("tensor_fill: unsupported dtype '{}'.",
                                      tensor_element_type_name(dtype));
    }
}
}// namespace detail

/// dst = value (fill). Iteration domain: dst.numel() threads.
inline void tensor_fill(const TensorOperand &dst, double value) noexcept {
    luisa::vector<const Expression *> args;
    args.reserve(8u);
    detail::tensor_push_operand(args, dst);
    args.emplace_back(detail::tensor_literal(detail::tensor_fill_bits(dst.desc.dtype, value)));
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(dst.desc.numel())));
    detail::tensor_emit(CallOp::TENSOR_FILL, args);
}

/// dst[dst_coords] = src[src_coords] where dst_coords[d] = src_coords[perm[d]]
/// (metadata-only view semantics; perm[i] is the source dim that becomes
/// destination dim i). Iteration domain: dst.numel() threads.
inline void tensor_permute(const TensorOperand &dst, const TensorOperand &src,
                           luisa::span<const uint32_t> perm) noexcept {
    LUISA_ASSERT(dst.desc.dtype == src.desc.dtype,
                 "tensor_permute: dtypes must match.");
    LUISA_ASSERT(perm.size() == dst.desc.rank && perm.size() == src.desc.rank,
                 "tensor_permute: permutation rank mismatch.");
    LUISA_ASSERT(dst.desc.numel() == src.desc.numel(),
                 "tensor_permute: element count mismatch.");
    uint4 p{0u, 0u, 0u, 0u};
    std::array<bool, tensor_max_rank> seen{};
    for (auto i = 0u; i < perm.size(); i++) {
        LUISA_ASSERT(perm[i] < src.desc.rank && !seen[perm[i]],
                     "tensor_permute: invalid permutation.");
        seen[perm[i]] = true;
        p[i] = perm[i];
    }
    luisa::vector<const Expression *> args;
    args.reserve(13u);
    detail::tensor_push_operand(args, dst);
    detail::tensor_push_operand(args, src);
    args.emplace_back(detail::tensor_literal(p));
    detail::tensor_emit(CallOp::TENSOR_PERMUTE, args);
}

/// dst = concat(src0, ..., srcN-1) along `dim`. The source dtypes must match
/// the destination dtype, and every non-`dim` extent must match the
/// destination. Iteration domain: dst.numel() threads.
inline void tensor_concat(const TensorOperand &dst, int dim,
                          luisa::span<const TensorOperand *const> srcs) noexcept {
    LUISA_ASSERT(dst.desc.rank >= 1u, "tensor_concat: invalid rank.");
    LUISA_ASSERT(dim >= 0 && static_cast<uint32_t>(dim) < dst.desc.rank,
                 "tensor_concat: invalid dim {}.", dim);
    LUISA_ASSERT(!srcs.empty() && srcs.size() <= 8u,
                 "tensor_concat: source count must be in [1, 8].");
    auto concat_extent = 0u;
    for (auto i = 0u; i < srcs.size(); i++) {
        auto &&s = srcs[i]->desc;
        LUISA_ASSERT(s.dtype == dst.desc.dtype && s.rank == dst.desc.rank,
                     "tensor_concat: source {} shape/dtype mismatch.", i);
        for (auto d = 0u; d < s.rank; d++) {
            LUISA_ASSERT(static_cast<uint32_t>(d) == static_cast<uint32_t>(dim) ||
                             s.extents[d] == dst.desc.extents[d],
                         "tensor_concat: source {} shape mismatch.", i);
        }
        concat_extent += s.extents[dim];
    }
    LUISA_ASSERT(concat_extent == dst.desc.extents[dim],
                 "tensor_concat: concatenated extent ({}) != destination extent ({}).",
                 concat_extent, dst.desc.extents[dim]);
    luisa::vector<const Expression *> args;
    args.reserve(56u);
    detail::tensor_push_operand(args, dst);
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(dim)));
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(srcs.size())));
    for (auto i = 0u; i < 8u; i++) {
        if (i < srcs.size()) {
            detail::tensor_push_operand(args, *srcs[i]);
        } else {
            // Sentinel descriptor marking an unused source slot.
            TensorDescriptor empty;
            empty.dtype = static_cast<TensorElementType>(0xFFu);
            empty.rank = 0u;
            TensorOperand dummy{empty, detail::tensor_literal(0ull)};
            detail::tensor_push_operand(args, dummy);
        }
    }
    detail::tensor_emit(CallOp::TENSOR_CONCAT, args);
}

/// dst = pad(src, pad_before) where `pad[d]` is the number of leading
/// elements inserted along dim d; trailing padding is derived from the
/// destination extents. Iteration domain: dst.numel() threads.
inline void tensor_pad(const TensorOperand &dst, const TensorOperand &src,
                       luisa::span<const uint32_t> pad) noexcept {
    LUISA_ASSERT(dst.desc.dtype == src.desc.dtype && dst.desc.rank == src.desc.rank,
                 "tensor_pad: shape/dtype mismatch.");
    LUISA_ASSERT(pad.size() == dst.desc.rank,
                 "tensor_pad: pad rank mismatch.");
    uint4 p{0u, 0u, 0u, 0u};
    for (auto i = 0u; i < pad.size(); i++) {
        p[i] = pad[i];
        LUISA_ASSERT(dst.desc.extents[i] >= src.desc.extents[i] + pad[i],
                     "tensor_pad: destination too small for padding.");
    }
    luisa::vector<const Expression *> args;
    args.reserve(13u);
    detail::tensor_push_operand(args, dst);
    detail::tensor_push_operand(args, src);
    args.emplace_back(detail::tensor_literal(p));
    detail::tensor_emit(CallOp::TENSOR_PAD, args);
}

// ---------------------------------------------------------------------------
// Element-wise unary
// ---------------------------------------------------------------------------

/// Declares an element-wise unary tensor op helper.
/// Iteration domain: out.numel() threads.
#define LUISA_DSL_TENSOR_UNARY_OP(NAME, OP)                                   \
    inline void tensor_##NAME(const TensorOperand &out,                       \
                              const TensorOperand &in) noexcept {             \
        detail::tensor_check_elementwise(out.desc, in.desc, "tensor_" #NAME); \
        luisa::vector<const Expression *> args;                               \
        args.reserve(13u);                                                    \
        detail::tensor_push_operand(args, out);                               \
        detail::tensor_push_operand(args, in);                                \
        args.emplace_back(detail::tensor_literal(                             \
            static_cast<uint32_t>(out.desc.numel())));                        \
        detail::tensor_emit(CallOp::TENSOR_##OP, args);                       \
    }

LUISA_DSL_TENSOR_UNARY_OP(neg, NEG)
LUISA_DSL_TENSOR_UNARY_OP(abs, ABS)
LUISA_DSL_TENSOR_UNARY_OP(exp, EXP)
LUISA_DSL_TENSOR_UNARY_OP(log, LOG)
LUISA_DSL_TENSOR_UNARY_OP(sqrt, SQRT)
LUISA_DSL_TENSOR_UNARY_OP(rsqrt, RSQRT)
LUISA_DSL_TENSOR_UNARY_OP(sin, SIN)
LUISA_DSL_TENSOR_UNARY_OP(cos, COS)
LUISA_DSL_TENSOR_UNARY_OP(tan, TAN)
LUISA_DSL_TENSOR_UNARY_OP(tanh, TANH)
LUISA_DSL_TENSOR_UNARY_OP(sigmoid, SIGMOID)
LUISA_DSL_TENSOR_UNARY_OP(gelu, GELU)
LUISA_DSL_TENSOR_UNARY_OP(relu, RELU)
LUISA_DSL_TENSOR_UNARY_OP(leaky_relu, LEAKY_RELU)
LUISA_DSL_TENSOR_UNARY_OP(erf, ERF)
LUISA_DSL_TENSOR_UNARY_OP(ceil, CEIL)
LUISA_DSL_TENSOR_UNARY_OP(floor, FLOOR)
LUISA_DSL_TENSOR_UNARY_OP(round, ROUND)
LUISA_DSL_TENSOR_UNARY_OP(isnan, ISNAN)
LUISA_DSL_TENSOR_UNARY_OP(isinf, ISINF)

#undef LUISA_DSL_TENSOR_UNARY_OP

// ---------------------------------------------------------------------------
// Element-wise binary
// ---------------------------------------------------------------------------

/// Declares an element-wise binary tensor op helper: out = a `op` b.
/// Iteration domain: out.numel() threads.
#define LUISA_DSL_TENSOR_BINARY_OP(NAME, OP)                                   \
    inline void tensor_##NAME(const TensorOperand &out,                        \
                              const TensorOperand &a,                          \
                              const TensorOperand &b) noexcept {               \
        detail::tensor_check_elementwise(out.desc, a.desc, b.desc,             \
                                         "tensor_" #NAME);                     \
        luisa::vector<const Expression *> args;                                \
        args.reserve(19u);                                                     \
        detail::tensor_push_operand(args, out);                                \
        detail::tensor_push_operand(args, a);                                  \
        detail::tensor_push_operand(args, b);                                  \
        args.emplace_back(detail::tensor_literal(                              \
            static_cast<uint32_t>(out.desc.numel())));                         \
        detail::tensor_emit(CallOp::TENSOR_##OP, args);                        \
    }

LUISA_DSL_TENSOR_BINARY_OP(add, ADD)
LUISA_DSL_TENSOR_BINARY_OP(sub, SUB)
LUISA_DSL_TENSOR_BINARY_OP(mul, MUL)
LUISA_DSL_TENSOR_BINARY_OP(div, DIV)
LUISA_DSL_TENSOR_BINARY_OP(pow, POW)
LUISA_DSL_TENSOR_BINARY_OP(min, MIN)
LUISA_DSL_TENSOR_BINARY_OP(max, MAX)

/// out = clamp(in, lo, hi). The bounds are encoded as raw dtype bit patterns
/// (see tensor_fill_bits). Iteration domain: out.numel() threads.
inline void tensor_clamp(const TensorOperand &out, const TensorOperand &in,
                         double lo, double hi) noexcept {
    detail::tensor_check_elementwise(out.desc, in.desc, "tensor_clamp");
    luisa::vector<const Expression *> args;
    args.reserve(16u);
    detail::tensor_push_operand(args, out);
    detail::tensor_push_operand(args, in);
    args.emplace_back(detail::tensor_literal(detail::tensor_fill_bits(out.desc.dtype, lo)));
    args.emplace_back(detail::tensor_literal(detail::tensor_fill_bits(out.desc.dtype, hi)));
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(out.desc.numel())));
    detail::tensor_emit(CallOp::TENSOR_CLAMP, args);
}

/// out = fma(a, b, c) (a*b+c). Iteration domain: out.numel() threads.
inline void tensor_fma(const TensorOperand &out, const TensorOperand &a,
                       const TensorOperand &b, const TensorOperand &c) noexcept {
    detail::tensor_check_elementwise(out.desc, a.desc, "tensor_fma");
    LUISA_ASSERT(b.desc.dtype == a.desc.dtype && b.desc.numel() == a.desc.numel() &&
                     c.desc.dtype == a.desc.dtype && c.desc.numel() == a.desc.numel(),
                 "tensor_fma: tensor shape/dtype mismatch.");
    luisa::vector<const Expression *> args;
    args.reserve(25u);
    detail::tensor_push_operand(args, out);
    detail::tensor_push_operand(args, a);
    detail::tensor_push_operand(args, b);
    detail::tensor_push_operand(args, c);
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(out.desc.numel())));
    detail::tensor_emit(CallOp::TENSOR_FMA, args);
}

#undef LUISA_DSL_TENSOR_BINARY_OP

// ---------------------------------------------------------------------------
// Contractions
// ---------------------------------------------------------------------------

/**
 * @brief C = alpha * op(A) * op(B) + beta * C (in-place epilogue reads the
 * existing C), with an optional fused element-wise epilogue.
 *
 * Iteration domain: F32 GEMM requires M*N threads; the F16 tensor-core path
 * requires ceil(M/16)*ceil(N/16) warps (i.e. 32x as many threads, one warp
 * per 16x16 output tile, plan.md §A.2). F16 GEMM requires an F32 accumulator
 * C (the FP32-accumulator rule of plan.md §A.3).
 */
inline void tensor_matmul(const TensorOperand &c, const TensorOperand &a,
                          const TensorOperand &b,
                          const GemmOptions &opts = {}) noexcept {
    auto &&ad = a.desc;
    auto &&bd = b.desc;
    auto &&cd = c.desc;
    LUISA_ASSERT(ad.rank == 2u && bd.rank == 2u && cd.rank == 2u,
                 "tensor_matmul: only rank-2 (matrix) operands are supported.");
    auto a_rows = ad.extents[opts.trans_a ? 1u : 0u];
    auto a_cols = ad.extents[opts.trans_a ? 0u : 1u];
    auto b_rows = bd.extents[opts.trans_b ? 1u : 0u];
    auto b_cols = bd.extents[opts.trans_b ? 0u : 1u];
    LUISA_ASSERT(a_cols == b_rows && a_rows == cd.extents[0] && b_cols == cd.extents[1],
                 "tensor_matmul: tensor shape mismatch (A {}x{}, B {}x{}, C {}x{}).",
                 a_rows, a_cols, b_rows, b_cols, cd.extents[0], cd.extents[1]);
    LUISA_ASSERT(ad.dtype == bd.dtype, "tensor_matmul: A and B dtypes must match.");
    LUISA_ASSERT(opts.trans_a == false && opts.trans_b == false,
                 "tensor_matmul: transposed operands are not implemented yet.");
    if (ad.dtype == TensorElementType::F16) {
        LUISA_ASSERT(cd.dtype == TensorElementType::F32,
                     "tensor_matmul: the F16 tensor-core GEMM writes an FP32 "
                     "accumulator; C must be f32.");
    } else {
        LUISA_ASSERT(cd.dtype == ad.dtype,
                     "tensor_matmul: C dtype must match the input dtype.");
    }
    luisa::vector<const Expression *> args;
    args.reserve(24u);
    detail::tensor_push_operand(args, c);
    detail::tensor_push_operand(args, a);
    detail::tensor_push_operand(args, b);
    args.emplace_back(detail::tensor_literal(luisa::to_underlying(opts.compute_dtype)));
    args.emplace_back(detail::tensor_literal(opts.trans_a ? 1u : 0u));
    args.emplace_back(detail::tensor_literal(opts.trans_b ? 1u : 0u));
    args.emplace_back(detail::tensor_literal(opts.alpha));
    args.emplace_back(detail::tensor_literal(opts.beta));
    args.emplace_back(detail::tensor_literal(luisa::to_underlying(opts.epilogue)));
    detail::tensor_emit(CallOp::TENSOR_MATMUL, args);
}

/// Batched GEMM: for b in [0, batch): C[b] = alpha * A[b] * B[b] + beta * C[b].
/// Each operand's descriptor must be rank 3, where dim 0 is the batch index
/// (stride = extents[1]*extents[2] for contiguous storage; arbitrary batch
/// strides are honored through the descriptor).
inline void tensor_batch_matmul(const TensorOperand &c, const TensorOperand &a,
                                const TensorOperand &b, uint32_t batch,
                                const GemmOptions &opts = {}) noexcept {
    auto &&ad = a.desc;
    auto &&bd = b.desc;
    auto &&cd = c.desc;
    LUISA_ASSERT(ad.rank == 3u && bd.rank == 3u && cd.rank == 3u,
                 "tensor_batch_matmul: only rank-3 (batched matrix) operands are supported.");
    LUISA_ASSERT(ad.extents[0] == bd.extents[0] && ad.extents[0] == cd.extents[0] &&
                     ad.extents[0] >= batch,
                 "tensor_batch_matmul: batch size mismatch.");
    auto a_rows = ad.extents[opts.trans_a ? 2u : 1u];
    auto a_cols = ad.extents[opts.trans_a ? 1u : 2u];
    auto b_rows = bd.extents[opts.trans_b ? 2u : 1u];
    auto b_cols = bd.extents[opts.trans_b ? 1u : 2u];
    LUISA_ASSERT(a_cols == b_rows && a_rows == cd.extents[1] && b_cols == cd.extents[2],
                 "tensor_batch_matmul: tensor shape mismatch.");
    LUISA_ASSERT(ad.dtype == bd.dtype, "tensor_batch_matmul: A and B dtypes must match.");
    if (ad.dtype == TensorElementType::F16) {
        LUISA_ASSERT(cd.dtype == TensorElementType::F32,
                     "tensor_batch_matmul: the F16 tensor-core GEMM writes an FP32 "
                     "accumulator; C must be f32.");
    } else {
        LUISA_ASSERT(cd.dtype == ad.dtype,
                     "tensor_batch_matmul: C dtype must match the input dtype.");
    }
    luisa::vector<const Expression *> args;
    args.reserve(25u);
    detail::tensor_push_operand(args, c);
    detail::tensor_push_operand(args, a);
    detail::tensor_push_operand(args, b);
    args.emplace_back(detail::tensor_literal(luisa::to_underlying(opts.compute_dtype)));
    args.emplace_back(detail::tensor_literal(opts.trans_a ? 1u : 0u));
    args.emplace_back(detail::tensor_literal(opts.trans_b ? 1u : 0u));
    args.emplace_back(detail::tensor_literal(opts.alpha));
    args.emplace_back(detail::tensor_literal(opts.beta));
    args.emplace_back(detail::tensor_literal(luisa::to_underlying(opts.epilogue)));
    args.emplace_back(detail::tensor_literal(batch));
    detail::tensor_emit(CallOp::TENSOR_BATCH_MATMUL, args);
}

/**
 * @brief Generalized contraction C[mode_c] = sum_reduce A[mode_a] * B[mode_b].
 *
 * Modes are einsum-style label lists: `mode_a[i]` is the label of A's dim i,
 * `mode_b[i]` of B's dim i and `mode_c[i]` of C's dim i. Labels are encoded as
 * small integers (0..25 for 'a'..'z'); unused slots are padded with 255. A
 * label that appears in both A and B but not in C is a reduction dimension;
 * labels appearing in C must appear in A or B. Iteration domain: c.numel()
 * threads (one thread per output element).
 */
inline void tensor_contract(const TensorOperand &c, const TensorOperand &a,
                            const TensorOperand &b,
                            luisa::span<const uint32_t> mode_a,
                            luisa::span<const uint32_t> mode_b,
                            luisa::span<const uint32_t> mode_c,
                            TensorElementType compute_dtype = TensorElementType::F32) noexcept {
    auto &&ad = a.desc;
    auto &&bd = b.desc;
    auto &&cd = c.desc;
    LUISA_ASSERT(mode_a.size() == ad.rank && mode_b.size() == bd.rank &&
                     mode_c.size() == cd.rank,
                 "tensor_contract: mode rank mismatch.");
    LUISA_ASSERT(ad.dtype == bd.dtype && cd.dtype == ad.dtype,
                 "tensor_contract: dtypes must match.");
    uint4 ma{255u, 255u, 255u, 255u};
    uint4 mb{255u, 255u, 255u, 255u};
    uint4 mc{255u, 255u, 255u, 255u};
    for (auto i = 0u; i < mode_a.size(); i++) { ma[i] = mode_a[i]; }
    for (auto i = 0u; i < mode_b.size(); i++) { mb[i] = mode_b[i]; }
    for (auto i = 0u; i < mode_c.size(); i++) { mc[i] = mode_c[i]; }
    luisa::vector<const Expression *> args;
    args.reserve(22u);
    detail::tensor_push_operand(args, c);
    detail::tensor_push_operand(args, a);
    detail::tensor_push_operand(args, b);
    args.emplace_back(detail::tensor_literal(ma));
    args.emplace_back(detail::tensor_literal(mb));
    args.emplace_back(detail::tensor_literal(mc));
    args.emplace_back(detail::tensor_literal(luisa::to_underlying(compute_dtype)));
    detail::tensor_emit(CallOp::TENSOR_CONTRACT, args);
}

// ---------------------------------------------------------------------------
// Reductions / scans
// ---------------------------------------------------------------------------

namespace detail {
inline void tensor_emit_reduce(CallOp op, const char *name,
                               const TensorOperand &out, const TensorOperand &in,
                               luisa::span<const int> reduce_dims) noexcept {
    LUISA_ASSERT(out.desc.dtype == in.desc.dtype,
                 "{}: input and output dtypes must match.", name);
    LUISA_ASSERT(!reduce_dims.empty() && reduce_dims.size() <= tensor_max_rank,
                 "{}: invalid reduce dims.", name);
    uint4 dims{0u, 0u, 0u, 0u};
    uint64_t out_numel = 1u;
    std::array<bool, tensor_max_rank> reduced{};
    for (auto d : reduce_dims) {
        LUISA_ASSERT(d >= 0 && static_cast<uint32_t>(d) < in.desc.rank && !reduced[d],
                     "{}: invalid reduce dim {}.", name, d);
        reduced[d] = true;
        dims[d] = 1u;
    }
    for (auto i = 0u; i < in.desc.rank; i++) {
        if (!reduced[i]) { out_numel *= in.desc.extents[i]; }
    }
    LUISA_ASSERT(out.desc.numel() == out_numel,
                 "{}: output numel ({}) does not match the reduced shape ({}).",
                 name, out.desc.numel(), out_numel);
    luisa::vector<const Expression *> args;
    args.reserve(15u);
    detail::tensor_push_operand(args, out);
    detail::tensor_push_operand(args, in);
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(reduce_dims.size())));
    args.emplace_back(detail::tensor_literal(dims));
    detail::tensor_emit(op, args);
}
}// namespace detail

/// out = sum of `in` over `reduce_dims`. Iteration domain: out.numel() threads.
inline void tensor_reduce_sum(const TensorOperand &out, const TensorOperand &in,
                              luisa::span<const int> reduce_dims) noexcept {
    detail::tensor_emit_reduce(CallOp::TENSOR_REDUCE_SUM, "tensor_reduce_sum",
                               out, in, reduce_dims);
}

/// out = max of `in` over `reduce_dims`. Iteration domain: out.numel() threads.
inline void tensor_reduce_max(const TensorOperand &out, const TensorOperand &in,
                              luisa::span<const int> reduce_dims) noexcept {
    detail::tensor_emit_reduce(CallOp::TENSOR_REDUCE_MAX, "tensor_reduce_max",
                               out, in, reduce_dims);
}

/// out = min of `in` over `reduce_dims`. Iteration domain: out.numel() threads.
inline void tensor_reduce_min(const TensorOperand &out, const TensorOperand &in,
                              luisa::span<const int> reduce_dims) noexcept {
    detail::tensor_emit_reduce(CallOp::TENSOR_REDUCE_MIN, "tensor_reduce_min",
                               out, in, reduce_dims);
}

/// Inclusive prefix sum of `in` along `dim` (out has the same shape as in).
/// Iteration domain: in.numel() / in.extents[dim] threads (one per fiber).
inline void tensor_cumsum(const TensorOperand &out, const TensorOperand &in, int dim) noexcept {
    LUISA_ASSERT(out.desc.dtype == in.desc.dtype && out.desc.numel() == in.desc.numel(),
                 "tensor_cumsum: input and output shape/dtype must match.");
    LUISA_ASSERT(dim >= 0 && static_cast<uint32_t>(dim) < in.desc.rank,
                 "tensor_cumsum: invalid dim {}.", dim);
    luisa::vector<const Expression *> args;
    args.reserve(14u);
    detail::tensor_push_operand(args, out);
    detail::tensor_push_operand(args, in);
    args.emplace_back(detail::tensor_literal(static_cast<uint32_t>(dim)));
    detail::tensor_emit(CallOp::TENSOR_CUMSUM, args);
}

}// namespace luisa::compute
