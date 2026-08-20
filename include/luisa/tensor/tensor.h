//
// Runtime tensor handle + ergonomic DSL surface (plan.md §2.3 / §2.4 and
// cuda_tensor/README.md §9).
//
// `Tensor` is a cheap, copyable handle to a dense strided device tensor. It
// carries a `TensorDescriptor` (dtype, rank, extents, strides, storage
// offset) plus the backing storage: an owned `ByteBuffer` for tensors created
// by `empty`/`zeros`/`ones`/`full`, or an external `ByteBufferView` for
// tensors created by `from_buffer`. Views (`view`, `permute`, `slice`) share
// the storage and only change metadata.
//
// Every convenience operation (`add`, `mul`, `exp`, `relu`, `matmul`,
// `reduce_sum`, `cumsum`, `contract`, ...) traces a zero-argument kernel that
// calls the low-level `tensor_*` DSL helpers from `<luisa/dsl/tensor_ops.h>`
// (which emit the first-class `TENSOR_*` CallOps), compiles it with the
// device, and enqueues the dispatch asynchronously on the caller's `Stream`.
// The operation returns a new `Tensor`; results are only visible after the
// stream is synchronized (or after subsequent commands on the same stream).
//

#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/tensor_ops.h>
#include <luisa/tensor/tensor_descriptor.h>

namespace luisa::compute {

/// Map a C++ scalar type to its tensor element type.
template<typename T>
[[nodiscard]] constexpr auto tensor_element_type_of() noexcept {
    if constexpr (std::is_same_v<T, half>) {
        return TensorElementType::F16;
    } else if constexpr (std::is_same_v<T, float>) {
        return TensorElementType::F32;
    } else if constexpr (std::is_same_v<T, int>) {
        return TensorElementType::I32;
    } else {
        static_assert(luisa::always_false_v<T>,
                      "Unsupported tensor element type (supported: half, float, int).");
    }
}

/// Runtime handle to a dense strided device tensor.
class Tensor {

public:
    /// Create an uninitialized contiguous tensor of the given shape/dtype.
    [[nodiscard]] static Tensor empty(luisa::span<const uint32_t> shape,
                                      TensorElementType dtype,
                                      Device &device) noexcept {
        LUISA_ASSERT(tensor_element_type_supported(dtype),
                     "Tensor element type '{}' is not supported by the "
                     "runtime tensor operators.",
                     tensor_element_type_name(dtype));
        auto desc = TensorDescriptor::contiguous(dtype, shape);
        auto bytes = desc.numel() * tensor_element_size(dtype);
        Tensor t;
        t.desc_ = desc;
        t.storage_ = luisa::make_shared<ByteBuffer>(device.create_byte_buffer(bytes));
        t.device_ = &device;
        return t;
    }

    /// Braced-shape overload, e.g. empty({4, 5}, F32, device).
    [[nodiscard]] static Tensor empty(std::initializer_list<uint32_t> shape,
                                      TensorElementType dtype,
                                      Device &device) noexcept {
        return empty(luisa::span<const uint32_t>{shape.begin(), shape.size()}, dtype, device);
    }

    /// Create a tensor filled with zero.
    [[nodiscard]] static Tensor zeros(luisa::span<const uint32_t> shape,
                                      TensorElementType dtype,
                                      Device &device,
                                      Stream &stream) noexcept {
        auto t = empty(shape, dtype, device);
        t.fill_(0.0, stream);
        return t;
    }

    /// Create a tensor filled with one.
    [[nodiscard]] static Tensor ones(luisa::span<const uint32_t> shape,
                                     TensorElementType dtype,
                                     Device &device,
                                     Stream &stream) noexcept {
        auto t = empty(shape, dtype, device);
        t.fill_(1.0, stream);
        return t;
    }

    /// Create a tensor filled with `value`.
    [[nodiscard]] static Tensor full(luisa::span<const uint32_t> shape,
                                     TensorElementType dtype,
                                     double value,
                                     Device &device,
                                     Stream &stream) noexcept {
        auto t = empty(shape, dtype, device);
        t.fill_(value, stream);
        return t;
    }

    /// Braced-shape overloads, e.g. zeros({4, 5}, F32, device, stream).
    [[nodiscard]] static Tensor zeros(std::initializer_list<uint32_t> shape,
                                      TensorElementType dtype,
                                      Device &device,
                                      Stream &stream) noexcept {
        return zeros(luisa::span<const uint32_t>{shape.begin(), shape.size()}, dtype, device, stream);
    }
    [[nodiscard]] static Tensor ones(std::initializer_list<uint32_t> shape,
                                     TensorElementType dtype,
                                     Device &device,
                                     Stream &stream) noexcept {
        return ones(luisa::span<const uint32_t>{shape.begin(), shape.size()}, dtype, device, stream);
    }
    [[nodiscard]] static Tensor full(std::initializer_list<uint32_t> shape,
                                     TensorElementType dtype,
                                     double value,
                                     Device &device,
                                     Stream &stream) noexcept {
        return full(luisa::span<const uint32_t>{shape.begin(), shape.size()}, dtype, value, device, stream);
    }

    /// Wrap external storage (e.g. a buffer view) with a full descriptor.
    [[nodiscard]] static Tensor from_buffer(ByteBufferView storage,
                                            TensorDescriptor desc,
                                            Device &device) noexcept {
        Tensor t;
        t.desc_ = desc;
        t.external_ = storage;
        t.device_ = &device;
        return t;
    }

    /// Wrap a typed buffer view as a contiguous tensor of the given shape.
    template<typename T>
    [[nodiscard]] static Tensor from_buffer(BufferView<T> storage,
                                            luisa::span<const uint32_t> shape,
                                            Device &device) noexcept {
        return from_buffer(ByteBufferView{storage},
                           TensorDescriptor::contiguous(tensor_element_type_of<T>(), shape),
                           device);
    }

    /// Braced-shape overload, e.g. from_buffer(buf.view(), {4, 5}, device).
    template<typename T>
    [[nodiscard]] static Tensor from_buffer(BufferView<T> storage,
                                            std::initializer_list<uint32_t> shape,
                                            Device &device) noexcept {
        return from_buffer(ByteBufferView{storage},
                           TensorDescriptor::contiguous(tensor_element_type_of<T>(), shape),
                           device);
    }

public:
    /// Rank of the tensor.
    [[nodiscard]] auto rank() const noexcept { return desc_.rank; }
    /// Extent of dimension `dim`.
    [[nodiscard]] auto size(int dim) const noexcept { return desc_.extents[dim]; }
    /// Stride (in elements) of dimension `dim`.
    [[nodiscard]] auto stride(int dim) const noexcept { return desc_.strides[dim]; }
    /// Number of logical elements.
    [[nodiscard]] auto numel() const noexcept { return desc_.numel(); }
    /// Element dtype.
    [[nodiscard]] auto dtype() const noexcept { return desc_.dtype; }
    /// Whether the descriptor is row-major contiguous.
    [[nodiscard]] auto is_contiguous() const noexcept { return desc_.is_contiguous(); }
    /// The underlying layout descriptor.
    [[nodiscard]] const auto &descriptor() const noexcept { return desc_; }
    /// Shape (extents) of the tensor.
    [[nodiscard]] luisa::span<const uint32_t> shape() const noexcept {
        return {desc_.extents.data(), desc_.rank};
    }
    /// Storage view (handle + byte offset + size) for host transfer / ops.
    [[nodiscard]] ByteBufferView storage_view() const noexcept {
        if (storage_) { return storage_->view(); }
        return external_;
    }
    /// Device used for result allocation.
    [[nodiscard]] Device &device() const noexcept {
        LUISA_ASSERT(device_ != nullptr,
                     "Tensor was created without a device; cannot allocate results.");
        return *device_;
    }

    /// Metadata-only reshape (requires contiguous storage, plan.md §9.1 view).
    [[nodiscard]] Tensor view(luisa::span<const uint32_t> shape) const noexcept {
        Tensor t = *this;
        t.desc_ = desc_.reshaped(shape);
        return t;
    }

    /// Metadata-only dimension permutation (no data movement).
    [[nodiscard]] Tensor permute(luisa::span<const uint32_t> dims) const noexcept {
        Tensor t = *this;
        t.desc_ = desc_.permuted(dims);
        return t;
    }

    /// Metadata-only slice along `dim` in [begin, end).
    [[nodiscard]] Tensor slice(int dim, uint32_t begin, uint32_t end) const noexcept {
        LUISA_ASSERT(dim >= 0 && static_cast<uint32_t>(dim) < desc_.rank &&
                         begin <= end && end <= desc_.extents[dim],
                     "Tensor::slice: invalid slice on dim {} [{}, {}).",
                     dim, begin, end);
        Tensor t = *this;
        t.desc_.extents[dim] = end - begin;
        t.desc_.storage_offset += begin * desc_.strides[dim];
        return t;
    }

    /// Materialize a contiguous copy whose storage exactly covers the logical
    /// tensor (metadata-only when the storage already starts at offset zero).
    [[nodiscard]] Tensor contiguous(Stream &stream) const {
        if (is_contiguous() && desc_.storage_offset == 0u) { return *this; }
        auto out = empty(shape(), dtype(), device());
        auto count = static_cast<uint32_t>(numel());
        auto o_desc = out.desc_;
        auto i_desc = desc_;
        auto o_view = out.storage_view();
        auto i_view = storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto in = tensor_operand(i_desc, i_view);
            tensor_copy(o, in);
        });
        return out;
    }

    /// Asynchronously copy host bytes into the storage (size must match).
    void copy_from(const void *host, Stream &stream) const noexcept {
        stream << storage_view().copy_from(host);
    }

    /// Asynchronously copy storage bytes to the host (size must match).
    void copy_to(void *host, Stream &stream) const noexcept {
        stream << storage_view().copy_to(host);
    }

    /// Fill the tensor in-place with `value` (async on the stream).
    void fill_(double value, Stream &stream) const {
        auto count = static_cast<uint32_t>(numel());
        auto d = desc_;
        auto v = storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto dst = tensor_operand(d, v);
            tensor_fill(dst, value);
        });
    }

    // ------------------------------------------------------------------
    // Element-wise unary ops (return a new tensor)
    // ------------------------------------------------------------------
#define LUISA_TENSOR_UNARY_MEMBER(NAME, FN)                                        \
    [[nodiscard]] Tensor NAME(Stream &stream) const {                              \
        return _unary(stream, [](const TensorOperand &o, const TensorOperand &i) { \
            FN(o, i);                                                              \
        });                                                                        \
    }
    LUISA_TENSOR_UNARY_MEMBER(neg, tensor_neg)
    LUISA_TENSOR_UNARY_MEMBER(abs, tensor_abs)
    LUISA_TENSOR_UNARY_MEMBER(exp, tensor_exp)
    LUISA_TENSOR_UNARY_MEMBER(log, tensor_log)
    LUISA_TENSOR_UNARY_MEMBER(sqrt, tensor_sqrt)
    LUISA_TENSOR_UNARY_MEMBER(rsqrt, tensor_rsqrt)
    LUISA_TENSOR_UNARY_MEMBER(sin, tensor_sin)
    LUISA_TENSOR_UNARY_MEMBER(cos, tensor_cos)
    LUISA_TENSOR_UNARY_MEMBER(tan, tensor_tan)
    LUISA_TENSOR_UNARY_MEMBER(tanh, tensor_tanh)
    LUISA_TENSOR_UNARY_MEMBER(sigmoid, tensor_sigmoid)
    LUISA_TENSOR_UNARY_MEMBER(gelu, tensor_gelu)
    LUISA_TENSOR_UNARY_MEMBER(relu, tensor_relu)
    LUISA_TENSOR_UNARY_MEMBER(leaky_relu, tensor_leaky_relu)
    LUISA_TENSOR_UNARY_MEMBER(erf, tensor_erf)
    LUISA_TENSOR_UNARY_MEMBER(ceil, tensor_ceil)
    LUISA_TENSOR_UNARY_MEMBER(floor, tensor_floor)
    LUISA_TENSOR_UNARY_MEMBER(round, tensor_round)
    LUISA_TENSOR_UNARY_MEMBER(isnan, tensor_isnan)
    LUISA_TENSOR_UNARY_MEMBER(isinf, tensor_isinf)
#undef LUISA_TENSOR_UNARY_MEMBER

    /// Element-wise clamp to [lo, hi].
    [[nodiscard]] Tensor clamp(double lo, double hi, Stream &stream) const {
        return _unary(stream, [lo, hi](const TensorOperand &o, const TensorOperand &i) {
            tensor_clamp(o, i, lo, hi);
        });
    }

    /// Element-wise copy into a new tensor.
    [[nodiscard]] Tensor copy(Stream &stream) const {
        return _unary(stream, [](const TensorOperand &o, const TensorOperand &i) {
            tensor_copy(o, i);
        });
    }

    /// Element-wise dtype conversion.
    [[nodiscard]] Tensor cast(TensorElementType dtype, Stream &stream) const {
        auto out = _empty_like(dtype);
        auto count = static_cast<uint32_t>(numel());
        auto o_desc = out.desc_;
        auto i_desc = desc_;
        auto o_view = out.storage_view();
        auto i_view = storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto in = tensor_operand(i_desc, i_view);
            tensor_cast(o, in);
        });
        return out;
    }

    // ------------------------------------------------------------------
    // Element-wise binary ops (return a new tensor)
    // ------------------------------------------------------------------
#define LUISA_TENSOR_BINARY_MEMBER(NAME, FN)                                            \
    [[nodiscard]] Tensor NAME(const Tensor &b, Stream &stream) const {                  \
        return _binary(b, stream, [](const TensorOperand &o, const TensorOperand &x,    \
                                     const TensorOperand &y) { FN(o, x, y); });         \
    }
    LUISA_TENSOR_BINARY_MEMBER(add, tensor_add)
    LUISA_TENSOR_BINARY_MEMBER(sub, tensor_sub)
    LUISA_TENSOR_BINARY_MEMBER(mul, tensor_mul)
    LUISA_TENSOR_BINARY_MEMBER(div, tensor_div)
    LUISA_TENSOR_BINARY_MEMBER(pow, tensor_pow)
    LUISA_TENSOR_BINARY_MEMBER(min, tensor_min)
    LUISA_TENSOR_BINARY_MEMBER(max, tensor_max)
#undef LUISA_TENSOR_BINARY_MEMBER

    /// out = fma(a, b, c) = a*b + c.
    [[nodiscard]] Tensor fma(const Tensor &b, const Tensor &c, Stream &stream) const {
        auto out = _like();
        auto count = static_cast<uint32_t>(out.numel());
        auto o_desc = out.desc_;
        auto a_desc = desc_;
        auto b_desc = b.desc_;
        auto c_desc = c.desc_;
        auto o_view = out.storage_view();
        auto a_view = storage_view();
        auto b_view = b.storage_view();
        auto c_view = c.storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto x = tensor_operand(a_desc, a_view);
            auto y = tensor_operand(b_desc, b_view);
            auto z = tensor_operand(c_desc, c_view);
            tensor_fma(o, x, y, z);
        });
        return out;
    }

    // ------------------------------------------------------------------
    // Reductions / scans
    // ------------------------------------------------------------------
#define LUISA_TENSOR_REDUCE_MEMBER(NAME, FN)                                             \
    [[nodiscard]] Tensor NAME(luisa::span<const int> dims, Stream &stream) const {       \
        auto out = _reduce_like(dims);                                                   \
        auto count = static_cast<uint32_t>(out.numel());                                 \
        auto o_desc = out.desc_;                                                         \
        auto i_desc = desc_;                                                             \
        auto o_view = out.storage_view();                                                \
        auto i_view = storage_view();                                                    \
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {                          \
            auto o = tensor_operand(o_desc, o_view);                                     \
            auto in = tensor_operand(i_desc, i_view);                                    \
            FN(o, in, dims);                                                             \
        });                                                                              \
        return out;                                                                      \
    }
    LUISA_TENSOR_REDUCE_MEMBER(reduce_sum, tensor_reduce_sum)
    LUISA_TENSOR_REDUCE_MEMBER(reduce_max, tensor_reduce_max)
    LUISA_TENSOR_REDUCE_MEMBER(reduce_min, tensor_reduce_min)
#undef LUISA_TENSOR_REDUCE_MEMBER

    /// Inclusive prefix sum along `dim` (same shape as the input).
    [[nodiscard]] Tensor cumsum(int dim, Stream &stream) const {
        auto out = _like();
        auto fibers = static_cast<uint32_t>(numel() / desc_.extents[dim]);
        auto o_desc = out.desc_;
        auto i_desc = desc_;
        auto o_view = out.storage_view();
        auto i_view = storage_view();
        _launch(stream, fibers, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto in = tensor_operand(i_desc, i_view);
            tensor_cumsum(o, in, dim);
        });
        return out;
    }

    // ------------------------------------------------------------------
    // Contractions
    // ------------------------------------------------------------------

    /// C = alpha * A * B (fresh output, zero-initialized). For an in-place
    /// epilogue with beta != 0 (C = alpha*A*B + beta*C) use `matmul_into`.
    [[nodiscard]] Tensor matmul(const Tensor &b, const GemmOptions &opts,
                                Stream &stream) const {
        LUISA_ASSERT(dtype() == b.dtype(),
                     "Tensor::matmul: A and B dtypes must match.");
        LUISA_ASSERT(opts.beta == 0.0f,
                     "Tensor::matmul returns a fresh output; use matmul_into "
                     "for an in-place epilogue with beta != 0.");
        auto m = size(opts.trans_a ? 1 : 0);
        auto k = size(opts.trans_a ? 0 : 1);
        auto n = b.size(opts.trans_b ? 0 : 1);
        LUISA_ASSERT(k == b.size(opts.trans_b ? 1 : 0),
                     "Tensor::matmul: inner dimensions mismatch ({} vs {}).",
                     k, b.size(opts.trans_b ? 1 : 0));
        auto out_dtype = dtype() == TensorElementType::F16 ?
                             TensorElementType::F32 : dtype();
        std::array<uint32_t, 2> out_shape{m, n};
        auto out = empty(luisa::span<const uint32_t>{out_shape.data(), 2u}, out_dtype, device());
        out.fill_(0.0, stream); // zero the accumulator so the epilogue is well-defined
        auto o_desc = out.desc_;
        auto a_desc = desc_;
        auto b_desc = b.desc_;
        auto o_view = out.storage_view();
        auto a_view = storage_view();
        auto b_view = b.storage_view();
        if (dtype() == TensorElementType::F16) {
            // Tensor-core path: one warp per 16x16 output tile (plan.md §A.2).
            auto tiles_n = (n + 15u) / 16u;
            auto tiles_m = (m + 15u) / 16u;
            _launch(stream, tiles_m * tiles_n * 32u, 32u, [&]() noexcept {
                auto o = tensor_operand(o_desc, o_view);
                auto a = tensor_operand(a_desc, a_view);
                auto bb = tensor_operand(b_desc, b_view);
                tensor_matmul(o, a, bb, opts);
            });
        } else {
            _launch(stream, m * n, _kDefaultBlock, [&]() noexcept {
                auto o = tensor_operand(o_desc, o_view);
                auto a = tensor_operand(a_desc, a_view);
                auto bb = tensor_operand(b_desc, b_view);
                tensor_matmul(o, a, bb, opts);
            });
        }
        return out;
    }

    /// In-place GEMM: writes alpha*A*B + beta*C into `c` (the epilogue reads the
    /// existing C, plan.md §A.1). Use this instead of `matmul` when beta != 0.
    void matmul_into(const Tensor &c, const Tensor &b, const GemmOptions &opts,
                     Stream &stream) const {
        auto o_desc = c.desc_;
        auto a_desc = desc_;
        auto b_desc = b.desc_;
        auto o_view = c.storage_view();
        auto a_view = storage_view();
        auto b_view = b.storage_view();
        auto m = c.size(0);
        auto n = c.size(1);
        if (dtype() == TensorElementType::F16) {
            auto tiles_n = (n + 15u) / 16u;
            auto tiles_m = (m + 15u) / 16u;
            _launch(stream, tiles_m * tiles_n * 32u, 32u, [&]() noexcept {
                auto o = tensor_operand(o_desc, o_view);
                auto a = tensor_operand(a_desc, a_view);
                auto bb = tensor_operand(b_desc, b_view);
                tensor_matmul(o, a, bb, opts);
            });
        } else {
            _launch(stream, m * n, _kDefaultBlock, [&]() noexcept {
                auto o = tensor_operand(o_desc, o_view);
                auto a = tensor_operand(a_desc, a_view);
                auto bb = tensor_operand(b_desc, b_view);
                tensor_matmul(o, a, bb, opts);
            });
        }
    }

    /// Batched GEMM (rank-3 operands, dim 0 is the batch).
    [[nodiscard]] Tensor batch_matmul(const Tensor &b, uint32_t batch,
                                      const GemmOptions &opts,
                                      Stream &stream) const {
        auto m = size(1);
        auto n = b.size(2);
        auto out_dtype = dtype() == TensorElementType::F16 ?
                             TensorElementType::F32 : dtype();
        std::array<uint32_t, 3> out_shape{batch, m, n};
        auto out = empty(luisa::span<const uint32_t>{out_shape.data(), 3u}, out_dtype, device());
        auto o_desc = out.desc_;
        auto a_desc = desc_;
        auto b_desc = b.desc_;
        auto o_view = out.storage_view();
        auto a_view = storage_view();
        auto b_view = b.storage_view();
        if (dtype() == TensorElementType::F16) {
            auto tiles_n = (n + 15u) / 16u;
            auto tiles_m = (m + 15u) / 16u;
            _launch(stream, batch * tiles_m * tiles_n * 32u, 32u, [&]() noexcept {
                auto o = tensor_operand(o_desc, o_view);
                auto a = tensor_operand(a_desc, a_view);
                auto bb = tensor_operand(b_desc, b_view);
                tensor_batch_matmul(o, a, bb, batch, opts);
            });
        } else {
            _launch(stream, batch * m * n, _kDefaultBlock, [&]() noexcept {
                auto o = tensor_operand(o_desc, o_view);
                auto a = tensor_operand(a_desc, a_view);
                auto bb = tensor_operand(b_desc, b_view);
                tensor_batch_matmul(o, a, bb, batch, opts);
            });
        }
        return out;
    }

    /// Generalized einsum-style contraction C[mode_c] = sum A[mode_a] * B[mode_b].
    [[nodiscard]] Tensor contract(const Tensor &b,
                                  luisa::span<const uint32_t> mode_a,
                                  luisa::span<const uint32_t> mode_b,
                                  luisa::span<const uint32_t> mode_c,
                                  Stream &stream) const {
        auto out_desc = _contract_out_desc(b, mode_a, mode_b, mode_c);
        auto out = empty(luisa::span<const uint32_t>{out_desc.extents.data(), out_desc.rank},
                         out_desc.dtype, device());
        auto count = static_cast<uint32_t>(out.numel());
        auto o_desc = out.desc_;
        auto a_desc = desc_;
        auto b_desc = b.desc_;
        auto o_view = out.storage_view();
        auto a_view = storage_view();
        auto b_view = b.storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto a = tensor_operand(a_desc, a_view);
            auto bb = tensor_operand(b_desc, b_view);
            tensor_contract(o, a, bb, mode_a, mode_b, mode_c);
        });
        return out;
    }

private:
    static constexpr uint32_t _kDefaultBlock = 256u;

    TensorDescriptor desc_;
    luisa::shared_ptr<ByteBuffer> storage_;
    ByteBufferView external_;
    Device *device_{};

    Tensor() noexcept = default;

    /// A contiguous tensor with the same shape/dtype as `this`.
    [[nodiscard]] Tensor _like() const {
        return empty(shape(), dtype(), device());
    }

    /// A contiguous tensor with the same shape but a new dtype.
    [[nodiscard]] Tensor _empty_like(TensorElementType dtype) const {
        return empty(shape(), dtype, device());
    }

    /// A contiguous tensor with the reduced shape (non-reduced dims remain).
    [[nodiscard]] Tensor _reduce_like(luisa::span<const int> dims) const {
        auto desc = _reduce_out_desc(dims);
        return empty(luisa::span<const uint32_t>{desc.extents.data(), desc.rank},
                     desc.dtype, device());
    }

    /// Output descriptor of a reduction over `dims` (non-reduced dims remain).
    [[nodiscard]] TensorDescriptor _reduce_out_desc(luisa::span<const int> dims) const {
        std::array<bool, tensor_max_rank> reduced{};
        for (auto d : dims) {
            LUISA_ASSERT(d >= 0 && static_cast<uint32_t>(d) < desc_.rank,
                         "Tensor::reduce: invalid dim {}.", d);
            reduced[d] = true;
        }
        std::array<uint32_t, tensor_max_rank> shape{};
        uint32_t r = 0u;
        for (auto i = 0u; i < desc_.rank; i++) {
            if (!reduced[i]) { shape[r++] = desc_.extents[i]; }
        }
        return TensorDescriptor::contiguous(
            desc_.dtype, luisa::span<const uint32_t>{shape.data(), r});
    }

    /// Output descriptor of a contraction from the mode lists.
    [[nodiscard]] TensorDescriptor _contract_out_desc(
        const Tensor &b, luisa::span<const uint32_t> mode_a,
        luisa::span<const uint32_t> mode_b,
        luisa::span<const uint32_t> mode_c) const {
        std::array<uint32_t, tensor_max_rank> shape{};
        for (auto j = 0u; j < mode_c.size(); j++) {
            auto label = mode_c[j];
            auto found = false;
            for (auto i = 0u; i < desc_.rank; i++) {
                if (mode_a[i] == label) {
                    shape[j] = desc_.extents[i];
                    found = true;
                    break;
                }
            }
            if (!found) {
                for (auto i = 0u; i < b.desc_.rank; i++) {
                    if (mode_b[i] == label) {
                        shape[j] = b.desc_.extents[i];
                        found = true;
                        break;
                    }
                }
            }
            LUISA_ASSERT(found, "Tensor::contract: output mode label {} not found.", label);
        }
        auto dtype = desc_.dtype == TensorElementType::F16 ?
                         TensorElementType::F32 : desc_.dtype;
        return TensorDescriptor::contiguous(
            dtype, luisa::span<const uint32_t>{shape.data(), mode_c.size()});
    }

    /// Trace a zero-argument kernel that calls `body()` and enqueue it.
    template<typename Body>
    void _launch(Stream &stream, uint32_t dispatch_size, uint32_t block_size,
                 Body &&body) const {
        Kernel1D kernel = [&]() noexcept {
            set_block_size(block_size, 1u, 1u);
            body();
        };
        auto shader = device().compile(kernel);
        stream << shader().dispatch(dispatch_size);
    }

    /// Trace an element-wise unary op producing a new tensor.
    template<typename Fn>
    [[nodiscard]] Tensor _unary(Stream &stream, Fn &&fn) const {
        auto out = _like();
        auto count = static_cast<uint32_t>(out.numel());
        auto o_desc = out.desc_;
        auto i_desc = desc_;
        auto o_view = out.storage_view();
        auto i_view = storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto in = tensor_operand(i_desc, i_view);
            fn(o, in);
        });
        return out;
    }

    /// Trace an element-wise binary op producing a new tensor.
    template<typename Fn>
    [[nodiscard]] Tensor _binary(const Tensor &b, Stream &stream, Fn &&fn) const {
        auto out = _like();
        auto count = static_cast<uint32_t>(out.numel());
        auto o_desc = out.desc_;
        auto a_desc = desc_;
        auto b_desc = b.desc_;
        auto o_view = out.storage_view();
        auto a_view = storage_view();
        auto b_view = b.storage_view();
        _launch(stream, count, _kDefaultBlock, [&]() noexcept {
            auto o = tensor_operand(o_desc, o_view);
            auto x = tensor_operand(a_desc, a_view);
            auto y = tensor_operand(b_desc, b_view);
            fn(o, x, y);
        });
        return out;
    }
};

} // namespace luisa::compute
