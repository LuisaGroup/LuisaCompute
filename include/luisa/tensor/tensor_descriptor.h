//
// Runtime tensor descriptor for the first-class tensor `CallOp`s
// (see plan.md — Tensor Operators as First-Class `CallOp`s).
//
// The descriptor is a plain host-side value describing the logical layout of
// a dense tensor: element dtype (reusing `TensorElementType` from the tile IR
// — no parallel dtype enum), rank, extents, strides (in elements) and a
// storage offset (in elements). The DSL encodes descriptors as scalar/vector
// constant arguments to the tensor `CallOp`s; backends map the dtype tag to
// their native element types.
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <numeric>
#include <initializer_list>

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/tensor.h>// TensorElementType (canonical dtype tag)

namespace luisa::compute {

/// Maximum rank supported by the tensor operator descriptors.
inline constexpr uint32_t tensor_max_rank = 4u;

/// Byte size of one element of the given tensor element type.
/// Sub-byte quantized types (I4/FP4) report 0 and are not yet supported by
/// the runtime tensor operators.
[[nodiscard]] constexpr size_t tensor_element_size(TensorElementType t) noexcept {
    switch (t) {
        case TensorElementType::F16: return 2u;
        case TensorElementType::F32: return 4u;
        case TensorElementType::I32: return 4u;
        case TensorElementType::I8: [[fallthrough]];
        case TensorElementType::FP8: return 1u;
        default: return 0u;
    }
}

/// Returns whether the runtime tensor operators currently support this dtype.
[[nodiscard]] constexpr auto tensor_element_type_supported(TensorElementType t) noexcept {
    return t == TensorElementType::F16 ||
           t == TensorElementType::F32 ||
           t == TensorElementType::I32;
}

/**
 * @brief Host-side descriptor of a dense, strided tensor layout.
 *
 * Mirrors the cuTENSOR descriptor / ATen `TensorImpl` metadata: dtypes,
 * extents and strides are decoupled from the backing storage, so views share
 * storage and only differ in metadata. Strides and the storage offset are
 * counted in elements, not bytes.
 */
struct TensorDescriptor {

    TensorElementType dtype{TensorElementType::F32};
    uint32_t rank{1u};
    std::array<uint32_t, tensor_max_rank> extents{1u, 1u, 1u, 1u};
    std::array<uint32_t, tensor_max_rank> strides{1u, 1u, 1u, 1u};
    uint32_t storage_offset{0u};

    /// Create a contiguous (row-major) descriptor for the given shape.
    [[nodiscard]] static TensorDescriptor contiguous(
        TensorElementType dtype, luisa::span<const uint32_t> shape) noexcept {
        LUISA_ASSERT(!shape.empty() && shape.size() <= tensor_max_rank,
                     "Tensor rank ({}) must be in [1, {}].",
                     shape.size(), tensor_max_rank);
        LUISA_ASSERT(tensor_element_type_supported(dtype),
                     "Tensor element type '{}' is not supported by the "
                     "runtime tensor operators.",
                     tensor_element_type_name(dtype));
        TensorDescriptor d{};
        d.dtype = dtype;
        d.rank = static_cast<uint32_t>(shape.size());
        uint32_t stride = 1u;
        for (auto i = shape.size(); i-- > 0u;) {
            d.extents[i] = shape[i];
            d.strides[i] = stride;
            stride *= shape[i];
        }
        return d;
    }

    /// Overload accepting a braced shape list, e.g. contiguous(F32, {4, 5}).
    [[nodiscard]] static TensorDescriptor contiguous(
        TensorElementType dtype, std::initializer_list<uint32_t> shape) noexcept {
        return contiguous(dtype, luisa::span<const uint32_t>{shape.begin(), shape.size()});
    }

    /// Number of logical elements.
    [[nodiscard]] auto numel() const noexcept {
        return std::accumulate(extents.begin(), extents.begin() + rank,
                               1ull, std::multiplies<>{});
    }

    /// Row-major contiguity check.
    [[nodiscard]] auto is_contiguous() const noexcept {
        uint32_t stride = 1u;
        for (auto i = rank; i-- > 0u;) {
            if (strides[i] != stride) { return false; }
            stride *= extents[i];
        }
        return true;
    }

    /// Size in bytes of the storage region touched by this descriptor,
    /// including the storage offset.
    [[nodiscard]] auto storage_size_bytes() const noexcept {
        uint64_t max_offset = storage_offset;
        for (auto i = 0u; i < rank; i++) {
            max_offset += static_cast<uint64_t>(extents[i] - 1u) * strides[i];
        }
        return (max_offset + 1u) * tensor_element_size(dtype);
    }

    [[nodiscard]] auto shape() const noexcept {
        return luisa::span{extents.data(), rank};
    }
    [[nodiscard]] auto stride_span() const noexcept {
        return luisa::span{strides.data(), rank};
    }

    /// Metadata-only view with permuted dimensions.
    [[nodiscard]] TensorDescriptor permuted(luisa::span<const uint32_t> perm) const noexcept {
        LUISA_ASSERT(perm.size() == rank,
                     "Permutation rank ({}) must equal the tensor rank ({}).",
                     perm.size(), rank);
        std::array<bool, tensor_max_rank> seen{};
        TensorDescriptor d = *this;
        for (auto i = 0u; i < rank; i++) {
            LUISA_ASSERT(perm[i] < rank && !seen[perm[i]],
                         "Invalid permutation.");
            seen[perm[i]] = true;
            d.extents[i] = extents[perm[i]];
            d.strides[i] = strides[perm[i]];
        }
        return d;
    }

    /// Metadata-only reshape. Requires the descriptor to be contiguous.
    [[nodiscard]] TensorDescriptor reshaped(luisa::span<const uint32_t> shape) const noexcept {
        LUISA_ASSERT(is_contiguous() && storage_offset == 0u,
                     "Reshape requires a contiguous tensor.");
        auto d = contiguous(dtype, shape);
        LUISA_ASSERT(d.numel() == numel(),
                     "Reshape must preserve the element count ({} vs {}).",
                     d.numel(), numel());
        return d;
    }

    [[nodiscard]] auto operator==(const TensorDescriptor &) const noexcept -> bool = default;
};

/// Epilogue tags for `TENSOR_MATMUL` (carried verbatim to the backend).
enum struct TensorMatmulEpilogue : uint32_t {
    NONE = 0u,
    RELU = 1u,
};

/// Options of a `tensor_matmul` call (plan.md §2.2 `GemmOptions`).
struct GemmOptions {
    float alpha{1.0f};
    float beta{0.0f};
    bool trans_a{false};
    bool trans_b{false};
    /// Compute dtype tag; for F16 inputs the accumulator is F32
    /// (the FP32-accumulator rule of plan.md §A.3).
    TensorElementType compute_dtype{TensorElementType::F32};
    TensorMatmulEpilogue epilogue{TensorMatmulEpilogue::NONE};
};

}// namespace luisa::compute
