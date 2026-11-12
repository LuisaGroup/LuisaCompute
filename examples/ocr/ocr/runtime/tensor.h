#pragma once
// OCR engine -- portable tensor metadata (Phase 2).
//
// Host-only logical tensor descriptors: shape/stride/dtype/layout plus a
// NON-OWNING BufferView descriptor that records where a logical tensor lives
// inside a device buffer. No host-side element values are stored here ("no
// hidden host values"): all data stays in device buffers or caller-owned
// staging spans. All element/byte/range math is checked 64-bit host
// arithmetic; converting to the 32-bit kernel envelope goes through
// runtime/dispatch.h (checked_u32), never through a silent truncation.
//
// Storage vs compute precision are explicit and separate:
// - TensorDType describes the COMPUTE element type of a logical tensor. The
//   Phase 2 correctness baseline computes in FP32 (activations, accumulators).
// - Weight storage encodings (FP16/BF16 packed, grouped INT4/INT8) are
//   described by ocr/package.h TensorEncoding + QuantKind and live in
//   runtime/weights.h; a packed weight buffer's word count is NOT its logical
//   element count. Nothing here reinterprets one storage format as another.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/quantization.h"
#include "ocr/status.h"

namespace luisa::ocr {

inline constexpr uint32_t kTensorMaxRank = 8u;

// Compute element types of logical tensors. Phase 2 kernels consume/produce
// fp32 (and uint32/int32 index buffers); half compute types are declared for
// later phases and rejected by the fp32 factories.
enum class TensorDType : uint32_t {
    f32 = 0,
    u32 = 1,
    i32 = 2,
    f16 = 3,// declared; NOT a Phase 2 compute type
    bf16 = 4, // declared; NOT a Phase 2 compute type
};

[[nodiscard]] luisa::string_view tensor_dtype_name(TensorDType dtype) noexcept;
// Bytes per COMPUTE element (storage encodings are a separate concept).
[[nodiscard]] uint64_t tensor_dtype_size(TensorDType dtype) noexcept;
[[nodiscard]] bool is_f32_compute_dtype(TensorDType dtype) noexcept;

// Semantic layout tags, aligned with the package layout tags (ocr/package.h).
enum class TensorLayoutTag : uint32_t {
    row_major = 0,
    kn = 1,       // linear weight [K, N]
    nk = 2,       // transposed linear weight [N, K]
    nchw = 3,     // [N, C, H, W]
    chw = 4,      // [C, H, W]
    nhwc = 5,     // [N, H, W, C]
    tokens_hidden = 6,// [T, H] packed token activations
};

[[nodiscard]] luisa::string_view tensor_layout_tag_name(TensorLayoutTag tag) noexcept;

// Logical tensor shape with explicit element strides. Rank <= kTensorMaxRank.
// Strides may describe non-contiguous views; every offset computation is
// checked. A zero dimension makes the tensor empty (element_count == 0).
struct TensorDesc {
    TensorDType dtype{TensorDType::f32};
    TensorLayoutTag layout{TensorLayoutTag::row_major};
    luisa::vector<uint64_t> shape;
    luisa::vector<uint64_t> strides;// element strides; empty = row-major

    [[nodiscard]] uint32_t rank() const noexcept { return static_cast<uint32_t>(shape.size()); }
    // Checked product of dimensions (empty shape = scalar = 1; any 0 dim = 0).
    [[nodiscard]] Result<uint64_t> element_count() const;
    // Row-major (contiguous) element strides for `shape`.
    [[nodiscard]] static Result<luisa::vector<uint64_t>> row_major_strides(luisa::span<const uint64_t> shape);
    // Fill `strides` with row-major strides when empty; validate rank/size.
    [[nodiscard]] Status normalize();
    // Validate rank, stride count, and that the described extent fits in
    // uint64 (max linear index + 1). Does not require contiguity.
    [[nodiscard]] Status validate() const;
    [[nodiscard]] bool is_contiguous() const;
    // Checked linear element offset of a multi-index (bounds-checked).
    [[nodiscard]] Result<uint64_t> linear_offset(luisa::span<const uint64_t> index) const;
    // Checked STRIDED extent: max linear element index + 1 (== element_count
    // for contiguous tensors, larger for padded/strided ones). Empty = 0.
    [[nodiscard]] Result<uint64_t> strided_extent() const;
};

// NON-OWNING descriptor of a logical tensor inside a device buffer.
// `byte_offset` is the absolute byte offset into the owning buffer; the
// referenced buffer must outlive every command that binds this view. This is
// a host descriptor only -- binding to a kernel goes through the typed
// compute::BufferView<T> the caller creates from the owning buffer.
struct BufferViewDesc {
    TensorDesc tensor;
    uint64_t byte_offset{0};

    // Checked total byte extent. Uses the STRIDED extent (max indexed
    // element + 1), so padded/strided views reserve their full footprint,
    // not just element_count * dtype size.
    [[nodiscard]] Result<uint64_t> byte_size() const;
    // Validate the view against an owning buffer of `buffer_bytes` bytes.
    [[nodiscard]] Status validate_within(uint64_t buffer_bytes) const;
    // True when [byte_offset, byte_offset+byte_size) intersects `other`.
    // Empty views never overlap. Used by aliasing checks (legal in-place
    // elementwise ops vs illegal overlapping GEMM operands).
    [[nodiscard]] Result<bool> overlaps(const BufferViewDesc &other) const;
};

// Checked byte size of `count` compute elements of `dtype`.
[[nodiscard]] Result<uint64_t> tensor_byte_size(TensorDType dtype, uint64_t count);

// Convenience: contiguous fp32 [rows, cols] descriptor.
[[nodiscard]] Result<TensorDesc> make_matrix_desc(uint64_t rows, uint64_t cols,
                                                  TensorLayoutTag layout = TensorLayoutTag::row_major);

}// namespace luisa::ocr