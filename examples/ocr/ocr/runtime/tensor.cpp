// OCR engine -- portable tensor metadata (Phase 2). Host-only.
#include "ocr/runtime/tensor.h"

#include <algorithm>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view tensor_dtype_name(TensorDType dtype) noexcept {
    switch (dtype) {
        case TensorDType::f32: return "f32";
        case TensorDType::u32: return "u32";
        case TensorDType::i32: return "i32";
        case TensorDType::f16: return "f16";
        case TensorDType::bf16: return "bf16";
        default: return "unknown";
    }
}

[[nodiscard]] uint64_t tensor_dtype_size(TensorDType dtype) noexcept {
    switch (dtype) {
        case TensorDType::f32:
        case TensorDType::u32:
        case TensorDType::i32: return 4u;
        case TensorDType::f16:
        case TensorDType::bf16: return 2u;
        default: return 0u;
    }
}

[[nodiscard]] bool is_f32_compute_dtype(TensorDType dtype) noexcept {
    return dtype == TensorDType::f32;
}

[[nodiscard]] luisa::string_view tensor_layout_tag_name(TensorLayoutTag tag) noexcept {
    switch (tag) {
        case TensorLayoutTag::row_major: return "row_major";
        case TensorLayoutTag::kn: return "kn";
        case TensorLayoutTag::nk: return "nk";
        case TensorLayoutTag::nchw: return "nchw";
        case TensorLayoutTag::chw: return "chw";
        case TensorLayoutTag::nhwc: return "nhwc";
        case TensorLayoutTag::tokens_hidden: return "tokens_hidden";
        default: return "unknown";
    }
}

[[nodiscard]] Result<uint64_t> TensorDesc::element_count() const {
    uint64_t count = 0;
    if (!checked_element_count(shape, count)) {
        return Result<uint64_t>::failure(StatusCode::invalid_argument,
                                         "tensor element count overflows uint64", "shape");
    }
    return Result<uint64_t>::success(count);
}

[[nodiscard]] Result<luisa::vector<uint64_t>> TensorDesc::row_major_strides(luisa::span<const uint64_t> shape) {
    using R = Result<luisa::vector<uint64_t>>;
    if (shape.size() > kTensorMaxRank) {
        return R::failure(StatusCode::invalid_argument, "tensor rank exceeds the maximum", "shape");
    }
    luisa::vector<uint64_t> strides(shape.size(), 1u);
    uint64_t acc = 1u;
    for (size_t i = shape.size(); i > 0u; i--) {
        strides[i - 1u] = acc;
        if (!checked_mul(acc, shape[i - 1u], acc)) {
            return R::failure(StatusCode::invalid_argument, "row-major stride overflows uint64", "shape");
        }
    }
    return R::success(std::move(strides));
}

[[nodiscard]] Status TensorDesc::normalize() {
    if (shape.size() > kTensorMaxRank) {
        return Status::error(StatusCode::invalid_argument, "tensor rank exceeds the maximum", "shape");
    }
    if (tensor_dtype_size(dtype) == 0u) {
        return Status::error(StatusCode::invalid_argument, "unknown tensor dtype", "dtype");
    }
    if (strides.empty()) {
        auto s = row_major_strides(shape);
        if (!s.ok()) { return s.status; }
        strides = std::move(*s.value);
    } else if (strides.size() != shape.size()) {
        return Status::error(StatusCode::invalid_argument, "stride count does not match rank", "strides");
    }
    return Status::success();
}

[[nodiscard]] Status TensorDesc::validate() const {
    if (shape.size() > kTensorMaxRank) {
        return Status::error(StatusCode::invalid_argument, "tensor rank exceeds the maximum", "shape");
    }
    if (tensor_dtype_size(dtype) == 0u) {
        return Status::error(StatusCode::invalid_argument, "unknown tensor dtype", "dtype");
    }
    if (strides.size() != shape.size()) {
        return Status::error(StatusCode::invalid_argument, "stride count does not match rank", "strides");
    }
    // Extent check: sum((dim-1)*stride) + 1 must fit uint64. Empty tensors
    // (any zero dim) have extent 0 and are always valid.
    uint64_t extent = 0u;
    bool empty = false;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0u) { empty = true; break; }
        uint64_t term = 0;
        if (!checked_mul(shape[i] - 1u, strides[i], term) ||
            !checked_add(extent, term, extent)) {
            return Status::error(StatusCode::invalid_argument, "tensor extent overflows uint64", "shape");
        }
    }
    if (!empty && !checked_add(extent, 1u, extent)) {
        return Status::error(StatusCode::invalid_argument, "tensor extent overflows uint64", "shape");
    }
    return Status::success();
}

[[nodiscard]] bool TensorDesc::is_contiguous() const {
    if (strides.size() != shape.size()) { return false; }
    uint64_t acc = 1u;
    for (size_t i = shape.size(); i > 0u; i--) {
        if (shape[i - 1u] == 0u) { return true; }// empty tensors are contiguous
        if (strides[i - 1u] != acc) { return false; }
        if (!checked_mul(acc, shape[i - 1u], acc)) { return false; }
    }
    return true;
}

[[nodiscard]] Result<uint64_t> TensorDesc::linear_offset(luisa::span<const uint64_t> index) const {
    using R = Result<uint64_t>;
    if (index.size() != shape.size()) {
        return R::failure(StatusCode::invalid_argument, "index rank does not match tensor rank", "index");
    }
    uint64_t offset = 0u;
    for (size_t i = 0; i < index.size(); i++) {
        if (index[i] >= shape[i]) {
            return R::failure(StatusCode::invalid_argument, "tensor index out of bounds", "index");
        }
        uint64_t term = 0;
        if (!checked_mul(index[i], strides[i], term) ||
            !checked_add(offset, term, offset)) {
            return R::failure(StatusCode::invalid_argument, "tensor offset overflows uint64", "index");
        }
    }
    return R::success(offset);
}

[[nodiscard]] Result<uint64_t> TensorDesc::strided_extent() const {
    using R = Result<uint64_t>;
    if (strides.size() != shape.size()) {
        return R::failure(StatusCode::invalid_argument, "stride count does not match rank", "strides");
    }
    uint64_t extent = 0u;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0u) { return R::success(0u); }// empty tensor
        uint64_t term = 0;
        if (!checked_mul(shape[i] - 1u, strides[i], term) ||
            !checked_add(extent, term, extent)) {
            return R::failure(StatusCode::invalid_argument, "strided extent overflows uint64", "shape");
        }
    }
    if (!checked_add(extent, 1u, extent)) {
        return R::failure(StatusCode::invalid_argument, "strided extent overflows uint64", "shape");
    }
    return R::success(extent);
}

[[nodiscard]] Result<uint64_t> tensor_byte_size(TensorDType dtype, uint64_t count) {
    using R = Result<uint64_t>;
    const auto es = tensor_dtype_size(dtype);
    if (es == 0u) {
        return R::failure(StatusCode::invalid_argument, "unknown tensor dtype", "dtype");
    }
    uint64_t bytes = 0;
    if (!checked_mul(count, es, bytes)) {
        return R::failure(StatusCode::invalid_argument, "tensor byte size overflows uint64", "count");
    }
    return R::success(bytes);
}

[[nodiscard]] Result<uint64_t> BufferViewDesc::byte_size() const {
    // S4: the view footprint is the STRIDED extent, not the logical element
    // count -- a padded view touches elements beyond count-1.
    auto extent = tensor.strided_extent();
    if (!extent.ok()) { return Result<uint64_t>::failure(extent.status); }
    return tensor_byte_size(tensor.dtype, *extent.value);
}

[[nodiscard]] Status BufferViewDesc::validate_within(uint64_t buffer_bytes) const {
    auto v = tensor.validate();
    if (!v.ok()) { return v; }
    auto bytes = byte_size();
    if (!bytes.ok()) { return bytes.status; }
    uint64_t end = 0;
    if (!checked_add(byte_offset, *bytes.value, end) || end > buffer_bytes) {
        return Status::error(StatusCode::invalid_argument,
                             "buffer view range exceeds the owning buffer", "view");
    }
    if (byte_offset % tensor_dtype_size(tensor.dtype) != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "buffer view offset is not element-aligned", "view");
    }
    return Status::success();
}

[[nodiscard]] Result<bool> BufferViewDesc::overlaps(const BufferViewDesc &other) const {
    using R = Result<bool>;
    auto a = byte_size();
    if (!a.ok()) { return R::failure(a.status); }
    auto b = other.byte_size();
    if (!b.ok()) { return R::failure(b.status); }
    if (*a.value == 0u || *b.value == 0u) { return R::success(false); }
    uint64_t a_end = 0, b_end = 0;
    if (!checked_add(byte_offset, *a.value, a_end) ||
        !checked_add(other.byte_offset, *b.value, b_end)) {
        return R::failure(StatusCode::invalid_argument, "view range overflows uint64", "view");
    }
    return R::success(byte_offset < b_end && other.byte_offset < a_end);
}

[[nodiscard]] Result<TensorDesc> make_matrix_desc(uint64_t rows, uint64_t cols, TensorLayoutTag layout) {
    using R = Result<TensorDesc>;
    TensorDesc desc;
    desc.dtype = TensorDType::f32;
    desc.layout = layout;
    desc.shape = {rows, cols};
    auto s = TensorDesc::row_major_strides(desc.shape);
    if (!s.ok()) { return R::failure(s.status); }
    desc.strides = std::move(*s.value);
    auto v = desc.validate();
    if (!v.ok()) { return R::failure(v); }
    return R::success(std::move(desc));
}

}// namespace luisa::ocr