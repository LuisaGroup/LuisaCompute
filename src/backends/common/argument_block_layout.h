#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace lc {

enum class ArgumentBlockLayoutStatus : uint8_t {
    SUCCESS,
    INVALID_ALIGNMENT,
    INVALID_STRIDE,
    ALIGNMENT_OVERFLOW,
    ADDITION_OVERFLOW,
    MULTIPLICATION_OVERFLOW,
    LIMIT_EXCEEDED,
    INCOMPATIBLE_TRAILERS
};

[[nodiscard]] constexpr const char *argument_block_layout_status_name(
    ArgumentBlockLayoutStatus status) noexcept {
    switch (status) {
        case ArgumentBlockLayoutStatus::SUCCESS: return "success";
        case ArgumentBlockLayoutStatus::INVALID_ALIGNMENT: return "invalid alignment";
        case ArgumentBlockLayoutStatus::INVALID_STRIDE: return "nonempty array has zero stride";
        case ArgumentBlockLayoutStatus::ALIGNMENT_OVERFLOW: return "alignment overflow";
        case ArgumentBlockLayoutStatus::ADDITION_OVERFLOW: return "addition overflow";
        case ArgumentBlockLayoutStatus::MULTIPLICATION_OVERFLOW: return "multiplication overflow";
        case ArgumentBlockLayoutStatus::LIMIT_EXCEEDED: return "limit exceeded";
        case ArgumentBlockLayoutStatus::INCOMPATIBLE_TRAILERS: return "incompatible trailers";
    }
    return "unknown";
}

struct ArgumentBlockTrailerLayout {
    size_t metadata_count{};
    size_t metadata_stride{};
    size_t metadata_alignment{1u};
    size_t validation_count{};
    size_t validation_stride{};
    size_t validation_alignment{alignof(uint32_t)};
    size_t word_alignment{sizeof(uint32_t)};
};

struct ArgumentBlockTrailerPlacement {
    size_t metadata_offset{};
    size_t metadata_size{};
    size_t validation_offset{};
    size_t validation_size{};
    size_t final_size{};
};

class ArgumentBlockLayout {
private:
    size_t _size{};
    size_t _limit;
    ArgumentBlockLayoutStatus _status{ArgumentBlockLayoutStatus::SUCCESS};

    [[nodiscard]] constexpr bool _fail(
        ArgumentBlockLayoutStatus status) noexcept {
        if (_status == ArgumentBlockLayoutStatus::SUCCESS) {
            _status = status;
        }
        return false;
    }

public:
    explicit constexpr ArgumentBlockLayout(
        size_t limit = std::numeric_limits<size_t>::max()) noexcept
        : _limit{limit} {}

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return _status == ArgumentBlockLayoutStatus::SUCCESS;
    }
    [[nodiscard]] constexpr auto status() const noexcept { return _status; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto limit() const noexcept { return _limit; }

    [[nodiscard]] static constexpr bool is_valid_alignment(
        size_t alignment) noexcept {
        return alignment != 0u &&
               (alignment & (alignment - 1u)) == 0u;
    }

    [[nodiscard]] constexpr bool align_to(size_t alignment) noexcept {
        if (!static_cast<bool>(*this)) { return false; }
        if (!is_valid_alignment(alignment)) {
            return _fail(ArgumentBlockLayoutStatus::INVALID_ALIGNMENT);
        }
        auto mask = alignment - 1u;
        if (_size > std::numeric_limits<size_t>::max() - mask) {
            return _fail(ArgumentBlockLayoutStatus::ALIGNMENT_OVERFLOW);
        }
        auto aligned = (_size + mask) & ~mask;
        if (aligned > _limit) {
            return _fail(ArgumentBlockLayoutStatus::LIMIT_EXCEEDED);
        }
        _size = aligned;
        return true;
    }

    [[nodiscard]] constexpr bool append(
        size_t byte_size, size_t alignment,
        size_t &offset) noexcept {
        if (!static_cast<bool>(*this)) { return false; }
        if (!is_valid_alignment(alignment)) {
            return _fail(ArgumentBlockLayoutStatus::INVALID_ALIGNMENT);
        }
        auto mask = alignment - 1u;
        if (_size > std::numeric_limits<size_t>::max() - mask) {
            return _fail(ArgumentBlockLayoutStatus::ALIGNMENT_OVERFLOW);
        }
        auto aligned = (_size + mask) & ~mask;
        if (byte_size > std::numeric_limits<size_t>::max() - aligned) {
            return _fail(ArgumentBlockLayoutStatus::ADDITION_OVERFLOW);
        }
        auto end = aligned + byte_size;
        if (end > _limit) {
            return _fail(ArgumentBlockLayoutStatus::LIMIT_EXCEEDED);
        }
        offset = aligned;
        _size = end;
        return true;
    }

    [[nodiscard]] constexpr bool append_array(
        size_t count, size_t stride, size_t alignment,
        size_t &offset) noexcept {
        if (!static_cast<bool>(*this)) { return false; }
        if (!is_valid_alignment(alignment)) {
            return _fail(ArgumentBlockLayoutStatus::INVALID_ALIGNMENT);
        }
        if (count != 0u && stride == 0u) {
            return _fail(ArgumentBlockLayoutStatus::INVALID_STRIDE);
        }
        if (stride != 0u &&
            count > std::numeric_limits<size_t>::max() / stride) {
            return _fail(
                ArgumentBlockLayoutStatus::MULTIPLICATION_OVERFLOW);
        }
        return append(count * stride, alignment, offset);
    }

    [[nodiscard]] constexpr bool append_padded(
        size_t byte_size, size_t alignment,
        size_t &offset) noexcept {
        if (!static_cast<bool>(*this)) { return false; }
        auto candidate = *this;
        size_t candidate_offset = 0u;
        if (!candidate.append(
                byte_size, alignment, candidate_offset) ||
            !candidate.align_to(alignment)) {
            _status = candidate.status();
            return false;
        }
        _size = candidate.size();
        offset = candidate_offset;
        return true;
    }

    [[nodiscard]] constexpr bool finalize_words(
        size_t word_alignment = sizeof(uint32_t)) noexcept {
        return align_to(word_alignment);
    }

    [[nodiscard]] constexpr bool append_trailers(
        ArgumentBlockTrailerLayout trailer,
        ArgumentBlockTrailerPlacement &placement) noexcept {
        if (!static_cast<bool>(*this)) { return false; }
        if (trailer.metadata_count != 0u &&
            trailer.validation_count != 0u) {
            return _fail(
                ArgumentBlockLayoutStatus::INCOMPATIBLE_TRAILERS);
        }

        auto candidate = *this;
        ArgumentBlockTrailerPlacement candidate_placement{};
        if (trailer.metadata_count != 0u) {
            if (!candidate.append_array(
                    trailer.metadata_count, trailer.metadata_stride,
                    trailer.metadata_alignment,
                    candidate_placement.metadata_offset)) {
                _status = candidate.status();
                return false;
            }
            candidate_placement.metadata_size =
                candidate.size() - candidate_placement.metadata_offset;
        } else {
            candidate_placement.metadata_offset = candidate.size();
        }
        if (trailer.validation_count != 0u) {
            if (!candidate.append_array(
                    trailer.validation_count, trailer.validation_stride,
                    trailer.validation_alignment,
                    candidate_placement.validation_offset)) {
                _status = candidate.status();
                return false;
            }
            candidate_placement.validation_size =
                candidate.size() - candidate_placement.validation_offset;
        } else {
            candidate_placement.validation_offset = candidate.size();
        }
        if (!candidate.finalize_words(trailer.word_alignment)) {
            _status = candidate.status();
            return false;
        }
        candidate_placement.final_size = candidate.size();
        _size = candidate.size();
        placement = candidate_placement;
        return true;
    }
};

[[nodiscard]] constexpr bool argument_block_validation_value(
    size_t byte_size, size_t element_size,
    uint32_t &value) noexcept {
    if (element_size != 0u && byte_size % element_size != 0u) {
        return false;
    }
    auto logical_size = element_size == 0u ?
                            byte_size :
                            byte_size / element_size;
    if (logical_size > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(logical_size);
    return true;
}

}// namespace lc
