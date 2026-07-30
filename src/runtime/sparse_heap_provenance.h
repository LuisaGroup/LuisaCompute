#pragma once

#include <cstdint>

namespace luisa::compute::detail {

enum class SparseHeapProvenanceStatus : uint8_t {
    SUCCESS,
    INVALID_HEAP,
    DEVICE_MISMATCH
};

[[nodiscard]] constexpr const char *
sparse_heap_provenance_status_name(
    SparseHeapProvenanceStatus status) noexcept {
    switch (status) {
        case SparseHeapProvenanceStatus::SUCCESS: return "success";
        case SparseHeapProvenanceStatus::INVALID_HEAP: return "invalid heap";
        case SparseHeapProvenanceStatus::DEVICE_MISMATCH: return "device mismatch";
    }
    return "unknown";
}

[[nodiscard]] constexpr SparseHeapProvenanceStatus
validate_sparse_heap_provenance(
    bool heap_valid,
    const void *resource_device,
    const void *heap_device) noexcept {
    if (!heap_valid) {
        return SparseHeapProvenanceStatus::INVALID_HEAP;
    }
    if (resource_device == nullptr ||
        heap_device == nullptr ||
        resource_device != heap_device) {
        return SparseHeapProvenanceStatus::DEVICE_MISMATCH;
    }
    return SparseHeapProvenanceStatus::SUCCESS;
}

}// namespace luisa::compute::detail
