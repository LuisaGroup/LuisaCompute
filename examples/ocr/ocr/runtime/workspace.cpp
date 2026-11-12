// OCR engine -- grow-only aligned activation arenas (Phase 2).
#include "ocr/runtime/workspace.h"

#include <limits>
#include <utility>

#include "ocr/quantization.h"

namespace luisa::ocr {

[[nodiscard]] Status WorkspacePlan::add_bytes(luisa::string name, uint64_t bytes) {
    if (bytes == 0u) {
        return Status::error(StatusCode::invalid_argument, "workspace region must be nonzero", name);
    }
    uint64_t aligned = 0;
    if (!checked_align_up(bytes, kWorkspaceAlignment, aligned)) {
        return Status::error(StatusCode::invalid_argument, "workspace region size overflows alignment", name);
    }
    uint64_t offset = 0, total = 0;
    if (!checked_align_up(_total, kWorkspaceAlignment, offset) ||
        !checked_add(offset, aligned, total)) {
        return Status::error(StatusCode::invalid_argument, "workspace plan overflows uint64", name);
    }
    WorkspaceRegion region;
    region.name = std::move(name);
    region.offset = offset;
    region.bytes = bytes;
    _regions.emplace_back(std::move(region));
    _total = total;
    return Status::success();
}

[[nodiscard]] Status WorkspacePlan::add_tensor(luisa::string name, TensorDesc tensor) {
    auto v = tensor.normalize();
    if (!v.ok()) { return v; }
    v = tensor.validate();
    if (!v.ok()) { return v; }
    auto count = tensor.element_count();
    if (!count.ok()) { return count.status; }
    auto bytes = tensor_byte_size(tensor.dtype, *count.value);
    if (!bytes.ok()) { return bytes.status; }
    auto s = add_bytes(std::move(name), *bytes.value);
    if (!s.ok()) { return s; }
    _regions.back().tensor = std::move(tensor);
    return Status::success();
}

[[nodiscard]] const WorkspaceRegion *WorkspacePlan::find(luisa::string_view name) const noexcept {
    for (const auto &region : _regions) {
        if (region.name == name) { return &region; }
    }
    return nullptr;
}

[[nodiscard]] Status DeviceWorkspace::reserve(uint64_t capacity_bytes) {
    if (_device == nullptr) {
        return Status::error(StatusCode::invalid_argument, "workspace has no device", "device");
    }
    if (capacity_bytes <= _capacity) { return Status::success(); }// grow-only
    uint64_t aligned = 0;
    if (!checked_align_up(capacity_bytes, kWorkspaceAlignment, aligned)) {
        return Status::error(StatusCode::invalid_argument, "workspace capacity overflows alignment", "capacity");
    }
    if (aligned > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return Status::error(StatusCode::out_of_memory, "workspace capacity exceeds size_t", "capacity");
    }
    auto new_buffer = _device->create_byte_buffer(static_cast<size_t>(aligned));
    if (!new_buffer) {
        return Status::error(StatusCode::out_of_memory, "failed to allocate workspace buffer", "capacity");
    }
    if (_buffer) {
        // Keep the old buffer alive until the caller proves the stream drained.
        _retired.emplace_back(std::move(_buffer));
    }
    _buffer = std::move(new_buffer);
    _capacity = aligned;
    _cursor = 0u;
    _generation++;
    return Status::success();
}

[[nodiscard]] Result<WorkspaceAllocation> DeviceWorkspace::allocate(uint64_t bytes) {
    using R = Result<WorkspaceAllocation>;
    if (bytes == 0u) {
        return R::failure(StatusCode::invalid_argument, "allocation must be nonzero", "bytes");
    }
    uint64_t aligned = 0, offset = 0, end = 0;
    if (!checked_align_up(bytes, kWorkspaceAlignment, aligned) ||
        !checked_align_up(_cursor, kWorkspaceAlignment, offset) ||
        !checked_add(offset, aligned, end)) {
        return R::failure(StatusCode::invalid_argument, "allocation overflows uint64", "bytes");
    }
    if (end > _capacity) {
        return R::failure(StatusCode::out_of_memory,
                          "workspace exhausted; reserve() between requests (grow-only)", "bytes");
    }
    WorkspaceAllocation alloc;
    alloc.offset = offset;
    alloc.bytes = bytes;
    alloc.generation = _generation;
    alloc.epoch = _epoch;
    _cursor = end;
    return R::success(alloc);
}

[[nodiscard]] Result<WorkspaceAllocation> DeviceWorkspace::allocate_tensor(const TensorDesc &tensor) {
    using R = Result<WorkspaceAllocation>;
    auto v = tensor.validate();
    if (!v.ok()) { return R::failure(v); }
    auto count = tensor.element_count();
    if (!count.ok()) { return R::failure(count.status); }
    auto bytes = tensor_byte_size(tensor.dtype, *count.value);
    if (!bytes.ok()) { return R::failure(bytes.status); }
    return allocate(*bytes.value);
}

void DeviceWorkspace::reset() noexcept {
    // Caller contract: no in-flight command references any region. The buffer
    // itself is untouched; the cursor rewinds and the epoch advances so
    // post-reset allocations are distinguishable from stale pre-reset ones.
    _cursor = 0u;
    _epoch++;
}

void DeviceWorkspace::retire_after_sync() noexcept {
    // The owning stream has drained: retired buffers are no longer referenced
    // by any command and can be destroyed (LuisaCompute defers the actual
    // device-side destruction safely).
    _retired.clear();
}

[[nodiscard]] uint64_t DeviceWorkspace::resident_bytes() const noexcept {
    uint64_t total = _capacity;
    for (const auto &buffer : _retired) {
        uint64_t next = 0;
        if (!checked_add(total, buffer.size_bytes(), next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

}// namespace luisa::ocr