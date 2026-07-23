#include "sparse_residency_registry.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace lc::vk::detail {

namespace {

[[nodiscard]] bool checked_end(
    uint64_t offset, uint64_t size, uint64_t &end) noexcept {
    if (size == 0u ||
        offset > std::numeric_limits<uint64_t>::max() - size) {
        return false;
    }
    end = offset + size;
    return true;
}

[[nodiscard]] bool valid_buffer_range(
    SparseBufferResidencyRange range) noexcept {
    uint64_t end{};
    return checked_end(range.offset, range.size, end);
}

[[nodiscard]] bool valid_image_box(
    SparseImageResidencyBox const &box) noexcept {
    for (auto axis = 0u; axis < 3u; ++axis) {
        uint64_t end{};
        if (!checked_end(box.offset[axis], box.extent[axis], end)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] SparseBufferResidencyRange buffer_intersection(
    SparseBufferResidencyRange lhs,
    SparseBufferResidencyRange rhs) noexcept {
    auto lhs_end = lhs.offset + lhs.size;
    auto rhs_end = rhs.offset + rhs.size;
    auto begin = std::max(lhs.offset, rhs.offset);
    auto end = std::min(lhs_end, rhs_end);
    return end > begin ? SparseBufferResidencyRange{begin, end - begin} :
                         SparseBufferResidencyRange{};
}

void append_buffer_subtraction(
    std::vector<SparseResidencyMapping> &output,
    SparseResidencyMapping const &mapping,
    SparseBufferResidencyRange cut) noexcept {
    auto intersection = buffer_intersection(mapping.buffer, cut);
    if (intersection.size == 0u) {
        output.emplace_back(mapping);
        return;
    }
    auto mapping_end = mapping.buffer.offset + mapping.buffer.size;
    auto intersection_end = intersection.offset + intersection.size;
    if (mapping.buffer.offset < intersection.offset) {
        auto fragment = mapping;
        fragment.buffer.size =
            intersection.offset - mapping.buffer.offset;
        output.emplace_back(fragment);
    }
    if (intersection_end < mapping_end) {
        auto fragment = mapping;
        fragment.buffer.offset = intersection_end;
        fragment.buffer.size = mapping_end - intersection_end;
        output.emplace_back(fragment);
    }
}

struct ImageBounds {
    std::array<uint64_t, 3u> lower{};
    std::array<uint64_t, 3u> upper{};
};

[[nodiscard]] ImageBounds image_bounds(
    SparseImageResidencyBox const &box) noexcept {
    auto bounds = ImageBounds{.lower = box.offset};
    for (auto axis = 0u; axis < 3u; ++axis) {
        bounds.upper[axis] = box.offset[axis] + box.extent[axis];
    }
    return bounds;
}

[[nodiscard]] bool image_intersection(
    SparseImageResidencyBox const &lhs,
    SparseImageResidencyBox const &rhs,
    ImageBounds &intersection) noexcept {
    if (lhs.mip_level != rhs.mip_level) { return false; }
    auto lhs_bounds = image_bounds(lhs);
    auto rhs_bounds = image_bounds(rhs);
    for (auto axis = 0u; axis < 3u; ++axis) {
        intersection.lower[axis] =
            std::max(lhs_bounds.lower[axis], rhs_bounds.lower[axis]);
        intersection.upper[axis] =
            std::min(lhs_bounds.upper[axis], rhs_bounds.upper[axis]);
        if (intersection.lower[axis] >= intersection.upper[axis]) {
            return false;
        }
    }
    return true;
}

void append_image_fragment(
    std::vector<SparseResidencyMapping> &output,
    SparseResidencyMapping const &mapping,
    ImageBounds const &bounds) noexcept {
    auto fragment = mapping;
    fragment.image.offset = bounds.lower;
    for (auto axis = 0u; axis < 3u; ++axis) {
        fragment.image.extent[axis] =
            bounds.upper[axis] - bounds.lower[axis];
    }
    output.emplace_back(fragment);
}

void append_image_subtraction(
    std::vector<SparseResidencyMapping> &output,
    SparseResidencyMapping const &mapping,
    SparseImageResidencyBox const &cut) noexcept {
    auto source = image_bounds(mapping.image);
    ImageBounds intersection{};
    if (!image_intersection(mapping.image, cut, intersection)) {
        output.emplace_back(mapping);
        return;
    }

    // Split source \ intersection into at most six disjoint orthogonal slabs.
    if (source.lower[0u] < intersection.lower[0u]) {
        append_image_fragment(
            output, mapping,
            {.lower = source.lower,
             .upper = {intersection.lower[0u], source.upper[1u],
                       source.upper[2u]}});
    }
    if (intersection.upper[0u] < source.upper[0u]) {
        append_image_fragment(
            output, mapping,
            {.lower = {intersection.upper[0u], source.lower[1u],
                       source.lower[2u]},
             .upper = source.upper});
    }

    auto middle_x_lower = intersection.lower[0u];
    auto middle_x_upper = intersection.upper[0u];
    if (source.lower[1u] < intersection.lower[1u]) {
        append_image_fragment(
            output, mapping,
            {.lower = {middle_x_lower, source.lower[1u],
                       source.lower[2u]},
             .upper = {middle_x_upper, intersection.lower[1u],
                       source.upper[2u]}});
    }
    if (intersection.upper[1u] < source.upper[1u]) {
        append_image_fragment(
            output, mapping,
            {.lower = {middle_x_lower, intersection.upper[1u],
                       source.lower[2u]},
             .upper = {middle_x_upper, source.upper[1u],
                       source.upper[2u]}});
    }

    auto middle_y_lower = intersection.lower[1u];
    auto middle_y_upper = intersection.upper[1u];
    if (source.lower[2u] < intersection.lower[2u]) {
        append_image_fragment(
            output, mapping,
            {.lower = {middle_x_lower, middle_y_lower,
                       source.lower[2u]},
             .upper = {middle_x_upper, middle_y_upper,
                       intersection.lower[2u]}});
    }
    if (intersection.upper[2u] < source.upper[2u]) {
        append_image_fragment(
            output, mapping,
            {.lower = {middle_x_lower, middle_y_lower,
                       intersection.upper[2u]},
             .upper = {middle_x_upper, middle_y_upper,
                       source.upper[2u]}});
    }
}

[[nodiscard]] bool buffer_range_fully_covered(
    std::vector<SparseResidencyMapping> const &mappings,
    uint64_t resource,
    SparseBufferResidencyRange range) noexcept {
    std::vector<SparseBufferResidencyRange> uncovered{range};
    for (auto const &mapping : mappings) {
        if (mapping.resource != resource ||
            mapping.resource_kind !=
                SparseResidencyResourceKind::BUFFER) {
            continue;
        }
        std::vector<SparseBufferResidencyRange> next;
        next.reserve(uncovered.size() + 1u);
        for (auto const &remaining : uncovered) {
            auto temporary = SparseResidencyMapping{
                .resource_kind =
                    SparseResidencyResourceKind::BUFFER,
                .buffer = remaining};
            std::vector<SparseResidencyMapping> fragments;
            append_buffer_subtraction(
                fragments, temporary, mapping.buffer);
            for (auto const &fragment : fragments) {
                next.emplace_back(fragment.buffer);
            }
        }
        uncovered = std::move(next);
        if (uncovered.empty()) { return true; }
    }
    return uncovered.empty();
}

[[nodiscard]] bool image_box_fully_covered(
    std::vector<SparseResidencyMapping> const &mappings,
    uint64_t resource,
    SparseImageResidencyBox const &box) noexcept {
    std::vector<SparseImageResidencyBox> uncovered{box};
    for (auto const &mapping : mappings) {
        if (mapping.resource != resource ||
            mapping.resource_kind !=
                SparseResidencyResourceKind::IMAGE ||
            mapping.image.mip_level != box.mip_level) {
            continue;
        }
        std::vector<SparseImageResidencyBox> next;
        next.reserve(uncovered.size() + 5u);
        for (auto const &remaining : uncovered) {
            auto temporary = SparseResidencyMapping{
                .resource_kind = SparseResidencyResourceKind::IMAGE,
                .image = remaining};
            std::vector<SparseResidencyMapping> fragments;
            append_image_subtraction(
                fragments, temporary, mapping.image);
            for (auto const &fragment : fragments) {
                next.emplace_back(fragment.image);
            }
        }
        uncovered = std::move(next);
        if (uncovered.empty()) { return true; }
    }
    return uncovered.empty();
}

}// namespace

bool sparse_buffer_residency_ranges_overlap(
    SparseBufferResidencyRange lhs,
    SparseBufferResidencyRange rhs) noexcept {
    if (!valid_buffer_range(lhs) || !valid_buffer_range(rhs)) {
        return false;
    }
    return buffer_intersection(lhs, rhs).size != 0u;
}

bool sparse_image_residency_boxes_overlap(
    SparseImageResidencyBox const &lhs,
    SparseImageResidencyBox const &rhs) noexcept {
    if (!valid_image_box(lhs) || !valid_image_box(rhs)) {
        return false;
    }
    ImageBounds intersection{};
    return image_intersection(lhs, rhs, intersection);
}

const char *sparse_residency_registry_status_name(
    SparseResidencyRegistryStatus status) noexcept {
    switch (status) {
        case SparseResidencyRegistryStatus::SUCCESS: return "success";
        case SparseResidencyRegistryStatus::INVALID_HANDLE: return "invalid handle";
        case SparseResidencyRegistryStatus::INVALID_REGION: return "invalid region";
        case SparseResidencyRegistryStatus::HEAP_ALREADY_REGISTERED: return "heap already registered";
        case SparseResidencyRegistryStatus::HEAP_NOT_REGISTERED: return "heap not registered";
        case SparseResidencyRegistryStatus::RESOURCE_ALREADY_REGISTERED: return "resource already registered";
        case SparseResidencyRegistryStatus::RESOURCE_NOT_REGISTERED: return "resource not registered";
        case SparseResidencyRegistryStatus::RESOURCE_KIND_MISMATCH: return "resource kind mismatch";
        case SparseResidencyRegistryStatus::HEAP_REUSED_IN_BATCH: return "heap reused in batch";
        case SparseResidencyRegistryStatus::HEAP_ALREADY_ACTIVE: return "heap already active";
        case SparseResidencyRegistryStatus::RESOURCE_RANGE_ALREADY_MAPPED: return "resource range already mapped";
        case SparseResidencyRegistryStatus::UNMAP_RANGE_NOT_FULLY_MAPPED: return "unmap range is not fully mapped";
        case SparseResidencyRegistryStatus::HEAP_HAS_ACTIVE_MAPPINGS: return "heap has active mappings";
        case SparseResidencyRegistryStatus::RESOURCE_HAS_ACTIVE_MAPPINGS: return "resource has active mappings";
        case SparseResidencyRegistryStatus::TRANSACTION_INACTIVE: return "transaction inactive";
    }
    return "unknown";
}

SparseResidencyRegistry::Transaction::Transaction(
    SparseResidencyRegistry &registry) noexcept
    : _registry{&registry},
      _lock{registry._mutex},
      _candidate{registry._state} {}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::_fail(
    SparseResidencyRegistryStatus status,
    uint64_t resource,
    uint64_t heap) noexcept {
    if (_result) {
        _result = {
            .status = status,
            .resource = resource,
            .heap = heap};
    }
    return _result;
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::_validate(
    uint64_t resource,
    SparseResidencyResourceKind resource_kind,
    uint64_t heap,
    bool is_map) noexcept {
    if (!_result) { return _result; }
    if (_registry == nullptr || !_lock.owns_lock()) {
        return _fail(
            SparseResidencyRegistryStatus::TRANSACTION_INACTIVE,
            resource, heap);
    }
    if (resource == 0u || (is_map && heap == 0u)) {
        return _fail(
            SparseResidencyRegistryStatus::INVALID_HANDLE,
            resource, heap);
    }
    auto resource_iter = _candidate.resources.find(resource);
    if (resource_iter == _candidate.resources.end()) {
        return _fail(
            SparseResidencyRegistryStatus::RESOURCE_NOT_REGISTERED,
            resource, heap);
    }
    if (resource_iter->second != resource_kind) {
        return _fail(
            SparseResidencyRegistryStatus::RESOURCE_KIND_MISMATCH,
            resource, heap);
    }
    if (is_map) {
        if (_candidate.heaps.find(heap) == _candidate.heaps.end()) {
            return _fail(
                SparseResidencyRegistryStatus::HEAP_NOT_REGISTERED,
                resource, heap);
        }
        if (!_mapped_heaps.emplace(heap).second) {
            return _fail(
                SparseResidencyRegistryStatus::HEAP_REUSED_IN_BATCH,
                resource, heap);
        }
    }
    return {};
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::validate_resource(
    uint64_t resource,
    SparseResidencyResourceKind resource_kind) noexcept {
    return _validate(resource, resource_kind, 0u, false);
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::map_buffer(
    uint64_t resource,
    uint64_t heap,
    SparseBufferResidencyRange range) noexcept {
    if (auto result = _validate(
            resource, SparseResidencyResourceKind::BUFFER,
            heap, true);
        !result) {
        return result;
    }
    if (!valid_buffer_range(range)) {
        return _fail(
            SparseResidencyRegistryStatus::INVALID_REGION,
            resource, heap);
    }
    for (auto const &mapping : _candidate.mappings) {
        if (mapping.resource == resource &&
            mapping.resource_kind ==
                SparseResidencyResourceKind::BUFFER &&
            sparse_buffer_residency_ranges_overlap(
                mapping.buffer, range)) {
            return _fail(
                SparseResidencyRegistryStatus::RESOURCE_RANGE_ALREADY_MAPPED,
                resource, heap);
        }
    }
    // A heap that was live at the start of the batch cannot be recycled by an
    // unmap and map in the same VkBindSparseInfo: the native binds do not form
    // an ordered release/acquire sequence.
    if (std::any_of(
            _registry->_state.mappings.cbegin(),
            _registry->_state.mappings.cend(),
            [heap](auto const &mapping) noexcept {
                return mapping.heap == heap;
            })) {
        return _fail(
            SparseResidencyRegistryStatus::HEAP_ALREADY_ACTIVE,
            resource, heap);
    }
    _candidate.mappings.emplace_back(SparseResidencyMapping{
        .heap = heap,
        .resource = resource,
        .resource_kind = SparseResidencyResourceKind::BUFFER,
        .buffer = range});
    return {};
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::unmap_buffer(
    uint64_t resource,
    SparseBufferResidencyRange range) noexcept {
    if (auto result = _validate(
            resource, SparseResidencyResourceKind::BUFFER,
            0u, false);
        !result) {
        return result;
    }
    if (!valid_buffer_range(range)) {
        return _fail(
            SparseResidencyRegistryStatus::INVALID_REGION,
            resource, 0u);
    }
    if (!buffer_range_fully_covered(
            _candidate.mappings, resource, range)) {
        return _fail(
            SparseResidencyRegistryStatus::UNMAP_RANGE_NOT_FULLY_MAPPED,
            resource, 0u);
    }
    std::vector<SparseResidencyMapping> next;
    next.reserve(_candidate.mappings.size() + 1u);
    for (auto const &mapping : _candidate.mappings) {
        if (mapping.resource == resource &&
            mapping.resource_kind ==
                SparseResidencyResourceKind::BUFFER) {
            append_buffer_subtraction(next, mapping, range);
        } else {
            next.emplace_back(mapping);
        }
    }
    _candidate.mappings = std::move(next);
    return {};
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::map_image(
    uint64_t resource,
    uint64_t heap,
    SparseImageResidencyBox const &box) noexcept {
    if (auto result = _validate(
            resource, SparseResidencyResourceKind::IMAGE,
            heap, true);
        !result) {
        return result;
    }
    if (!valid_image_box(box)) {
        return _fail(
            SparseResidencyRegistryStatus::INVALID_REGION,
            resource, heap);
    }
    for (auto const &mapping : _candidate.mappings) {
        if (mapping.resource == resource &&
            mapping.resource_kind ==
                SparseResidencyResourceKind::IMAGE &&
            sparse_image_residency_boxes_overlap(
                mapping.image, box)) {
            return _fail(
                SparseResidencyRegistryStatus::RESOURCE_RANGE_ALREADY_MAPPED,
                resource, heap);
        }
    }
    if (std::any_of(
            _registry->_state.mappings.cbegin(),
            _registry->_state.mappings.cend(),
            [heap](auto const &mapping) noexcept {
                return mapping.heap == heap;
            })) {
        return _fail(
            SparseResidencyRegistryStatus::HEAP_ALREADY_ACTIVE,
            resource, heap);
    }
    _candidate.mappings.emplace_back(SparseResidencyMapping{
        .heap = heap,
        .resource = resource,
        .resource_kind = SparseResidencyResourceKind::IMAGE,
        .image = box});
    return {};
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::unmap_image(
    uint64_t resource,
    SparseImageResidencyBox const &box) noexcept {
    if (auto result = _validate(
            resource, SparseResidencyResourceKind::IMAGE,
            0u, false);
        !result) {
        return result;
    }
    if (!valid_image_box(box)) {
        return _fail(
            SparseResidencyRegistryStatus::INVALID_REGION,
            resource, 0u);
    }
    if (!image_box_fully_covered(
            _candidate.mappings, resource, box)) {
        return _fail(
            SparseResidencyRegistryStatus::UNMAP_RANGE_NOT_FULLY_MAPPED,
            resource, 0u);
    }
    std::vector<SparseResidencyMapping> next;
    next.reserve(_candidate.mappings.size() + 5u);
    for (auto const &mapping : _candidate.mappings) {
        if (mapping.resource == resource &&
            mapping.resource_kind ==
                SparseResidencyResourceKind::IMAGE) {
            append_image_subtraction(next, mapping, box);
        } else {
            next.emplace_back(mapping);
        }
    }
    _candidate.mappings = std::move(next);
    return {};
}

SparseResidencyRegistryResult
SparseResidencyRegistry::Transaction::commit() noexcept {
    if (!_result) { return _result; }
    if (_registry == nullptr || !_lock.owns_lock()) {
        return _fail(
            SparseResidencyRegistryStatus::TRANSACTION_INACTIVE,
            0u, 0u);
    }
    _registry->_state = std::move(_candidate);
    _registry = nullptr;
    _lock.unlock();
    return {};
}

SparseResidencyRegistryResult SparseResidencyRegistry::register_heap(
    uint64_t heap) noexcept {
    std::lock_guard lock{_mutex};
    if (heap == 0u) {
        return {
            .status = SparseResidencyRegistryStatus::INVALID_HANDLE,
            .heap = heap};
    }
    if (!_state.heaps.emplace(heap).second) {
        return {
            .status =
                SparseResidencyRegistryStatus::HEAP_ALREADY_REGISTERED,
            .heap = heap};
    }
    return {};
}

SparseResidencyRegistryResult SparseResidencyRegistry::unregister_heap(
    uint64_t heap) noexcept {
    std::lock_guard lock{_mutex};
    if (_state.heaps.find(heap) == _state.heaps.end()) {
        return {
            .status = SparseResidencyRegistryStatus::HEAP_NOT_REGISTERED,
            .heap = heap};
    }
    if (std::any_of(
            _state.mappings.cbegin(), _state.mappings.cend(),
            [heap](auto const &mapping) noexcept {
                return mapping.heap == heap;
            })) {
        return {
            .status =
                SparseResidencyRegistryStatus::HEAP_HAS_ACTIVE_MAPPINGS,
            .heap = heap};
    }
    _state.heaps.erase(heap);
    return {};
}

SparseResidencyRegistryResult SparseResidencyRegistry::register_resource(
    uint64_t resource,
    SparseResidencyResourceKind resource_kind) noexcept {
    std::lock_guard lock{_mutex};
    if (resource == 0u) {
        return {
            .status = SparseResidencyRegistryStatus::INVALID_HANDLE,
            .resource = resource};
    }
    if (!_state.resources.emplace(resource, resource_kind).second) {
        return {
            .status =
                SparseResidencyRegistryStatus::RESOURCE_ALREADY_REGISTERED,
            .resource = resource};
    }
    return {};
}

SparseResidencyRegistryResult SparseResidencyRegistry::unregister_resource(
    uint64_t resource) noexcept {
    std::lock_guard lock{_mutex};
    if (_state.resources.find(resource) == _state.resources.end()) {
        return {
            .status = SparseResidencyRegistryStatus::RESOURCE_NOT_REGISTERED,
            .resource = resource};
    }
    if (std::any_of(
            _state.mappings.cbegin(), _state.mappings.cend(),
            [resource](auto const &mapping) noexcept {
                return mapping.resource == resource;
            })) {
        return {
            .status =
                SparseResidencyRegistryStatus::RESOURCE_HAS_ACTIVE_MAPPINGS,
            .resource = resource};
    }
    _state.resources.erase(resource);
    return {};
}

SparseResidencyRegistry::Transaction
SparseResidencyRegistry::begin_transaction() noexcept {
    return Transaction{*this};
}

std::vector<SparseResidencyMapping>
SparseResidencyRegistry::mapping_snapshot() const noexcept {
    std::lock_guard lock{_mutex};
    return _state.mappings;
}

bool SparseResidencyRegistry::contains_heap(uint64_t heap) const noexcept {
    std::lock_guard lock{_mutex};
    return _state.heaps.find(heap) != _state.heaps.end();
}

bool SparseResidencyRegistry::contains_resource(
    uint64_t resource) const noexcept {
    std::lock_guard lock{_mutex};
    return _state.resources.find(resource) != _state.resources.end();
}

bool SparseResidencyRegistry::heap_is_active(uint64_t heap) const noexcept {
    std::lock_guard lock{_mutex};
    return std::any_of(
        _state.mappings.cbegin(), _state.mappings.cend(),
        [heap](auto const &mapping) noexcept {
            return mapping.heap == heap;
        });
}

bool SparseResidencyRegistry::resource_is_active(
    uint64_t resource) const noexcept {
    std::lock_guard lock{_mutex};
    return std::any_of(
        _state.mappings.cbegin(), _state.mappings.cend(),
        [resource](auto const &mapping) noexcept {
            return mapping.resource == resource;
        });
}

}// namespace lc::vk::detail
