#include <luisa/tile/target.h>

namespace luisa::compute::tile {

void TargetModel::_resize_contains(size_t old_count) noexcept {
    auto new_count = _execution_scope_names.size();
    luisa::vector<uint8_t> resized(new_count * new_count, uint8_t{0u});
    for (auto i = 0u; i < old_count; i++) {
        for (auto j = 0u; j < old_count; j++) {
            resized[i * new_count + j] = _execution_contains[i * old_count + j];
        }
    }
    for (auto i = 0u; i < new_count; i++) { resized[i * new_count + i] = 1u; }
    _execution_contains = std::move(resized);
}

void TargetModel::_resize_access(size_t old_scope_count, size_t old_resource_count) noexcept {
    auto scope_count = _execution_scope_names.size();
    auto resource_count = _resource_class_names.size();
    auto kind_count = static_cast<size_t>(MemoryAccessKind::COUNT);
    luisa::vector<uint8_t> resized(scope_count * resource_count * kind_count, uint8_t{0u});
    for (auto s = 0u; s < old_scope_count; s++) {
        for (auto r = 0u; r < old_resource_count; r++) {
            for (auto k = 0u; k < kind_count; k++) {
                auto old_index = (s * old_resource_count + r) * kind_count + k;
                auto new_index = (s * resource_count + r) * kind_count + k;
                resized[new_index] = _access_capabilities[old_index];
            }
        }
    }
    _access_capabilities = std::move(resized);
}

void TargetModel::_close_contains() noexcept {
    auto count = _execution_scope_names.size();
    for (auto k = 0u; k < count; k++) {
        for (auto i = 0u; i < count; i++) {
            if (_execution_contains[i * count + k] == 0u) { continue; }
            for (auto j = 0u; j < count; j++) {
                _execution_contains[i * count + j] |= _execution_contains[k * count + j];
            }
        }
    }
}

size_t TargetModel::_access_index(ExecutionScope scope, ResourceClass resource, MemoryAccessKind kind) const noexcept {
    auto resource_count = _resource_class_names.size();
    auto kind_count = static_cast<size_t>(MemoryAccessKind::COUNT);
    return (static_cast<size_t>(scope.index()) * resource_count + resource.index()) * kind_count + static_cast<size_t>(kind);
}

ExecutionScope TargetModel::add_execution_scope(luisa::string_view name) noexcept {
    if (name.empty()) { return {}; }
    if (auto existing = find_execution_scope(name)) { return *existing; }
    auto old_scope_count = _execution_scope_names.size();
    auto old_resource_count = _resource_class_names.size();
    _execution_scope_names.emplace_back(name.data(), name.size());
    _resize_contains(old_scope_count);
    _resize_access(old_scope_count, old_resource_count);
    return ExecutionScope{this, static_cast<uint32_t>(_execution_scope_names.size() - 1u)};
}

ResourceClass TargetModel::add_resource_class(luisa::string_view name) noexcept {
    if (name.empty()) { return {}; }
    if (auto existing = find_resource_class(name)) { return *existing; }
    auto old_scope_count = _execution_scope_names.size();
    auto old_resource_count = _resource_class_names.size();
    _resource_class_names.emplace_back(name.data(), name.size());
    _resize_access(old_scope_count, old_resource_count);
    return ResourceClass{this, static_cast<uint32_t>(_resource_class_names.size() - 1u)};
}

bool TargetModel::add_contains(ExecutionScope parent, ExecutionScope child) noexcept {
    if (!owns(parent) || !owns(child)) { return false; }
    if (parent == child) { return true; }
    if (contains(child, parent)) { return false; }
    auto count = _execution_scope_names.size();
    _execution_contains[parent.index() * count + child.index()] = 1u;
    _close_contains();
    return true;
}

bool TargetModel::allow_access(ExecutionScope scope, ResourceClass resource, MemoryAccessKind kind) noexcept {
    if (!owns(scope) || !owns(resource) || kind >= MemoryAccessKind::COUNT) { return false; }
    _access_capabilities[_access_index(scope, resource, kind)] = 1u;
    return true;
}

bool TargetModel::owns(ExecutionScope scope) const noexcept {
    return scope._target == this && scope._index < _execution_scope_names.size();
}

bool TargetModel::owns(ResourceClass resource) const noexcept {
    return resource._target == this && resource._index < _resource_class_names.size();
}

bool TargetModel::contains(ExecutionScope parent, ExecutionScope child) const noexcept {
    if (!owns(parent) || !owns(child)) { return false; }
    auto count = _execution_scope_names.size();
    return _execution_contains[parent.index() * count + child.index()] != 0u;
}

bool TargetModel::can_access(ExecutionScope scope, ResourceClass resource, MemoryAccessKind kind) const noexcept {
    return owns(scope) && owns(resource) && kind < MemoryAccessKind::COUNT &&
           _access_capabilities[_access_index(scope, resource, kind)] != 0u;
}

luisa::optional<ExecutionScope> TargetModel::find_execution_scope(luisa::string_view name) const noexcept {
    for (auto i = 0u; i < _execution_scope_names.size(); i++) {
        if (luisa::string_view{_execution_scope_names[i]} == name) { return ExecutionScope{this, static_cast<uint32_t>(i)}; }
    }
    return luisa::nullopt;
}

luisa::optional<ResourceClass> TargetModel::find_resource_class(luisa::string_view name) const noexcept {
    for (auto i = 0u; i < _resource_class_names.size(); i++) {
        if (luisa::string_view{_resource_class_names[i]} == name) { return ResourceClass{this, static_cast<uint32_t>(i)}; }
    }
    return luisa::nullopt;
}

luisa::string_view TargetModel::name(ExecutionScope scope) const noexcept {
    return owns(scope) ? luisa::string_view{_execution_scope_names[scope.index()]} : luisa::string_view{};
}

luisa::string_view TargetModel::name(ResourceClass resource) const noexcept {
    return owns(resource) ? luisa::string_view{_resource_class_names[resource.index()]} : luisa::string_view{};
}

}// namespace luisa::compute::tile
