#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::tile {

class TargetModel;

class ExecutionScope final {

private:
    friend class TargetModel;
    const TargetModel *_target{nullptr};
    uint32_t _index{~0u};

    explicit ExecutionScope(const TargetModel *target, uint32_t index) noexcept
        : _target{target}, _index{index} {}

public:
    ExecutionScope() noexcept = default;

    [[nodiscard]] explicit operator bool() const noexcept { return _target != nullptr; }
    [[nodiscard]] auto target() const noexcept { return _target; }
    [[nodiscard]] auto index() const noexcept { return _index; }

    [[nodiscard]] friend bool operator==(ExecutionScope lhs, ExecutionScope rhs) noexcept {
        return lhs._target == rhs._target && lhs._index == rhs._index;
    }
};

class ResourceClass final {

private:
    friend class TargetModel;
    const TargetModel *_target{nullptr};
    uint32_t _index{~0u};

    explicit ResourceClass(const TargetModel *target, uint32_t index) noexcept
        : _target{target}, _index{index} {}

public:
    ResourceClass() noexcept = default;

    [[nodiscard]] explicit operator bool() const noexcept { return _target != nullptr; }
    [[nodiscard]] auto target() const noexcept { return _target; }
    [[nodiscard]] auto index() const noexcept { return _index; }

    [[nodiscard]] friend bool operator==(ResourceClass lhs, ResourceClass rhs) noexcept {
        return lhs._target == rhs._target && lhs._index == rhs._index;
    }
};

enum class MemoryAccessKind : uint8_t {
    LOAD,
    STORE,
    ATOMIC,
    COPY_SOURCE,
    COPY_DESTINATION,
    MMA_OPERAND,
    COUNT
};

class LUISA_TILE_API TargetModel final {

private:
    luisa::vector<luisa::string> _execution_scope_names;
    luisa::vector<uint8_t> _execution_contains;
    luisa::vector<luisa::string> _resource_class_names;
    luisa::vector<uint8_t> _access_capabilities;

private:
    void _resize_contains(size_t old_count) noexcept;
    void _resize_access(size_t old_scope_count, size_t old_resource_count) noexcept;
    void _close_contains() noexcept;
    [[nodiscard]] size_t _access_index(ExecutionScope scope, ResourceClass resource, MemoryAccessKind kind) const noexcept;

public:
    TargetModel() noexcept = default;
    TargetModel(TargetModel &&) noexcept = delete;
    TargetModel(const TargetModel &) noexcept = delete;
    TargetModel &operator=(TargetModel &&) noexcept = delete;
    TargetModel &operator=(const TargetModel &) noexcept = delete;
    ~TargetModel() noexcept = default;

    [[nodiscard]] ExecutionScope add_execution_scope(luisa::string_view name) noexcept;
    [[nodiscard]] ResourceClass add_resource_class(luisa::string_view name) noexcept;
    [[nodiscard]] bool add_contains(ExecutionScope parent, ExecutionScope child) noexcept;
    [[nodiscard]] bool allow_access(ExecutionScope scope, ResourceClass resource, MemoryAccessKind kind) noexcept;

    [[nodiscard]] bool owns(ExecutionScope scope) const noexcept;
    [[nodiscard]] bool owns(ResourceClass resource) const noexcept;
    [[nodiscard]] bool contains(ExecutionScope parent, ExecutionScope child) const noexcept;
    [[nodiscard]] bool can_access(ExecutionScope scope, ResourceClass resource, MemoryAccessKind kind) const noexcept;

    [[nodiscard]] luisa::optional<ExecutionScope> find_execution_scope(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::optional<ResourceClass> find_resource_class(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::string_view name(ExecutionScope scope) const noexcept;
    [[nodiscard]] luisa::string_view name(ResourceClass resource) const noexcept;
    [[nodiscard]] auto execution_scope_count() const noexcept { return _execution_scope_names.size(); }
    [[nodiscard]] auto resource_class_count() const noexcept { return _resource_class_names.size(); }
};

}// namespace luisa::compute::tile
