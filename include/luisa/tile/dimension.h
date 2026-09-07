#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::tile {

class DimensionContext;

class Dim final {

private:
    friend class DimensionContext;
    const DimensionContext *_context{nullptr};
    uint32_t _index{~0u};

    explicit Dim(const DimensionContext *context, uint32_t index) noexcept
        : _context{context}, _index{index} {}

public:
    Dim() noexcept = default;

    [[nodiscard]] explicit operator bool() const noexcept { return _context != nullptr; }
    [[nodiscard]] auto context() const noexcept { return _context; }
    [[nodiscard]] auto index() const noexcept { return _index; }

    [[nodiscard]] friend bool operator==(Dim lhs, Dim rhs) noexcept {
        return lhs._context == rhs._context && lhs._index == rhs._index;
    }
};

class DynamicExtent final {

private:
    friend class DimensionContext;
    const DimensionContext *_context{nullptr};
    uint32_t _index{~0u};

    explicit DynamicExtent(const DimensionContext *context, uint32_t index) noexcept
        : _context{context}, _index{index} {}

public:
    DynamicExtent() noexcept = default;

    [[nodiscard]] explicit operator bool() const noexcept { return _context != nullptr; }
    [[nodiscard]] auto context() const noexcept { return _context; }
    [[nodiscard]] auto index() const noexcept { return _index; }

    [[nodiscard]] friend bool operator==(DynamicExtent lhs, DynamicExtent rhs) noexcept {
        return lhs._context == rhs._context && lhs._index == rhs._index;
    }
};

class LUISA_TILE_API DimensionContext final {

private:
    luisa::vector<luisa::string> _dimension_names;
    luisa::vector<luisa::string> _dynamic_extent_names;

public:
    DimensionContext() noexcept = default;
    DimensionContext(DimensionContext &&) noexcept = delete;
    DimensionContext(const DimensionContext &) noexcept = delete;
    DimensionContext &operator=(DimensionContext &&) noexcept = delete;
    DimensionContext &operator=(const DimensionContext &) noexcept = delete;
    ~DimensionContext() noexcept = default;

    [[nodiscard]] Dim create_dimension(luisa::string_view name = {}) noexcept;
    [[nodiscard]] DynamicExtent create_dynamic_extent(luisa::string_view name = {}) noexcept;

    [[nodiscard]] bool owns(Dim dimension) const noexcept;
    [[nodiscard]] bool owns(DynamicExtent extent) const noexcept;
    [[nodiscard]] luisa::string_view name(Dim dimension) const noexcept;
    [[nodiscard]] luisa::string_view name(DynamicExtent extent) const noexcept;
    [[nodiscard]] auto dimension_count() const noexcept { return _dimension_names.size(); }
    [[nodiscard]] auto dynamic_extent_count() const noexcept { return _dynamic_extent_names.size(); }
};

class Extent final {

public:
    enum class Kind : uint8_t {
        INVALID,
        CONSTANT,
        DYNAMIC
    };

private:
    Kind _kind{Kind::INVALID};
    uint64_t _constant{0u};
    DynamicExtent _dynamic;

    explicit Extent(uint64_t value) noexcept
        : _kind{Kind::CONSTANT}, _constant{value} {}
    explicit Extent(DynamicExtent value) noexcept
        : _kind{Kind::DYNAMIC}, _dynamic{value} {}

public:
    Extent() noexcept = default;

    [[nodiscard]] static Extent constant(uint64_t value) noexcept { return Extent{value}; }
    [[nodiscard]] static Extent dynamic(DynamicExtent value) noexcept { return Extent{value}; }

    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] bool is_valid() const noexcept { return _kind != Kind::INVALID; }
    [[nodiscard]] bool is_constant() const noexcept { return _kind == Kind::CONSTANT; }
    [[nodiscard]] bool is_dynamic() const noexcept { return _kind == Kind::DYNAMIC; }
    [[nodiscard]] auto constant_value() const noexcept { return _constant; }
    [[nodiscard]] auto dynamic_value() const noexcept { return _dynamic; }

    [[nodiscard]] friend bool operator==(const Extent &lhs, const Extent &rhs) noexcept {
        if (lhs._kind != rhs._kind) { return false; }
        switch (lhs._kind) {
            case Kind::INVALID: return true;
            case Kind::CONSTANT: return lhs._constant == rhs._constant;
            case Kind::DYNAMIC: return lhs._dynamic == rhs._dynamic;
        }
        return false;
    }
};

struct IndexAxis {
    Dim dimension;
    Extent extent;

    [[nodiscard]] friend bool operator==(const IndexAxis &, const IndexAxis &) noexcept = default;
};

class LUISA_TILE_API IndexSpace final {

private:
    luisa::vector<IndexAxis> _axes;

public:
    IndexSpace() noexcept = default;
    explicit IndexSpace(luisa::span<const IndexAxis> axes) noexcept;

    [[nodiscard]] bool add(Dim dimension, Extent extent) noexcept;
    [[nodiscard]] bool add(Dim dimension, uint64_t extent) noexcept {
        return add(dimension, Extent::constant(extent));
    }

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return _axes.empty(); }
    [[nodiscard]] size_t rank() const noexcept { return _axes.size(); }
    [[nodiscard]] luisa::span<const IndexAxis> axes() const noexcept { return _axes; }
    [[nodiscard]] const IndexAxis &axis(size_t index) const noexcept { return _axes[index]; }
    [[nodiscard]] bool contains(Dim dimension) const noexcept;
    [[nodiscard]] luisa::optional<size_t> axis_index(Dim dimension) const noexcept;
    [[nodiscard]] luisa::optional<uint64_t> static_volume() const noexcept;

    [[nodiscard]] friend bool operator==(const IndexSpace &, const IndexSpace &) noexcept = default;
};

}// namespace luisa::compute::tile
