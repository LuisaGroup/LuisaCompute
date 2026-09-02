#include <limits>

#include <luisa/tile/dimension.h>

namespace luisa::compute::tile {

Dim DimensionContext::create_dimension(luisa::string_view name) noexcept {
    auto index = static_cast<uint32_t>(_dimension_names.size());
    _dimension_names.emplace_back(name.data(), name.size());
    return Dim{this, index};
}

DynamicExtent DimensionContext::create_dynamic_extent(luisa::string_view name) noexcept {
    auto index = static_cast<uint32_t>(_dynamic_extent_names.size());
    _dynamic_extent_names.emplace_back(name.data(), name.size());
    return DynamicExtent{this, index};
}

bool DimensionContext::owns(Dim dimension) const noexcept {
    return dimension._context == this && dimension._index < _dimension_names.size();
}

bool DimensionContext::owns(DynamicExtent extent) const noexcept {
    return extent._context == this && extent._index < _dynamic_extent_names.size();
}

luisa::string_view DimensionContext::name(Dim dimension) const noexcept {
    return owns(dimension) ? luisa::string_view{_dimension_names[dimension._index]} : luisa::string_view{};
}

luisa::string_view DimensionContext::name(DynamicExtent extent) const noexcept {
    return owns(extent) ? luisa::string_view{_dynamic_extent_names[extent._index]} : luisa::string_view{};
}

IndexSpace::IndexSpace(luisa::span<const IndexAxis> axes) noexcept
    : _axes{axes.begin(), axes.end()} {}

bool IndexSpace::add(Dim dimension, Extent extent) noexcept {
    if (!dimension || !extent.is_valid() || contains(dimension)) { return false; }
    if (extent.is_dynamic() && extent.dynamic_value().context() != dimension.context()) { return false; }
    _axes.emplace_back(IndexAxis{dimension, extent});
    return true;
}

bool IndexSpace::is_valid() const noexcept {
    const DimensionContext *context = nullptr;
    for (auto i = 0u; i < _axes.size(); i++) {
        auto &&axis = _axes[i];
        if (!axis.dimension || !axis.extent.is_valid()) { return false; }
        if (context == nullptr) {
            context = axis.dimension.context();
        } else if (axis.dimension.context() != context) {
            return false;
        }
        if (axis.extent.is_dynamic() && axis.extent.dynamic_value().context() != context) { return false; }
        for (auto j = 0u; j < i; j++) {
            if (_axes[j].dimension == axis.dimension) { return false; }
        }
    }
    return true;
}

bool IndexSpace::contains(Dim dimension) const noexcept {
    return axis_index(dimension).has_value();
}

luisa::optional<size_t> IndexSpace::axis_index(Dim dimension) const noexcept {
    for (auto i = 0u; i < _axes.size(); i++) {
        if (_axes[i].dimension == dimension) { return i; }
    }
    return luisa::nullopt;
}

luisa::optional<uint64_t> IndexSpace::static_volume() const noexcept {
    uint64_t volume = 1u;
    for (auto &&axis : _axes) {
        if (!axis.extent.is_constant()) { return luisa::nullopt; }
        auto extent = axis.extent.constant_value();
        if (extent != 0u && volume > std::numeric_limits<uint64_t>::max() / extent) { return luisa::nullopt; }
        volume *= extent;
    }
    return volume;
}

}// namespace luisa::compute::tile
