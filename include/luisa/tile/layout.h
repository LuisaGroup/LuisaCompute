#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/dimension.h>

namespace luisa::compute::tile {

namespace detail {
struct IndexExprNode;
}// namespace detail

class IndexMap;

class LUISA_TILE_API IndexExpr final {

private:
    friend class IndexMap;
    luisa::shared_ptr<const detail::IndexExprNode> _node;

    explicit IndexExpr(luisa::shared_ptr<const detail::IndexExprNode> node) noexcept
        : _node{std::move(node)} {}
    [[nodiscard]] static luisa::shared_ptr<const detail::IndexExprNode> _substitute_node(
        const detail::IndexExprNode *node,
        const IndexSpace &variables,
        luisa::span<const IndexExpr> replacements) noexcept;
    [[nodiscard]] IndexExpr _substitute(const IndexSpace &variables, luisa::span<const IndexExpr> replacements) const noexcept;

public:
    IndexExpr() noexcept = default;
    IndexExpr(const IndexExpr &) noexcept = default;
    IndexExpr(IndexExpr &&) noexcept = default;
    IndexExpr &operator=(const IndexExpr &) noexcept = default;
    IndexExpr &operator=(IndexExpr &&) noexcept = default;
    ~IndexExpr() noexcept;

    [[nodiscard]] static IndexExpr constant(int64_t value) noexcept;
    [[nodiscard]] static IndexExpr coordinate(Dim dimension) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept { return _node != nullptr; }
    [[nodiscard]] bool verify(const IndexSpace &domain) const noexcept;
    [[nodiscard]] luisa::optional<int64_t> evaluate(const IndexSpace &domain, luisa::span<const int64_t> point) const noexcept;

    friend LUISA_TILE_API IndexExpr operator+(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr operator-(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr operator*(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr floor_div(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr modulo(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr bit_xor(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr bit_and(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr shift_left(IndexExpr lhs, IndexExpr rhs) noexcept;
    friend LUISA_TILE_API IndexExpr shift_right(IndexExpr lhs, IndexExpr rhs) noexcept;
};

struct LayoutProperties {
    bool enumerated{false};
    bool total{false};
    bool in_bounds{false};
    bool injective{false};
    bool surjective{false};
    uint64_t domain_points{0u};
    uint64_t codomain_points{0u};
};

class LUISA_TILE_API IndexMap final {

private:
    IndexSpace _domain;
    IndexSpace _codomain;
    luisa::vector<IndexExpr> _outputs;

public:
    IndexMap() noexcept = default;
    IndexMap(IndexSpace domain, IndexSpace codomain, luisa::span<const IndexExpr> outputs) noexcept;

    [[nodiscard]] const IndexSpace &domain() const noexcept { return _domain; }
    [[nodiscard]] const IndexSpace &codomain() const noexcept { return _codomain; }
    [[nodiscard]] luisa::span<const IndexExpr> outputs() const noexcept { return _outputs; }
    [[nodiscard]] bool verify() const noexcept;
    [[nodiscard]] luisa::optional<luisa::vector<int64_t>> apply(luisa::span<const int64_t> point) const noexcept;
    [[nodiscard]] LayoutProperties analyze_finite(uint64_t max_points = 1024u * 1024u) const noexcept;

    [[nodiscard]] static IndexMap identity(const IndexSpace &space) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> compose(const IndexMap &outer, const IndexMap &inner) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> permute(const IndexSpace &domain, luisa::span<const Dim> order) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> reshape(const IndexSpace &domain, const IndexSpace &codomain) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> strided(const IndexSpace &domain, Dim storage_dimension, luisa::span<const uint64_t> strides) noexcept;
};

}// namespace luisa::compute::tile
