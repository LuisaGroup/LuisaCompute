#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

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

enum class IndexExprKind : uint8_t {
    INVALID,
    CONSTANT,
    COORDINATE,
    ADD,
    SUBTRACT,
    MULTIPLY,
    FLOOR_DIVIDE,
    MODULO,
    BIT_XOR,
    BIT_AND,
    SHIFT_LEFT,
    SHIFT_RIGHT
};

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
    // Immutable DAG inspection for analyses and native bridges. These expose
    // typed operations, not a serialization or an opaque user callback.
    [[nodiscard]] IndexExprKind kind() const noexcept;
    [[nodiscard]] luisa::optional<int64_t> constant_value() const noexcept;
    [[nodiscard]] Dim dimension() const noexcept;
    [[nodiscard]] IndexExpr lhs() const noexcept;
    [[nodiscard]] IndexExpr rhs() const noexcept;
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

enum class ProofStatus : uint8_t {
    UNKNOWN,
    PROVEN,
    DISPROVEN
};

struct LayoutProof {
    // Same predicates as analyze_finite(): total means checked apply()
    // succeeds throughout the declared domain, including codomain bounds.
    ProofStatus total{ProofStatus::UNKNOWN};
    ProofStatus in_bounds{ProofStatus::UNKNOWN};
    ProofStatus injective{ProofStatus::UNKNOWN};
    ProofStatus surjective{ProofStatus::UNKNOWN};
    bool enumerated{false};

    [[nodiscard]] bool is_storage_safe() const noexcept {
        return total == ProofStatus::PROVEN && in_bounds == ProofStatus::PROVEN && injective == ProofStatus::PROVEN;
    }
    [[nodiscard]] bool is_storage_invalid() const noexcept {
        return total == ProofStatus::DISPROVEN || in_bounds == ProofStatus::DISPROVEN || injective == ProofStatus::DISPROVEN;
    }
};

class LUISA_TILE_API IndexMap final {

private:
    IndexSpace _domain;
    IndexSpace _codomain;
    luisa::vector<IndexExpr> _outputs;

    [[nodiscard]] LayoutProof _prove_bit_linear() const noexcept;

public:
    IndexMap() noexcept = default;
    IndexMap(IndexSpace domain, IndexSpace codomain, luisa::span<const IndexExpr> outputs) noexcept;

    [[nodiscard]] const IndexSpace &domain() const noexcept { return _domain; }
    [[nodiscard]] const IndexSpace &codomain() const noexcept { return _codomain; }
    [[nodiscard]] luisa::span<const IndexExpr> outputs() const noexcept { return _outputs; }
    [[nodiscard]] bool verify() const noexcept;
    [[nodiscard]] luisa::optional<luisa::vector<int64_t>> apply(luisa::span<const int64_t> point) const noexcept;
    // Checked affine and GF(2) proofs, followed by bounded exhaustive fallback.
    // Passing zero disables enumeration; UNKNOWN is never a safety proof.
    [[nodiscard]] LayoutProof prove(uint64_t max_fallback_points = 1024u * 1024u) const noexcept;
    // Kept independent of prove() as an exact small-domain semantic oracle.
    [[nodiscard]] LayoutProperties analyze_finite(uint64_t max_points = 1024u * 1024u) const noexcept;

    [[nodiscard]] static IndexMap identity(const IndexSpace &space) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> compose(const IndexMap &outer, const IndexMap &inner) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> permute(const IndexSpace &domain, luisa::span<const Dim> order) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> reshape(const IndexSpace &domain, const IndexSpace &codomain) noexcept;
    [[nodiscard]] static luisa::optional<IndexMap> strided(const IndexSpace &domain, Dim storage_dimension, luisa::span<const uint64_t> strides) noexcept;
};

// A layout correspondence is represented by a span F and two ordinary maps:
//
//   left  : F -> logical coordinates
//   right : F -> physical coordinates
//
// Unlike an inverse layout map, this representation is exact for replication,
// broadcast, and non-bijective placement. The fiber witnesses in F are an IR
// construction detail; placements() returns a set and removes duplicate
// witnesses.
struct LayoutCorrespondenceProperties {
    bool enumerated{false};
    bool total{false};
    bool covers_logical_space{false};
    uint64_t fiber_points{0u};
    uint64_t logical_points{0u};
    uint64_t physical_points{0u};
    // Number of distinct physical placements per logical point. Duplicate
    // fiber witnesses do not count as extra replicas.
    uint64_t minimum_replication{0u};
    uint64_t maximum_replication{0u};
};

class LUISA_TILE_API LayoutCorrespondence final {

private:
    IndexMap _left;
    IndexMap _right;

public:
    LayoutCorrespondence() noexcept = default;
    LayoutCorrespondence(IndexMap left, IndexMap right) noexcept
        : _left{std::move(left)}, _right{std::move(right)} {}

    [[nodiscard]] const IndexSpace &fiber_space() const noexcept { return _left.domain(); }
    [[nodiscard]] const IndexSpace &logical_space() const noexcept { return _left.codomain(); }
    [[nodiscard]] const IndexSpace &physical_space() const noexcept { return _right.codomain(); }
    [[nodiscard]] const IndexMap &left_leg() const noexcept { return _left; }
    [[nodiscard]] const IndexMap &right_leg() const noexcept { return _right; }

    [[nodiscard]] bool verify() const noexcept;
    [[nodiscard]] LayoutCorrespondence converse() const noexcept { return LayoutCorrespondence{_right, _left}; }
    [[nodiscard]] luisa::optional<luisa::vector<luisa::vector<int64_t>>> placements(
        luisa::span<const int64_t> logical_point,
        uint64_t max_fiber_points = 1024u * 1024u) const noexcept;
    [[nodiscard]] LayoutCorrespondenceProperties analyze_finite(
        uint64_t max_fiber_points = 1024u * 1024u) const noexcept;
};

}// namespace luisa::compute::tile
