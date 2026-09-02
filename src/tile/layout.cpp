#include <limits>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/layout.h>

namespace luisa::compute::tile {

namespace detail {

enum class IndexExprKind : uint8_t {
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

struct IndexExprNode {
    IndexExprKind kind{IndexExprKind::CONSTANT};
    int64_t constant{0};
    Dim dimension;
    luisa::shared_ptr<const IndexExprNode> lhs;
    luisa::shared_ptr<const IndexExprNode> rhs;
};

[[nodiscard]] auto make_constant(int64_t value) noexcept {
    auto node = luisa::make_shared<IndexExprNode>();
    node->constant = value;
    return node;
}

[[nodiscard]] auto make_coordinate(Dim dimension) noexcept {
    auto node = luisa::make_shared<IndexExprNode>();
    node->kind = IndexExprKind::COORDINATE;
    node->dimension = dimension;
    return node;
}

[[nodiscard]] auto make_binary(IndexExprKind kind,
                               luisa::shared_ptr<const IndexExprNode> lhs,
                               luisa::shared_ptr<const IndexExprNode> rhs) noexcept {
    if (lhs == nullptr || rhs == nullptr) { return luisa::shared_ptr<const IndexExprNode>{}; }
    auto node = luisa::make_shared<IndexExprNode>();
    node->kind = kind;
    node->lhs = std::move(lhs);
    node->rhs = std::move(rhs);
    return luisa::shared_ptr<const IndexExprNode>{std::move(node)};
}

[[nodiscard]] bool verify_expr(const IndexExprNode *node, const IndexSpace &domain) noexcept {
    if (node == nullptr) { return false; }
    switch (node->kind) {
        case IndexExprKind::CONSTANT: return true;
        case IndexExprKind::COORDINATE: return domain.contains(node->dimension);
        default: return verify_expr(node->lhs.get(), domain) && verify_expr(node->rhs.get(), domain);
    }
}

[[nodiscard]] luisa::optional<int64_t> checked_add(int64_t lhs, int64_t rhs) noexcept {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) { return luisa::nullopt; }
    return lhs + rhs;
}

[[nodiscard]] luisa::optional<int64_t> checked_subtract(int64_t lhs, int64_t rhs) noexcept {
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
        (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs)) { return luisa::nullopt; }
    return lhs - rhs;
}

[[nodiscard]] luisa::optional<int64_t> checked_multiply(int64_t lhs, int64_t rhs) noexcept {
    if (lhs == 0 || rhs == 0) { return 0; }
    if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
        (rhs == -1 && lhs == std::numeric_limits<int64_t>::min())) { return luisa::nullopt; }
    if (lhs > 0) {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs)) { return luisa::nullopt; }
    } else {
        if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
            (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs)) { return luisa::nullopt; }
    }
    return lhs * rhs;
}

[[nodiscard]] luisa::optional<int64_t> floor_quotient(int64_t lhs, int64_t rhs) noexcept {
    if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) { return luisa::nullopt; }
    auto quotient = lhs / rhs;
    auto remainder = lhs % rhs;
    if (remainder != 0 && ((remainder < 0) != (rhs < 0))) { quotient--; }
    return quotient;
}

[[nodiscard]] luisa::optional<int64_t> floor_remainder(int64_t lhs, int64_t rhs) noexcept {
    auto quotient = floor_quotient(lhs, rhs);
    if (!quotient) { return luisa::nullopt; }
    auto product = checked_multiply(*quotient, rhs);
    return product ? checked_subtract(lhs, *product) : luisa::nullopt;
}

[[nodiscard]] luisa::optional<int64_t> evaluate_expr(const IndexExprNode *node,
                                                     const IndexSpace &domain,
                                                     luisa::span<const int64_t> point) noexcept {
    if (node == nullptr) { return luisa::nullopt; }
    switch (node->kind) {
        case IndexExprKind::CONSTANT: return node->constant;
        case IndexExprKind::COORDINATE: {
            auto index = domain.axis_index(node->dimension);
            return index && *index < point.size() ? luisa::optional<int64_t>{point[*index]} : luisa::nullopt;
        }
        default: break;
    }
    auto lhs = evaluate_expr(node->lhs.get(), domain, point);
    auto rhs = evaluate_expr(node->rhs.get(), domain, point);
    if (!lhs || !rhs) { return luisa::nullopt; }
    switch (node->kind) {
        case IndexExprKind::ADD: return checked_add(*lhs, *rhs);
        case IndexExprKind::SUBTRACT: return checked_subtract(*lhs, *rhs);
        case IndexExprKind::MULTIPLY: return checked_multiply(*lhs, *rhs);
        case IndexExprKind::FLOOR_DIVIDE: return floor_quotient(*lhs, *rhs);
        case IndexExprKind::MODULO: return floor_remainder(*lhs, *rhs);
        case IndexExprKind::BIT_XOR: return *lhs ^ *rhs;
        case IndexExprKind::BIT_AND: return *lhs & *rhs;
        case IndexExprKind::SHIFT_LEFT:
            if (*rhs < 0 || *rhs >= 64) { return luisa::nullopt; }
            return static_cast<int64_t>(static_cast<uint64_t>(*lhs) << static_cast<uint64_t>(*rhs));
        case IndexExprKind::SHIFT_RIGHT:
            if (*rhs < 0 || *rhs >= 64) { return luisa::nullopt; }
            return static_cast<int64_t>(static_cast<uint64_t>(*lhs) >> static_cast<uint64_t>(*rhs));
        default: break;
    }
    return luisa::nullopt;
}

[[nodiscard]] bool decode_point(uint64_t linear, const IndexSpace &space, luisa::vector<int64_t> &point) noexcept {
    point.resize(space.rank());
    for (auto i = space.rank(); i != 0u; i--) {
        auto extent = space.axis(i - 1u).extent;
        if (!extent.is_constant() || extent.constant_value() == 0u) { return false; }
        point[i - 1u] = static_cast<int64_t>(linear % extent.constant_value());
        linear /= extent.constant_value();
    }
    return linear == 0u;
}

[[nodiscard]] luisa::optional<uint64_t> encode_point(luisa::span<const int64_t> point, const IndexSpace &space) noexcept {
    if (point.size() != space.rank()) { return luisa::nullopt; }
    uint64_t linear = 0u;
    for (auto i = 0u; i < point.size(); i++) {
        auto extent = space.axis(i).extent;
        if (!extent.is_constant() || point[i] < 0 || static_cast<uint64_t>(point[i]) >= extent.constant_value()) { return luisa::nullopt; }
        auto n = extent.constant_value();
        if (n != 0u && linear > std::numeric_limits<uint64_t>::max() / n) { return luisa::nullopt; }
        linear *= n;
        if (linear > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(point[i])) { return luisa::nullopt; }
        linear += static_cast<uint64_t>(point[i]);
    }
    return linear;
}

}// namespace detail

IndexExpr::~IndexExpr() noexcept = default;

IndexExpr IndexExpr::constant(int64_t value) noexcept {
    return IndexExpr{detail::make_constant(value)};
}

IndexExpr IndexExpr::coordinate(Dim dimension) noexcept {
    return dimension ? IndexExpr{detail::make_coordinate(dimension)} : IndexExpr{};
}

bool IndexExpr::verify(const IndexSpace &domain) const noexcept {
    return domain.is_valid() && detail::verify_expr(_node.get(), domain);
}

luisa::optional<int64_t> IndexExpr::evaluate(const IndexSpace &domain, luisa::span<const int64_t> point) const noexcept {
    return point.size() == domain.rank() ? detail::evaluate_expr(_node.get(), domain, point) : luisa::nullopt;
}

IndexExpr IndexExpr::_substitute(const IndexSpace &variables, luisa::span<const IndexExpr> replacements) const noexcept {
    return IndexExpr{_substitute_node(_node.get(), variables, replacements)};
}

luisa::shared_ptr<const detail::IndexExprNode> IndexExpr::_substitute_node(
    const detail::IndexExprNode *node,
    const IndexSpace &variables,
    luisa::span<const IndexExpr> replacements) noexcept {
    if (node == nullptr) { return nullptr; }
    switch (node->kind) {
        case detail::IndexExprKind::CONSTANT: return detail::make_constant(node->constant);
        case detail::IndexExprKind::COORDINATE: {
            auto index = variables.axis_index(node->dimension);
            return index && *index < replacements.size() ? replacements[*index]._node : nullptr;
        }
        default: break;
    }
    return detail::make_binary(
        node->kind,
        _substitute_node(node->lhs.get(), variables, replacements),
        _substitute_node(node->rhs.get(), variables, replacements));
}

IndexExpr operator+(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::ADD, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr operator-(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::SUBTRACT, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr operator*(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::MULTIPLY, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr floor_div(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::FLOOR_DIVIDE, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr modulo(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::MODULO, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr bit_xor(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::BIT_XOR, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr bit_and(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::BIT_AND, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr shift_left(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::SHIFT_LEFT, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr shift_right(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(detail::IndexExprKind::SHIFT_RIGHT, std::move(lhs._node), std::move(rhs._node))};
}

IndexMap::IndexMap(IndexSpace domain, IndexSpace codomain, luisa::span<const IndexExpr> outputs) noexcept
    : _domain{std::move(domain)}, _codomain{std::move(codomain)}, _outputs{outputs.begin(), outputs.end()} {}

bool IndexMap::verify() const noexcept {
    if (!_domain.is_valid() || !_codomain.is_valid() || _outputs.size() != _codomain.rank()) { return false; }
    for (auto &&output : _outputs) {
        if (!output.verify(_domain)) { return false; }
    }
    return true;
}

luisa::optional<luisa::vector<int64_t>> IndexMap::apply(luisa::span<const int64_t> point) const noexcept {
    if (!verify() || point.size() != _domain.rank()) { return luisa::nullopt; }
    for (auto i = 0u; i < point.size(); i++) {
        if (point[i] < 0) { return luisa::nullopt; }
        auto extent = _domain.axis(i).extent;
        if (extent.is_constant() && static_cast<uint64_t>(point[i]) >= extent.constant_value()) { return luisa::nullopt; }
    }
    luisa::vector<int64_t> result;
    result.reserve(_outputs.size());
    for (auto i = 0u; i < _outputs.size(); i++) {
        auto value = _outputs[i].evaluate(_domain, point);
        if (!value || *value < 0) { return luisa::nullopt; }
        auto extent = _codomain.axis(i).extent;
        if (extent.is_constant() && static_cast<uint64_t>(*value) >= extent.constant_value()) { return luisa::nullopt; }
        result.emplace_back(*value);
    }
    return result;
}

LayoutProperties IndexMap::analyze_finite(uint64_t max_points) const noexcept {
    LayoutProperties properties;
    auto domain_volume = _domain.static_volume();
    auto codomain_volume = _codomain.static_volume();
    if (!verify() || !domain_volume || !codomain_volume || *domain_volume > max_points) { return properties; }
    properties.enumerated = true;
    properties.total = true;
    properties.in_bounds = true;
    properties.injective = true;
    properties.domain_points = *domain_volume;
    properties.codomain_points = *codomain_volume;
    luisa::unordered_set<uint64_t> images;
    luisa::vector<int64_t> point;
    for (uint64_t i = 0u; i < *domain_volume; i++) {
        if (!detail::decode_point(i, _domain, point)) {
            properties.total = false;
            properties.in_bounds = false;
            properties.injective = false;
            break;
        }
        auto mapped = apply(point);
        if (!mapped) {
            properties.total = false;
            properties.in_bounds = false;
            properties.injective = false;
            break;
        }
        auto linear = detail::encode_point(*mapped, _codomain);
        if (!linear) {
            properties.in_bounds = false;
            properties.injective = false;
            break;
        }
        if (!images.emplace(*linear).second) { properties.injective = false; }
    }
    properties.surjective = properties.total && properties.in_bounds && images.size() == *codomain_volume;
    return properties;
}

IndexMap IndexMap::identity(const IndexSpace &space) noexcept {
    luisa::vector<IndexExpr> outputs;
    outputs.reserve(space.rank());
    for (auto &&axis : space.axes()) { outputs.emplace_back(IndexExpr::coordinate(axis.dimension)); }
    return IndexMap{space, space, outputs};
}

luisa::optional<IndexMap> IndexMap::compose(const IndexMap &outer, const IndexMap &inner) noexcept {
    if (!outer.verify() || !inner.verify() || !(inner.codomain() == outer.domain())) { return luisa::nullopt; }
    luisa::vector<IndexExpr> outputs;
    outputs.reserve(outer._outputs.size());
    for (auto &&output : outer._outputs) {
        auto composed = output._substitute(outer._domain, inner._outputs);
        if (!composed) { return luisa::nullopt; }
        outputs.emplace_back(std::move(composed));
    }
    return IndexMap{inner._domain, outer._codomain, outputs};
}

luisa::optional<IndexMap> IndexMap::permute(const IndexSpace &domain, luisa::span<const Dim> order) noexcept {
    if (!domain.is_valid() || order.size() != domain.rank()) { return luisa::nullopt; }
    IndexSpace codomain;
    luisa::vector<IndexExpr> outputs;
    outputs.reserve(order.size());
    for (auto dimension : order) {
        auto index = domain.axis_index(dimension);
        if (!index || !codomain.add(dimension, domain.axis(*index).extent)) { return luisa::nullopt; }
        outputs.emplace_back(IndexExpr::coordinate(dimension));
    }
    return IndexMap{domain, std::move(codomain), outputs};
}

luisa::optional<IndexMap> IndexMap::reshape(const IndexSpace &domain, const IndexSpace &codomain) noexcept {
    auto domain_volume = domain.static_volume();
    auto codomain_volume = codomain.static_volume();
    if (!domain.is_valid() || !codomain.is_valid() || !domain_volume || !codomain_volume || *domain_volume == 0u || *domain_volume != *codomain_volume) { return luisa::nullopt; }
    auto linear = IndexExpr::constant(0);
    for (auto &&axis : domain.axes()) {
        linear = linear * IndexExpr::constant(static_cast<int64_t>(axis.extent.constant_value())) + IndexExpr::coordinate(axis.dimension);
    }
    luisa::vector<IndexExpr> outputs(codomain.rank());
    auto suffix = uint64_t{1u};
    for (auto i = codomain.rank(); i != 0u; i--) {
        auto extent = codomain.axis(i - 1u).extent.constant_value();
        if (extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return luisa::nullopt; }
        outputs[i - 1u] = modulo(floor_div(linear, IndexExpr::constant(static_cast<int64_t>(suffix))), IndexExpr::constant(static_cast<int64_t>(extent)));
        if (extent != 0u && suffix > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / extent) { return luisa::nullopt; }
        suffix *= extent;
    }
    return IndexMap{domain, codomain, outputs};
}

luisa::optional<IndexMap> IndexMap::strided(const IndexSpace &domain, Dim storage_dimension, luisa::span<const uint64_t> strides) noexcept {
    if (!domain.is_valid() || !storage_dimension || strides.size() != domain.rank()) { return luisa::nullopt; }
    auto offset = IndexExpr::constant(0);
    uint64_t storage_extent = domain.empty() ? 1u : 0u;
    bool empty_domain = false;
    for (auto i = 0u; i < domain.rank(); i++) {
        auto extent = domain.axis(i).extent;
        if (!extent.is_constant() || strides[i] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return luisa::nullopt; }
        offset = offset + IndexExpr::coordinate(domain.axis(i).dimension) * IndexExpr::constant(static_cast<int64_t>(strides[i]));
        if (extent.constant_value() == 0u) {
            empty_domain = true;
            continue;
        }
        if (strides[i] != 0u && extent.constant_value() - 1u > std::numeric_limits<uint64_t>::max() / strides[i]) { return luisa::nullopt; }
        auto term = (extent.constant_value() - 1u) * strides[i];
        if (storage_extent > std::numeric_limits<uint64_t>::max() - term) { return luisa::nullopt; }
        storage_extent += term;
    }
    if (empty_domain) {
        storage_extent = 0u;
    } else if (!domain.empty()) {
        if (storage_extent == std::numeric_limits<uint64_t>::max()) { return luisa::nullopt; }
        storage_extent++;
    }
    IndexSpace codomain;
    if (!codomain.add(storage_dimension, storage_extent)) { return luisa::nullopt; }
    IndexExpr outputs[]{std::move(offset)};
    return IndexMap{domain, std::move(codomain), outputs};
}

}// namespace luisa::compute::tile
