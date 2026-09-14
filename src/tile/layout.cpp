#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <numeric>
#include <utility>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/layout.h>

namespace luisa::compute::tile {

namespace detail {

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

[[nodiscard]] luisa::optional<int64_t> evaluate_binary(IndexExprKind kind, int64_t lhs, int64_t rhs) noexcept {
    switch (kind) {
        case IndexExprKind::ADD: return checked_add(lhs, rhs);
        case IndexExprKind::SUBTRACT: return checked_subtract(lhs, rhs);
        case IndexExprKind::MULTIPLY: return checked_multiply(lhs, rhs);
        case IndexExprKind::FLOOR_DIVIDE: return floor_quotient(lhs, rhs);
        case IndexExprKind::MODULO: return floor_remainder(lhs, rhs);
        case IndexExprKind::BIT_XOR: return lhs ^ rhs;
        case IndexExprKind::BIT_AND: return lhs & rhs;
        case IndexExprKind::SHIFT_LEFT:
            if (rhs < 0 || rhs >= 64) { return luisa::nullopt; }
            return static_cast<int64_t>(static_cast<uint64_t>(lhs) << static_cast<uint64_t>(rhs));
        case IndexExprKind::SHIFT_RIGHT:
            if (rhs < 0 || rhs >= 64) { return luisa::nullopt; }
            return static_cast<int64_t>(static_cast<uint64_t>(lhs) >> static_cast<uint64_t>(rhs));
        default: return luisa::nullopt;
    }
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
    return evaluate_binary(node->kind, *lhs, *rhs);
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

struct AffineForm {
    int64_t offset{0};
    luisa::vector<int64_t> coefficients;
    int64_t minimum{0};
    int64_t maximum{0};

    [[nodiscard]] bool is_constant() const noexcept {
        return std::all_of(coefficients.begin(), coefficients.end(), [](auto value) noexcept { return value == 0; });
    }
};

class AffineAnalyzer {

private:
    const IndexSpace &_domain;
    luisa::span<const int64_t> _limits;
    luisa::unordered_map<const IndexExprNode *, luisa::optional<AffineForm>> _cache;

    [[nodiscard]] luisa::optional<AffineForm> _finish(AffineForm form) const noexcept {
        form.minimum = form.offset;
        form.maximum = form.offset;
        for (auto i = 0u; i < _limits.size(); i++) {
            auto term = checked_multiply(form.coefficients[i], _limits[i]);
            if (!term) { return luisa::nullopt; }
            auto &bound = *term < 0 ? form.minimum : form.maximum;
            auto sum = checked_add(bound, *term);
            if (!sum) { return luisa::nullopt; }
            bound = *sum;
        }
        return form;
    }

    [[nodiscard]] luisa::optional<AffineForm> _compute(const IndexExprNode *node) noexcept {
        if (node == nullptr) { return luisa::nullopt; }
        AffineForm form;
        if (node->kind == IndexExprKind::CONSTANT || node->kind == IndexExprKind::COORDINATE) {
            form.coefficients.resize(_domain.rank(), 0);
            if (node->kind == IndexExprKind::CONSTANT) {
                form.offset = node->constant;
            } else {
                auto axis = _domain.axis_index(node->dimension);
                if (!axis) { return luisa::nullopt; }
                // A unit axis is identically zero on this specialized domain.
                form.coefficients[*axis] = _limits[*axis] == 0 ? 0 : 1;
            }
            return _finish(std::move(form));
        }
        if (node->kind != IndexExprKind::ADD && node->kind != IndexExprKind::SUBTRACT &&
            node->kind != IndexExprKind::MULTIPLY) { return luisa::nullopt; }
        auto lhs = analyze(node->lhs.get());
        auto rhs = analyze(node->rhs.get());
        // Every original subexpression must be safe before normalization can
        // cancel it. In particular, 0 * (1 / 0) is not the constant zero map.
        if (!lhs || !rhs) { return luisa::nullopt; }
        if (node->kind == IndexExprKind::MULTIPLY) {
            if (!rhs->is_constant()) { std::swap(lhs, rhs); }
            if (!rhs->is_constant()) { return luisa::nullopt; }
            auto factor = rhs->offset;
            auto offset = checked_multiply(lhs->offset, factor);
            if (!offset) { return luisa::nullopt; }
            lhs->offset = *offset;
            for (auto &coefficient : lhs->coefficients) {
                auto product = checked_multiply(coefficient, factor);
                if (!product) { return luisa::nullopt; }
                coefficient = *product;
            }
        } else {
            auto combine = node->kind == IndexExprKind::ADD ? checked_add : checked_subtract;
            auto offset = combine(lhs->offset, rhs->offset);
            if (!offset) { return luisa::nullopt; }
            lhs->offset = *offset;
            for (auto i = 0u; i < lhs->coefficients.size(); i++) {
                auto coefficient = combine(lhs->coefficients[i], rhs->coefficients[i]);
                if (!coefficient) { return luisa::nullopt; }
                lhs->coefficients[i] = *coefficient;
            }
        }
        return _finish(std::move(*lhs));
    }

public:
    AffineAnalyzer(const IndexSpace &domain, luisa::span<const int64_t> limits) noexcept
        : _domain{domain}, _limits{limits} {}

    [[nodiscard]] luisa::optional<AffineForm> analyze(const IndexExprNode *node) noexcept {
        if (auto iter = _cache.find(node); iter != _cache.end()) { return iter->second; }
        auto result = _compute(node);
        _cache.emplace(node, result);
        return result;
    }
};

// One 64-bit output word, affine over GF(2). Column i is the contribution
// of input bit i; offset is XOR translation, never integer addition.
// This is derived analysis state, not a second stored layout representation.
struct BitLinearForm {
    uint64_t offset{0u};
    luisa::vector<uint64_t> columns;

    [[nodiscard]] bool is_constant() const noexcept {
        return std::all_of(columns.begin(), columns.end(), [](auto value) noexcept { return value == 0u; });
    }

    [[nodiscard]] uint64_t possible_bits() const noexcept {
        auto bits = offset;
        for (auto column : columns) { bits |= column; }
        return bits;
    }

    [[nodiscard]] uint64_t maximum() const noexcept {
        std::array<uint64_t, 64u> pivots{};
        for (auto column : columns) {
            while (column != 0u) {
                auto pivot = std::bit_width(column) - 1u;
                if (pivots[pivot] == 0u) {
                    pivots[pivot] = column;
                    break;
                }
                column ^= pivots[pivot];
            }
        }
        // Greedy reduction in a highest-bit triangular XOR basis gives the
        // exact unsigned maximum of this affine image, including correlations.
        auto value = offset;
        for (auto i = pivots.size(); i != 0u; i--) { value = std::max(value, value ^ pivots[i - 1u]); }
        return value;
    }
};

class BitLinearAnalyzer {

private:
    const IndexSpace &_domain;
    luisa::vector<size_t> _bit_offsets;
    luisa::unordered_map<const IndexExprNode *, luisa::optional<BitLinearForm>> _cache;

    [[nodiscard]] BitLinearForm _constant(uint64_t value) const noexcept {
        return {value, luisa::vector<uint64_t>(input_bits(), 0u)};
    }

    template<typename F>
    [[nodiscard]] static BitLinearForm _transform(BitLinearForm form, F &&transform) noexcept {
        form.offset = transform(form.offset);
        for (auto &column : form.columns) { column = transform(column); }
        return form;
    }

    [[nodiscard]] luisa::optional<BitLinearForm> _compute(const IndexExprNode *node) noexcept {
        if (node == nullptr) { return luisa::nullopt; }
        if (node->kind == IndexExprKind::CONSTANT) { return _constant(static_cast<uint64_t>(node->constant)); }
        if (node->kind == IndexExprKind::COORDINATE) {
            auto axis = _domain.axis_index(node->dimension);
            if (!axis) { return luisa::nullopt; }
            auto form = _constant(0u);
            auto begin = _bit_offsets[*axis];
            auto end = _bit_offsets[*axis + 1u];
            for (auto i = begin; i < end; i++) { form.columns[i] = uint64_t{1u} << (i - begin); }
            return form;
        }
        auto lhs = analyze(node->lhs.get());
        auto rhs = analyze(node->rhs.get());
        // Prove every original subtree safe. Masking, XOR cancellation, and
        // multiplication by zero must not erase an undefined computation.
        if (!lhs || !rhs) { return luisa::nullopt; }
        if (lhs->is_constant() && rhs->is_constant()) {
            auto value = evaluate_binary(node->kind, std::bit_cast<int64_t>(lhs->offset), std::bit_cast<int64_t>(rhs->offset));
            return value ? luisa::optional<BitLinearForm>{_constant(static_cast<uint64_t>(*value))} : luisa::nullopt;
        }
        switch (node->kind) {
            case IndexExprKind::ADD:
                // Disjoint bit support rules out every carry, and hence both
                // signed overflow and any difference between addition and XOR.
                if ((lhs->possible_bits() & rhs->possible_bits()) != 0u) { return luisa::nullopt; }
                [[fallthrough]];
            case IndexExprKind::BIT_XOR:
                lhs->offset ^= rhs->offset;
                for (auto i = 0u; i < lhs->columns.size(); i++) { lhs->columns[i] ^= rhs->columns[i]; }
                return lhs;
            case IndexExprKind::BIT_AND:
                if (!rhs->is_constant()) { std::swap(lhs, rhs); }
                if (!rhs->is_constant()) { return luisa::nullopt; }
                return _transform(std::move(*lhs), [mask = rhs->offset](auto value) noexcept { return value & mask; });
            case IndexExprKind::SHIFT_LEFT:
            case IndexExprKind::SHIFT_RIGHT: {
                if (!rhs->is_constant() || rhs->offset >= 64u) { return luisa::nullopt; }
                auto shift = rhs->offset;
                return _transform(std::move(*lhs), [shift, kind = node->kind](auto value) noexcept {
                    return kind == IndexExprKind::SHIFT_LEFT ? value << shift : value >> shift;
                });
            }
            case IndexExprKind::MULTIPLY: {
                if (!rhs->is_constant()) { std::swap(lhs, rhs); }
                if (!rhs->is_constant()) { return luisa::nullopt; }
                auto factor = rhs->offset;
                if (factor == 0u) { return _constant(0u); }
                // A Boolean times any signed constant is safe, and can
                // duplicate one input bit into an arbitrary output bit basis.
                if (lhs->possible_bits() <= 1u) {
                    return _transform(std::move(*lhs), [factor](auto value) noexcept { return value == 0u ? 0u : factor; });
                }
                constexpr auto maximum = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
                if (!std::has_single_bit(factor) || factor > maximum || lhs->maximum() > maximum / factor) { return luisa::nullopt; }
                auto shift = std::countr_zero(factor);
                return _transform(std::move(*lhs), [shift](auto value) noexcept { return value << shift; });
            }
            case IndexExprKind::FLOOR_DIVIDE:
            case IndexExprKind::MODULO: {
                if (!rhs->is_constant() || !std::has_single_bit(rhs->offset) ||
                    rhs->offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                    (lhs->possible_bits() >> 63u) != 0u) { return luisa::nullopt; }
                auto divisor = rhs->offset;
                auto shift = std::countr_zero(divisor);
                return _transform(std::move(*lhs), [divisor, shift, kind = node->kind](auto value) noexcept {
                    return kind == IndexExprKind::FLOOR_DIVIDE ? value >> shift : value & (divisor - 1u);
                });
            }
            case IndexExprKind::SUBTRACT:
                if (rhs->is_constant() && rhs->offset == 0u) { return lhs; }
                if (lhs->offset == rhs->offset && lhs->columns == rhs->columns) { return _constant(0u); }
                return luisa::nullopt;
            default: return luisa::nullopt;
        }
    }

public:
    BitLinearAnalyzer(const IndexSpace &domain, luisa::span<const int64_t> limits) noexcept
        : _domain{domain} {
        _bit_offsets.reserve(limits.size() + 1u);
        _bit_offsets.emplace_back(0u);
        for (auto limit : limits) { _bit_offsets.emplace_back(_bit_offsets.back() + std::bit_width(static_cast<uint64_t>(limit))); }
    }

    [[nodiscard]] size_t input_bits() const noexcept { return _bit_offsets.back(); }

    [[nodiscard]] luisa::optional<BitLinearForm> analyze(const IndexExprNode *node) noexcept {
        if (auto iter = _cache.find(node); iter != _cache.end()) { return iter->second; }
        auto result = _compute(node);
        _cache.emplace(node, result);
        return result;
    }
};

[[nodiscard]] size_t bit_linear_rank(luisa::span<const BitLinearForm> forms, size_t input_bits) noexcept {
    // Columns span all output coordinates, not just one uint64. Layouts with
    // more than 64 input or output bits have the same exact rank semantics.
    luisa::vector<luisa::vector<uint64_t>> pivots(forms.size() * 64u);
    auto rank = size_t{0u};
    for (auto i = 0u; i < input_bits; i++) {
        luisa::vector<uint64_t> column;
        column.reserve(forms.size());
        for (auto &&form : forms) { column.emplace_back(form.columns[i]); }
        for (auto word = column.size(); word != 0u; word--) {
            auto inserted = false;
            while (column[word - 1u] != 0u) {
                auto bit = std::bit_width(column[word - 1u]) - 1u;
                auto &pivot = pivots[(word - 1u) * 64u + bit];
                if (pivot.empty()) {
                    pivot = std::move(column);
                    rank++;
                    inserted = true;
                    break;
                }
                for (auto j = 0u; j < word; j++) { column[j] ^= pivot[j]; }
            }
            if (inserted) { break; }
        }
    }
    return rank;
}

[[nodiscard]] uint64_t magnitude(int64_t value) noexcept {
    return value < 0 ? uint64_t{0u} - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
}

[[nodiscard]] bool full_column_rank(luisa::span<const AffineForm> forms, luisa::span<const uint8_t> resolved) noexcept {
    auto columns = static_cast<size_t>(std::count(resolved.begin(), resolved.end(), uint8_t{0u}));
    if (columns == 0u) { return true; }
    if (forms.size() < columns) { return false; }
    // A nonzero minor modulo this prime is a nonzero integer minor. This is
    // an exact sufficient proof over Q, not a floating-point rank estimate.
    // Products of residues fit uint64_t because p < 2^31.
    constexpr uint64_t prime = 2147483647u;
    auto power = [](uint64_t base, uint64_t exponent) noexcept {
        auto result = uint64_t{1u};
        while (exponent != 0u) {
            if ((exponent & 1u) != 0u) { result = result * base % prime; }
            base = base * base % prime;
            exponent >>= 1u;
        }
        return result;
    };
    luisa::vector<luisa::vector<uint64_t>> matrix;
    matrix.reserve(forms.size());
    for (auto &&form : forms) {
        auto &row = matrix.emplace_back();
        row.reserve(columns);
        for (auto i = 0u; i < resolved.size(); i++) {
            if (resolved[i] != 0u) { continue; }
            auto value = form.coefficients[i] % static_cast<int64_t>(prime);
            row.emplace_back(static_cast<uint64_t>(value < 0 ? value + static_cast<int64_t>(prime) : value));
        }
    }
    auto rank = size_t{0u};
    for (auto column = size_t{0u}; column < columns; column++) {
        auto pivot = rank;
        while (pivot < matrix.size() && matrix[pivot][column] == 0u) { pivot++; }
        if (pivot == matrix.size()) { continue; }
        std::swap(matrix[rank], matrix[pivot]);
        auto inverse = power(matrix[rank][column], prime - 2u);
        for (auto row = rank + 1u; row < matrix.size(); row++) {
            auto factor = matrix[row][column] * inverse % prime;
            for (auto j = column; j < columns; j++) {
                auto term = factor * matrix[rank][j] % prime;
                matrix[row][j] = (matrix[row][j] + prime - term) % prime;
            }
        }
        rank++;
    }
    return rank == columns;
}

[[nodiscard]] bool prove_affine_injective(luisa::span<const AffineForm> forms, luisa::span<const int64_t> limits) noexcept {
    luisa::vector<uint8_t> resolved(limits.size(), 0u);
    for (auto i = 0u; i < limits.size(); i++) { resolved[i] = limits[i] == 0; }
    auto changed = true;
    while (changed) {
        changed = false;
        for (auto &&form : forms) {
            auto span = uint64_t{0u};
            auto overflow = false;
            for (auto i = 0u; i < limits.size(); i++) {
                if (resolved[i] != 0u) { continue; }
                auto weight = magnitude(form.coefficients[i]);
                auto extent = static_cast<uint64_t>(limits[i]);
                if (extent != 0u && weight > (std::numeric_limits<uint64_t>::max() - span) / extent) {
                    overflow = true;
                    break;
                }
                span += weight * extent;
            }
            if (overflow) { continue; }
            for (auto i = 0u; i < limits.size(); i++) {
                if (resolved[i] != 0u) { continue; }
                auto weight = magnitude(form.coefficients[i]);
                auto term = weight * static_cast<uint64_t>(limits[i]);
                // Equality of output coordinates cannot cancel a nonzero
                // delta_i if its smallest step exceeds every other delta.
                if (weight > span - term) {
                    resolved[i] = 1u;
                    changed = true;
                    span -= term;
                }
            }
        }
    }
    return full_column_rank(forms, resolved);
}

[[nodiscard]] bool has_affine_collision(const IndexMap &map, luisa::span<const AffineForm> forms,
                                        luisa::span<const int64_t> limits) noexcept {
    for (auto i = 0u; i < limits.size(); i++) {
        if (limits[i] == 0) { continue; }
        if (std::all_of(forms.begin(), forms.end(), [i](auto &&form) noexcept { return form.coefficients[i] == 0; })) { return true; }
        for (auto j = i + 1u; j < limits.size(); j++) {
            if (limits[j] == 0) { continue; }
            for (auto &&form : forms) {
                auto a = form.coefficients[i];
                auto b = form.coefficients[j];
                if (a == 0 && b == 0) { continue; }
                auto divisor = std::gcd(magnitude(a), magnitude(b));
                auto di = magnitude(b) / divisor;
                auto dj = magnitude(a) / divisor;
                if (di <= static_cast<uint64_t>(limits[i]) && dj <= static_cast<uint64_t>(limits[j])) {
                    luisa::vector<int64_t> positive(limits.size(), 0);
                    luisa::vector<int64_t> negative(limits.size(), 0);
                    (b < 0 ? negative : positive)[i] = static_cast<int64_t>(di);
                    (a > 0 ? negative : positive)[j] = static_cast<int64_t>(dj);
                    auto lhs = map.apply(positive);
                    auto rhs = map.apply(negative);
                    if (lhs && rhs && *lhs == *rhs) { return true; }
                }
                break;
            }
        }
    }
    return false;
}

[[nodiscard]] bool equal_static_cardinality(const IndexSpace &lhs, const IndexSpace &rhs) noexcept {
    auto a = lhs.static_volume();
    auto b = rhs.static_volume();
    if (a && b) { return *a == *b; }
    // Preserve useful facts for products larger than uint64 without treating
    // overflow as zero. This sufficient test also recognizes permutations.
    auto extents = [](const IndexSpace &space) noexcept {
        luisa::vector<uint64_t> result;
        for (auto &&axis : space.axes()) {
            if (axis.extent.constant_value() != 1u) { result.emplace_back(axis.extent.constant_value()); }
        }
        std::sort(result.begin(), result.end());
        return result;
    };
    return extents(lhs) == extents(rhs);
}

}// namespace detail

IndexExpr::~IndexExpr() noexcept = default;

IndexExpr IndexExpr::constant(int64_t value) noexcept {
    return IndexExpr{detail::make_constant(value)};
}

IndexExpr IndexExpr::coordinate(Dim dimension) noexcept {
    return dimension ? IndexExpr{detail::make_coordinate(dimension)} : IndexExpr{};
}

IndexExprKind IndexExpr::kind() const noexcept {
    return _node == nullptr ? IndexExprKind::INVALID : _node->kind;
}

luisa::optional<int64_t> IndexExpr::constant_value() const noexcept {
    return kind() == IndexExprKind::CONSTANT ? luisa::optional<int64_t>{_node->constant} : luisa::nullopt;
}

Dim IndexExpr::dimension() const noexcept {
    return kind() == IndexExprKind::COORDINATE ? _node->dimension : Dim{};
}

IndexExpr IndexExpr::lhs() const noexcept {
    return _node == nullptr ? IndexExpr{} : IndexExpr{_node->lhs};
}

IndexExpr IndexExpr::rhs() const noexcept {
    return _node == nullptr ? IndexExpr{} : IndexExpr{_node->rhs};
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
        case IndexExprKind::CONSTANT: return detail::make_constant(node->constant);
        case IndexExprKind::COORDINATE: {
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
    return IndexExpr{detail::make_binary(IndexExprKind::ADD, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr operator-(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::SUBTRACT, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr operator*(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::MULTIPLY, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr floor_div(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::FLOOR_DIVIDE, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr modulo(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::MODULO, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr bit_xor(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::BIT_XOR, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr bit_and(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::BIT_AND, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr shift_left(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::SHIFT_LEFT, std::move(lhs._node), std::move(rhs._node))};
}

IndexExpr shift_right(IndexExpr lhs, IndexExpr rhs) noexcept {
    return IndexExpr{detail::make_binary(IndexExprKind::SHIFT_RIGHT, std::move(lhs._node), std::move(rhs._node))};
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

LayoutProof IndexMap::_prove_bit_linear() const noexcept {
    constexpr auto yes = ProofStatus::PROVEN;
    constexpr auto no = ProofStatus::DISPROVEN;
    constexpr auto unknown = ProofStatus::UNKNOWN;
    luisa::vector<int64_t> limits;
    limits.reserve(_domain.rank());
    auto full_cube = true;
    for (auto &&axis : _domain.axes()) {
        if (!axis.extent.is_constant()) { return {}; }
        auto extent = axis.extent.constant_value();
        if (extent == 0u || extent - 1u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return {}; }
        limits.emplace_back(static_cast<int64_t>(extent - 1u));
        full_cube &= std::has_single_bit(extent);
    }
    for (auto &&axis : _codomain.axes()) {
        if (!axis.extent.is_constant()) { return {}; }
    }
    detail::BitLinearAnalyzer analyzer{_domain, limits};
    luisa::vector<detail::BitLinearForm> forms;
    forms.reserve(_outputs.size());
    for (auto i = 0u; i < _outputs.size(); i++) {
        auto form = analyzer.analyze(_outputs[i]._node.get());
        if (!form) { return {}; }
        // The basis lives on the smallest enclosing power-of-two box. Origin
        // and every unit-bit point also belong to the original (possibly
        // ragged) domain, so any invalid one is an actual counterexample.
        auto extent = std::min(_codomain.axis(i).extent.constant_value(), uint64_t{1u} << 63u);
        if (form->offset >= extent ||
            std::any_of(form->columns.begin(), form->columns.end(), [&](auto column) noexcept { return (form->offset ^ column) >= extent; })) {
            return {no, no, no, no};
        }
        if (form->maximum() >= extent) {
            // An envelope counterexample need not be inside a ragged domain.
            // Success at the basis points is never a proof of range safety.
            return full_cube ? LayoutProof{no, no, no, no} : LayoutProof{};
        }
        forms.emplace_back(std::move(*form));
    }
    auto rank = detail::bit_linear_rank(forms, analyzer.input_bits());
    // Rank deficiency also proves a collision on a ragged prefix box: for
    // [0,n) with b=ceil(log2(n)), every b-bit delta is x XOR y for valid x,y.
    // If delta<n use (delta,0); otherwise use (2^(b-1),delta XOR 2^(b-1)).
    // Apply this per coordinate to any nonzero vector in the matrix kernel.
    LayoutProof proof{yes, yes, rank == analyzer.input_bits() ? yes : no, unknown};
    auto output_bits = size_t{0u};
    auto power_of_two_output = true;
    for (auto &&axis : _codomain.axes()) {
        auto extent = axis.extent.constant_value();
        power_of_two_output &= std::has_single_bit(extent);
        output_bits += std::bit_width(extent - 1u);
    }
    if (full_cube) {
        // A GF(2) affine image has exactly 2^rank points, even when the
        // cardinality cannot be held in uint64. Translation preserves rank.
        proof.surjective = power_of_two_output && rank == output_bits ? yes : no;
    } else {
        auto domain_volume = _domain.static_volume();
        auto codomain_volume = _codomain.static_volume();
        if (codomain_volume && *codomain_volume == 1u) {
            proof.surjective = yes;
        } else if ((domain_volume && (!codomain_volume || *domain_volume < *codomain_volume)) ||
                   (power_of_two_output && rank < output_bits)) {
            proof.surjective = no;
        } else if (detail::equal_static_cardinality(_domain, _codomain)) {
            proof.surjective = proof.injective;
        }
    }
    return proof;
}

LayoutProof IndexMap::prove(uint64_t max_fallback_points) const noexcept {
    constexpr auto yes = ProofStatus::PROVEN;
    constexpr auto no = ProofStatus::DISPROVEN;
    constexpr auto unknown = ProofStatus::UNKNOWN;
    auto fallback = [&](LayoutProof proof) noexcept {
        auto complete = [](const LayoutProof &candidate) noexcept {
            return candidate.total != unknown && candidate.in_bounds != unknown &&
                   candidate.injective != unknown && candidate.surjective != unknown;
        };
        if (complete(proof)) { return proof; }
        auto bits = _prove_bit_linear();
        auto merge = [](ProofStatus &fact, ProofStatus other) noexcept {
            if (fact == unknown) { fact = other; }
        };
        merge(proof.total, bits.total);
        merge(proof.in_bounds, bits.in_bounds);
        merge(proof.injective, bits.injective);
        merge(proof.surjective, bits.surjective);
        if (complete(proof)) { return proof; }
        auto finite = analyze_finite(max_fallback_points);
        return finite.enumerated ? LayoutProof{finite.total ? yes : no, finite.in_bounds ? yes : no,
                                               finite.injective ? yes : no, finite.surjective ? yes : no, true} :
                                   proof;
    };
    if (!verify()) { return {no, no, no, no}; }
    auto empty = [](const IndexSpace &space) noexcept {
        return std::any_of(space.axes().begin(), space.axes().end(), [](auto &&axis) noexcept {
            return axis.extent.is_constant() && axis.extent.constant_value() == 0u;
        });
    };
    auto is_static = [](const IndexSpace &space) noexcept {
        return std::all_of(space.axes().begin(), space.axes().end(), [](auto &&axis) noexcept { return axis.extent.is_constant(); });
    };
    if (empty(_domain)) {
        return {yes, yes, yes, empty(_codomain) ? yes : (is_static(_codomain) ? no : unknown)};
    }
    if (!is_static(_domain) || !is_static(_codomain)) { return {}; }
    luisa::vector<int64_t> limits;
    limits.reserve(_domain.rank());
    for (auto &&axis : _domain.axes()) {
        auto limit = axis.extent.constant_value() - 1u;
        if (limit > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return fallback({}); }
        limits.emplace_back(static_cast<int64_t>(limit));
    }
    luisa::vector<int64_t> origin(_domain.rank(), 0);
    auto first = apply(origin);
    auto last = apply(limits);
    // A concrete counterexample disproves totality; success at these points
    // does not prove anything about the rest of the domain.
    if (!first || !last) { return {no, no, no, no}; }
    LayoutProof proof;
    if (origin != limits && *first == *last) { proof.injective = no; }
    detail::AffineAnalyzer analyzer{_domain, limits};
    luisa::vector<detail::AffineForm> forms;
    forms.reserve(_outputs.size());
    for (auto i = 0u; i < _outputs.size(); i++) {
        auto form = analyzer.analyze(_outputs[i]._node.get());
        if (!form) { return fallback(proof); }
        auto extent = _codomain.axis(i).extent.constant_value();
        if (form->minimum < 0 || static_cast<uint64_t>(form->maximum) >= extent) { return {no, no, no, no}; }
        forms.emplace_back(std::move(*form));
    }
    proof.total = yes;
    proof.in_bounds = yes;
    auto domain_volume = _domain.static_volume();
    auto codomain_volume = _codomain.static_volume();
    if (proof.injective != no) {
        if (codomain_volume && (!domain_volume || *domain_volume > *codomain_volume)) {
            proof.injective = no;
        } else if (detail::prove_affine_injective(forms, limits)) {
            proof.injective = yes;
        } else if (detail::has_affine_collision(*this, forms, limits)) {
            proof.injective = no;
        }
    }
    auto same_cardinality = detail::equal_static_cardinality(_domain, _codomain);
    if (codomain_volume && *codomain_volume == 1u) {
        proof.surjective = yes;
    } else if (domain_volume && (!codomain_volume || *domain_volume < *codomain_volume)) {
        proof.surjective = no;
    } else if (proof.injective == yes) {
        if (same_cardinality) {
            proof.surjective = yes;
        } else if (domain_volume && codomain_volume) {
            proof.surjective = no;
        }
    } else if (proof.injective == no && same_cardinality) {
        proof.surjective = no;
    }
    return fallback(proof);
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

bool LayoutCorrespondence::verify() const noexcept {
    return _left.verify() && _right.verify() && _left.domain() == _right.domain();
}

luisa::optional<luisa::vector<luisa::vector<int64_t>>> LayoutCorrespondence::placements(
    luisa::span<const int64_t> logical_point,
    uint64_t max_fiber_points) const noexcept {
    auto fiber_volume = fiber_space().static_volume();
    if (!verify() || !fiber_volume || *fiber_volume > max_fiber_points ||
        !detail::encode_point(logical_point, logical_space())) { return luisa::nullopt; }
    luisa::vector<luisa::vector<int64_t>> result;
    luisa::vector<int64_t> fiber_point;
    for (uint64_t linear = 0u; linear < *fiber_volume; linear++) {
        if (!detail::decode_point(linear, fiber_space(), fiber_point)) { return luisa::nullopt; }
        auto logical = _left.apply(fiber_point);
        if (!logical) { return luisa::nullopt; }
        if (logical->size() == logical_point.size() &&
            std::equal(logical->begin(), logical->end(), logical_point.begin())) {
            auto physical = _right.apply(fiber_point);
            if (!physical) { return luisa::nullopt; }
            if (std::find(result.begin(), result.end(), *physical) == result.end()) {
                result.emplace_back(std::move(*physical));
            }
        }
    }
    return result;
}

LayoutCorrespondenceProperties LayoutCorrespondence::analyze_finite(uint64_t max_fiber_points) const noexcept {
    LayoutCorrespondenceProperties properties;
    auto fiber_volume = fiber_space().static_volume();
    auto logical_volume = logical_space().static_volume();
    auto physical_volume = physical_space().static_volume();
    if (!verify() || !fiber_volume || !logical_volume || !physical_volume || *fiber_volume > max_fiber_points) {
        return properties;
    }
    properties.enumerated = true;
    properties.total = true;
    properties.fiber_points = *fiber_volume;
    properties.logical_points = *logical_volume;
    properties.physical_points = *physical_volume;
    luisa::vector<std::pair<uint64_t, uint64_t>> placements;
    placements.reserve(*fiber_volume);
    luisa::vector<int64_t> fiber_point;
    for (uint64_t linear = 0u; linear < *fiber_volume; linear++) {
        if (!detail::decode_point(linear, fiber_space(), fiber_point)) {
            properties.total = false;
            break;
        }
        auto logical = _left.apply(fiber_point);
        auto physical = _right.apply(fiber_point);
        if (!logical || !physical) {
            properties.total = false;
            break;
        }
        auto logical_linear = detail::encode_point(*logical, logical_space());
        auto physical_linear = detail::encode_point(*physical, physical_space());
        if (!logical_linear || !physical_linear) {
            properties.total = false;
            break;
        }
        placements.emplace_back(*logical_linear, *physical_linear);
    }
    if (properties.total) {
        std::sort(placements.begin(), placements.end());
        placements.erase(std::unique(placements.begin(), placements.end()), placements.end());
        luisa::vector<uint64_t> multiplicity(*logical_volume, 0u);
        for (auto &&placement : placements) { multiplicity[placement.first]++; }
        properties.covers_logical_space = true;
        properties.minimum_replication = std::numeric_limits<uint64_t>::max();
        for (auto count : multiplicity) {
            properties.covers_logical_space &= count != 0u;
            properties.minimum_replication = std::min(properties.minimum_replication, count);
            properties.maximum_replication = std::max(properties.maximum_replication, count);
        }
        if (multiplicity.empty()) { properties.minimum_replication = 0u; }
    }
    return properties;
}

}// namespace luisa::compute::tile
