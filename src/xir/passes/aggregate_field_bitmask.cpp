#include <luisa/core/logging.h>
#include <luisa/core/concepts.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>

#include <algorithm>
#include <limits>

namespace luisa::compute::xir {

namespace detail {

struct AggregateFieldBitRange {
    size_t offset;
    size_t size;
};

class AggregateFieldTree : public concepts::Noncopyable {

private:
    const Type *_type;
    size_t _size;
    luisa::fixed_vector<const AggregateFieldTree *, 4u> _children;
    luisa::fixed_vector<size_t, 4u> _child_offsets;

public:
    explicit AggregateFieldTree(const Type *type) noexcept;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    [[nodiscard]] auto bit_range(luisa::span<const size_t> access_chain) const noexcept {
        auto node = this;
        auto offset = size_t{0};
        for (auto i : access_chain) {
            switch (node->type()->tag()) {
                case Type::Tag::VECTOR: [[fallthrough]];
                case Type::Tag::MATRIX: [[fallthrough]];
                case Type::Tag::ARRAY: {
                    LUISA_DEBUG_ASSERT(i < node->type()->dimension(), "Invalid access chain.");
                    auto child = node->_children.front();
                    node = child;
                    offset += i * child->size();
                    break;
                }
                case Type::Tag::STRUCTURE: {
                    LUISA_DEBUG_ASSERT(i < node->_children.size(), "Invalid access chain.");
                    offset += node->_child_offsets[i];
                    node = node->_children[i];
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Invalid access chain.");
            }
        }
        return AggregateFieldBitRange{offset, node->_size};
    }
};

[[nodiscard]] static const AggregateFieldTree *register_aggregate_type(const Type *type) noexcept {
    static thread_local unordered_map<const Type *, luisa::unique_ptr<AggregateFieldTree>> _cache;
    if (auto iter = _cache.find(type); iter != _cache.end()) {
        return iter->second.get();
    }
    auto node = luisa::make_unique<AggregateFieldTree>(type);
    auto ptr = node.get();
    _cache.emplace(type, std::move(node));
    return ptr;
}

AggregateFieldTree::AggregateFieldTree(const Type *type) noexcept : _type{type}, _size{} {
    switch (type->tag()) {
        case Type::Tag::VECTOR: [[fallthrough]];
        case Type::Tag::MATRIX: [[fallthrough]];
        case Type::Tag::ARRAY: {
            auto dim = type->dimension();
            auto child_type = type->element();
            if (type->tag() == Type::Tag::MATRIX) {
                child_type = Type::vector(child_type, dim);
            }
            auto child = register_aggregate_type(child_type);
            LUISA_ASSERT(
                child->size() == 0u ||
                    dim <= std::numeric_limits<size_t>::max() /
                               child->size(),
                "Aggregate field count overflows size_t.");
            _size = dim * child->size();
            _children.emplace_back(child);
            break;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            _child_offsets.reserve(members.size());
            _children.reserve(members.size());
            auto offset = size_t{0};
            for (auto m : members) {
                auto child = register_aggregate_type(m);
                LUISA_ASSERT(
                    child->size() <=
                        std::numeric_limits<size_t>::max() - offset,
                    "Aggregate field count overflows size_t.");
                _child_offsets.emplace_back(offset);
                _children.emplace_back(child);
                offset += child->size();
            }
            _size = offset;
            break;
        }
        default: {
            _size = size_t{1};
            break;
        }
    }
}

}// namespace detail

inline bool AggregateFieldBitmask::_is_small() const noexcept {
    return size() <= sizeof(uint64_t) * 8u;
}

uint64_t *AggregateFieldBitmask::raw_bits() noexcept {
    return _is_small() ? &_bits_small : _bits_large;
}

const uint64_t *AggregateFieldBitmask::raw_bits() const noexcept {
    return const_cast<AggregateFieldBitmask *>(this)->raw_bits();
}

AggregateFieldBitmask::AggregateFieldBitmask(const Type *type) noexcept
    : _field_tree{detail::register_aggregate_type(type)}, _bits_small{} {
    if (!_is_small()) {
        _bits_large = luisa::allocate_with_allocator<uint64_t>(size_buckets());
        std::memset(_bits_large, 0, size_bytes());
    }
}

AggregateFieldBitmask::~AggregateFieldBitmask() {
    if (!_is_small() && _bits_large != nullptr) {
        luisa::deallocate_with_allocator(_bits_large);
        _bits_large = nullptr;
    }
}

AggregateFieldBitmask::AggregateFieldBitmask(const AggregateFieldBitmask &other) noexcept
    : _field_tree{other._field_tree}, _bits_small{} {
    if (_is_small()) {
        _bits_small = other._bits_small;
    } else {
        _bits_large = luisa::allocate_with_allocator<uint64_t>(size_buckets());
        std::memcpy(_bits_large, other._bits_large, size_bytes());
    }
}

AggregateFieldBitmask::AggregateFieldBitmask(AggregateFieldBitmask &&other) noexcept
    : _field_tree{other._field_tree}, _bits_small{} {
    if (_is_small()) {
        _bits_small = other._bits_small;
    } else {
        _bits_large = other._bits_large;
        other._bits_large = nullptr;
    }
}

AggregateFieldBitmask &AggregateFieldBitmask::operator=(const AggregateFieldBitmask &other) noexcept {
    if (this != &other) [[likely]] {
        LUISA_ASSERT(type() == other.type(), "Type mismatch.");
        if (_is_small()) {
            _bits_small = other._bits_small;
        } else {
            if (_bits_large == nullptr) {
                _bits_large = luisa::allocate_with_allocator<uint64_t>(size_buckets());
            }
            std::memcpy(_bits_large, other._bits_large, size_bytes());
        }
    }
    return *this;
}

AggregateFieldBitmask &AggregateFieldBitmask::operator=(AggregateFieldBitmask &&other) noexcept {
    if (this != &other) [[likely]] {
        LUISA_ASSERT(type() == other.type(), "Type mismatch.");
        if (_is_small()) {
            _bits_small = other._bits_small;
        } else {
            std::swap(_bits_large, other._bits_large);
        }
    }
    return *this;
}

void AggregateFieldBitmask::set(bool value) noexcept {
    if (_is_small()) {
        _bits_small = value ? ~0ull : 0ull;
    } else {
        std::memset(_bits_large, value ? 0xff : 0, size_bytes());
    }
}

size_t AggregateFieldBitmask::size() const noexcept { return _field_tree->size(); }

size_t AggregateFieldBitmask::size_bytes() const noexcept {
    return size_buckets() * sizeof(uint64_t);
}

size_t AggregateFieldBitmask::size_buckets() const noexcept {
    return size() / 64u + static_cast<size_t>(size() % 64u != 0u);
}

const Type *AggregateFieldBitmask::type() const noexcept { return _field_tree->type(); }

AggregateFieldBitmask AggregateFieldBitmask::operator|(const AggregateFieldBitmask &rhs) const noexcept {
    auto copy = *this;
    copy |= rhs;
    return copy;
}

AggregateFieldBitmask AggregateFieldBitmask::operator&(const AggregateFieldBitmask &rhs) const noexcept {
    auto copy = *this;
    copy &= rhs;
    return copy;
}

AggregateFieldBitmask AggregateFieldBitmask::operator^(const AggregateFieldBitmask &rhs) const noexcept {
    auto copy = *this;
    copy ^= rhs;
    return copy;
}

void AggregateFieldBitmask::flip() noexcept {
    if (_is_small()) {
        _bits_small = ~_bits_small;
    } else {
        for (size_t i = 0; i < size_buckets(); i++) {
            _bits_large[i] = ~_bits_large[i];
        }
    }
}

namespace {

[[nodiscard]] constexpr uint64_t low_bits_mask(size_t count) noexcept {
    return count == 0u ? 0ull :
           count >= 64u ? ~0ull :
                          (1ull << count) - 1ull;
}

[[nodiscard]] constexpr uint64_t span_bucket_mask(size_t offset,
                                                  size_t size,
                                                  size_t bucket) noexcept {
    auto bucket_begin = bucket * 64u;
    auto bucket_end = bucket_begin + 64u;
    auto span_begin = offset;
    auto span_end = span_begin + size;
    auto begin = std::max(bucket_begin, span_begin);
    auto end = std::min(bucket_end, span_end);
    if (begin >= end) { return 0ull; }
    auto local_begin = begin - bucket_begin;
    auto local_end = end - bucket_begin;
    return low_bits_mask(local_end) & (~0ull << local_begin);
}

[[nodiscard]] bool span_bit(const AggregateFieldBitmask::ConstBitSpan &span,
                            size_t index) noexcept {
    auto bit = span.offset() + index;
    return ((span.raw_bits()[bit / 64u] >> (bit % 64u)) & 1ull) != 0ull;
}

}// namespace

void AggregateFieldBitmask::BitSpan::set(bool value) noexcept {
    if (_size == 0u) { return; }
    auto lower = _offset / 64u;
    auto upper = (_offset + _size - 1u) / 64u;
    for (auto bucket = lower; bucket <= upper; ++bucket) {
        auto mask = span_bucket_mask(_offset, _size, bucket);
        if (value) {
            _bits[bucket] |= mask;
        } else {
            _bits[bucket] &= ~mask;
        }
    }
}

void AggregateFieldBitmask::BitSpan::flip() noexcept {
    if (_size == 0u) { return; }
    auto lower = _offset / 64u;
    auto upper = (_offset + _size - 1u) / 64u;
    for (auto bucket = lower; bucket <= upper; ++bucket) {
        _bits[bucket] ^= span_bucket_mask(_offset, _size, bucket);
    }
}

// TODO: Implement the following methods in a SIMD-friendly way
AggregateFieldBitmask::BitSpan &AggregateFieldBitmask::BitSpan::operator|=(const ConstBitSpan &rhs) noexcept {
    LUISA_DEBUG_ASSERT(_size == rhs.size(), "Size mismatch.");
    for (size_t i = 0; i < _size; i++) {
        if (auto rhs_bucket = rhs.raw_bits()[(i + rhs.offset()) / 64];
            (rhs_bucket >> ((i + rhs.offset()) % 64)) & 1ull) {
            _bits[(_offset + i) / 64] |= 1ull << ((_offset + i) % 64);
        }
    }
    return *this;
}

AggregateFieldBitmask::BitSpan &AggregateFieldBitmask::BitSpan::operator&=(const ConstBitSpan &rhs) noexcept {
    LUISA_DEBUG_ASSERT(_size == rhs.size(), "Size mismatch.");
    for (size_t i = 0; i < _size; i++) {
        if (!span_bit(rhs, i)) {
            _bits[(_offset + i) / 64u] &= ~(1ull << ((_offset + i) % 64u));
        }
    }
    return *this;
}

AggregateFieldBitmask::BitSpan &AggregateFieldBitmask::BitSpan::operator^=(const ConstBitSpan &rhs) noexcept {
    LUISA_DEBUG_ASSERT(_size == rhs.size(), "Size mismatch.");
    for (size_t i = 0; i < _size; i++) {
        if (auto rhs_bucket = rhs.raw_bits()[(i + rhs.offset()) / 64];
            (rhs_bucket >> ((i + rhs.offset()) % 64)) & 1ull) {
            _bits[(_offset + i) / 64] ^= 1ull << ((_offset + i) % 64);
        }
    }
    return *this;
}

bool AggregateFieldBitmask::BitSpan::operator==(const ConstBitSpan &rhs) const noexcept {
    if (_size != rhs.size()) { return false; }
    if (this != &rhs) {
        for (size_t i = 0; i < _size; i++) {
            auto lhs_bit = ((_bits[(_offset + i) / 64u] >> ((_offset + i) % 64u)) & 1ull) != 0ull;
            if (lhs_bit != span_bit(rhs, i)) { return false; }
        }
    }
    return true;
}

bool AggregateFieldBitmask::BitSpan::operator!=(const ConstBitSpan &rhs) const noexcept {
    return !(*this == rhs);
}

bool AggregateFieldBitmask::ConstBitSpan::all() const noexcept {
    if (_size == 0u) { return true; }
    auto lower = _offset / 64u;
    auto upper = (_offset + _size - 1u) / 64u;
    for (auto bucket = lower; bucket <= upper; ++bucket) {
        auto mask = span_bucket_mask(_offset, _size, bucket);
        if ((_bits[bucket] & mask) != mask) { return false; }
    }
    return true;
}

bool AggregateFieldBitmask::ConstBitSpan::any() const noexcept {
    if (_size == 0u) { return false; }
    auto lower = _offset / 64u;
    auto upper = (_offset + _size - 1u) / 64u;
    for (auto bucket = lower; bucket <= upper; ++bucket) {
        if ((_bits[bucket] & span_bucket_mask(_offset, _size, bucket)) != 0ull) {
            return true;
        }
    }
    return false;
}

bool AggregateFieldBitmask::ConstBitSpan::none() const noexcept {
    return !any();
}

AggregateFieldBitmask::BitSpan AggregateFieldBitmask::access(luisa::span<const size_t> access_chain) noexcept {
    auto bits = raw_bits();
    auto range = _field_tree->bit_range(access_chain);
    return {bits, range.offset, range.size};
}

AggregateFieldBitmask::ConstBitSpan AggregateFieldBitmask::access(luisa::span<const size_t> access_chain) const noexcept {
    return const_cast<AggregateFieldBitmask *>(this)->access(access_chain);
}

AggregateFieldBitmask::BitSpan AggregateFieldBitmask::access(std::initializer_list<size_t> access_chain) noexcept {
    return access(luisa::span{access_chain.begin(), access_chain.end()});
}

AggregateFieldBitmask::ConstBitSpan AggregateFieldBitmask::access(std::initializer_list<size_t> access_chain) const noexcept {
    return access(luisa::span{access_chain.begin(), access_chain.end()});
}

bool AggregateFieldBitmask::mark_access_pattern(
    luisa::span<const luisa::optional<size_t>> access_pattern) noexcept {
    AggregateFieldBitmask selected{type()};
    luisa::vector<size_t> concrete_path;
    concrete_path.reserve(access_pattern.size());
    auto visit = [&](auto &&self, const Type *node_type,
                     size_t depth) noexcept -> bool {
        if (depth == access_pattern.size()) {
            selected.access(luisa::span{concrete_path}).set(true);
            return true;
        }
        if (node_type == nullptr) { return false; }
        auto descend = [&](size_t index, const Type *child_type) noexcept {
            concrete_path.emplace_back(index);
            auto valid = self(self, child_type, depth + 1u);
            concrete_path.pop_back();
            return valid;
        };
        auto index = access_pattern[depth];
        switch (node_type->tag()) {
            case Type::Tag::VECTOR:
            case Type::Tag::ARRAY: {
                auto dimension = node_type->dimension();
                auto *child_type = node_type->element();
                if (index) {
                    return *index < dimension &&
                           descend(*index, child_type);
                }
                for (size_t i = 0u; i < dimension; ++i) {
                    if (!descend(i, child_type)) { return false; }
                }
                return true;
            }
            case Type::Tag::MATRIX: {
                auto dimension = node_type->dimension();
                auto *child_type = Type::vector(
                    node_type->element(), dimension);
                if (index) {
                    return *index < dimension &&
                           descend(*index, child_type);
                }
                for (size_t i = 0u; i < dimension; ++i) {
                    if (!descend(i, child_type)) { return false; }
                }
                return true;
            }
            case Type::Tag::STRUCTURE: {
                if (!index) { return false; }
                auto members = node_type->members();
                return *index < members.size() &&
                       descend(*index, members[*index]);
            }
            default: return false;
        }
    };
    if (!visit(visit, type(), 0u)) { return false; }
    *this |= selected;
    return true;
}

AggregateFieldBitmask &AggregateFieldBitmask::operator|=(const AggregateFieldBitmask &rhs) noexcept {
    LUISA_DEBUG_ASSERT(type() == rhs.type(), "Type mismatch.");
    auto n_buckets = size_buckets();
    auto lhs_bits = raw_bits();
    auto rhs_bits = rhs.raw_bits();
    for (size_t i = 0; i < n_buckets; i++) {
        lhs_bits[i] |= rhs_bits[i];
    }
    return *this;
}

AggregateFieldBitmask &AggregateFieldBitmask::operator&=(const AggregateFieldBitmask &rhs) noexcept {
    LUISA_DEBUG_ASSERT(type() == rhs.type(), "Type mismatch.");
    auto n_buckets = size_buckets();
    auto lhs_bits = raw_bits();
    auto rhs_bits = rhs.raw_bits();
    for (size_t i = 0; i < n_buckets; i++) {
        lhs_bits[i] &= rhs_bits[i];
    }
    return *this;
}

AggregateFieldBitmask &AggregateFieldBitmask::operator^=(const AggregateFieldBitmask &rhs) noexcept {
    LUISA_DEBUG_ASSERT(type() == rhs.type(), "Type mismatch.");
    auto n_buckets = size_buckets();
    auto lhs_bits = raw_bits();
    auto rhs_bits = rhs.raw_bits();
    for (size_t i = 0; i < n_buckets; i++) {
        lhs_bits[i] ^= rhs_bits[i];
    }
    return *this;
}

AggregateFieldBitmask AggregateFieldBitmask::operator~() const noexcept {
    auto copy = *this;
    copy.flip();
    return copy;
}

bool AggregateFieldBitmask::operator==(const AggregateFieldBitmask &rhs) const noexcept {
    if (this == &rhs) { return true; }
    if (type() != rhs.type() || size() != rhs.size()) { return false; }
    auto lhs_bits = raw_bits();
    auto rhs_bits = rhs.raw_bits();
    auto n_complete_buckets = size() / 64;
    for (size_t i = 0; i < n_complete_buckets; i++) {
        if (lhs_bits[i] != rhs_bits[i]) { return false; }
    }
    if (auto n_remaining_bits = size() % 64; n_remaining_bits != 0) {
        auto mask = (1ull << n_remaining_bits) - 1ull;
        if ((lhs_bits[n_complete_buckets] & mask) != (rhs_bits[n_complete_buckets] & mask)) { return false; }
    }
    return true;
}

bool AggregateFieldBitmask::operator!=(const AggregateFieldBitmask &rhs) const noexcept {
    return !(*this == rhs);
}

}// namespace luisa::compute::xir
