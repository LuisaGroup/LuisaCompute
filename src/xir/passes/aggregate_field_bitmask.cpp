#include <luisa/core/logging.h>
#include <luisa/core/concepts.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>

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
    : _field_tree{other._field_tree}, _bits_small{} { *this = other; }

AggregateFieldBitmask::AggregateFieldBitmask(AggregateFieldBitmask &&other) noexcept
    : _field_tree{other._field_tree}, _bits_small{other._bits_small} {
    if (!_is_small()) { other._bits_large = nullptr; }
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
        this->~AggregateFieldBitmask();
        new (this) AggregateFieldBitmask{std::move(other)};
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
    return (size() + 63) / 64;
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

void AggregateFieldBitmask::BitSpan::set(bool value) noexcept {
    uint32_t lower = _offset / 64;
    uint32_t upper = (_offset + _size - 1) / 64;
    if (lower == upper) {// all selected bits are in the same bucket
        uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull);
        uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull;
        auto mask = lower_mask & upper_mask;
        if (value) {
            _bits[lower] |= mask;
        } else {
            _bits[lower] &= ~mask;
        }
    } else {// selected bits are in different buckets
        // process the first bucket
        if (uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull); lower_mask != 0) {
            if (value) {
                _bits[lower] |= lower_mask;
            } else {
                _bits[lower] &= ~lower_mask;
            }
            lower++;
        }
        // process the middle buckets
        for (size_t i = lower; i < upper; i++) {
            if (value) {
                _bits[i] = ~0ull;
            } else {
                _bits[i] = 0ull;
            }
        }
        // process the last bucket
        if (uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull; upper_mask != 0) {
            if (value) {
                _bits[upper] |= upper_mask;
            } else {
                _bits[upper] &= ~upper_mask;
            }
        }
    }
}

void AggregateFieldBitmask::BitSpan::flip() noexcept {
    uint32_t lower = _offset / 64;
    uint32_t upper = (_offset + _size - 1) / 64;
    if (lower == upper) {// all selected bits are in the same bucket
        uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull);
        uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull;
        auto mask = lower_mask & upper_mask;
        _bits[lower] ^= mask;
    } else {// selected bits are in different buckets
        // process the first bucket
        if (uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull); lower_mask != 0) {
            _bits[lower] ^= lower_mask;
            lower++;
        }
        // process the middle buckets
        for (size_t i = lower; i < upper; i++) {
            _bits[i] = ~_bits[i];
        }
        // process the last bucket
        if (uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull; upper_mask != 0) {
            _bits[upper] ^= upper_mask;
        }
    }
}

// TODO: Implement the following methods in a SIMD-friendly way
AggregateFieldBitmask::BitSpan &AggregateFieldBitmask::BitSpan::operator|=(const ConstBitSpan &rhs) noexcept {
    LUISA_DEBUG_ASSERT(_size == rhs.size(), "Size mismatch.");
    for (uint32_t i = 0; i < _size; i++) {
        if (auto rhs_bucket = rhs.raw_bits()[(i + rhs.offset()) / 64];
            (rhs_bucket >> ((i + rhs.offset()) % 64)) & 1ull) {
            _bits[(_offset + i) / 64] |= 1ull << ((_offset + i) % 64);
        }
    }
    return *this;
}

AggregateFieldBitmask::BitSpan &AggregateFieldBitmask::BitSpan::operator&=(const ConstBitSpan &rhs) noexcept {
    LUISA_DEBUG_ASSERT(_size == rhs.size(), "Size mismatch.");
    for (uint32_t i = 0; i < _size; i++) {
        if (auto rhs_bucket = rhs.raw_bits()[(i + rhs.offset()) / 64];
            (rhs_bucket >> ((i + rhs.offset()) % 64)) & 1ull) {
            _bits[(_offset + i) / 64] &= 1ull << ((_offset + i) % 64);
        }
    }
    return *this;
}

AggregateFieldBitmask::BitSpan &AggregateFieldBitmask::BitSpan::operator^=(const ConstBitSpan &rhs) noexcept {
    LUISA_DEBUG_ASSERT(_size == rhs.size(), "Size mismatch.");
    for (uint32_t i = 0; i < _size; i++) {
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
        for (uint32_t i = 0; i < _size; i++) {
            uint64_t lhs_bit = (_bits[(_offset + i) / 64] >> ((_offset + i) % 64)) & 1ull;
            uint64_t rhs_bit = (rhs.raw_bits()[i / 64] >> (i % 64)) & 1ull;
            if (lhs_bit != rhs_bit) { return false; }
        }
    }
    return true;
}

bool AggregateFieldBitmask::BitSpan::operator!=(const ConstBitSpan &rhs) const noexcept {
    return !(*this == rhs);
}

bool AggregateFieldBitmask::ConstBitSpan::all() const noexcept {
    uint32_t lower = _offset / 64;
    uint32_t upper = (_offset + _size - 1) / 64;
    if (lower == upper) {// all selected bits are in the same bucket
        uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull);
        uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull;
        auto mask = lower_mask & upper_mask;
        return (_bits[lower] & mask) == mask;
    }
    // selected bits are in different buckets
    // process the first bucket
    if (uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull); lower_mask != 0) {
        if ((_bits[lower] & lower_mask) != lower_mask) { return false; }
        lower++;
    }
    // process the middle buckets
    for (size_t i = lower; i < upper; i++) {
        if (_bits[i] != ~0ull) { return false; }
    }
    // process the last bucket
    if (uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull; upper_mask != 0) {
        auto mask = ~upper_mask;
        return (_bits[upper] & mask) == mask;
    }
    return true;
}

bool AggregateFieldBitmask::ConstBitSpan::any() const noexcept {
    uint32_t lower = _offset / 64;
    uint32_t upper = (_offset + _size - 1) / 64;
    if (lower == upper) {// all selected bits are in the same bucket
        uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull);
        uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull;
        auto mask = lower_mask & upper_mask;
        return (_bits[lower] & mask) != 0ull;
    }
    // selected bits are in different buckets
    // process the first bucket
    if (uint64_t lower_mask = ~((1ull << (_offset % 64)) - 1ull); lower_mask != 0) {
        if ((_bits[lower] & lower_mask) != 0ull) { return true; }
        lower++;
    }
    // process the middle buckets
    for (size_t i = lower; i < upper; i++) {
        if (_bits[i] != 0ull) { return true; }
    }
    // process the last bucket
    if (uint64_t upper_mask = (1ull << ((_offset + _size) % 64)) - 1ull; upper_mask != 0) {
        auto mask = ~upper_mask;
        return (_bits[upper] & mask) != 0ull;
    }
    return false;
}

bool AggregateFieldBitmask::ConstBitSpan::none() const noexcept {
    return !any();
}

AggregateFieldBitmask::BitSpan AggregateFieldBitmask::access(luisa::span<const size_t> access_chain) noexcept {
    auto bits = raw_bits();
    auto range = _field_tree->bit_range(access_chain);
    return {bits, static_cast<uint32_t>(range.offset), static_cast<uint32_t>(range.size)};
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
