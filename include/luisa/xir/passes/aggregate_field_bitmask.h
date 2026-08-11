#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

namespace detail {
class AggregateFieldTree;
}// namespace detail

class LUISA_XIR_API AggregateFieldBitmask {

private:
    alignas(16) const detail::AggregateFieldTree *_field_tree;
    union {
        uint64_t _bits_small;
        uint64_t *_bits_large;
    };

private:
    [[nodiscard]] bool _is_small() const noexcept;

public:
    explicit AggregateFieldBitmask(const Type *type) noexcept;
    ~AggregateFieldBitmask();

    AggregateFieldBitmask(const AggregateFieldBitmask &other) noexcept;
    AggregateFieldBitmask(AggregateFieldBitmask &&other) noexcept;
    AggregateFieldBitmask &operator=(const AggregateFieldBitmask &other) noexcept;
    AggregateFieldBitmask &operator=(AggregateFieldBitmask &&other) noexcept;

    void set(bool value = true) noexcept;
    void flip() noexcept;
    [[nodiscard]] uint64_t *raw_bits() noexcept;
    [[nodiscard]] const uint64_t *raw_bits() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] size_t size_bytes() const noexcept;
    [[nodiscard]] size_t size_buckets() const noexcept;
    [[nodiscard]] const Type *type() const noexcept;

public:
    class LUISA_XIR_API ConstBitSpan {
    protected:
        uint64_t *_bits;
        size_t _offset;
        size_t _size;
    public:
        ConstBitSpan(uint64_t *bits, size_t offset, size_t size) noexcept
            : _bits{bits}, _offset{offset}, _size{size} {}
        [[nodiscard]] const uint64_t *raw_bits() const noexcept { return _bits; }
        [[nodiscard]] size_t offset() const noexcept { return _offset; }
        [[nodiscard]] size_t size() const noexcept { return _size; }
        [[nodiscard]] bool all() const noexcept;
        [[nodiscard]] bool any() const noexcept;
        [[nodiscard]] bool none() const noexcept;
    };
    class LUISA_XIR_API BitSpan : public ConstBitSpan {
    public:
        using ConstBitSpan::ConstBitSpan;
        [[nodiscard]] uint64_t *raw_bits() noexcept { return _bits; }

        void set(bool value = true) noexcept;
        void flip() noexcept;

        BitSpan &operator|=(const ConstBitSpan &rhs) noexcept;
        BitSpan &operator&=(const ConstBitSpan &rhs) noexcept;
        BitSpan &operator^=(const ConstBitSpan &rhs) noexcept;
        [[nodiscard]] bool operator==(const ConstBitSpan &rhs) const noexcept;
        [[nodiscard]] bool operator!=(const ConstBitSpan &rhs) const noexcept;

        BitSpan &operator|=(const AggregateFieldBitmask &rhs) noexcept { return *this |= rhs.access(); }
        BitSpan &operator&=(const AggregateFieldBitmask &rhs) noexcept { return *this &= rhs.access(); }
        BitSpan &operator^=(const AggregateFieldBitmask &rhs) noexcept { return *this ^= rhs.access(); }
        [[nodiscard]] bool operator==(const AggregateFieldBitmask &rhs) const noexcept { return *this == rhs.access(); }
        [[nodiscard]] bool operator!=(const AggregateFieldBitmask &rhs) const noexcept { return *this != rhs.access(); }
    };
    [[nodiscard]] BitSpan access(luisa::span<const size_t> access_chain) noexcept;
    [[nodiscard]] ConstBitSpan access(luisa::span<const size_t> access_chain) const noexcept;
    [[nodiscard]] BitSpan access(std::initializer_list<size_t> access_chain) noexcept;
    [[nodiscard]] ConstBitSpan access(std::initializer_list<size_t> access_chain) const noexcept;

    // Marks every primitive leaf selected by an access pattern. A present
    // index selects one child; nullopt denotes a runtime index and selects
    // the union of every possible child. Runtime indexing is valid only for
    // homogeneous vector/matrix/array nodes. Failure is atomic and leaves the
    // mask unchanged.
    [[nodiscard]] bool mark_access_pattern(
        luisa::span<const luisa::optional<size_t>> access_pattern) noexcept;

    template<typename... I>
        requires(std::conjunction_v<std::is_integral<I>...>)
    [[nodiscard]] BitSpan access(I... indices) noexcept {
        return access({static_cast<size_t>(indices)...});
    }

    template<typename... I>
        requires(std::conjunction_v<std::is_integral<I>...>)
    [[nodiscard]] ConstBitSpan access(I... indices) const noexcept {
        return access({static_cast<size_t>(indices)...});
    }

    [[nodiscard]] AggregateFieldBitmask operator|(const AggregateFieldBitmask &rhs) const noexcept;
    [[nodiscard]] AggregateFieldBitmask operator&(const AggregateFieldBitmask &rhs) const noexcept;
    [[nodiscard]] AggregateFieldBitmask operator^(const AggregateFieldBitmask &rhs) const noexcept;
    [[nodiscard]] AggregateFieldBitmask operator~() const noexcept;

    AggregateFieldBitmask &operator|=(const AggregateFieldBitmask &rhs) noexcept;
    AggregateFieldBitmask &operator&=(const AggregateFieldBitmask &rhs) noexcept;
    AggregateFieldBitmask &operator^=(const AggregateFieldBitmask &rhs) noexcept;

    [[nodiscard]] bool operator==(const AggregateFieldBitmask &rhs) const noexcept;
    [[nodiscard]] bool operator!=(const AggregateFieldBitmask &rhs) const noexcept;
};

}// namespace luisa::compute::xir
