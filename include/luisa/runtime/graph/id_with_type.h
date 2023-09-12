#pragma once
#include <cstdint>
#include <luisa/core/logging.h>

namespace luisa::compute::graph {
class IdWithType {
public:
    using type = uint64_t;
    static constexpr auto invalid_id = std::numeric_limits<uint64_t>::max();
    explicit IdWithType(uint64_t value) noexcept : _value{value} {}
    explicit IdWithType() noexcept : _value{invalid_id} {}
    uint64_t value() const noexcept { return _value; }
    friend std::ostream &operator<<(std::ostream &os, const IdWithType &id) {
        os << id._value;
        return os;
    }
    friend bool operator==(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs._value == rhs._value; }
    friend bool operator!=(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs._value != rhs._value; }
    friend bool operator<(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs._value < rhs._value; }
    friend bool operator>(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs._value > rhs._value; }

    bool is_valid() const noexcept { return _value != invalid_id; }

protected:
    uint64_t _value{invalid_id};
};
}// namespace luisa::compute::graph

template<>
struct fmt::formatter<luisa::compute::graph::IdWithType> : formatter<luisa::compute::graph::IdWithType::type> {
    auto format(luisa::compute::graph::IdWithType c, format_context &ctx) {
        return formatter<luisa::compute::graph::IdWithType::type>::format(c.value(), ctx);
    }
};