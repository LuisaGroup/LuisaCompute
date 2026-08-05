#pragma once

#include <luisa/runtime/dynamic_buffer.h>
#include <luisa/dsl/sugar.h>

namespace luisa::compute {

/// Atomically reserves a byte range in a DynamicBuffer arena.
template<typename ByteCount, typename Capacity>
    requires is_integral_expr_v<ByteCount> && is_integral_expr_v<Capacity>
[[nodiscard]] inline auto dynamic_buffer_allocate(
    Expr<Buffer<uint>> counter, Expr<Buffer<uint>> overflow,
    ByteCount &&byte_count, Capacity &&capacity_bytes) noexcept {
    auto bytes = def(cast<uint>(std::forward<ByteCount>(byte_count)));
    auto capacity = def(cast<uint>(std::forward<Capacity>(capacity_bytes)));
    UInt offset = dynamic_buffer_invalid_offset;
    UInt current = counter.read(0u);
    $loop {
        auto size_fits = bytes <= capacity;
        auto maximum_offset = select(0u, capacity - bytes, size_fits);
        $if (!size_fits | current > maximum_offset) {
            overflow.atomic(0u).fetch_or(1u);
            $break;
        };
        auto observed = counter.atomic(0u).compare_exchange(current, current + bytes);
        $if (observed == current) {
            offset = current;
            $break;
        };
        current = observed;
    };
    return offset;
}

}// namespace luisa::compute
