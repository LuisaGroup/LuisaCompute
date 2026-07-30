// Test for aggregate-field bitmask layout, projection, and set operations.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>

#include <utility>
#include <limits>

using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {

    "bitmask_span_coordinates_do_not_truncate_at_uint32"_test = [] {
        auto above_u32 =
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 17u;
        AggregateFieldBitmask::ConstBitSpan span{
            nullptr, above_u32, above_u32 + 9u};
        expect(span.offset() == above_u32);
        expect(span.size() == above_u32 + 9u);
    };

    "bitmask_exact_bucket_span"_test = [] {
        AggregateFieldBitmask mask{Type::array(Type::of<int>(), 64u)};
        expect(mask.access().none());
        mask.access().set();
        expect(mask.access().all());
        expect(mask.access(0u).all());
        expect(mask.access(63u).all());
        mask.access().flip();
        expect(mask.access().none());
    };

    "bitmask_cross_bucket_span_isolated"_test = [] {
        auto *inner = Type::array(Type::of<int>(), 96u);
        AggregateFieldBitmask mask{Type::array(inner, 2u)};

        mask.access(1u).set();
        expect(mask.access(0u).none());
        expect(mask.access(1u).all());
        expect(mask.access(1u, 0u).all());
        expect(mask.access(1u, 95u).all());

        mask.set(false);
        mask.access(1u, 4u).set();
        expect(mask.access(0u).none())
            << "bits outside a cross-bucket span must not affect any()";
        expect(mask.access(1u).any());
    };

    "bitmask_span_and_clears_only_selected_bits"_test = [] {
        auto *inner = Type::array(Type::of<int>(), 96u);
        auto *outer = Type::array(inner, 3u);
        AggregateFieldBitmask lhs{outer};
        AggregateFieldBitmask rhs{outer};

        lhs.access(0u, 7u).set();
        lhs.access(1u).set();
        lhs.access(2u, 11u).set();
        rhs.access(2u, 0u).set();
        rhs.access(2u, 65u).set();

        lhs.access(1u) &= rhs.access(2u);
        expect(lhs.access(0u, 7u).all());
        expect(lhs.access(2u, 11u).all());
        expect(lhs.access(1u, 0u).all());
        expect(lhs.access(1u, 65u).all());
        expect(lhs.access(1u, 1u).none());
        expect(lhs.access(1u, 64u).none());
        expect(lhs.access(1u, 95u).none());
    };

    "bitmask_offset_span_equality"_test = [] {
        auto *inner = Type::array(Type::of<int>(), 96u);
        auto *outer = Type::array(inner, 3u);
        AggregateFieldBitmask lhs{outer};
        AggregateFieldBitmask rhs{outer};
        lhs.access(1u, 3u).set();
        lhs.access(1u, 70u).set();
        rhs.access(2u, 3u).set();
        rhs.access(2u, 70u).set();

        expect(lhs.access(1u) == rhs.access(2u));
        rhs.access(2u, 95u).set();
        expect(lhs.access(1u) != rhs.access(2u));
    };

    "bitmask_large_copy_and_move_preserve_storage"_test = [] {
        auto *type = Type::array(Type::of<int>(), 130u);
        AggregateFieldBitmask source{type};
        source.access(0u).set();
        source.access(64u).set();
        source.access(129u).set();

        AggregateFieldBitmask copied{source};
        expect(copied == source);
        copied.access(1u).set();
        expect(copied != source);

        AggregateFieldBitmask copy_assigned{type};
        copy_assigned = source;
        expect(copy_assigned == source);

        AggregateFieldBitmask moved{std::move(copied)};
        expect(moved.access(0u).all());
        expect(moved.access(1u).all());
        expect(moved.access(64u).all());
        expect(moved.access(129u).all());

        AggregateFieldBitmask move_assigned{type};
        move_assigned = std::move(copy_assigned);
        expect(move_assigned == source);
    };
}
