// Test CoroFrameDesc: field layout, offsets, lookup, from_materialize_info
#include <luisa/ast/type_registry.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/passes/coro_materialize.h>

#include "ut/ut.hpp"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void reg_coro_frame_desc_manual() {

    "field_count_empty"_test = [] {
        CoroFrameDesc desc;
        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);
    };

    "add_two_uint_fields"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("a", Type::of<uint32_t>());
        desc.add_field("b", Type::of<uint32_t>());

        expect(desc.field_count() == 2u);
        expect(desc.total_size() == 8u);

        auto &a = desc.field(0u);
        expect(a.name == "a");
        expect(a.offset == 0u);
        expect(a.size == 4u);

        auto &b = desc.field(1u);
        expect(b.name == "b");
        expect(b.offset == 4u);
        expect(b.size == 4u);
    };

    "user_fields_with_alignment"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("a", Type::of<uint32_t>());// offset 0, size 4
        desc.add_field("b", Type::of<uint32_t>());// offset 4, size 4
        desc.add_field("x", Type::of<float>());   // offset 8, size 4
        desc.add_field("y", Type::of<int>());     // offset 12, size 4

        expect(desc.field_count() == 4u);
        expect(desc.total_size() == 16u);

        expect(desc.field(2u).name == "x");
        expect(desc.field(2u).offset == 8u);
        expect(desc.field(3u).name == "y");
        expect(desc.field(3u).offset == 12u);
    };

    "field_lookup_by_name"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("a", Type::of<uint32_t>());
        desc.add_field("b", Type::of<uint32_t>());
        desc.add_field("x", Type::of<float>());
        desc.add_field("y", Type::of<int>());

        auto *x_field = desc.field("x");
        expect(x_field != nullptr);
        expect(x_field->name == "x");
        expect(x_field->offset == 8u);
        expect(x_field->type->tag() == Type::Tag::FLOAT32);

        auto *y_field = desc.field("y");
        expect(y_field != nullptr);
        expect(y_field->offset == 12u);
        expect(y_field->type->tag() == Type::Tag::INT32);
    };

    "field_lookup_invalid_name"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("a", Type::of<uint32_t>());
        desc.add_field("b", Type::of<uint32_t>());

        auto *not_found = desc.field("nonexistent");
        expect(not_found == nullptr);
    };

    "field_lookup_by_index"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("u0", Type::of<uint32_t>());
        desc.add_field("u1", Type::of<uint32_t>());
        desc.add_field("a", Type::of<float>());
        desc.add_field("b", Type::of<int>());
        desc.add_field("c", Type::of<uint32_t>());

        expect(desc.field_count() == 5u);

        expect(desc.field(0u).name == "u0");
        expect(desc.field(1u).name == "u1");
        expect(desc.field(2u).name == "a");
        expect(desc.field(3u).name == "b");
        expect(desc.field(4u).name == "c");
    };

    "total_size_is_aligned"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("a", Type::of<uint32_t>());// 4 bytes
        desc.add_field("b", Type::of<uint32_t>());// 4 bytes
        // float3 has alignment 16, size 12
        desc.add_field("v", Type::of<float3>());// offset 16, size 12

        expect(desc.field_count() == 3u);
        // v should be aligned to 16
        expect(desc.field(2u).offset == 16u);
        // total_size = 16 + 12 = 28, aligned to 16 → 32
        expect(desc.total_size() == 32u);
    };

    "reserved_fields_are_scalar_uints"_test = [] {
        CoroFrameDesc desc;
        expect(desc.frame_field_count() == CoroFrameDesc::reserved_field_count);
        expect(desc.frame_field_type(0u) == Type::of<uint>());
        expect(desc.frame_field_type(1u) == Type::of<uint>());
        expect(desc.frame_field_type(2u) == Type::of<uint>());
        expect(desc.frame_field_type(3u) == Type::of<uint>());

        desc.add_field("payload", Type::of<float>());
        expect(desc.frame_field_count() == CoroFrameDesc::reserved_field_count + 1u);
        expect(desc.frame_field_type(CoroFrameDesc::reserved_field_count) == Type::of<float>());
    };
}

void reg_coro_frame_desc_from_materialize() {

    "from_materialize_info_basic"_test = [] {
        // given: CoroMaterializeInfo with 2 user registers
        CoroMaterializeInfo info;
        info.name_to_field.emplace("x", CoroFrameDesc::reserved_field_count);
        info.name_to_field.emplace("y", CoroFrameDesc::reserved_field_count + 1u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());

        // when
        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // then: user fields x, y (sorted by field index)
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "x");
        expect(desc.field(0u).offset == 0u);
        expect(desc.field(1u).name == "y");
        expect(desc.field(1u).offset == 4u);
        expect(desc.total_size() == 8u);

        // Lookup by name works
        auto *xf = desc.field("x");
        expect(xf != nullptr);
        expect(xf->type->tag() == Type::Tag::FLOAT32);

        auto *yf = desc.field("y");
        expect(yf != nullptr);
        expect(yf->type->tag() == Type::Tag::INT32);
    };

    "from_materialize_info_sorted_by_index"_test = [] {
        // given: entries added out of order
        CoroMaterializeInfo info;
        info.name_to_field.emplace("z", CoroFrameDesc::reserved_field_count + 2u);
        info.name_to_field.emplace("x", CoroFrameDesc::reserved_field_count);
        info.name_to_field.emplace("y", CoroFrameDesc::reserved_field_count + 1u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());
        info.name_to_type.emplace("z", Type::of<uint32_t>());

        // when
        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // then: fields ordered by index (x, y, z)
        expect(desc.field_count() == 3u);
        expect(desc.field(0u).name == "x");
        expect(desc.field(1u).name == "y");
        expect(desc.field(2u).name == "z");
    };

    "from_materialize_info_no_user_fields"_test = [] {
        // given: empty info (no registers)
        CoroMaterializeInfo info;

        // when
        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // then: no fields (system fields managed by CoroFrame)
        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);
    };

    "from_materialize_info_preserves_logical_field_aliases"_test = [] {
        auto field_index = CoroFrameDesc::reserved_field_count;
        CoroMaterializeInfo info;
        info.frame_field_count = field_index + 1u;
        info.frame_fields.emplace_back(
            CoroMaterializeInfo::FrameField{
                .name = "physical_state",
                .type = Type::of<uint>(),
                .index = field_index});
        info.name_to_field.emplace("physical_state", field_index);
        info.name_to_field.emplace("coro_hint", field_index);
        info.name_to_type.emplace("physical_state", Type::of<uint>());
        info.name_to_type.emplace("coro_hint", Type::of<uint>());

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        expect(desc.field_count() == 1u);
        expect(desc.field_index("physical_state") == 0u);
        expect(desc.field_index("coro_hint") == 0u);
        expect(desc.field("coro_hint") == &desc.field(0u));
        expect(desc.field("coro_hint")->type == Type::of<uint>());
    };
}

}// namespace

int main(int /* argc */, char * /* argv */[]) {
    reg_coro_frame_desc_manual();
    reg_coro_frame_desc_from_materialize();
    return 0;
}
