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

    "add_token_skip_fields"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("token", Type::of<uint32_t>());
        desc.add_field("skip", Type::of<uint32_t>());

        expect(desc.field_count() == 2u);
        expect(desc.total_size() == 8u);

        // token at offset 0
        auto &token_f = desc.field(0u);
        expect(token_f.name == "token");
        expect(token_f.offset == 0u);
        expect(token_f.size == 4u);

        // skip at offset 4
        auto &skip_f = desc.field(1u);
        expect(skip_f.name == "skip");
        expect(skip_f.offset == 4u);
        expect(skip_f.size == 4u);
    };

    "user_fields_with_alignment"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("token", Type::of<uint32_t>());// offset 0, size 4
        desc.add_field("skip", Type::of<uint32_t>()); // offset 4, size 4
        desc.add_field("x", Type::of<float>());       // offset 8, size 4
        desc.add_field("y", Type::of<int>());          // offset 12, size 4

        expect(desc.field_count() == 4u);
        expect(desc.total_size() == 16u);

        expect(desc.field(2u).name == "x");
        expect(desc.field(2u).offset == 8u);
        expect(desc.field(3u).name == "y");
        expect(desc.field(3u).offset == 12u);
    };

    "field_lookup_by_name"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("token", Type::of<uint32_t>());
        desc.add_field("skip", Type::of<uint32_t>());
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
        desc.add_field("token", Type::of<uint32_t>());
        desc.add_field("skip", Type::of<uint32_t>());

        auto *not_found = desc.field("nonexistent");
        expect(not_found == nullptr);
    };

    "field_lookup_by_index"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("token", Type::of<uint32_t>());
        desc.add_field("skip", Type::of<uint32_t>());
        desc.add_field("a", Type::of<float>());
        desc.add_field("b", Type::of<int>());
        desc.add_field("c", Type::of<uint32_t>());

        expect(desc.field_count() == 5u);

        expect(desc.field(0u).name == "token");
        expect(desc.field(1u).name == "skip");
        expect(desc.field(2u).name == "a");
        expect(desc.field(3u).name == "b");
        expect(desc.field(4u).name == "c");
    };

    "total_size_is_aligned"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("token", Type::of<uint32_t>());// 4 bytes
        desc.add_field("skip", Type::of<uint32_t>()); // 4 bytes
        // float3 has alignment 16, size 12
        desc.add_field("v", Type::of<float3>());       // offset 16, size 12

        expect(desc.field_count() == 3u);
        // v should be aligned to 16
        expect(desc.field(2u).offset == 16u);
        // total_size = 16 + 12 = 28, aligned to 16 → 32
        expect(desc.total_size() == 32u);
    };
}

void reg_coro_frame_desc_from_materialize() {

    "from_materialize_info_basic"_test = [] {
        // given: CoroMaterializeInfo with 2 user registers
        CoroMaterializeInfo info;
        info.name_to_field.emplace("x", 2u);
        info.name_to_field.emplace("y", 3u);
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
        info.name_to_field.emplace("z", 4u);
        info.name_to_field.emplace("x", 2u);
        info.name_to_field.emplace("y", 3u);
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
}

}// namespace

int main(int /* argc */, char * /* argv */[]) {
    reg_coro_frame_desc_manual();
    reg_coro_frame_desc_from_materialize();
    return 0;
}
