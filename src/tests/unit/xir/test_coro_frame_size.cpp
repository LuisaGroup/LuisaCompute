#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/dead_field_elimination.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void reg_coro_frame_size() {

    "frame_size_reduction_5_fields_3_dead"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: reserved fields precede a, b, c, d, e
        // a and d are loaded -> alive; b, c, e are dead
        CoroMaterializeInfo info;
        info.frame_field_count = r + 5u;
        info.name_to_field.emplace("a", r);
        info.name_to_field.emplace("b", r + 1u);
        info.name_to_field.emplace("c", r + 2u);
        info.name_to_field.emplace("d", r + 3u);
        info.name_to_field.emplace("e", r + 4u);
        info.name_to_type.emplace("a", Type::of<float>());
        info.name_to_type.emplace("b", Type::of<int>());
        info.name_to_type.emplace("c", Type::of<uint32_t>());
        info.name_to_type.emplace("d", Type::of<float>());
        info.name_to_type.emplace("e", Type::of<float>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r, r + 3u};// only a and d loaded
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);
        auto size_before = desc.total_size();
        expect(size_before > 0u);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: 3 fields eliminated, frame size reduced
        expect(result.original_field_count == r + 5u);
        expect(result.eliminated_field_count == 3u);
        expect(result.remaining_field_count == r + 2u);
        expect(result.original_frame_size == size_before);
        expect(result.new_frame_size < result.original_frame_size);
        expect(result.eliminated_field_indices.contains(r + 1u));
        expect(result.eliminated_field_indices.contains(r + 2u));
        expect(result.eliminated_field_indices.contains(r + 4u));

        // desc rebuilt with a, d only
        expect(desc.total_size() == result.new_frame_size);
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "a");
        expect(desc.field(1u).name == "d");

        // info remapped correctly
        expect(info.frame_field_count == r + 2u);
        expect(info.name_to_field.contains("a"));
        expect(info.name_to_field.contains("d"));
        expect(!info.name_to_field.contains("b"));
        expect(!info.name_to_field.contains("c"));
        expect(!info.name_to_field.contains("e"));
    };

    "frame_size_unchanged_all_fields_used"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: 3 user fields, all loaded → no elimination
        CoroMaterializeInfo info;
        info.frame_field_count = r + 3u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_field.emplace("z", r + 2u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());
        info.name_to_type.emplace("z", Type::of<double>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r, r + 1u, r + 2u};// all loaded
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);
        auto size_before = desc.total_size();
        expect(size_before > 0u);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: no elimination, size unchanged
        expect(result.original_field_count == r + 3u);
        expect(result.eliminated_field_count == 0u);
        expect(result.remaining_field_count == r + 3u);
        expect(result.eliminated_field_indices.empty());
        expect(result.new_frame_size == result.original_frame_size);
        expect(desc.total_size() == size_before);
        expect(desc.field_count() == 3u);
    };

    "frame_size_header_only_no_reduction"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: only reserved fields, no user fields
        CoroMaterializeInfo info;
        info.frame_field_count = r;

        // no edges -> no loads, but reserved fields are always kept
        CoroFrameDesc desc;
        desc.from_materialize_info(info);
        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: nothing to eliminate, no size change
        expect(result.original_field_count == r);
        expect(result.eliminated_field_count == 0u);
        expect(result.remaining_field_count == r);
        expect(result.eliminated_field_indices.empty());
        expect(result.new_frame_size == result.original_frame_size);
        expect(desc.total_size() == 0u);
    };

    "frame_size_empty_frame"_test = [] {
        // given: empty CoroFrameDesc with no fields at all
        CoroMaterializeInfo info;
        info.frame_field_count = 0u;

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: no fields to eliminate
        expect(result.original_field_count == 0u);
        expect(result.eliminated_field_count == 0u);
        expect(result.remaining_field_count == 0u);
        expect(result.eliminated_field_indices.empty());
        expect(result.original_frame_size == 0u);
        expect(result.new_frame_size == 0u);
        expect(desc.total_size() == 0u);
    };
}

}// namespace

int main(int /* argc */, char * /* argv */[]) {
    reg_coro_frame_size();
    return 0;
}
