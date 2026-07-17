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

void reg_coro_dead_field() {

    "one_dead_field_eliminated"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: 3 user fields, only x and y are loaded
        CoroMaterializeInfo info;
        info.frame_field_count = r + 3u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_field.emplace("z", r + 2u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());
        info.name_to_type.emplace("z", Type::of<uint32_t>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r, r + 1u};// x and y are loaded, z is never loaded
        edge.store_fields = {r, r + 1u, r + 2u};
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: z eliminated
        expect(result.original_field_count == r + 3u);
        expect(result.eliminated_field_count == 1u);
        expect(result.remaining_field_count == r + 2u);
        expect(result.eliminated_field_indices.contains(r + 2u));

        // info.name_to_field has z removed, x and y remain after reserved fields
        expect(info.name_to_field.contains("x"));
        expect(info.name_to_field.contains("y"));
        expect(!info.name_to_field.contains("z"));
        expect(info.name_to_type.contains("x"));
        expect(info.name_to_type.contains("y"));
        expect(!info.name_to_type.contains("z"));
        expect(info.name_to_field.at("x") == r);
        expect(info.name_to_field.at("y") == r + 1u);
        expect(info.frame_field_count == r + 2u);

        // desc rebuilt with only x, y
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "x");
        expect(desc.field(1u).name == "y");

        // edge fields remapped
        expect(info.edges.size() == 1u);
        expect(info.edges[0].load_fields.size() == 2u);
        expect(info.edges[0].load_fields[0] == r);
        expect(info.edges[0].load_fields[1] == r + 1u);
        expect(info.edges[0].store_fields.size() == 2u);
        expect(info.edges[0].store_fields[0] == r);
        expect(info.edges[0].store_fields[1] == r + 1u);
    };

    "all_fields_used_no_elimination"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: 2 user fields, both loaded
        CoroMaterializeInfo info;
        info.frame_field_count = r + 2u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r, r + 1u};
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: no elimination
        expect(result.original_field_count == r + 2u);
        expect(result.eliminated_field_count == 0u);
        expect(result.remaining_field_count == r + 2u);
        expect(result.eliminated_field_indices.empty());

        expect(info.frame_field_count == r + 2u);
        expect(info.name_to_field.at("x") == r);
        expect(info.name_to_field.at("y") == r + 1u);
        expect(desc.field_count() == 2u);
    };

    "store_only_field_eliminated"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: field y is stored but never loaded by any edge
        CoroMaterializeInfo info;
        info.frame_field_count = r + 2u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r};         // only x loaded
        edge.store_fields = {r, r + 1u};// y stored but never loaded -> dead
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: y eliminated (store-only, never loaded)
        expect(result.eliminated_field_count == 1u);
        expect(result.eliminated_field_indices.contains(r + 1u));
        expect(result.remaining_field_count == r + 1u);

        expect(info.frame_field_count == r + 1u);
        expect(info.name_to_field.contains("x"));
        expect(!info.name_to_field.contains("y"));
        expect(info.name_to_field.at("x") == r);
        expect(desc.field_count() == 1u);
        expect(desc.field(0u).name == "x");

        // edge fields remapped
        expect(info.edges[0].load_fields.size() == 1u);
        expect(info.edges[0].load_fields[0] == r);
        expect(info.edges[0].store_fields.size() == 1u);
        expect(info.edges[0].store_fields[0] == r);
    };

    "reserved_fields_never_eliminated"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: only 1 user field, never loaded → dead
        // all reserved fields must survive
        CoroMaterializeInfo info;
        info.frame_field_count = r + 1u;
        info.name_to_field.emplace("x", r);
        info.name_to_type.emplace("x", Type::of<float>());

        // no edges at all → x is never loaded → dead
        // but reserved fields must survive

        CoroFrameDesc desc;
        desc.from_materialize_info(info);
        auto orig_size = desc.total_size();

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: x eliminated, but reserved fields survive
        expect(result.original_field_count == r + 1u);
        expect(result.eliminated_field_count == 1u);
        expect(result.remaining_field_count == r);
        expect(result.eliminated_field_indices.contains(r));
        for (auto i = 0u; i < r; i++) {
            expect(!result.eliminated_field_indices.contains(i));
        }

        expect(info.frame_field_count == r);
        expect(!info.name_to_field.contains("x"));
        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);
    };

    "frame_size_reduction_verified"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: 3 user fields, middle one dead
        CoroMaterializeInfo info;
        info.frame_field_count = r + 3u;
        info.name_to_field.emplace("a", r);
        info.name_to_field.emplace("b", r + 1u);
        info.name_to_field.emplace("c", r + 2u);
        info.name_to_type.emplace("a", Type::of<float>());
        info.name_to_type.emplace("b", Type::of<uint32_t>());
        info.name_to_type.emplace("c", Type::of<float>());

        // a loaded, b dead, c loaded
        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r, r + 2u};
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);
        auto orig_size = desc.total_size();
        expect(orig_size > 0u);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: b eliminated, size decreased
        expect(result.eliminated_field_count == 1u);
        expect(result.new_frame_size < result.original_frame_size);
        expect(desc.total_size() == result.new_frame_size);
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "a");
        expect(desc.field(1u).name == "c");
    };

    "multiple_edges_union_load_fields"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: field y loaded only in edge 2, x loaded in edge 1, z never loaded
        CoroMaterializeInfo info;
        info.frame_field_count = r + 3u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_field.emplace("z", r + 2u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());
        info.name_to_type.emplace("z", Type::of<uint32_t>());

        CoroMaterializeInfo::TransitionEdge edge1;
        edge1.from_scope = 0u;
        edge1.to_scope = 1u;
        edge1.load_fields = {r};// x is loaded

        CoroMaterializeInfo::TransitionEdge edge2;
        edge2.from_scope = 1u;
        edge2.to_scope = 2u;
        edge2.load_fields = {r + 1u};// y is loaded

        info.edges.push_back(edge1);
        info.edges.push_back(edge2);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: only z is dead (never loaded in any edge)
        expect(result.eliminated_field_count == 1u);
        expect(result.eliminated_field_indices.contains(r + 2u));
        expect(!result.eliminated_field_indices.contains(r));
        expect(!result.eliminated_field_indices.contains(r + 1u));

        expect(info.frame_field_count == r + 2u);
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "x");
        expect(desc.field(1u).name == "y");

        // edge1: x remains at the first user field
        expect(info.edges[0].load_fields[0] == r);
        // edge2: y remains at the second user field
        expect(info.edges[1].load_fields[0] == r + 1u);
    };

    "empty_edges_all_user_fields_dead"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: edges exist but have no load_fields
        CoroMaterializeInfo info;
        info.frame_field_count = r + 2u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        // no load_fields at all
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: all user fields dead, reserved fields survive
        expect(result.eliminated_field_count == 2u);
        expect(result.remaining_field_count == r);
        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);
    };

    "no_edges_fallback"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: no edges at all
        CoroMaterializeInfo info;
        info.frame_field_count = r + 2u;
        info.name_to_field.emplace("x", r);
        info.name_to_field.emplace("y", r + 1u);
        info.name_to_type.emplace("x", Type::of<float>());
        info.name_to_type.emplace("y", Type::of<int>());

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: all user fields dead (nothing loads them)
        expect(result.eliminated_field_count == 2u);
        expect(result.remaining_field_count == r);
        expect(desc.field_count() == 0u);
    };

    "index_remapping_preserves_gaps"_test = [] {
        auto r = CoroFrameDesc::reserved_field_count;
        // given: 4 user fields; b and c are dead
        CoroMaterializeInfo info;
        info.frame_field_count = r + 4u;
        info.name_to_field.emplace("a", r);
        info.name_to_field.emplace("b", r + 1u);
        info.name_to_field.emplace("c", r + 2u);
        info.name_to_field.emplace("d", r + 3u);
        info.name_to_type.emplace("a", Type::of<float>());
        info.name_to_type.emplace("b", Type::of<int>());
        info.name_to_type.emplace("c", Type::of<uint32_t>());
        info.name_to_type.emplace("d", Type::of<float>());

        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = 0u;
        edge.to_scope = 1u;
        edge.load_fields = {r, r + 3u};// a and d are loaded; b and c are dead
        info.edges.push_back(edge);

        CoroFrameDesc desc;
        desc.from_materialize_info(info);

        // when
        auto result = dead_field_elimination_pass_run(info, desc);

        // then: b and c eliminated
        expect(result.eliminated_field_count == 2u);
        expect(result.eliminated_field_indices.contains(r + 1u));
        expect(result.eliminated_field_indices.contains(r + 2u));
        expect(result.remaining_field_count == r + 2u);
        expect(info.frame_field_count == r + 2u);

        // name_to_field: a remains first user field, d moves to second user field
        expect(info.name_to_field.contains("a"));
        expect(info.name_to_field.contains("d"));
        expect(!info.name_to_field.contains("b"));
        expect(!info.name_to_field.contains("c"));
        expect(info.name_to_field.at("a") == r);
        expect(info.name_to_field.at("d") == r + 1u);

        // edge remapped: a stays at r, d moves from r+3 to r+1
        expect(info.edges[0].load_fields.size() == 2u);
        expect(info.edges[0].load_fields[0] == r);
        expect(info.edges[0].load_fields[1] == r + 1u);

        // desc rebuilt with a, d only
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "a");
        expect(desc.field(1u).name == "d");
    };
}

}// namespace

int main(int /* argc */, char * /* argv */[]) {
    reg_coro_dead_field();
    return 0;
}
