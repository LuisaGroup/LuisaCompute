#pragma once
#include "attributes.hpp"
#include "types/ray.hpp"

namespace luisa::shader {
struct [[builtin("ray_query_all")]] RayQueryAll {
    [[callop("RAY_QUERY_PROCEED")]] bool proceed();
    [[callop("RAY_QUERY_IS_TRIANGLE_CANDIDATE")]] bool is_triangle_candidate();
    [[callop("RAY_QUERY_IS_PROCEDURAL_CANDIDATE")]] bool is_procedural_candidate();
    [[callop("RAY_QUERY_WORLD_SPACE_RAY")]] Ray world_ray();
    [[callop("RAY_QUERY_PROCEDURAL_CANDIDATE_HIT")]] ProceduralHit procedural_candidate();
    [[callop("RAY_QUERY_TRIANGLE_CANDIDATE_HIT")]] TriangleHit triangle_candidate();
    [[callop("RAY_QUERY_COMMITTED_HIT")]] CommittedHit committed_hit();
    [[callop("RAY_QUERY_COMMIT_TRIANGLE")]] void commit_triangle();
    [[callop("RAY_QUERY_COMMIT_PROCEDURAL")]] void commit_procedural(float distance);
    [[callop("RAY_QUERY_TERMINATE")]] void terminate();
};
}// namespace luisa::shader