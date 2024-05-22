#pragma once
#include "./../types/matrix.hpp"
#include "./../types/ray.hpp"
#include "./../raytracing/ray_query.hpp"

namespace luisa::shader {
struct [[builtin("accel")]] Accel {
    [[callop("RAY_TRACING_INSTANCE_TRANSFORM")]] float4x4 instance_transform(uint32 index);
    [[callop("RAY_TRACING_INSTANCE_USER_ID")]] uint32 instance_user_id(uint32 index);
    [[callop("RAY_TRACING_INSTANCE_VISIBILITY_MASK")]] uint32 instance_visibility_mask(uint32 index);
    [[callop("RAY_TRACING_SET_INSTANCE_TRANSFORM")]] void set_instance_transform(uint32 index, float4x4 transform);
    [[callop("RAY_TRACING_SET_INSTANCE_VISIBILITY")]] void set_instance_visibility(uint32 index, uint32 visibility);
    [[callop("RAY_TRACING_SET_INSTANCE_OPACITY")]] void set_instance_opacity(uint32 index, bool opacity);
    [[callop("RAY_TRACING_SET_INSTANCE_USER_ID")]] void set_instance_user_id(uint32 index, uint32 user_id);
    [[callop("RAY_TRACING_TRACE_CLOSEST")]] TriangleHit trace_closest(Ray ray, uint32 mask = ~0);
    [[callop("RAY_TRACING_TRACE_ANY")]] bool trace_any(Ray ray, uint32 mask = ~0);
    // TODO: how to design ray_query? That's a problem.
    [[callop("RAY_TRACING_QUERY_ALL")]] RayQueryAll query_all(Ray ray, uint32 mask = ~0);
    [[callop("RAY_TRACING_QUERY_ANY")]] RayQueryAny query_any(Ray ray, uint32 mask = ~0);
};
}// namespace luisa::shader