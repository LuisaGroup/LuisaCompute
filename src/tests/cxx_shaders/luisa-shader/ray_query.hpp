#pragma once
#include "attributes.hpp"
#include "types/ray.hpp"

namespace luisa::shader {
struct SurfaceCandidate {
    [[expr("ray")]] Ray ray() const { return Ray(); }
    [[expr("tri_hit")]] TriangleHit hit() const { return TriangleHit(); }
    [[expr("tri_commit")]] void commit() const {}
    [[expr("terminate")]] void terminate() const {}
    float _;
};

struct ProceduralCandidate {
    [[expr("ray")]] Ray ray() const { return Ray(); }
    [[expr("procedual_hit")]] ProceduralHit hit() const { return ProceduralHit(); }
    [[expr("procedual_commit")]] void commit(float distance) const {}
    [[expr("terminate")]] void terminate() const {}
    float _;
};

struct RayQueryTracer {
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct RayQueryProceduralProxy {
    template<typename Func>
    [[noignore]] RayQueryTracer on_procedural_candidate(Func &&func) {
        ProceduralCandidate s;
        func(s);
        return RayQueryTracer();
    }
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct RayQuerySurfaceProxy {
    template<typename Func>
    [[noignore]] RayQueryTracer on_surface_candidate(Func &&func) {
        SurfaceCandidate s;
        func(s);
        return RayQueryTracer();
    }
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct [[builtin("ray_query_all")]] RayQueryAll {
    template<typename Func>
    [[noignore]] RayQueryProceduralProxy on_surface_candidate(Func &&func) {
        SurfaceCandidate s;
        func(s);
        return RayQueryProceduralProxy();
    }
    template<typename Func>
    [[noignore]] RayQuerySurfaceProxy on_procedural_candidate(Func &&func) {
        ProceduralCandidate s;
        func(s);
        return RayQuerySurfaceProxy();
    }
};

struct [[builtin("ray_query_any")]] RayQueryAny {
    template<typename Func>
    [[noignore]] RayQueryProceduralProxy on_surface_candidate(Func &&func) {
        SurfaceCandidate s;
        func(s);
        return RayQueryProceduralProxy();
    }
    template<typename Func>
    [[noignore]] RayQuerySurfaceProxy on_procedural_candidate(Func &&func) {
        ProceduralCandidate s;
        func(s);
        return RayQuerySurfaceProxy();
    }
};

}// namespace luisa::shader