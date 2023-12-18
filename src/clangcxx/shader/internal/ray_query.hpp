#pragma once
#include "attributes.hpp"

namespace luisa::shader {
struct RayQueryTracer {
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct RayQueryProceduralProxy {
    template<typename Func>
    [[scope("on_procedural_candidate")]] RayQueryTracer on_procedural_candidate(Func &&func) {
        func();
        return RayQueryTracer();
    }
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct RayQuerySurfaceProxy {
    template<typename Func>
    [[scope("on_surface_candidate")]] RayQueryTracer on_surface_candidate(Func &&func) {
        func();
        return RayQueryTracer();
    }
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct [[builtin("ray_query_all")]] RayQueryAll {
    template<typename Func>
    [[scope("on_surface_candidate")]] RayQueryProceduralProxy on_surface_candidate(Func &&func) {
        func();
        return RayQueryProceduralProxy();
    }
    template<typename Func>
    [[scope("on_procedural_candidate")]] RayQuerySurfaceProxy on_procedural_candidate(Func &&func) {
        func();
        return RayQuerySurfaceProxy();
    }
};

struct [[builtin("ray_query_any")]] RayQueryAny {
    template<typename Func>
    [[scope("on_surface_candidate")]] RayQueryProceduralProxy on_surface_candidate(Func &&func) {
        func();
        return RayQueryProceduralProxy();
    }
    template<typename Func>
    [[scope("on_procedural_candidate")]] RayQuerySurfaceProxy on_procedural_candidate(Func &&func) {
        func();
        return RayQuerySurfaceProxy();
    }
};
}// namespace luisa::shader