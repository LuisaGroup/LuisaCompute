#pragma once
#include "attributes.hpp"

namespace luisa::shader {
struct RayQueryTracer {
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct RayQueryProceduralProxy {
    template<typename Func>
    [[noignore]] RayQueryTracer on_procedural_candidate(Func &&func) {
        func();
        return RayQueryTracer();
    }
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct RayQuerySurfaceProxy {
    template<typename Func>
    [[noignore]] RayQueryTracer on_surface_candidate(Func &&func) {
        func();
        return RayQueryTracer();
    }
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
    float _;
};

struct [[builtin("ray_query_all")]] RayQueryAll {
    template<typename Func>
    [[noignore]] RayQueryProceduralProxy on_surface_candidate(Func &&func) {
        func();
        return RayQueryProceduralProxy();
    }
    template<typename Func>
    [[noignore]] RayQuerySurfaceProxy on_procedural_candidate(Func &&func) {
        func();
        return RayQuerySurfaceProxy();
    }
};

struct [[builtin("ray_query_any")]] RayQueryAny {
    template<typename Func>
    [[noignore]] RayQueryProceduralProxy on_surface_candidate(Func &&func) {
        func();
        return RayQueryProceduralProxy();
    }
    template<typename Func>
    [[noignore]] RayQuerySurfaceProxy on_procedural_candidate(Func &&func) {
        func();
        return RayQuerySurfaceProxy();
    }
};
}// namespace luisa::shader