#pragma once
#include "attributes.hpp"

namespace luisa::shader {
struct RayQueryTracer {
    void trace();
};

struct RayQueryProceduralProxy {
    template<typename Func>
    RayQueryTracer on_procedural_candidate(Func &&func);
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
};

struct RayQuerySurfaceProxy {
    template<typename Func>
    RayQueryTracer on_surface_candidate(Func &&func);
    [[callop("RAY_QUERY_COMMITTED_HIT")]] void trace();
};

struct [[builtin("ray_query_all")]] RayQueryAll {
    template<typename Func>
    RayQueryProceduralProxy on_surface_candidate(Func &&func);
    template<typename Func>
    RayQuerySurfaceProxy on_procedural_candidate(Func &&func);
};

struct [[builtin("ray_query_any")]] RayQueryAny {
    template<typename Func>
    RayQueryProceduralProxy on_surface_candidate(Func &&func);
    template<typename Func>
    RayQuerySurfaceProxy on_procedural_candidate(Func &&func);
};
}// namespace luisa::shader