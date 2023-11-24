#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {
struct RayQueryProceduralProxy {
    template<typename Func>
    void on_procedural_candidate(Func &&func);
};
struct RayQuerySurfaceProxy {
    template<typename Func>
    void on_surface_candidate(Func &&func);
};
struct [[builtin("ray_query")]] RayQuery {
    template<typename Func>
    RayQueryProceduralProxy on_surface_candidate(Func &&func);
    template<typename Func>
    RayQuerySurfaceProxy on_procedural_candidate(Func &&func);
};
}// namespace luisa::shader