#pragma once

#include <luisa/core/stl/functional.h>
#include <luisa/ast/statement.h>
#include <luisa/dsl/rtx/hit.h>
#include <luisa/dsl/rtx/ray.h>

namespace luisa::compute {

namespace detail {
template<bool terminate_on_first>
class RayQueryBase;
}// namespace detail

// RayQuery DSL module, see test_procedural.cpp as example

class LC_DSL_API SurfaceCandidate {

private:
    const Expression *_query;

private:
    template<bool terminate_on_first>
    friend class detail::RayQueryBase;
    explicit SurfaceCandidate(const Expression *query) noexcept
        : _query{query} {}

public:
    SurfaceCandidate(SurfaceCandidate const &) noexcept = delete;
    SurfaceCandidate(SurfaceCandidate &&) noexcept = delete;
    SurfaceCandidate &operator=(SurfaceCandidate const &) noexcept = delete;
    SurfaceCandidate &operator=(SurfaceCandidate &&) noexcept = delete;

public:
    [[nodiscard]] Var<Ray> ray() const noexcept;
    [[nodiscard]] Var<TriangleHit> hit() const noexcept;
    void commit() const noexcept;
    void terminate() const noexcept;
};

// legacy names, provided for compatibility
using TriangleCandidate = SurfaceCandidate;

class LC_DSL_API ProceduralCandidate {

private:
    const Expression *_query;

private:
    template<bool terminate_on_first>
    friend class detail::RayQueryBase;
    explicit ProceduralCandidate(const Expression *query) noexcept
        : _query{query} {}

public:
    ProceduralCandidate(ProceduralCandidate const &) noexcept = delete;
    ProceduralCandidate(ProceduralCandidate &&) noexcept = delete;
    ProceduralCandidate &operator=(ProceduralCandidate const &) noexcept = delete;
    ProceduralCandidate &operator=(ProceduralCandidate &&) noexcept = delete;

public:
    [[nodiscard]] Var<Ray> ray() const noexcept;
    [[nodiscard]] Var<ProceduralHit> hit() const noexcept;
    void commit(Expr<float> distance) const noexcept;
    void terminate() const noexcept;
};

namespace detail {

template<bool terminate_on_first>
class LC_DSL_API RayQueryBase {

private:
    RayQueryStmt *_stmt;
    bool _surface_handler_set{false};
    bool _procedural_handler_set{false};
    bool _inside_surface_handler{false};
    bool _inside_procedural_handler{false};

public:
    using SurfaceCandidateHandler = luisa::function<void(SurfaceCandidate &)>;
    using ProceduralCandidateHandler = luisa::function<void(ProceduralCandidate &)>;

    // legacy names, provided for compatibility
    using TriangleCandidateHandler = SurfaceCandidateHandler;

private:
    friend struct Expr<Accel>;
    friend class compute::SurfaceCandidate;
    friend class compute::ProceduralCandidate;
    RayQueryBase(const Expression *accel,
                 const Expression *ray,
                 const Expression *mask) noexcept;
    RayQueryBase(RayQueryBase &&) noexcept;

public:
    virtual ~RayQueryBase() noexcept = default;
    RayQueryBase(RayQueryBase const &) noexcept = delete;
    RayQueryBase &operator=(RayQueryBase &&) noexcept = delete;
    RayQueryBase &operator=(RayQueryBase const &) noexcept = delete;

public:
    [[nodiscard, deprecated("Please use on_surface_candidate(), which unifies the code paths for triangles and curves.")]]// deprecated
    RayQueryBase
    on_triangle_candidate(const TriangleCandidateHandler &handler) && noexcept;

public:
    [[nodiscard]] RayQueryBase on_surface_candidate(const SurfaceCandidateHandler &handler) && noexcept;
    [[nodiscard]] RayQueryBase on_procedural_candidate(const ProceduralCandidateHandler &handler) && noexcept;
    [[nodiscard]] Var<CommittedHit> trace() const noexcept;
};

}// namespace detail

using RayQueryAny = detail::RayQueryBase<true>;
using RayQueryAll = detail::RayQueryBase<false>;

}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQueryAll, "LC_RayQueryAll")
LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQueryAny, "LC_RayQueryAny")
