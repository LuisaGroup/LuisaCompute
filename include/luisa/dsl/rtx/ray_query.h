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
class RayQuerySurfaceProxy;

template<bool terminate_on_first>
class RayQueryProceduralProxy;

template<bool terminate_on_first>
class LC_DSL_API RayQueryBase {

protected:
    RayQueryStmt *_stmt;
    bool _inside_surface_handler{false};
    bool _inside_procedural_handler{false};

public:
    using SurfaceCandidateHandler = luisa::function<void(SurfaceCandidate &)>;
    using ProceduralCandidateHandler = luisa::function<void(ProceduralCandidate &)>;

protected:
    RayQueryBase(const Expression *accel,
                 const Expression *ray,
                 const Expression *mask) noexcept;

    RayQueryBase(const Expression *accel,
                 const Expression *ray,
                 const Expression *time,
                 const Expression *mask) noexcept;

    RayQueryBase(RayQueryStmt *stmt,
                 bool inside_surface_handler,
                 bool inside_procedural_handler) noexcept
        : _stmt(stmt),
          _inside_surface_handler(inside_surface_handler),
          _inside_procedural_handler(inside_procedural_handler) {}

public:
    virtual ~RayQueryBase() noexcept = default;
    RayQueryBase(RayQueryBase &&) noexcept;
    RayQueryBase(RayQueryBase const &) noexcept = delete;
    RayQueryBase &operator=(RayQueryBase &&) noexcept = delete;
    RayQueryBase &operator=(RayQueryBase const &) noexcept = delete;

public:
    [[nodiscard]] Var<CommittedHit> trace() const noexcept;

protected:
    [[nodiscard]] RayQueryProceduralProxy<terminate_on_first> _on_surface_candidate(const SurfaceCandidateHandler &handler) noexcept;
    [[nodiscard]] RayQuerySurfaceProxy<terminate_on_first> _on_procedural_candidate(const ProceduralCandidateHandler &handler) noexcept;
};

template<bool terminate_on_first>
class RayQueryProceduralProxy : public RayQueryBase<terminate_on_first> {
private:
    friend struct Expr<Accel>;
    friend class compute::SurfaceCandidate;
    friend class compute::ProceduralCandidate;
    friend class RayQueryBase<terminate_on_first>;
    RayQueryProceduralProxy(RayQueryStmt *stmt,
                            bool inside_surface_handler,
                            bool inside_procedural_handler) noexcept
        : RayQueryBase<terminate_on_first>(stmt, inside_surface_handler, inside_procedural_handler) {}
    RayQueryProceduralProxy(RayQueryProceduralProxy &&rhs) noexcept = default;
public:
    using ProceduralCandidateHandler = luisa::function<void(ProceduralCandidate &)>;

    // legacy names, provided for compatibility

    RayQueryProceduralProxy(RayQueryProceduralProxy const &) noexcept = delete;
    RayQueryProceduralProxy &operator=(RayQueryProceduralProxy &&) noexcept = delete;
    RayQueryProceduralProxy &operator=(RayQueryProceduralProxy const &) noexcept = delete;
    [[nodiscard]] RayQueryBase<terminate_on_first> on_procedural_candidate(const ProceduralCandidateHandler &handler) && noexcept;
};

template<bool terminate_on_first>
class RayQuerySurfaceProxy : public RayQueryBase<terminate_on_first> {

private:
    friend struct Expr<Accel>;
    friend class compute::SurfaceCandidate;
    friend class compute::ProceduralCandidate;
    friend class RayQueryBase<terminate_on_first>;

    RayQuerySurfaceProxy(RayQueryStmt *stmt,
                         bool inside_surface_handler,
                         bool inside_procedural_handler) noexcept
        : RayQueryBase<terminate_on_first>(stmt, inside_surface_handler, inside_procedural_handler) {}
    RayQuerySurfaceProxy(RayQuerySurfaceProxy &&rhs) noexcept = default;

public:
    using SurfaceCandidateHandler = luisa::function<void(SurfaceCandidate &)>;
    // legacy names, provided for compatibility
    using TriangleCandidateHandler = SurfaceCandidateHandler;

    RayQuerySurfaceProxy(RayQuerySurfaceProxy const &) noexcept = delete;
    RayQuerySurfaceProxy &operator=(RayQuerySurfaceProxy &&) noexcept = delete;
    RayQuerySurfaceProxy &operator=(RayQuerySurfaceProxy const &) noexcept = delete;
    [[nodiscard]] RayQueryBase<terminate_on_first> on_surface_candidate(const SurfaceCandidateHandler &handler) && noexcept;
    [[nodiscard, deprecated("Please use on_surface_candidate(), which unifies the code paths for triangles and curves.")]]// deprecated
    RayQueryBase<terminate_on_first>
    on_triangle_candidate(const TriangleCandidateHandler &handler) && noexcept;
};

template<bool terminate_on_first>
class RayQueryProxy : public RayQueryBase<terminate_on_first> {

private:
    friend struct Expr<Accel>;
    friend class compute::SurfaceCandidate;
    friend class compute::ProceduralCandidate;
    friend class RayQueryBase<terminate_on_first>;

    RayQueryProxy(const Expression *accel,
                  const Expression *ray,
                  const Expression *mask) noexcept
        : RayQueryBase<terminate_on_first>(accel, ray, mask) {}

    RayQueryProxy(const Expression *accel,
                  const Expression *ray,
                  const Expression *time,
                  const Expression *mask) noexcept
        : RayQueryBase<terminate_on_first>(accel, ray, time, mask) {}

    RayQueryProxy(RayQueryProxy &&rhs) noexcept = default;

public:
    using SurfaceCandidateHandler = luisa::function<void(SurfaceCandidate &)>;
    using ProceduralCandidateHandler = luisa::function<void(ProceduralCandidate &)>;

    // legacy names, provided for compatibility
    using TriangleCandidateHandler = SurfaceCandidateHandler;

    RayQueryProxy(RayQueryProxy const &) noexcept = delete;
    RayQueryProxy &operator=(RayQueryProxy &&) noexcept = delete;
    RayQueryProxy &operator=(RayQueryProxy const &) noexcept = delete;
    [[nodiscard]] RayQueryProceduralProxy<terminate_on_first> on_surface_candidate(const SurfaceCandidateHandler &handler) && noexcept;
    [[nodiscard]] RayQuerySurfaceProxy<terminate_on_first> on_procedural_candidate(const ProceduralCandidateHandler &handler) && noexcept;
    [[nodiscard, deprecated("Please use on_surface_candidate(), which unifies the code paths for triangles and curves.")]]// deprecated
    RayQueryProceduralProxy<terminate_on_first>
    on_triangle_candidate(const TriangleCandidateHandler &handler) && noexcept;
};

template<bool terminate_on_first>
inline RayQueryProceduralProxy<terminate_on_first> RayQueryProxy<terminate_on_first>::on_surface_candidate(const SurfaceCandidateHandler &handler) && noexcept {
    return this->_on_surface_candidate(handler);
}
template<bool terminate_on_first>
inline RayQueryProceduralProxy<terminate_on_first> RayQueryProxy<terminate_on_first>::on_triangle_candidate(const TriangleCandidateHandler &handler) && noexcept {
    return this->_on_surface_candidate(handler);
}
template<bool terminate_on_first>
inline RayQuerySurfaceProxy<terminate_on_first> RayQueryProxy<terminate_on_first>::on_procedural_candidate(const ProceduralCandidateHandler &handler) && noexcept {
    return this->_on_procedural_candidate(handler);
}
template<bool terminate_on_first>
inline RayQueryBase<terminate_on_first> RayQueryProceduralProxy<terminate_on_first>::on_procedural_candidate(const ProceduralCandidateHandler &handler) && noexcept {
    return this->_on_procedural_candidate(handler);
}
template<bool terminate_on_first>
inline RayQueryBase<terminate_on_first> RayQuerySurfaceProxy<terminate_on_first>::on_surface_candidate(const SurfaceCandidateHandler &handler) && noexcept {
    return this->_on_surface_candidate(handler);
}
template<bool terminate_on_first>
inline RayQueryBase<terminate_on_first> RayQuerySurfaceProxy<terminate_on_first>::on_triangle_candidate(const TriangleCandidateHandler &handler) && noexcept {
    return this->_on_surface_candidate(handler);
}
}// namespace detail

using RayQueryAny = detail::RayQueryProxy<true>;
using RayQueryAll = detail::RayQueryProxy<false>;

}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQueryAll, "LC_RayQueryAll")
LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQueryAny, "LC_RayQueryAny")
