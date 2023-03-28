#pragma once

#include <core/stl/functional.h>
#include <ast/statement.h>
#include <dsl/rtx/hit.h>

namespace luisa::compute {

namespace detail {
class RayQueryBuilder;
}

// RayQuery DSL module, see test_procedural.cpp as example
class LC_DSL_API RayQuery {

private:
    const Expression *_expr;

private:
    friend class detail::RayQueryBuilder;
    explicit RayQuery(const Expression *expr) noexcept
        : _expr{expr} {}

public:
    RayQuery(RayQuery &&) noexcept = default;
    RayQuery(RayQuery const &) noexcept = delete;
    RayQuery &operator=(RayQuery &&) noexcept = delete;
    RayQuery &operator=(RayQuery const &) noexcept = delete;
    [[nodiscard]] Var<CommittedHit> committed_hit() const noexcept;
};

class LC_DSL_API TriangleCandidate {

private:
    const detail::RayQueryBuilder *_builder;
    const Expression *_query;

private:
    friend class detail::RayQueryBuilder;
    TriangleCandidate(const detail::RayQueryBuilder *builder,
                      const Expression *query) noexcept
        : _builder{builder}, _query{query} {}

public:
    TriangleCandidate(TriangleCandidate const &) noexcept = delete;
    TriangleCandidate(TriangleCandidate &&) noexcept = delete;
    TriangleCandidate &operator=(TriangleCandidate const &) noexcept = delete;
    TriangleCandidate &operator=(TriangleCandidate &&) noexcept = delete;

public:
    [[nodiscard]] Var<TriangleHit> hit() const noexcept;
    void commit() const noexcept;
    void terminate() const noexcept;
};

class LC_DSL_API ProceduralCandidate {

private:
    const detail::RayQueryBuilder *_builder;
    const Expression *_query;

private:
    friend class detail::RayQueryBuilder;
    ProceduralCandidate(const detail::RayQueryBuilder *builder,
                        const Expression *query) noexcept
        : _builder{builder}, _query{query} {}

public:
    ProceduralCandidate(ProceduralCandidate const &) noexcept = delete;
    ProceduralCandidate(ProceduralCandidate &&) noexcept = delete;
    ProceduralCandidate &operator=(ProceduralCandidate const &) noexcept = delete;
    ProceduralCandidate &operator=(ProceduralCandidate &&) noexcept = delete;

public:
    [[nodiscard]] Var<ProceduralHit> hit() const noexcept;
    void commit(Expr<float> distance) const noexcept;
    void terminate() const noexcept;
};

namespace detail {

class LC_DSL_API RayQueryBuilder {

private:
    RayQueryStmt *_stmt;
    bool _triangle_handler_set{false};
    bool _procedural_handler_set{false};
    bool _inside_triangle_handler{false};
    bool _inside_procedural_handler{false};
    bool _queried{false};

public:
    using TriangleCandidateHandler = luisa::function<void(const TriangleCandidate &)>;
    using ProceduralCandidateHandler = luisa::function<void(const ProceduralCandidate &)>;

private:
    friend class Expr<Accel>;
    friend class compute::TriangleCandidate;
    friend class compute::ProceduralCandidate;
    RayQueryBuilder(const Expression *accel,
                    const Expression *ray,
                    const Expression *mask) noexcept;
    RayQueryBuilder(RayQueryBuilder &&) noexcept;

public:
    ~RayQueryBuilder() noexcept;
    RayQueryBuilder(RayQueryBuilder const &) noexcept = delete;
    RayQueryBuilder &operator=(RayQueryBuilder &&) noexcept = delete;
    RayQueryBuilder &operator=(RayQueryBuilder const &) noexcept = delete;

public:
    [[nodiscard]] RayQuery query() &&noexcept;
    [[nodiscard]] RayQueryBuilder on_triangle_candidate(const TriangleCandidateHandler &handler) &&noexcept;
    [[nodiscard]] RayQueryBuilder on_procedural_candidate(const ProceduralCandidateHandler &handler) &&noexcept;
};

}// namespace detail

}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQuery, "LC_RayQuery")
