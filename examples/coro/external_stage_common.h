#pragma once

#include <algorithm>
#include <initializer_list>

#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>

namespace luisa::compute::coro::example {

/// A host-side view that keeps the normalized extension, its data-flow stage,
/// and the owner-indexed binding access plans together. The graph remains the
/// storage owner; this view never creates extension-private frame fields.
struct ExternalStageView {
    const CoroGraph::Boundary *boundary{nullptr};
    const CoroSuspendExtension *extension{nullptr};
    const CoroGraph::Stage *stage{nullptr};

    [[nodiscard]] const CoroSlotAccess &binding(
        luisa::string_view name) const noexcept {
        LUISA_ASSERT(boundary != nullptr && extension != nullptr,
                     "Invalid coroutine external-stage view.");
        for (auto &&descriptor : extension->bindings()) {
            if (descriptor.name == name) {
                LUISA_ASSERT(
                    descriptor.index < boundary->bindings.size(),
                    "Coroutine extension binding '{}' has invalid owner "
                    "index {}.",
                    name, descriptor.index);
                return boundary->bindings[descriptor.index];
            }
        }
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine extension '{}' has no binding named '{}'.",
            extension->schema(), name);
    }
};

/// Resolve exactly one schema instance. Examples deliberately reject
/// ambiguity instead of silently coalescing static suspend boundaries.
[[nodiscard]] inline luisa::vector<ExternalStageView> find_external_stages(
    const CoroGraph &graph, luisa::string_view schema) noexcept {
    luisa::vector<ExternalStageView> result;
    for (auto &&boundary : graph.boundaries()) {
        for (auto &&stage : boundary.stages) {
            LUISA_ASSERT(stage.extension_index < boundary.extensions.size(),
                         "Coroutine stage has invalid extension index {}.",
                         stage.extension_index);
            auto *extension =
                boundary.extensions[stage.extension_index].get();
            if (extension != nullptr && extension->schema() == schema) {
                result.emplace_back(ExternalStageView{
                    .boundary = &boundary,
                    .extension = extension,
                    .stage = &stage});
            }
        }
    }
    LUISA_ASSERT(!result.empty(),
                 "Coroutine has no external stage with schema '{}'.",
                 schema);
    return result;
}

[[nodiscard]] inline ExternalStageView find_external_stage(
    const CoroGraph &graph, luisa::string_view schema) noexcept {
    auto stages = find_external_stages(graph, schema);
    LUISA_ASSERT(
        stages.size() == 1u,
        "Coroutine external-stage schema '{}' has {} static boundaries; "
        "select one boundary before scheduling it.",
        schema, stages.size());
    return stages.front();
}

/// Combine the compiler-proved slot set with storage required by the user's
/// scheduler or application. The result is sorted and unique, so an existing
/// colored frame slot is never duplicated merely because several bindings or
/// policies refer to it.
[[nodiscard]] inline luisa::vector<size_t> merge_stage_slots(
    luisa::span<const size_t> proved,
    std::initializer_list<size_t> application_slots = {}) noexcept {
    luisa::vector<size_t> result{proved.begin(), proved.end()};
    result.insert(result.end(), application_slots.begin(),
                  application_slots.end());
    luisa::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}// namespace luisa::compute::coro::example
