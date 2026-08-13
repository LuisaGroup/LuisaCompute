//
// HIP-specific transformations of LLVM pass-pipeline descriptions.
//

#include "hip_llvm_pipeline.h"

#include <algorithm>
#include <limits>
#include <string_view>

namespace luisa::compute::hip {

bool preserve_generated_callable_boundary(
    size_t instruction_count) noexcept {
    return instruction_count >
           generated_callable_inline_instruction_budget;
}

std::vector<std::uint8_t>
select_generated_callable_boundaries(
    std::span<const GeneratedCallableInlineGraphNode> graph,
    size_t instruction_budget) noexcept {
    const auto node_count = graph.size();
    auto preserve = std::vector<std::uint8_t>(node_count, false);
    for (auto node_index = size_t{0u}; node_index < node_count;
         node_index++) {
        preserve[node_index] =
            graph[node_index].instruction_count > instruction_budget;
    }

    // Tarjan SCC decomposition. LLVM cannot finitely inline a recursive SCC;
    // making every member a boundary gives the residual-size recurrence below
    // a well-founded DAG and agrees with that semantic limitation.
    constexpr auto unvisited = std::numeric_limits<size_t>::max();
    auto discovery = std::vector<size_t>(node_count, unvisited);
    auto low_link = std::vector<size_t>(node_count, unvisited);
    auto on_stack = std::vector<std::uint8_t>(node_count, false);
    auto stack = std::vector<size_t>{};
    stack.reserve(node_count);
    auto next_discovery = size_t{0u};
    auto visit_scc = [&](auto &&self, size_t node_index) noexcept -> void {
        discovery[node_index] = next_discovery;
        low_link[node_index] = next_discovery;
        next_discovery++;
        stack.emplace_back(node_index);
        on_stack[node_index] = true;
        for (auto callee : graph[node_index].callees) {
            if (callee >= node_count) { continue; }
            if (discovery[callee] == unvisited) {
                self(self, callee);
                low_link[node_index] =
                    std::min(low_link[node_index], low_link[callee]);
            } else if (on_stack[callee]) {
                low_link[node_index] =
                    std::min(low_link[node_index], discovery[callee]);
            }
        }
        if (low_link[node_index] != discovery[node_index]) { return; }
        auto component = std::vector<size_t>{};
        while (true) {
            auto member = stack.back();
            stack.pop_back();
            on_stack[member] = false;
            component.emplace_back(member);
            if (member == node_index) { break; }
        }
        auto recursive = component.size() > 1u;
        if (!recursive) {
            const auto member = component.front();
            recursive = std::find(
                            graph[member].callees.begin(),
                            graph[member].callees.end(),
                            member) != graph[member].callees.end();
        }
        if (recursive) {
            for (auto member : component) { preserve[member] = true; }
        }
    };
    for (auto node_index = size_t{0u}; node_index < node_count;
         node_index++) {
        if (discovery[node_index] == unvisited) {
            visit_scc(visit_scc, node_index);
        }
    }

    constexpr auto saturation_limit =
        std::numeric_limits<size_t>::max();
    auto saturated_add = [saturation_limit](size_t lhs,
                                            size_t rhs) noexcept {
        if (lhs >= saturation_limit ||
            rhs >= saturation_limit - lhs) {
            return saturation_limit;
        }
        return lhs + rhs;
    };
    enum struct VisitState : std::uint8_t {
        unvisited,
        visiting,
        complete,
    };
    // Boundary selection is a monotone fixed point. A callee selected by one
    // caller can also reduce the modeled expansion of an already visited
    // caller, so recomputing all residuals between selection rounds makes the
    // result independent of graph storage/traversal order. At least one new
    // node is preserved per non-terminal round, hence termination in at most
    // |V| rounds.
    while (true) {
        auto residual_size = std::vector<size_t>(node_count, 1u);
        auto state = std::vector<VisitState>(
            node_count, VisitState::unvisited);
        auto compute_residual =
            [&](auto &&self, size_t node_index) noexcept -> size_t {
            if (state[node_index] == VisitState::complete) {
                return residual_size[node_index];
            }
            // Recursive edges already have preserved endpoints, so they do
            // not participate in expansion. This is a conservative guard for
            // malformed graphs that violate that invariant.
            if (state[node_index] == VisitState::visiting) { return 1u; }
            state[node_index] = VisitState::visiting;
            const auto &node = graph[node_index];
            auto expanded_size =
                std::min(node.instruction_count, saturation_limit);
            for (auto callee : node.callees) {
                if (callee >= node_count || preserve[callee]) { continue; }
                const auto callee_size = self(self, callee);
                // The call instruction itself is already part of I(v).
                const auto callee_expansion =
                    callee_size > 1u ? callee_size - 1u : 0u;
                expanded_size = saturated_add(
                    expanded_size, callee_expansion);
            }
            residual_size[node_index] = expanded_size;
            state[node_index] = VisitState::complete;
            return expanded_size;
        };
        for (auto node_index = size_t{0u}; node_index < node_count;
             node_index++) {
            static_cast<void>(compute_residual(
                compute_residual, node_index));
        }

        auto next_preserve = preserve;
        for (auto node_index = size_t{0u}; node_index < node_count;
             node_index++) {
            if (residual_size[node_index] <= instruction_budget) {
                continue;
            }
            const auto &node = graph[node_index];
            auto best_callees = std::vector<size_t>{};
            auto best_expansion = size_t{0u};
            for (const auto &group : node.alternative_call_groups) {
                if (group.size() < 2u) { continue; }
                auto valid = true;
                auto candidate_callees = std::vector<size_t>{};
                auto expansion = size_t{0u};
                for (auto call_site : group) {
                    if (call_site >= node.callees.size() ||
                        node.callees[call_site] >= node_count) {
                        valid = false;
                        break;
                    }
                    const auto callee = node.callees[call_site];
                    if (preserve[callee]) { continue; }
                    if (std::find(candidate_callees.begin(),
                                  candidate_callees.end(),
                                  callee) == candidate_callees.end()) {
                        candidate_callees.emplace_back(callee);
                    }
                    const auto call_site_expansion =
                        residual_size[callee] > 1u ?
                            residual_size[callee] - 1u :
                            0u;
                    expansion = saturated_add(
                        expansion, call_site_expansion);
                }
                if (valid && !candidate_callees.empty() &&
                    (best_callees.empty() ||
                     expansion > best_expansion)) {
                    best_callees = std::move(candidate_callees);
                    best_expansion = expansion;
                }
            }
            if (best_callees.empty() || best_expansion == 0u) {
                continue;
            }
            for (auto callee : best_callees) {
                next_preserve[callee] = true;
            }
        }
        if (next_preserve == preserve) { break; }
        preserve = std::move(next_preserve);
    }
    return preserve;
}

size_t preserve_hardware_ray_query_loop_form(
    std::string &pipeline) noexcept {
    // LLVM serializes SimplifyCFG options as semicolon-delimited tokens inside
    // angle brackets. Matching the delimiters is intentional: a pass name or
    // a future longer option containing this text must remain untouched.
    constexpr std::string_view noncanonical{";no-keep-loops;"};
    constexpr std::string_view canonical{";keep-loops;"};
    auto replacement_count = size_t{0u};
    auto offset = size_t{0u};
    while ((offset = pipeline.find(noncanonical, offset)) != std::string::npos) {
        pipeline.replace(offset, noncanonical.size(), canonical);
        offset += canonical.size();
        replacement_count++;
    }
    return replacement_count;
}

}// namespace luisa::compute::hip
