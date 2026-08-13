#include "irreducible_cfg_analysis.h"

#include <utility>

#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir::detail {

namespace {

[[nodiscard]] luisa::vector<CFGIrreducibleRegion>
find_irreducible_regions(
    const CFGStronglyConnectedComponents &cfg,
    size_t body_index) noexcept {
    // Maximal SCCs hide nested irreducibility. For every cyclic component with
    // one entry, remove that unique header and decompose the remaining induced
    // subgraph again. This is the standard recursive definition of a reducible
    // flow graph. The implementation is iterative so generated shader CFG
    // depth cannot overflow the host stack.
    luisa::vector<CFGIrreducibleRegion> irreducible_regions;
    luisa::vector<luisa::vector<size_t>> tasks;
    for (auto &&component : cfg.components) {
        if (!component.cyclic) { continue; }
        if (component.entry_nodes.size() > 1u) {
            irreducible_regions.emplace_back(CFGIrreducibleRegion{
                component.nodes,
                component.entry_nodes});
            continue;
        }
        if (component.entry_nodes.size() == 1u) {
            auto header = component.entry_nodes.front();
            auto &task = tasks.emplace_back();
            task.reserve(component.nodes.size() - 1u);
            for (auto node : component.nodes) {
                if (node != header) { task.emplace_back(node); }
            }
            if (task.empty()) { tasks.pop_back(); }
        }
    }

    const auto block_count = cfg.blocks.size();
    luisa::vector<size_t> active_epoch(block_count, 0u);
    luisa::vector<size_t> visited_epoch(block_count, 0u);
    luisa::vector<size_t> component_ids(
        block_count,
        CFGStronglyConnectedComponents::invalid_component);
    auto epoch = size_t{0u};

    while (!tasks.empty()) {
        auto nodes = std::move(tasks.back());
        tasks.pop_back();
        if (nodes.empty()) { continue; }
        ++epoch;
        LUISA_ASSERT(epoch != 0u,
                     "Irreducible-CFG analysis epoch overflow.");
        for (auto node : nodes) {
            active_epoch[node] = epoch;
            component_ids[node] =
                CFGStronglyConnectedComponents::invalid_component;
        }

        luisa::vector<size_t> finish_order;
        finish_order.reserve(nodes.size());
        for (auto root : nodes) {
            if (visited_epoch[root] == epoch) { continue; }
            visited_epoch[root] = epoch;
            luisa::vector<std::pair<size_t, size_t>> stack;
            stack.emplace_back(root, 0u);
            while (!stack.empty()) {
                auto &[node, successor_index] = stack.back();
                while (successor_index <
                           cfg.successors[node].size() &&
                       active_epoch[cfg.successors[node]
                                                  [successor_index]] !=
                           epoch) {
                    ++successor_index;
                }
                if (successor_index <
                    cfg.successors[node].size()) {
                    auto successor =
                        cfg.successors[node][successor_index++];
                    if (visited_epoch[successor] != epoch) {
                        visited_epoch[successor] = epoch;
                        stack.emplace_back(successor, 0u);
                    }
                } else {
                    finish_order.emplace_back(node);
                    stack.pop_back();
                }
            }
        }

        luisa::vector<CFGStronglyConnectedComponent> components;
        for (auto order_index = finish_order.size();
             order_index != 0u; --order_index) {
            auto root = finish_order[order_index - 1u];
            if (component_ids[root] !=
                CFGStronglyConnectedComponents::invalid_component) {
                continue;
            }
            auto component_id = components.size();
            components.emplace_back();
            component_ids[root] = component_id;
            luisa::vector<size_t> worklist{root};
            while (!worklist.empty()) {
                auto node = worklist.back();
                worklist.pop_back();
                components[component_id].nodes.emplace_back(node);
                for (auto predecessor : cfg.predecessors[node]) {
                    if (active_epoch[predecessor] == epoch &&
                        component_ids[predecessor] ==
                            CFGStronglyConnectedComponents::
                                invalid_component) {
                        component_ids[predecessor] = component_id;
                        worklist.emplace_back(predecessor);
                    }
                }
            }
        }

        for (auto component_id = size_t{0u};
             component_id < components.size(); ++component_id) {
            auto &component = components[component_id];
            component.cyclic = component.nodes.size() > 1u;
            if (!component.cyclic && !component.nodes.empty()) {
                auto node = component.nodes.front();
                for (auto successor : cfg.successors[node]) {
                    if (successor == node) {
                        component.cyclic = true;
                        break;
                    }
                }
            }
            if (!component.cyclic) { continue; }

            for (auto node : component.nodes) {
                auto is_entry = node == body_index;
                if (!is_entry) {
                    for (auto predecessor : cfg.predecessors[node]) {
                        if (active_epoch[predecessor] != epoch ||
                            component_ids[predecessor] !=
                                component_id) {
                            is_entry = true;
                            break;
                        }
                    }
                }
                if (is_entry) {
                    component.entry_nodes.emplace_back(node);
                }
            }

            if (component.entry_nodes.size() > 1u) {
                irreducible_regions.emplace_back(CFGIrreducibleRegion{
                    std::move(component.nodes),
                    std::move(component.entry_nodes)});
                continue;
            }
            // Every cyclic region in a reachable flow graph has an entry. A
            // missing entry would mean the input traversal admitted an
            // unreachable closed component, which is outside this analysis
            // domain; leave it to the verifier rather than guessing a header.
            if (component.entry_nodes.size() != 1u) { continue; }
            auto header = component.entry_nodes.front();
            auto &child = tasks.emplace_back();
            child.reserve(component.nodes.size() - 1u);
            for (auto node : component.nodes) {
                if (node != header) { child.emplace_back(node); }
            }
            if (child.empty()) { tasks.pop_back(); }
        }
    }
    return irreducible_regions;
}

}// namespace

size_t CFGStronglyConnectedComponents::
    irreducible_region_count() const noexcept {
    return irreducible_regions.size();
}

CFGStronglyConnectedComponents
analyze_cfg_strongly_connected_components(
    FunctionDefinition *definition) noexcept {
    CFGStronglyConnectedComponents result;
    if (definition == nullptr) { return result; }

    definition->traverse_basic_blocks(
        [&](BasicBlock *block) noexcept {
            result.blocks.emplace_back(block);
        });
    if (result.blocks.empty()) { return result; }

    result.block_indices.reserve(result.blocks.size());
    for (auto index = size_t{0u};
         index < result.blocks.size(); ++index) {
        result.block_indices.emplace(
            result.blocks[index], index);
    }

    result.successors.resize(result.blocks.size());
    result.predecessors.resize(result.blocks.size());
    for (auto source = size_t{0u};
         source < result.blocks.size(); ++source) {
        auto *block = result.blocks[source];
        if (!block->is_terminated()) { continue; }
        block->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                if (auto iter =
                        result.block_indices.find(successor);
                    iter != result.block_indices.end()) {
                    auto target = iter->second;
                    result.successors[source].emplace_back(target);
                    result.predecessors[target].emplace_back(source);
                }
            });
    }

    luisa::vector<uint8_t> visited(
        result.blocks.size(), 0u);
    luisa::vector<size_t> finish_order;
    finish_order.reserve(result.blocks.size());
    for (auto root = size_t{0u};
         root < result.blocks.size(); ++root) {
        if (visited[root] != 0u) { continue; }
        visited[root] = 1u;
        luisa::vector<std::pair<size_t, size_t>> stack;
        stack.emplace_back(root, 0u);
        while (!stack.empty()) {
            auto &[node, successor_index] = stack.back();
            if (successor_index <
                result.successors[node].size()) {
                auto successor = result.successors[node]
                                                  [successor_index++];
                if (visited[successor] == 0u) {
                    visited[successor] = 1u;
                    stack.emplace_back(successor, 0u);
                }
            } else {
                finish_order.emplace_back(node);
                stack.pop_back();
            }
        }
    }

    result.component_ids.assign(
        result.blocks.size(),
        CFGStronglyConnectedComponents::invalid_component);
    for (auto order_index = finish_order.size();
         order_index != 0u; --order_index) {
        auto root = finish_order[order_index - 1u];
        if (result.component_ids[root] !=
            CFGStronglyConnectedComponents::invalid_component) {
            continue;
        }
        auto component_id = result.components.size();
        result.components.emplace_back();
        luisa::vector<size_t> worklist{root};
        result.component_ids[root] = component_id;
        while (!worklist.empty()) {
            auto node = worklist.back();
            worklist.pop_back();
            result.components[component_id]
                .nodes.emplace_back(node);
            for (auto predecessor :
                 result.predecessors[node]) {
                if (result.component_ids[predecessor] ==
                    CFGStronglyConnectedComponents::
                        invalid_component) {
                    result.component_ids[predecessor] =
                        component_id;
                    worklist.emplace_back(predecessor);
                }
            }
        }
    }

    for (auto component_id = size_t{0u};
         component_id < result.components.size();
         ++component_id) {
        auto &component = result.components[component_id];
        component.cyclic = component.nodes.size() > 1u;
        if (!component.cyclic && !component.nodes.empty()) {
            auto node = component.nodes.front();
            for (auto successor : result.successors[node]) {
                if (successor == node) {
                    component.cyclic = true;
                    break;
                }
            }
        }
    }

    luisa::vector<uint8_t> entry_node(
        result.blocks.size(), 0u);
    if (auto iter = result.block_indices.find(
            definition->body_block());
        iter != result.block_indices.end()) {
        entry_node[iter->second] = 1u;
    }
    for (auto source = size_t{0u};
         source < result.blocks.size(); ++source) {
        for (auto target : result.successors[source]) {
            if (result.component_ids[source] !=
                result.component_ids[target]) {
                entry_node[target] = 1u;
            }
        }
    }
    for (auto node = size_t{0u};
         node < result.blocks.size(); ++node) {
        if (entry_node[node] != 0u) {
            result.components[result.component_ids[node]]
                .entry_nodes.emplace_back(node);
        }
    }
    auto body_iter = result.block_indices.find(
        definition->body_block());
    LUISA_ASSERT(body_iter != result.block_indices.end(),
                 "Reachable CFG analysis lost the function body.");
    result.irreducible_regions =
        find_irreducible_regions(
            result, body_iter->second);
    return result;
}

}// namespace luisa::compute::xir::detail
