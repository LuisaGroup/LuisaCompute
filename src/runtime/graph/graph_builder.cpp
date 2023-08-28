#include <luisa/runtime/graph/graph_builder.h>
#include <luisa/runtime/graph/kernel_node.h>

using namespace luisa::compute::graph;

// # Add Node API:
KernelNode *GraphBuilder::add_kernel_node(span<uint64_t> arg_ids, const Resource *shader_resource) noexcept {
    auto node = make_shared<KernelNode>(current(), arg_ids, shader_resource);
    auto ptr = node.get();
    _current()->_kernel_nodes.emplace_back(std::move(node));
    _current()->_nodes.emplace_back(ptr);
    return ptr;
}

// # Build Deps API:
// - build deps from nodes and their arg usages

static void process_node(
    luisa::vector<GraphDependency> &deps,
    luisa::vector<uint64_t> &last_read_or_write_nodes,
    luisa::vector<uint64_t> &last_write_nodes,
    const luisa::vector<luisa::unique_ptr<GraphVarBase>> &vars,
    luisa::vector<GraphNode *> &nodes,
    uint64_t current_node_id,
    uint64_t &dep_begin,
    uint64_t &dep_count) {
    using namespace luisa::compute;

    auto is_read_write = [](Usage usage) { return (usage == Usage::READ_WRITE) || (usage == Usage::WRITE); };
    auto is_read_only = [](Usage usage) { return usage == Usage::READ; };

    luisa::unordered_set<uint64_t> unique_deps;
    auto node = nodes[current_node_id];

    for (auto &[arg_id, usage] : node->arg_usage()) {
        // if this is a written resource,
        // this should depend on any write and read before it
        // to get newest data or to avoid data corruption
        if (is_read_write(usage)) {
            auto dst_nid = last_read_or_write_nodes[arg_id];
            if (dst_nid != GraphNode::invalid_node_id()) {
                // the last accessing node reads or writes this resrouce, so I should depend on it
                if (unique_deps.find(dst_nid) == unique_deps.end()) {
                    // record this dependency
                    unique_deps.insert(dst_nid);
                }
            }
        }
        // if this is a read resource,
        // this should depend on any write before it
        // to get newest data
        // but it has no need to depend on any read before it
        else if (is_read_only(usage)) {
            auto dst_nid = last_write_nodes[arg_id];
            if (dst_nid != GraphNode::invalid_node_id()) {
                // the last accessing node writes this resrouce, so I should depend on it
                if (unique_deps.find(dst_nid) == unique_deps.end()) {
                    // record this dependency
                    unique_deps.insert(dst_nid);
                }
            }
        }
    }

    // set up res node map with pair [res, node]
    for (auto &[arg_id, usage] : node->arg_usage()) {
        // if this is a write resource,
        // the latter read/write kernel should depend on this
        // to get the newest data.
        if (is_read_write(usage)) {
            last_read_or_write_nodes[arg_id] = current_node_id;
            last_write_nodes[arg_id] = current_node_id;
        }
        // if this is a read resource,
        // the latter write kernel should depend on this
        // to avoid data corruption.
        else if (is_read_only(usage)) {
            last_read_or_write_nodes[arg_id] = current_node_id;
        }
    }

    // add dependencies to deps
    dep_begin = deps.size();
    for (auto dep : unique_deps) deps.emplace_back(GraphDependency{current_node_id, dep});
    dep_count = unique_deps.size();
};

void luisa::compute::graph::GraphBuilder::_build_deps() noexcept {
    auto &nodes = _nodes;
    auto &deps = _deps;
    auto &vars = _vars;
    deps.clear();
    // map: arg_id -> node_id, uint64_t{-1} means no write node yet
    auto last_write_nodes = luisa::vector<uint64_t>(_vars.size(), GraphNode::invalid_node_id());
    // map: arg_id -> node_id, uint64_t{-1} means no read node yet
    auto last_read_nodes = luisa::vector<uint64_t>(_vars.size(), GraphNode::invalid_node_id());

    // process all nodes
    for (uint64_t i = 0u; i < nodes.size(); i++) {
        uint64_t dep_begin, dep_count;
        process_node(deps, last_read_nodes, last_write_nodes, vars, nodes, i, dep_begin, dep_count);
        nodes[i]->set_dep_range(dep_begin, dep_count);
    }
}

void GraphBuilder::_build_var_accessors() noexcept {
    auto &nodes = _nodes;
    auto &vars = _vars;
    auto &var_accessors = _var_accessors;

    var_accessors.resize(vars.size());
    for (auto &node : nodes) {
        for (auto &[arg_id, usage] : node->arg_usage()) {
            auto &var = vars[arg_id];
            var_accessors[var->arg_id()].emplace_back(node->node_id());
        }
    }
}

// # Getter/Setter:
void GraphBuilder::set_var_count(size_t size) noexcept { current()->_vars.resize(size); }

GraphBuilder::U<GraphBuilder> &GraphBuilder::_current() noexcept {
    static thread_local U<GraphBuilder> _builder = nullptr;
    return _builder;
}

GraphBuilder *GraphBuilder::current() noexcept { return _current().get(); }

bool GraphBuilder::is_building() noexcept { return _current() != nullptr && _current()->_is_building; }

void GraphBuilder::clear_need_update_flags() noexcept {
    for (auto &var : _vars) var->clear_need_update_flag();
}

const GraphNode *luisa::compute::graph::GraphBuilder::graph_node(uint64_t id) const noexcept {
    LUISA_ASSERT(id < _nodes.size(), "GraphNode id out of range.");
    return _nodes[id];
}

const GraphVarBase *GraphBuilder::graph_var(uint64_t id) const noexcept {
    LUISA_ASSERT(id < _vars.size(), "GraphVarBase id out of range.");
    return _vars[id].get();
}
const luisa::vector<GraphBuilder::node_id_t> &GraphBuilder::accessor_node_ids(const GraphVarBase *graph_var) const noexcept {
    return _var_accessors[graph_var->arg_id()];
}

// # Constructor/Destructor:
GraphBuilder::GraphBuilder() noexcept {}

GraphBuilder::~GraphBuilder() noexcept {}

luisa::compute::graph::GraphBuilder::GraphBuilder(const GraphBuilder &other) noexcept {
    LUISA_ASSERT(!is_building() && !other.is_building(), "Copy is not allowed when building");

    //auto set_nodes = [&]<typename T>(luisa::vector<U<T>> &concrete_nodes) {
    //    static_assert(std::is_base_of_v<GraphNode, T>, "T must be derived from GraphNode");
    //    for (size_t i = 0; i < concrete_nodes.size(); ++i) {
    //        const auto &other_node = other._nodes[i];
    //        auto node = make_unique<T>(*other_node);
    //        _nodes[node->node_id()] = node.get();
    //        concrete_nodes[i] = std::move(node);
    //    }
    //};

    // graph vars can be different in different graph (e.g. different input)
    _vars.reserve(other._vars.size());
    for (size_t i = 0; i < other._vars.size(); ++i) {
        auto &var = other._vars[i];
        LUISA_ASSERT(var->is_virtual(), "Only virtual GraphVar (in GraphDef) can be cloned.");
        auto cloned_var = var->clone();
        cloned_var->_is_virtual = false;
        _vars.emplace_back(std::move(cloned_var));
    }

    // nodes should be identical, so we just copy the shared pointers
    _kernel_nodes = other._kernel_nodes;
    _nodes = other._nodes;
    _deps = other._deps;
    _var_accessors = other._var_accessors;
}