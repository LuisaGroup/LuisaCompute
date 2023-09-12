#include <luisa/runtime/graph/graph_builder.h>
#include <luisa/runtime/graph/kernel_node.h>
#include <luisa/runtime/graph/capture_node.h>
#include <luisa/runtime/graph/graph_basic_var.h>
#include <luisa/runtime/graph/graph_buffer_var.h>
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/kernel_node.h>
#include <luisa/runtime/graph/memory_node.h>

using namespace luisa::compute::graph;

// # Add Node API:
KernelNode *GraphBuilder::add_kernel_node(span<GraphSubVarId> arg_ids,
                                          const Resource *shader_resource,
                                          U<KernelNodeCmdEncoder> &&encoder,
                                          size_t dimension,
                                          const uint3 &block_size) noexcept {
    auto node = make_shared<KernelNode>(current(), arg_ids, shader_resource, dimension, block_size);
    auto ptr = node.get();
    _current()->_kernel_nodes.emplace_back(std::move(node));
    _current()->_kernel_node_cmd_encoders.emplace_back(std::move(encoder));
    _current()->_graph_nodes.emplace_back(ptr);
    return ptr;
}

MemoryNode *GraphBuilder::add_memory_node(GraphSubVarId src_var_id, GraphSubVarId dst_var_id, MemoryNodeDirection direction) noexcept {
    auto node = make_shared<MemoryNode>(current(), src_var_id, dst_var_id, direction);
    auto ptr = node.get();
    _current()->_memory_nodes.emplace_back(std::move(node));
    _current()->_graph_nodes.emplace_back(ptr);
    return ptr;
}

// # Build Deps API:
// - build deps from nodes and their arg usages

static void process_node(
    luisa::vector<GraphDependency> &deps,
    luisa::vector<GraphInputVarId> &last_read_or_write_nodes,
    luisa::vector<GraphInputVarId> &last_write_nodes,
    const luisa::span<GraphVarBase *> input_vars,
    luisa::span<GraphNode *> nodes,
    GraphInputVarId current_node_id,
    uint64_t &dep_begin,
    uint64_t &dep_count) {
    using namespace luisa::compute;

    auto is_read_write = [](Usage usage) { return (usage == Usage::READ_WRITE) || (usage == Usage::WRITE); };
    auto is_read_only = [](Usage usage) { return usage == Usage::READ; };

    luisa::unordered_set<GraphInputVarId::type> unique_deps;
    auto node = nodes[current_node_id.value()];

    for (auto &[arg_id, usage] : node->input_var_usage()) {
        // if this is a written resource,
        // this should depend on any write and read before it
        // to get newest data or to avoid data corruption
        if (is_read_write(usage)) {
            auto dst_nid = last_read_or_write_nodes[arg_id.value()];
            if (dst_nid.is_valid()) {
                // the last accessing node reads or writes this resrouce, so I should depend on it
                if (unique_deps.find(dst_nid.value()) == unique_deps.end()) {
                    // record this dependency
                    unique_deps.insert(dst_nid.value());
                }
            }
        }
        // if this is a read resource,
        // this should depend on any write before it
        // to get newest data
        // but it has no need to depend on any read before it
        else if (is_read_only(usage)) {
            auto dst_nid = last_write_nodes[arg_id.value()];
            if (dst_nid.is_valid()) {
                // the last accessing node writes this resrouce, so I should depend on it
                if (unique_deps.find(dst_nid.value()) == unique_deps.end()) {
                    // record this dependency
                    unique_deps.insert(dst_nid.value());
                }
            }
        }
    }

    // set up res node map with pair [res, node]
    for (auto &[arg_id, usage] : node->input_var_usage()) {
        // if this is a write resource,
        // the latter read/write kernel should depend on this
        // to get the newest data.
        if (is_read_write(usage)) {
            last_read_or_write_nodes[arg_id.value()] = current_node_id;
            last_write_nodes[arg_id.value()] = current_node_id;
        }
        // if this is a read resource,
        // the latter write kernel should depend on this
        // to avoid data corruption.
        else if (is_read_only(usage)) {
            last_read_or_write_nodes[arg_id.value()] = current_node_id;
        }
    }

    // add dependencies to deps
    dep_begin = deps.size();
    for (auto dep : unique_deps) deps.emplace_back(GraphDependency{current_node_id, GraphInputVarId{dep}});
    dep_count = unique_deps.size();
};

void GraphBuilder::_build_deps() noexcept {
    auto nodes = this->graph_nodes();
    auto &deps = _deps;
    auto input_vars = this->input_vars();
    deps.clear();
    // map: arg_id -> node_id, uint64_t{-1} means no write node yet
    auto last_write_nodes = luisa::vector<GraphInputVarId>(input_vars.size());
    // map: arg_id -> node_id, uint64_t{-1} means no read node yet
    auto last_read_nodes = luisa::vector<GraphInputVarId>(input_vars.size());

    // process all nodes
    for (uint64_t i = 0u; i < nodes.size(); i++) {
        uint64_t dep_begin, dep_count;
        process_node(deps, last_read_nodes, last_write_nodes, input_vars, nodes, GraphInputVarId{i}, dep_begin, dep_count);
        nodes[i]->set_dep_range(dep_begin, dep_count);
    }
}

void GraphBuilder::_build_var_accessors() noexcept {
    auto &nodes = _graph_nodes;
    auto vars = this->input_vars();
    luisa::vector<luisa::vector<GraphSubVarId>> input_var_to_sub_vars;
    input_var_to_sub_vars.resize(vars.size());

    for (auto sub : pure_sub_vars()) {
        for (auto dep_sub_id : sub->_other_dependent_var_ids) {
            auto input_var_id = sub_var(dep_sub_id)->input_var_id();
            input_var_to_sub_vars[input_var_id.value()].emplace_back(sub->sub_var_id());
        }
    }
    _input_var_to_sub_vars.set(input_var_to_sub_vars);

    luisa::vector<luisa::vector<GraphNodeId>> sub_var_to_nodes;
    sub_var_to_nodes.resize(_sub_vars.size());

    for (auto &node : nodes) {
        for (auto &&sub_var_id : node->sub_var_ids()) {
            sub_var_to_nodes[sub_var_id.value()].emplace_back(node->node_id());
        }
    }
    _sub_var_to_nodes.set(sub_var_to_nodes);
}

// # Update Node API:
void GraphBuilder::_setup_node_need_update_flags() noexcept {
    auto node_count = _graph_nodes.size();
    _node_need_update_flags.resize(node_count, 1);
}

bool GraphBuilder::propagate_need_update_flag_from_vars_to_nodes() noexcept {
    auto need_update = false;
    auto input_vars = this->input_vars();
    auto sub_vars = this->sub_vars();
    auto graph_nodes = this->graph_nodes();
    for (auto &input_var : input_vars) {
        if (input_var->need_update()) {// if an input var need update, all sub vars that depend on this var should be updated
            for (auto sub_var_id : dep_sub_vars(input_var->input_var_id()))
                this->sub_var(sub_var_id)->sub_var_update_check(this);
            need_update = true;
        }
    }

    if (need_update) {
        for (auto &sub_var : sub_vars)
            if (sub_var->need_update()) {// if a sub var need update, all nodes that use this var should be updated
                for (auto nid : dep_nodes(sub_var->sub_var_id()))
                    this->node_need_update_flag(nid) = true;
                need_update = true;
            }
    }

    if (need_update) {
        for (auto node : graph_nodes)
            if (node_need_update(node->node_id())) {
                switch (node->type()) {
                    case GraphNodeType::Kernel:
                        // if it's a kernel node, we need update the argument buffer in cmd encoder
                        _update_kernel_node_cmd_encoders(dynamic_cast<const KernelNode *>(node));
                        break;
                    default:
                        break;
                }
            }
    }
    return need_update;
}

void GraphBuilder::_update_kernel_node_cmd_encoders(const KernelNode *node) noexcept {
    auto encoder = kernel_node_cmd_encoder(node->kernel_node_id());
    for (size_t i = 0; i < node->kernel_args().size(); ++i) {
        auto arg_idx_in_kernel_parms = i;
        auto sub_var_id = node->sub_var_id(arg_idx_in_kernel_parms);
        sub_var(sub_var_id)->update_kernel_node_cmd_encoder(arg_idx_in_kernel_parms, encoder);
    }
    uint3 dispatch_size = {1, 1, 1};
    for (size_t i = 0; i < node->dispatch_args().size(); ++i) {
        auto sub_var_id = node->dispatch_args()[i];
        dispatch_size[i] = dynamic_cast<const GraphVar<uint> *>(sub_var(sub_var_id))->eval();
    }
    encoder->_dispatch_size = dispatch_size;
}

void GraphBuilder::check_var_overlap() noexcept {
    _check_buffer_var_overlap();
}

void GraphBuilder::_check_buffer_var_overlap() noexcept {
    auto buffer_vars = _buffer_vars.input_vars();
    struct Data {
        uint64_t id;
        uint64_t value;
        bool operator<(const Data &rhs) const noexcept { return value < rhs.value; }
    };
    luisa::vector<Data> begins(buffer_vars.size());
    luisa::vector<Data> ends(buffer_vars.size());
    for (size_t i = 0; i < buffer_vars.size(); ++i) {
        auto bv = buffer_vars[i]->eval_buffer_view_base(this);
        begins[i] = {i, reinterpret_cast<uint64_t>(bv.native_handle()) + bv.offset_bytes()};
        ends[i] = {i, reinterpret_cast<uint64_t>(bv.native_handle()) + bv.size_bytes()};
    }
    std::sort(begins.begin(), begins.end());
    std::sort(ends.begin(), ends.end());
    for (size_t i = 0; i < buffer_vars.size() - 1; ++i) {
        auto &this_end = ends[i];
        auto &next_begin = begins[i + 1];
        if (this_end.value > next_begin.value) {
            const auto &this_var = buffer_vars[this_end.id];
            const auto &next_var = buffer_vars[next_begin.id];
            LUISA_ERROR_WITH_LOCATION(
                "Graph Buffer Var Overlap Detected: Var{}[name={}] and Var{}[name={}] overlap",
                this_var->input_var_id().value(), this_var->var_name(),
                next_var->input_var_id().value(), next_var->var_name());
        }
    }
}

// # Getter/Setter:
void GraphBuilder::set_var_count(size_t size) noexcept {
    current()->_input_var_count = size;
    current()->_sub_vars.resize(size);
}

GraphBuilder::U<GraphBuilder> &GraphBuilder::_current() noexcept {
    static thread_local U<GraphBuilder> _builder = nullptr;
    return _builder;
}

GraphBuilder *GraphBuilder::current() noexcept { return _current().get(); }

bool GraphBuilder::is_building() noexcept { return _current() != nullptr && _current()->_is_building; }

void GraphBuilder::clear_need_update_flags() noexcept {
    for (auto &var : _sub_vars) var->clear_need_update_flag();
    for (auto &i : _node_need_update_flags) i = 0;
}

const GraphNode *luisa::compute::graph::GraphBuilder::graph_node(GraphNodeId id) const noexcept {
    LUISA_ASSERT(id.value() < _graph_nodes.size(), "GraphNode id out of range.");
    return _graph_nodes[id.value()];
}

bool luisa::compute::graph::GraphBuilder::node_need_update(GraphNodeId id) const noexcept {
    return _node_need_update_flags[id.value()];
}

// # Constructor/Destructor:
GraphBuilder::GraphBuilder() noexcept {}

GraphBuilder::~GraphBuilder() noexcept {}

GraphBuilder::GraphBuilder(const GraphBuilder &other) noexcept
    : _input_var_count{other._input_var_count},
      // vars:
      // alway be different in different graph, so we need to deep copy
      _basic_vars{other._basic_vars},
      _buffer_vars{other._buffer_vars},
      _host_memory_vars{other._host_memory_vars},

      // nodes:
      // should be identical w.r.t a GraphDef, so we just copy the shared pointers
      _graph_nodes{other._graph_nodes},
      _kernel_nodes{other._kernel_nodes},
      _memory_nodes{other._memory_nodes},
      _capture_nodes{other._capture_nodes},

      // for graph update:
      _input_var_to_sub_vars{other._input_var_to_sub_vars},
      _sub_var_to_nodes{other._sub_var_to_nodes},
      _node_need_update_flags{other._node_need_update_flags},

      // deps:
      _deps{other._deps} {
    LUISA_ASSERT(!is_building() && !other.is_building(), "Copy is not allowed when building");

    // graph vars can be different in different graph (e.g. different input)
    _sub_vars.resize(other._sub_vars.size(), nullptr);
    // fill the pointer in _sub_vars
    fill_sub_vars(_basic_vars);
    fill_sub_vars(_buffer_vars);
    fill_sub_vars(_host_memory_vars);

    //
    _kernel_node_cmd_encoders.resize(other._kernel_node_cmd_encoders.size());
    std::transform(other._kernel_node_cmd_encoders.begin(), other._kernel_node_cmd_encoders.end(),
                   _kernel_node_cmd_encoders.begin(),
                   [](const auto &encoder) { return make_unique<KernelNodeCmdEncoder>(*encoder); });
}

void GraphBuilder::_def_input_var(GraphBuilder::U<GraphBasicVarBase> &&var) noexcept { _basic_vars.def_input_var(std::move(var)); }
void GraphBuilder::_def_sub_var(GraphBuilder::U<GraphBasicVarBase> &&var) noexcept { _basic_vars.def_sub_var(std::move(var)); }
void GraphBuilder::_def_input_var(GraphBuilder::U<GraphBufferVarBase> &&buffer) noexcept { _buffer_vars.def_input_var(std::move(buffer)); }
void GraphBuilder::_def_sub_var(GraphBuilder::U<GraphBufferVarBase> &&buffer) noexcept { _buffer_vars.def_sub_var(std::move(buffer)); }
void GraphBuilder::_def_input_var(GraphBuilder::U<GraphVar<void *>> &&var) noexcept { _host_memory_vars.def_input_var(std::move(var)); }
void GraphBuilder::_def_sub_var(GraphBuilder::U<GraphVar<void *>> &&var) noexcept { _host_memory_vars.def_sub_var(std::move(var)); }
//# Graphviz
void GraphBuilder::graphviz(std::ostream &o, GraphvizOptions options) noexcept {
    o << "digraph G {\n";
    if (options.show_vars) {
        o << "// vars: \n";
        for (auto &&var : sub_vars()) {
            var->graphviz_def(o);
            o << "\n";
        }
        o << "\n";
        o << "// var deps: \n";
        for (auto &&var : pure_sub_vars()) {
            var->graphviz_arg_usage(o);
            o << "\n";
        }
    }

    if (options.show_nodes) {
        o << "// nodes: \n";
        for (auto &&node : _graph_nodes) {
            node->graphviz_def(o);
            o << "\n";
        }
        o << "\n";
        if (options.show_vars) {
            o << "// node var usages: \n";
            for (auto &&node : _graph_nodes) {
                node->graphviz_arg_usages(o);
                o << "\n";
            }
            o << "\n";
        }
        o << "// node deps: \n";
        for (auto dep : _deps) {
            auto src = _graph_nodes[dep.src.value()];
            auto dst = _graph_nodes[dep.dst.value()];
            dst->graphviz_id(o);
            o << "->";
            src->graphviz_id(o);
            o << "\n";
        }
    }
    o << "}\n";
}
