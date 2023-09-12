#pragma once
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_basic_var.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/vstl/hash_map.h>
#include <luisa/runtime/rhi/command_encoder.h>

namespace luisa::compute::graph {
class GraphBuilder;
class KernelNodeCmdEncoder;
class LC_RUNTIME_API KernelNode final : public GraphNode {
    template<size_t N>
    friend class GraphShaderInvoke;
    friend class GraphBuilder;
public:
    KernelNode(GraphBuilder *builder, span<GraphSubVarId> arg_ids,
               const Resource *shader_resource,
               size_t dimension, const uint3 &block_size) noexcept;
    virtual ~KernelNode() noexcept {}
    auto kernel_node_id() const noexcept { return _kernel_id; }
    auto dimension() const noexcept { return _dimension; }
    auto kernel_arg_count() const noexcept { return _kernel_arg_count; }
    const auto kernel_args() const noexcept { return span{sub_var_ids()}.subspan(0, kernel_arg_count()); }
    const auto dispatch_args() const noexcept { return span{sub_var_ids()}.subspan(kernel_arg_count(), dimension()); }
    auto shader_resource() const noexcept { return _shader_resource; }
    auto block_size() const noexcept { return _block_size; }
protected:
    //virtual U<GraphNode> clone() const noexcept override;
private:
    void add_dispatch_arg(GraphSubVarId x_arg_id) noexcept;
    void add_dispatch_arg(GraphSubVarId x_arg_id, GraphSubVarId y_arg_id) noexcept;
    void add_dispatch_arg(GraphSubVarId x_arg_id, GraphSubVarId y_arg_id, GraphSubVarId z_arg_id) noexcept;

    const Resource *_shader_resource = nullptr;
    GraphNodeId _kernel_id{GraphNodeId::invalid_id};

    // private: never access in friend class >>>
    size_t _dimension = 0;
    size_t _kernel_arg_count = 0;
    uint3 _block_size;
    // <<< private: never access in friend class <<<
};

class GraphShaderInvokeBase {
public:
    GraphShaderInvokeBase(KernelNode *node) noexcept
        : _node{node} {}
protected:
    KernelNode *_node = nullptr;
};

template<size_t N>
class GraphShaderInvoke;

template<>
class GraphShaderInvoke<1> : public GraphShaderInvokeBase {
public:
    GraphShaderInvoke(KernelNode *node) noexcept
        : GraphShaderInvokeBase{node} {}
    auto &dispatch(GraphUInt dispatch_x) noexcept {
        LUISA_ASSERT(GraphBuilder::is_building(), "This function is invocable in GraphDef only");
        _node->add_dispatch_arg(dispatch_x.sub_var_id());
        return *_node;
    }
};

template<>
class GraphShaderInvoke<2> : public GraphShaderInvokeBase {
public:
    GraphShaderInvoke(KernelNode *node) noexcept
        : GraphShaderInvokeBase{node} {}
    auto &dispatch(GraphUInt dispatch_x,
                   GraphUInt dispatch_y) noexcept {
        LUISA_ASSERT(GraphBuilder::is_building(), "This function is invocable in GraphDef only");
        _node->add_dispatch_arg(dispatch_x.sub_var_id(), dispatch_y.sub_var_id());
        return *_node;
    }
};

template<>
class GraphShaderInvoke<3> : public GraphShaderInvokeBase {
public:
    GraphShaderInvoke(KernelNode *node) noexcept
        : GraphShaderInvokeBase{node} {}

    auto &dispatch(GraphVar<uint32_t> dispatch_x,
                   GraphVar<uint32_t> dispatch_y,
                   GraphVar<uint32_t> dispatch_z) noexcept {
        LUISA_ASSERT(GraphBuilder::is_building(), "This function is invocable in GraphDef only");
        _node->add_dispatch_arg(dispatch_x.sub_var_id(), dispatch_y.sub_var_id(), dispatch_z.sub_var_id());
        return *_node;
    }
};
}// namespace luisa::compute::graph