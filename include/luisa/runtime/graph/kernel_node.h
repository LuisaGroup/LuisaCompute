#pragma once
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/vstl/hash_map.h>
namespace luisa::compute::graph {
class KernelNode : public GraphNode {
    template<size_t N>
    friend class GraphShaderInvoke;
public:
    KernelNode(GraphBuilder *builder, span<uint64_t> arg_ids, const Resource *shader_resource) noexcept
        : GraphNode{builder, arg_ids}, _shader_resource{shader_resource} {
    }
    virtual ~KernelNode() noexcept {}
private:
    const Resource *_shader_resource;
    uint3 _dispatch_size;
};

template<size_t N>
class GraphShaderInvoke;

template<>
class GraphShaderInvoke<1> {
public:
    GraphShaderInvoke(KernelNode *node) noexcept
        : _node{node} {}
    KernelNode *dispatch(uint32_t dispatch_x) noexcept {
        _node->_dispatch_size = {dispatch_x, 1, 1};
        return _node;
    }
private:
    KernelNode *_node = nullptr;
};

template<>
class GraphShaderInvoke<2> {
public:
    GraphShaderInvoke(KernelNode *node) noexcept
        : _node{node} {}
    KernelNode *dispatch(uint32_t dispatch_x, uint32_t dispatch_y) noexcept {
        _node->_dispatch_size = {dispatch_x, dispatch_y, 1};
        return _node;
    }
private:
    KernelNode *_node = nullptr;
};

template<>
class GraphShaderInvoke<3> {
public:
	GraphShaderInvoke(KernelNode *node) noexcept
		: _node{node} {}
	KernelNode *dispatch(uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z) noexcept {
		_node->_dispatch_size = {dispatch_x, dispatch_y, dispatch_z};
		return _node;
	}
private:
KernelNode *_node = nullptr;
};
}// namespace luisa::compute::graph