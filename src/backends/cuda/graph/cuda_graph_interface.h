#pragma once
#include <cuda_runtime.h>
#include <luisa/runtime/graph/graph.h>
#include "../cuda_device.h"
namespace luisa::compute::graph {
class GraphBuilder;
}// namespace luisa::compute::graph
namespace luisa::compute::cuda::graph {
class CUDAGraphInterface : public luisa::compute::graph::GraphInterface {
    friend class CUDAGraphExt;
protected:
    virtual void create_graph_instance(luisa::compute::graph::GraphBuilder *builder) noexcept override;
    virtual void destroy_graph_instance(luisa::compute::graph::GraphBuilder *builder) noexcept override;
    virtual void update_graph_instance_node_parms(luisa::compute::graph::GraphBuilder *builder) noexcept override;
    virtual void launch_graph_instance(Stream *stream) noexcept override;
private:
    CUgraph _cuda_graph = nullptr;
    CUgraphExec _cuda_graph_exec = nullptr;
    luisa::vector<CUgraphNode> _cuda_graph_nodes;
    luisa::vector<CUgraphNode> _cuda_kernel_nodes;
    luisa::vector<luisa::vector<void *>> _kernel_parms_cache;
    CUDADevice *_device = nullptr;

    void build_graph(luisa::compute::graph::GraphBuilder *builder) noexcept;
    void _add_kernel_nodes(luisa::compute::graph::GraphBuilder *builder) noexcept;
    void _add_deps(luisa::compute::graph::GraphBuilder *builder) noexcept;

    void _update_graph_kernel_node(const luisa::compute::graph::KernelNode *kernel, luisa::compute::graph::GraphBuilder *builder) noexcept;
public:
    CUDAGraphInterface(CUDADevice *device) noexcept;
    ~CUDAGraphInterface() noexcept;
};
}// namespace luisa::compute::cuda::graph
