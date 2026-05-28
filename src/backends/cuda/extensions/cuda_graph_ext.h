#pragma once

#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/spin_mutex.h>
#include <cuda.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CudaGraphExtImpl final : public CudaGraphExt {

    CUDADevice *_device;

    struct GraphData {
        luisa::vector<CUgraphNode> nodes;
        luisa::vector<void *> host_allocations;// pinned memory kept alive for graph lifetime
    };

    mutable spin_mutex _mutex;
    luisa::unordered_map<uint64_t, GraphData> _graph_data;
    luisa::unordered_map<uint64_t, uint64_t> _exec_to_graph;// exec_handle -> graph_handle

    [[nodiscard]] CUgraphNode _get_node(GraphExecHandle exec, size_t node_index) const noexcept;

public:
    explicit CudaGraphExtImpl(CUDADevice *device) noexcept;
    ~CudaGraphExtImpl() noexcept;

    ResourceCreationInfo create_graph(CommandList &&cmdlist) noexcept override;
    void destroy_graph(GraphHandle graph) noexcept override;
    ResourceCreationInfo instantiate(GraphHandle graph, InstantiateFlag flags) noexcept override;
    void destroy_exec(GraphExecHandle exec) noexcept override;
    void launch(GraphExecHandle exec, uint64_t stream_handle) noexcept override;
    void upload(GraphExecHandle exec, uint64_t stream_handle) noexcept override;
    bool update(GraphExecHandle exec, CommandList &&cmdlist) noexcept override;
    bool update_kernel_node(GraphExecHandle exec, size_t node_index,
                            uint3 dispatch_size, luisa::span<const Argument> arguments) noexcept override;
    bool update_upload_node(GraphExecHandle exec, size_t node_index,
                            const void *data, size_t size) noexcept override;
    bool update_download_node(GraphExecHandle exec, size_t node_index,
                              void *data, size_t size) noexcept override;
    bool update_buffer_copy_node(GraphExecHandle exec, size_t node_index,
                                 uint64_t src_handle, size_t src_offset,
                                 uint64_t dst_handle, size_t dst_offset,
                                 size_t size) noexcept override;
    void set_node_enabled(GraphExecHandle exec, size_t node_index, bool enabled) noexcept override;
    bool is_node_enabled(GraphExecHandle exec, size_t node_index) const noexcept override;
};

}// namespace luisa::compute::cuda
