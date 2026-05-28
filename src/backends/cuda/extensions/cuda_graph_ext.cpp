#include "cuda_graph_ext.h"
#include "../cuda_device.h"
#include "../cuda_stream.h"
#include "../cuda_buffer.h"
#include "../cuda_error.h"
#include <cuda.h>

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] CUstream to_cu_stream(uint64_t stream_handle) noexcept {
    return reinterpret_cast<CUDAStream *>(stream_handle)->handle();
}

[[nodiscard]] unsigned long long to_instantiate_flags(CudaGraphExt::InstantiateFlag flags) noexcept {
    unsigned long long f = 0u;
    auto v = static_cast<uint32_t>(flags);
    if (v & static_cast<uint32_t>(CudaGraphExt::InstantiateFlag::AUTO_FREE_ON_LAUNCH)) {
        f |= CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    }
    if (v & static_cast<uint32_t>(CudaGraphExt::InstantiateFlag::DEVICE_LAUNCH)) {
        f |= CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
    }
    return f;
}

}// namespace

CudaGraphExtImpl::CudaGraphExtImpl(CUDADevice *device) noexcept
    : _device{device} {}

CudaGraphExtImpl::~CudaGraphExtImpl() noexcept {
    for (auto &[exec_handle, graph_handle] : _exec_to_graph) {
        cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(exec_handle));
    }
    for (auto &[graph_handle, data] : _graph_data) {
        for (auto *ptr : data.host_allocations) {
            cuMemFreeHost(ptr);
        }
        cuGraphDestroy(reinterpret_cast<CUgraph>(graph_handle));
    }
}

CUgraphNode CudaGraphExtImpl::_get_node(GraphExecHandle exec, size_t node_index) const noexcept {
    auto it = _exec_to_graph.find(exec);
    if (it == _exec_to_graph.end()) { return nullptr; }
    auto git = _graph_data.find(it->second);
    if (git == _graph_data.end() || node_index >= git->second.nodes.size()) { return nullptr; }
    return git->second.nodes[node_index];
}

ResourceCreationInfo CudaGraphExtImpl::create_graph(CommandList &&cmdlist) noexcept {
    if (cmdlist.empty()) { return {invalid_handle, nullptr}; }

    return _device->with_handle([&]() -> ResourceCreationInfo {
        CUstream cap_stream = nullptr;
        if (auto err = cuStreamCreate(&cap_stream, CU_STREAM_DEFAULT); err != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(err, &err_name);
            LUISA_WARNING_WITH_LOCATION("Failed to create capture stream: {}", err_name ? err_name : "unknown");
            return {invalid_handle, nullptr};
        }

        auto ret = cuStreamBeginCapture(cap_stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuStreamBeginCapture failed: {}", err_name ? err_name : "unknown");
            cuStreamDestroy(cap_stream);
            return {invalid_handle, nullptr};
        }

        luisa::vector<void *> host_allocs;
        auto commands = cmdlist.steal_commands();
        auto user_callbacks = cmdlist.steal_callbacks();
        CUgraph graph = nullptr;

        for (auto &cmd : commands) {
            if (auto *upload = dynamic_cast<BufferUploadCommand *>(cmd.get())) {
                auto *buffer = reinterpret_cast<const CUDABuffer *>(upload->handle());
                auto address = buffer->device_address() + upload->offset();
                auto data = upload->data();
                auto size = upload->size();
                void *host_mem = nullptr;
                if (cuMemAllocHost(&host_mem, size) != CUDA_SUCCESS) {
                    cuStreamEndCapture(cap_stream, &graph);
                    cuStreamDestroy(cap_stream);
                    for (auto *ptr : host_allocs) { cuMemFreeHost(ptr); }
                    if (graph) { cuGraphDestroy(graph); }
                    return {invalid_handle, nullptr};
                }
                std::memcpy(host_mem, data, size);
                cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(address), host_mem, size, cap_stream);
                host_allocs.push_back(host_mem);
            } else if (auto *download = dynamic_cast<BufferDownloadCommand *>(cmd.get())) {
                auto *buffer = reinterpret_cast<const CUDABuffer *>(download->handle());
                auto address = buffer->device_address() + download->offset();
                auto data = download->data();
                auto size = download->size();
                cuMemcpyDtoHAsync(data, static_cast<CUdeviceptr>(address), size, cap_stream);
            } else if (auto *copy = dynamic_cast<BufferCopyCommand *>(cmd.get())) {
                auto *src = reinterpret_cast<const CUDABuffer *>(copy->src_handle());
                auto *dst = reinterpret_cast<const CUDABuffer *>(copy->dst_handle());
                cuMemcpyDtoDAsync(
                    static_cast<CUdeviceptr>(dst->device_address() + copy->dst_offset()),
                    static_cast<CUdeviceptr>(src->device_address() + copy->src_offset()),
                    copy->size(), cap_stream);
            }
        }

        for (auto &cb : user_callbacks) { cb(); }

        ret = cuStreamEndCapture(cap_stream, &graph);
        cuStreamDestroy(cap_stream);

        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuStreamEndCapture failed: {}", err_name ? err_name : "unknown");
            for (auto *ptr : host_allocs) { cuMemFreeHost(ptr); }
            if (graph) { cuGraphDestroy(graph); }
            return {invalid_handle, nullptr};
        }

        size_t num_nodes = 0;
        cuGraphGetNodes(graph, nullptr, &num_nodes);
        luisa::vector<CUgraphNode> nodes(num_nodes);
        if (num_nodes > 0) {
            cuGraphGetNodes(graph, nodes.data(), &num_nodes);
        }

        auto graph_handle = reinterpret_cast<uint64_t>(graph);
        {
            std::scoped_lock lock{_mutex};
            _graph_data[graph_handle] = GraphData{std::move(nodes), std::move(host_allocs)};
        }

        return {graph_handle, graph};
    });
}

void CudaGraphExtImpl::destroy_graph(GraphHandle graph) noexcept {
    if (graph == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        if (auto it = _graph_data.find(graph); it != _graph_data.end()) {
            for (auto *ptr : it->second.host_allocations) {
                cuMemFreeHost(ptr);
            }
            _graph_data.erase(it);
        }
        cuGraphDestroy(reinterpret_cast<CUgraph>(graph));
    });
}

ResourceCreationInfo CudaGraphExtImpl::instantiate(GraphHandle graph, InstantiateFlag flags) noexcept {
    if (graph == invalid_handle) { return {invalid_handle, nullptr}; }
    return _device->with_handle([&]() -> ResourceCreationInfo {
        CUgraphExec exec = nullptr;
        auto ret = cuGraphInstantiateWithFlags(&exec, reinterpret_cast<CUgraph>(graph),
                                               to_instantiate_flags(flags));
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphInstantiateWithFlags failed: {}", err_name ? err_name : "unknown");
            return {invalid_handle, nullptr};
        }
        auto exec_handle = reinterpret_cast<uint64_t>(exec);
        {
            std::scoped_lock lock{_mutex};
            _exec_to_graph.emplace(exec_handle, graph);
        }
        return {exec_handle, exec};
    });
}

void CudaGraphExtImpl::destroy_exec(GraphExecHandle exec) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        {
            std::scoped_lock lock{_mutex};
            _exec_to_graph.erase(exec);
        }
        cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(exec));
    });
}

void CudaGraphExtImpl::launch(GraphExecHandle exec, uint64_t stream_handle) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        auto ret = cuGraphLaunch(reinterpret_cast<CUgraphExec>(exec), to_cu_stream(stream_handle));
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphLaunch failed: {}", err_name ? err_name : "unknown");
        }
    });
}

void CudaGraphExtImpl::upload(GraphExecHandle exec, uint64_t stream_handle) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        auto ret = cuGraphUpload(reinterpret_cast<CUgraphExec>(exec), to_cu_stream(stream_handle));
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphUpload failed: {}", err_name ? err_name : "unknown");
        }
    });
}

bool CudaGraphExtImpl::update(GraphExecHandle exec, CommandList &&cmdlist) noexcept {
    if (exec == invalid_handle || cmdlist.empty()) { return false; }
    return _device->with_handle([&] {
        auto new_graph_info = create_graph(std::move(cmdlist));
        if (new_graph_info.handle == invalid_handle) { return false; }

        CUgraphExecUpdateResultInfo result_info{};
        auto ret = cuGraphExecUpdate(reinterpret_cast<CUgraphExec>(exec),
                                     reinterpret_cast<CUgraph>(new_graph_info.handle),
                                     &result_info);

        {
            std::scoped_lock lock{_mutex};
            if (auto it = _graph_data.find(new_graph_info.handle); it != _graph_data.end()) {
                for (auto *ptr : it->second.host_allocations) {
                    cuMemFreeHost(ptr);
                }
                _graph_data.erase(it);
            }
        }
        cuGraphDestroy(reinterpret_cast<CUgraph>(new_graph_info.handle));
        return ret == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_kernel_node(GraphExecHandle exec, size_t node_index,
                                           uint3 dispatch_size,
                                           luisa::span<const Argument> arguments) noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        CUDA_KERNEL_NODE_PARAMS params{};
        if (cuGraphKernelNodeGetParams(node, &params) != CUDA_SUCCESS) { return false; }

        luisa::vector<void *> kernel_args;
        for (const auto &arg : arguments) {
            kernel_args.push_back(const_cast<void *>(static_cast<const void *>(&arg)));
        }

        params.gridDimX = dispatch_size.x;
        params.gridDimY = dispatch_size.y;
        params.gridDimZ = dispatch_size.z;
        params.kernelParams = kernel_args.data();

        return cuGraphExecKernelNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params) == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_upload_node(GraphExecHandle exec, size_t node_index,
                                           const void *data, size_t size) noexcept {
    if (exec == invalid_handle || !data || size == 0) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        CUDA_MEMCPY3D params{};
        if (cuGraphMemcpyNodeGetParams(node, &params) != CUDA_SUCCESS) { return false; }

        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        params.srcHost = data;
        params.srcPitch = size;
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        // dstDevice is preserved from the original node params
        params.dstPitch = size;
        params.dstHeight = 1;
        params.WidthInBytes = size;
        params.Height = 1;
        params.Depth = 1;

        return cuGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params, nullptr) == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_download_node(GraphExecHandle exec, size_t node_index,
                                             void *data, size_t size) noexcept {
    if (exec == invalid_handle || !data || size == 0) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        CUDA_MEMCPY3D params{};
        if (cuGraphMemcpyNodeGetParams(node, &params) != CUDA_SUCCESS) { return false; }

        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        // srcDevice is preserved from the original node params
        params.srcPitch = size;
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_HOST;
        params.dstHost = data;
        params.dstPitch = size;
        params.dstHeight = 1;
        params.WidthInBytes = size;
        params.Height = 1;
        params.Depth = 1;

        return cuGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params, nullptr) == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_buffer_copy_node(GraphExecHandle exec, size_t node_index,
                                                uint64_t src_handle, size_t src_offset,
                                                uint64_t dst_handle, size_t dst_offset,
                                                size_t size) noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        auto src_addr = reinterpret_cast<const CUDABuffer *>(src_handle)->device_address() + src_offset;
        auto dst_addr = reinterpret_cast<const CUDABuffer *>(dst_handle)->device_address() + dst_offset;

        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice = static_cast<CUdeviceptr>(src_addr);
        params.srcPitch = size;
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        params.dstDevice = static_cast<CUdeviceptr>(dst_addr);
        params.dstPitch = size;
        params.dstHeight = 1;
        params.WidthInBytes = size;
        params.Height = 1;
        params.Depth = 1;

        return cuGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params, nullptr) == CUDA_SUCCESS;
    });
}

void CudaGraphExtImpl::set_node_enabled(GraphExecHandle exec, size_t node_index, bool enabled) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return; }
        auto ret = cuGraphNodeSetEnabled(reinterpret_cast<CUgraphExec>(exec), node,
                                         enabled ? 1u : 0u);
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphNodeSetEnabled failed: {}", err_name ? err_name : "unknown");
        }
    });
}

bool CudaGraphExtImpl::is_node_enabled(GraphExecHandle exec, size_t node_index) const noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = const_cast<CudaGraphExtImpl *>(this)->_get_node(exec, node_index);
        if (!node) { return false; }
        unsigned int is_enabled = 0u;
        auto ret = cuGraphNodeGetEnabled(reinterpret_cast<CUgraphExec>(exec), node, &is_enabled);
        return ret == CUDA_SUCCESS && is_enabled != 0u;
    });
}

}// namespace luisa::compute::cuda
