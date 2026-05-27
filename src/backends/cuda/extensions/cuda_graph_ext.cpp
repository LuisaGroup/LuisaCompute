#include "cuda_graph_ext.h"
#include "../cuda_device.h"
#include "../cuda_stream.h"
#include "../cuda_buffer.h"
#include <cuda_runtime_api.h>

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] CUstream to_cu_stream(uint64_t stream_handle) noexcept {
    return reinterpret_cast<CUDAStream *>(stream_handle)->handle();
}

[[nodiscard]] unsigned long long to_instantiate_flags(CudaGraphExt::InstantiateFlag flags) noexcept {
    unsigned long long f = 0u;
    auto v = static_cast<uint32_t>(flags);
    if (v & static_cast<uint32_t>(CudaGraphExt::InstantiateFlag::AUTO_FREE_ON_LAUNCH)) {
        f |= cudaGraphInstantiateFlagAutoFreeOnLaunch;
    }
    if (v & static_cast<uint32_t>(CudaGraphExt::InstantiateFlag::DEVICE_LAUNCH)) {
        f |= cudaGraphInstantiateFlagDeviceLaunch;
    }
    return f;
}

}// namespace

CudaGraphExtImpl::CudaGraphExtImpl(CUDADevice *device) noexcept
    : _device{device} {}

CudaGraphExtImpl::~CudaGraphExtImpl() noexcept {
    for (auto &[exec_handle, graph_handle] : _exec_to_graph) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec_handle));
    }
    for (auto &[graph_handle, data] : _graph_data) {
        for (auto *ptr : data.host_allocations) {
            cudaFreeHost(ptr);
        }
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph_handle));
    }
}

cudaGraphNode_t CudaGraphExtImpl::_get_node(GraphExecHandle exec, size_t node_index) const noexcept {
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
            LUISA_WARNING_WITH_LOCATION("Failed to create capture stream: {}", static_cast<int>(err));
            return {invalid_handle, nullptr};
        }

        auto cuda_cap_stream = static_cast<cudaStream_t>(cap_stream);
        auto ret = cudaStreamBeginCapture(cuda_cap_stream, cudaStreamCaptureModeGlobal);
        if (ret != cudaSuccess) {
            LUISA_WARNING_WITH_LOCATION("cudaStreamBeginCapture failed: {}", cudaGetErrorName(ret));
            cuStreamDestroy(cap_stream);
            return {invalid_handle, nullptr};
        }

        luisa::vector<void *> host_allocs;
        auto commands = cmdlist.steal_commands();
        auto user_callbacks = cmdlist.steal_callbacks();
        cudaGraph_t graph = nullptr;

        for (auto &cmd : commands) {
            if (auto *upload = dynamic_cast<BufferUploadCommand *>(cmd.get())) {
                auto *buffer = reinterpret_cast<const CUDABuffer *>(upload->handle());
                auto address = buffer->device_address() + upload->offset();
                auto data = upload->data();
                auto size = upload->size();
                void *host_mem = nullptr;
                if (cudaMallocHost(&host_mem, size) != cudaSuccess) {
                    cudaStreamEndCapture(cuda_cap_stream, &graph);
                    cuStreamDestroy(cap_stream);
                    for (auto *ptr : host_allocs) { cudaFreeHost(ptr); }
                    return {invalid_handle, nullptr};
                }
                std::memcpy(host_mem, data, size);
                cudaMemcpyAsync(reinterpret_cast<void *>(address), host_mem, size,
                                cudaMemcpyHostToDevice, cuda_cap_stream);
                host_allocs.push_back(host_mem);
            } else if (auto *download = dynamic_cast<BufferDownloadCommand *>(cmd.get())) {
                auto *buffer = reinterpret_cast<const CUDABuffer *>(download->handle());
                auto address = buffer->device_address() + download->offset();
                auto data = download->data();
                auto size = download->size();
                cudaMemcpyAsync(data, reinterpret_cast<const void *>(address), size,
                                cudaMemcpyDeviceToHost, cuda_cap_stream);
            } else if (auto *copy = dynamic_cast<BufferCopyCommand *>(cmd.get())) {
                auto *src = reinterpret_cast<const CUDABuffer *>(copy->src_handle());
                auto *dst = reinterpret_cast<const CUDABuffer *>(copy->dst_handle());
                cudaMemcpyAsync(
                    reinterpret_cast<void *>(dst->device_address() + copy->dst_offset()),
                    reinterpret_cast<const void *>(src->device_address() + copy->src_offset()),
                    copy->size(), cudaMemcpyDeviceToDevice, cuda_cap_stream);
            }
        }

        for (auto &cb : user_callbacks) { cb(); }

        ret = cudaStreamEndCapture(cuda_cap_stream, &graph);
        cuStreamDestroy(cap_stream);

        if (ret != cudaSuccess) {
            LUISA_WARNING_WITH_LOCATION("cudaStreamEndCapture failed: {}", cudaGetErrorName(ret));
            for (auto *ptr : host_allocs) { cudaFreeHost(ptr); }
            return {invalid_handle, nullptr};
        }

        size_t num_nodes = 0;
        cudaGraphGetNodes(graph, nullptr, &num_nodes);
        luisa::vector<cudaGraphNode_t> nodes(num_nodes);
        if (num_nodes > 0) {
            cudaGraphGetNodes(graph, nodes.data(), &num_nodes);
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
                cudaFreeHost(ptr);
            }
            _graph_data.erase(it);
        }
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph));
    });
}

ResourceCreationInfo CudaGraphExtImpl::instantiate(GraphHandle graph, InstantiateFlag flags) noexcept {
    if (graph == invalid_handle) { return {invalid_handle, nullptr}; }
    return _device->with_handle([&]() -> ResourceCreationInfo {
        cudaGraphExec_t exec = nullptr;
        auto ret = cudaGraphInstantiate(&exec, reinterpret_cast<cudaGraph_t>(graph),
                                        to_instantiate_flags(flags));
        if (ret != cudaSuccess) {
            LUISA_WARNING_WITH_LOCATION("cudaGraphInstantiate failed: {}", cudaGetErrorName(ret));
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
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec));
    });
}

void CudaGraphExtImpl::launch(GraphExecHandle exec, uint64_t stream_handle) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        auto ret = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(exec),
                                   static_cast<cudaStream_t>(to_cu_stream(stream_handle)));
        if (ret != cudaSuccess) {
            LUISA_WARNING_WITH_LOCATION("cudaGraphLaunch failed: {}", cudaGetErrorName(ret));
        }
    });
}

void CudaGraphExtImpl::upload(GraphExecHandle exec, uint64_t stream_handle) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        auto ret = cudaGraphUpload(reinterpret_cast<cudaGraphExec_t>(exec),
                                   static_cast<cudaStream_t>(to_cu_stream(stream_handle)));
        if (ret != cudaSuccess) {
            LUISA_WARNING_WITH_LOCATION("cudaGraphUpload failed: {}", cudaGetErrorName(ret));
        }
    });
}

bool CudaGraphExtImpl::update(GraphExecHandle exec, CommandList &&cmdlist) noexcept {
    if (exec == invalid_handle || cmdlist.empty()) { return false; }
    return _device->with_handle([&] {
        auto new_graph_info = create_graph(std::move(cmdlist));
        if (new_graph_info.handle == invalid_handle) { return false; }

        cudaGraphExecUpdateResultInfo result_info{};
        auto ret = cudaGraphExecUpdate(reinterpret_cast<cudaGraphExec_t>(exec),
                                       reinterpret_cast<cudaGraph_t>(new_graph_info.handle),
                                       &result_info);

        {
            std::scoped_lock lock{_mutex};
            if (auto it = _graph_data.find(new_graph_info.handle); it != _graph_data.end()) {
                for (auto *ptr : it->second.host_allocations) {
                    cudaFreeHost(ptr);
                }
                _graph_data.erase(it);
            }
        }
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(new_graph_info.handle));
        return ret == cudaSuccess;
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

        cudaKernelNodeParams params{};
        if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) { return false; }

        luisa::vector<void *> kernel_args;
        for (const auto &arg : arguments) {
            kernel_args.push_back(const_cast<void *>(static_cast<const void *>(&arg)));
        }

        params.gridDim = dim3{dispatch_size.x, dispatch_size.y, dispatch_size.z};
        params.kernelParams = kernel_args.data();

        return cudaGraphExecKernelNodeSetParams(
                   reinterpret_cast<cudaGraphExec_t>(exec), node, &params) == cudaSuccess;
    });
}

bool CudaGraphExtImpl::update_upload_node(GraphExecHandle exec, size_t node_index,
                                       const void *data, size_t size) noexcept {
    if (exec == invalid_handle || !data || size == 0) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        cudaMemcpy3DParms params{};
        if (cudaGraphMemcpyNodeGetParams(node, &params) != cudaSuccess) { return false; }

        params.srcPtr = cudaPitchedPtr{const_cast<void *>(data), size, size, 1};
        params.srcPos = cudaPos{0, 0, 0};
        params.extent = cudaExtent{size, 1, 1};
        params.kind = cudaMemcpyHostToDevice;

        return cudaGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<cudaGraphExec_t>(exec), node, &params) == cudaSuccess;
    });
}

bool CudaGraphExtImpl::update_download_node(GraphExecHandle exec, size_t node_index,
                                         void *data, size_t size) noexcept {
    if (exec == invalid_handle || !data || size == 0) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        cudaMemcpy3DParms params{};
        if (cudaGraphMemcpyNodeGetParams(node, &params) != cudaSuccess) { return false; }

        params.dstPtr = cudaPitchedPtr{data, size, size, 1};
        params.dstPos = cudaPos{0, 0, 0};
        params.extent = cudaExtent{size, 1, 1};
        params.kind = cudaMemcpyDeviceToHost;

        return cudaGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<cudaGraphExec_t>(exec), node, &params) == cudaSuccess;
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

        cudaMemcpy3DParms params{};
        params.srcPtr = cudaPitchedPtr{reinterpret_cast<void *>(src_addr), size, size, 1};
        params.srcPos = cudaPos{0, 0, 0};
        params.dstPtr = cudaPitchedPtr{reinterpret_cast<void *>(dst_addr), size, size, 1};
        params.dstPos = cudaPos{0, 0, 0};
        params.extent = cudaExtent{size, 1, 1};
        params.kind = cudaMemcpyDeviceToDevice;

        return cudaGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<cudaGraphExec_t>(exec), node, &params) == cudaSuccess;
    });
}

void CudaGraphExtImpl::set_node_enabled(GraphExecHandle exec, size_t node_index, bool enabled) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return; }
        auto ret = cudaGraphNodeSetEnabled(reinterpret_cast<cudaGraphExec_t>(exec), node,
                                           enabled ? 1u : 0u);
        if (ret != cudaSuccess) {
            LUISA_WARNING_WITH_LOCATION("cudaGraphNodeSetEnabled failed: {}", cudaGetErrorName(ret));
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
        auto ret = cudaGraphNodeGetEnabled(reinterpret_cast<cudaGraphExec_t>(exec), node, &is_enabled);
        return ret == cudaSuccess && is_enabled != 0u;
    });
}

}// namespace luisa::compute::cuda
