#include "cuda_graph_ext.h"
#include "../cuda_device.h"
#include "../cuda_stream.h"
#include "../cuda_buffer.h"
#include "../cuda_bindless_array.h"
#include "../cuda_shader.h"
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

void CUDA_CB cuda_graph_host_copy_callback(void *user_data) noexcept {
    auto *data = static_cast<CudaGraphHostCopyData *>(user_data);
    std::memcpy(data->dst, data->src, data->size);
}

}// namespace

CudaGraphExtImpl::CudaGraphExtImpl(CUDADevice *device) noexcept
    : _device{device} {}

CudaGraphExtImpl::~CudaGraphExtImpl() noexcept {
    for (auto &[exec_handle, data] : _exec_data) {
        for (auto *ptr : data.host_allocations) {
            cuMemFreeHost(ptr);
        }
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
    auto it = _exec_data.find(exec);
    if (it == _exec_data.end()) { return nullptr; }
    auto git = _graph_data.find(it->second.graph_handle);
    if (git == _graph_data.end() || node_index >= git->second.nodes.size()) { return nullptr; }
    return git->second.nodes[node_index];
}

ResourceCreationInfo CudaGraphExtImpl::_create_graph(CommandList &&cmdlist) noexcept {
    if (cmdlist.empty()) { return {invalid_handle, nullptr}; }

    return _device->with_handle([&]() -> ResourceCreationInfo {
        luisa::vector<void *> host_allocs;
        luisa::vector<CudaGraphHostCopyData> host_copies;
        auto commands = cmdlist.steal_commands();
        auto user_callbacks = cmdlist.steal_callbacks();

        struct UploadStagingVisitor final : MutableCommandVisitor {
            luisa::vector<void *> &host_allocs;
            luisa::vector<CudaGraphHostCopyData> &host_copies;
            bool ok{true};

            UploadStagingVisitor(luisa::vector<void *> &host_allocs,
                                 luisa::vector<CudaGraphHostCopyData> &host_copies) noexcept
                : host_allocs{host_allocs}, host_copies{host_copies} {}

            void visit(BufferUploadCommand *upload) noexcept override {
                if (!ok) { return; }
                auto size = upload->size();
                void *host_mem = nullptr;
                if (cuMemAllocHost(&host_mem, size) != CUDA_SUCCESS) {
                    ok = false;
                    return;
                }
                std::memcpy(host_mem, upload->data(), size);
                upload->set_data(host_mem);
                host_allocs.push_back(host_mem);
            }

            void visit(BufferDownloadCommand *download) noexcept override {
                if (!ok) { return; }
                auto size = download->size();
                void *host_mem = nullptr;
                if (cuMemAllocHost(&host_mem, size) != CUDA_SUCCESS) {
                    ok = false;
                    return;
                }
                host_copies.push_back(CudaGraphHostCopyData{
                    .dst = download->data(),
                    .src = host_mem,
                    .size = size,
                });
                download->set_data(host_mem);
                host_allocs.push_back(host_mem);
            }
            void visit(BufferCopyCommand *) noexcept override {}
            void visit(BufferToTextureCopyCommand *) noexcept override {}
            void visit(ShaderDispatchCommand *) noexcept override {}
            void visit(TextureUploadCommand *) noexcept override {}
            void visit(TextureDownloadCommand *) noexcept override {}
            void visit(TextureCopyCommand *) noexcept override {}
            void visit(TextureToBufferCopyCommand *) noexcept override {}
            void visit(AccelBuildCommand *) noexcept override {}
            void visit(MeshBuildCommand *) noexcept override {}
            void visit(CurveBuildCommand *) noexcept override {}
            void visit(ProceduralPrimitiveBuildCommand *) noexcept override {}
            void visit(MotionInstanceBuildCommand *) noexcept override {}
            void visit(BindlessArrayUpdateCommand *) noexcept override {}
            void visit(CustomCommand *) noexcept override {}
        };

        UploadStagingVisitor staging_visitor{host_allocs, host_copies};
        for (auto &cmd : commands) {
            cmd->accept(staging_visitor);
            if (!staging_visitor.ok) { break; }
        }
        if (!staging_visitor.ok) {
            for (auto *ptr : host_allocs) { cuMemFreeHost(ptr); }
            return {invalid_handle, nullptr};
        }

        CUstream cap_stream = nullptr;
        if (auto err = cuStreamCreate(&cap_stream, CU_STREAM_DEFAULT); err != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(err, &err_name);
            LUISA_WARNING_WITH_LOCATION("Failed to create capture stream: {}", err_name ? err_name : "unknown");
            for (auto *ptr : host_allocs) { cuMemFreeHost(ptr); }
            return {invalid_handle, nullptr};
        }

        auto ret = cuStreamBeginCapture(cap_stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuStreamBeginCapture failed: {}", err_name ? err_name : "unknown");
            cuStreamDestroy(cap_stream);
            for (auto *ptr : host_allocs) { cuMemFreeHost(ptr); }
            return {invalid_handle, nullptr};
        }

        CUgraph graph = nullptr;

        struct GraphCaptureVisitor final : MutableCommandVisitor {
            CUstream stream;
            luisa::vector<CudaGraphHostCopyData> &host_copies;
            size_t download_index{0u};
            bool ok{true};
            GraphCaptureVisitor(CUstream stream, luisa::vector<CudaGraphHostCopyData> &host_copies) noexcept
                : stream{stream}, host_copies{host_copies} {}

            void visit(BufferUploadCommand *upload) noexcept override {
                if (!ok) { return; }
                auto *buffer = reinterpret_cast<const CUDABuffer *>(upload->handle());
                auto address = buffer->device_address() + upload->offset();
                auto data = upload->data();
                auto size = upload->size();
                if (cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(address), data, size, stream) != CUDA_SUCCESS) {
                    ok = false;
                }
            }

            void visit(BufferDownloadCommand *download) noexcept override {
                if (!ok) { return; }
                auto *buffer = reinterpret_cast<const CUDABuffer *>(download->handle());
                auto address = buffer->device_address() + download->offset();
                auto data = download->data();
                auto size = download->size();
                if (cuMemcpyDtoHAsync(data, static_cast<CUdeviceptr>(address), size, stream) != CUDA_SUCCESS) {
                    ok = false;
                    return;
                }
                if (download_index >= host_copies.size() ||
                    cuLaunchHostFunc(stream, cuda_graph_host_copy_callback,
                                     &host_copies[download_index++]) != CUDA_SUCCESS) {
                    ok = false;
                }
            }

            void visit(BufferCopyCommand *copy) noexcept override {
                if (!ok) { return; }
                auto *src = reinterpret_cast<const CUDABuffer *>(copy->src_handle());
                auto *dst = reinterpret_cast<const CUDABuffer *>(copy->dst_handle());
                if (cuMemcpyDtoDAsync(
                        static_cast<CUdeviceptr>(dst->device_address() + copy->dst_offset()),
                        static_cast<CUdeviceptr>(src->device_address() + copy->src_offset()),
                        copy->size(), stream) != CUDA_SUCCESS) {
                    ok = false;
                }
            }

            void visit(BufferToTextureCopyCommand *) noexcept override {}

            /// Rebuild the by-value `Params` struct argument exactly like
            /// `CUDAShaderNative::_launch` does (16-byte-aligned slots, buffer
            /// arguments stored as their `CUDABuffer::Binding`, uniforms copied
            /// verbatim, trailing launch-size-and-kernel-id slot), then replay
            /// the very same `cuLaunchKernel` on the capturing stream. CUDA
            /// records the launch as a kernel node of the graph-in-progress,
            /// so the captured kernel reads byte-identical arguments to a live
            /// dispatch. Any dispatch the normal path would treat specially
            /// (printing, bound arguments, indirect or multiple dispatch, a
            /// non-native shader, Accel arguments) is refused instead of
            /// approximated, so a captured graph is never subtly different
            /// from the ordinary submission path.
            void visit(ShaderDispatchCommand *dispatch) noexcept override {
                if (!ok) { return; }
                auto *shader = reinterpret_cast<const CUDAShader *>(dispatch->handle());
                if (shader == nullptr ||
                    !shader->is_native() ||
                    !shader->is_graph_compatible() ||
                    shader->requires_printing() ||
                    shader->bound_argument_count() != 0u ||
                    dispatch->is_indirect() ||
                    dispatch->is_multiple_dispatch()) [[unlikely]] {
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: capturing a dispatch of a shader that is not "
                        "a plain single-launch native CUDA kernel (printing, bound "
                        "arguments, indirect or multiple dispatch). Refusing the "
                        "capture - a graph created from it could not reproduce the "
                        "ordinary launch semantics.");
                    ok = false;
                    return;
                }
                auto dispatch_size = dispatch->dispatch_size();
                if (any(dispatch_size == make_uint3(0u))) [[unlikely]] {
                    // The normal path ignores empty launches; silently turning
                    // one into a missing graph node would change what the
                    // command list captures, so refuse instead.
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: empty (zero-sized) dispatch cannot be "
                        "captured into a graph. Refusing the capture.");
                    ok = false;
                    return;
                }
                auto func = static_cast<CUfunction>(shader->handle());
                auto block_size = shader->block_size();
                // Same alignment rule as _launch's allocate_argument().
                static constexpr auto cuda_shader_native_alignment = 16u;
                auto argument_buffer_offset = size_t{0u};
                auto allocate_argument = [&argument_buffer_offset](size_t bytes) noexcept {
                    auto offset = (argument_buffer_offset + cuda_shader_native_alignment - 1u) / cuda_shader_native_alignment * cuda_shader_native_alignment;
                    argument_buffer_offset = offset + bytes;
                    return offset;
                };
                luisa::vector<std::byte> argument_buffer;
                auto store = [&](size_t offset, const void *src, size_t bytes) noexcept {
                    if (argument_buffer.size() < offset + bytes) {
                        argument_buffer.resize(offset + bytes);
                    }
                    std::memcpy(argument_buffer.data() + offset, src, bytes);
                };
                for (auto &&arg : dispatch->arguments()) {
                    using Tag = ShaderDispatchCommand::Argument::Tag;
                    switch (arg.tag) {
                        case Tag::BUFFER: {
                            auto offset = allocate_argument(sizeof(CUDABuffer::Binding));
                            auto *buffer = reinterpret_cast<const CUDABuffer *>(arg.buffer.handle);
                            auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                            store(offset, &binding, sizeof(binding));
                            break;
                        }
                        case Tag::TEXTURE: {
                            auto offset = allocate_argument(sizeof(CUDATexture::Binding));
                            auto *texture = reinterpret_cast<const CUDATexture *>(arg.texture.handle);
                            auto binding = texture->binding(arg.texture.level);
                            store(offset, &binding, sizeof(binding));
                            break;
                        }
                        case Tag::UNIFORM: {
                            auto uniform = dispatch->uniform(arg.uniform);
                            auto offset = allocate_argument(uniform.size_bytes());
                            store(offset, uniform.data(), uniform.size_bytes());
                            break;
                        }
                        case Tag::BINDLESS_ARRAY: {
                            auto offset = allocate_argument(sizeof(CUDABindlessArray::Binding));
                            auto *array = reinterpret_cast<const CUDABindlessArray *>(arg.bindless_array.handle);
                            auto binding = array->binding();
                            store(offset, &binding, sizeof(binding));
                            break;
                        }
                        case Tag::ACCEL: {
                            // OptiX traversable handles must not be baked into
                            // a CUDA graph (they may be rebuilt between launches).
                            LUISA_WARNING_WITH_LOCATION(
                                "CudaGraphExt: capturing a dispatch with an Accel "
                                "argument is refused (traversable handles must not "
                                "be baked into a CUDA graph).");
                            ok = false;
                            return;
                        }
                    }
                }
                // The trailing launch-size-and-kernel-id slot (uint4), exactly
                // like _launch: kernel id 0 for a single dispatch.
                auto launch_size_offset = allocate_argument(sizeof(uint4));
                auto launch_size_and_kernel_id = make_uint4(dispatch_size, 0u);
                store(launch_size_offset, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));
                // The launch configuration _launch computes for this dispatch.
                auto blocks = (dispatch_size + block_size - 1u) / block_size;
                // Launch on the capturing stream with the identical argument
                // encoding: CUDA records it as a kernel node.
                void *kernel_args = static_cast<void *>(argument_buffer.data());
                if (auto err = cuLaunchKernel(
                        func,
                        blocks.x, blocks.y, blocks.z,
                        block_size.x, block_size.y, block_size.z,
                        0u, stream,
                        &kernel_args, nullptr); err != CUDA_SUCCESS) {
                    const char *err_name = nullptr;
                    cuGetErrorName(err, &err_name);
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: cuLaunchKernel on the capture stream failed: {}",
                        err_name ? err_name : "unknown");
                    ok = false;
                }
            }

            void visit(TextureUploadCommand *) noexcept override {}
            void visit(TextureDownloadCommand *) noexcept override {}
            void visit(TextureCopyCommand *) noexcept override {}
            void visit(TextureToBufferCopyCommand *) noexcept override {}
            void visit(AccelBuildCommand *) noexcept override {}
            void visit(MeshBuildCommand *) noexcept override {}
            void visit(CurveBuildCommand *) noexcept override {}
            void visit(ProceduralPrimitiveBuildCommand *) noexcept override {}
            void visit(MotionInstanceBuildCommand *) noexcept override {}
            void visit(BindlessArrayUpdateCommand *) noexcept override {}
            void visit(CustomCommand *) noexcept override {}
        };

        GraphCaptureVisitor visitor{cap_stream, host_copies};
        for (auto &cmd : commands) {
            cmd->accept(visitor);
            if (!visitor.ok) { break; }
        }

        if (!visitor.ok) {
            cuStreamEndCapture(cap_stream, &graph);
            cuStreamDestroy(cap_stream);
            for (auto *ptr : host_allocs) { cuMemFreeHost(ptr); }
            if (graph) { cuGraphDestroy(graph); }
            return {invalid_handle, nullptr};
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
            _graph_data[graph_handle] = GraphData{std::move(nodes), std::move(host_allocs), std::move(host_copies)};
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

ResourceCreationInfo CudaGraphExtImpl::_instantiate(GraphHandle graph, InstantiateFlag flags) noexcept {
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
            _exec_data.emplace(exec_handle, ExecData{graph, {}, {}});
        }
        return {exec_handle, exec};
    });
}

void CudaGraphExtImpl::destroy_exec(GraphExecHandle exec) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        {
            std::scoped_lock lock{_mutex};
            if (auto it = _exec_data.find(exec); it != _exec_data.end()) {
                for (auto *ptr : it->second.host_allocations) {
                    cuMemFreeHost(ptr);
                }
                _exec_data.erase(it);
            }
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
        auto new_graph_info = _create_graph(std::move(cmdlist));
        if (new_graph_info.handle == invalid_handle) { return false; }

        CUgraphExecUpdateResultInfo result_info{};
        auto ret = cuGraphExecUpdate(reinterpret_cast<CUgraphExec>(exec),
                                     reinterpret_cast<CUgraph>(new_graph_info.handle),
                                     &result_info);

        auto updated = ret == CUDA_SUCCESS;
        {
            std::scoped_lock lock{_mutex};
            auto new_graph_it = _graph_data.find(new_graph_info.handle);
            if (new_graph_it != _graph_data.end()) {
                if (updated) {
                    auto exec_it = _exec_data.find(exec);
                    if (exec_it != _exec_data.end()) {
                        for (auto *ptr : exec_it->second.host_allocations) {
                            cuMemFreeHost(ptr);
                        }
                        exec_it->second.host_allocations = std::move(new_graph_it->second.host_allocations);
                        exec_it->second.host_copies = std::move(new_graph_it->second.host_copies);
                    } else {
                        updated = false;
                    }
                }
                if (!updated) {
                    for (auto *ptr : new_graph_it->second.host_allocations) {
                        cuMemFreeHost(ptr);
                    }
                }
                _graph_data.erase(new_graph_it);
            }
        }
        cuGraphDestroy(reinterpret_cast<CUgraph>(new_graph_info.handle));
        return updated;
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
