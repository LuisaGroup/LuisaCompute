#include "cuda_graph_interface.h"
#include <luisa/runtime/graph/graph_builder.h>
#include "../cuda_error.h"
#include "../cuda_shader.h"
#include <luisa/runtime/stream.h>
#include <cuda.h>
#include <luisa/runtime/graph/graph_basic_var.h>
#include <luisa/runtime/graph/graph_buffer_var.h>

using namespace luisa::compute::graph;
using namespace luisa::compute::cuda::graph;
// # build backend graph:
void CUDAGraphInterface::build_graph(GraphBuilder *builder) noexcept {
    _cuda_graph_nodes.resize(builder->graph_nodes().size());

    // add kernel nodes:
    _add_kernel_nodes(builder);
    // add capture nodes:
    _add_capture_nodes(builder);
    // add memory nodes:
    _add_memory_nodes(builder);

    // build deps:
    _add_deps(builder);
    builder->clear_need_update_flags();
}

void CUDAGraphInterface::_add_kernel_nodes(GraphBuilder *builder) noexcept {
    auto kernels = builder->kernel_nodes();
    _kernel_parms_cache.reserve(kernels.size());
    _cuda_kernel_nodes.reserve(kernels.size());

    for (auto &&k : kernels) {
        auto res = k->shader_resource();
        // func
        auto cuda_shader = reinterpret_cast<CUDAShader *>(res->handle());

        // dispatch_args
        auto dispatch_args = k->dispatch_args();
        auto block_size = k->block_size();
        std::array grid_size = {1u, 1u, 1u};
        auto round_up = [](uint dispatch, uint block) { return (dispatch + block - 1) / block; };
        for (size_t i = 0; i < dispatch_args.size(); i++) {
            // LUISA_ASSERT(dispatch_args[i].second == luisa::compute::Usage::NONE, "dispatch_arg's usage should be NONE!");
            auto dispatch_arg = builder->sub_var(dispatch_args[i])->cast<GraphVar<uint>>();
            grid_size[i] = round_up(dispatch_arg->eval(), block_size[i]);
        }

        auto encoder = builder->kernel_node_cmd_encoder(k->kernel_node_id());

        CUgraphNode node;
        cuda_shader->encode_kernel_node_parms(
            [&](auto parms) {
                _device->with_handle([&] {
                    LUISA_CHECK_CUDA(cuGraphAddKernelNode(&node, _cuda_graph, nullptr, 0, parms));
                });
            },
            encoder);

        // record the node
        _cuda_kernel_nodes.push_back(node);
        _cuda_graph_nodes[k->node_id().value()] = node;
    };
}

void CUDAGraphInterface::_add_capture_nodes(GraphBuilder *builder) noexcept {
    auto s = capture_stream();
    auto capture_nodes = builder->capture_nodes();
    _cuda_capture_node_graphs.reserve(capture_nodes.size());
    _cuda_capture_nodes.reserve(capture_nodes.size());

    _device->with_handle([&] {
        for (auto &&n : capture_nodes) {
            CUgraph child_graph = nullptr;
            LUISA_CHECK_CUDA_RUNTIME_ERROR(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            n->capture(builder, reinterpret_cast<uint64_t>(s));
            LUISA_CHECK_CUDA_RUNTIME_ERROR(cudaStreamEndCapture(s, &child_graph));
            _cuda_capture_node_graphs.emplace_back(child_graph);// record the child graph, for later update or delete
            CUgraphNode node;
            LUISA_CHECK_CUDA(cuGraphAddChildGraphNode(&node, _cuda_graph, nullptr, 0, child_graph));
            // capture info
            cudaGraphNode_t *nodes = NULL;
            _cuda_capture_nodes.emplace_back(node);
            _cuda_graph_nodes[n->node_id().value()] = node;
        }
    });
}

static CUDA_MEMCPY3D create_memory_parms(GraphBuilder *builder, const MemoryNode &node) noexcept;

void CUDAGraphInterface::_add_memory_nodes(GraphBuilder *builder) noexcept {
    auto memory_nodes = builder->memory_nodes();
    _cuda_memory_nodes.reserve(memory_nodes.size());

    _device->with_handle([&] {
        for (auto &&n : memory_nodes) {
            CUgraphNode node;
            auto parms = create_memory_parms(builder, *n);
            LUISA_CHECK_CUDA(
                cuGraphAddMemcpyNode(&node, _cuda_graph, nullptr, 0, &parms, ctx()));

            _cuda_memory_nodes.push_back(node);
            _cuda_graph_nodes[n->node_id().value()] = node;
        }
    });
}

void CUDAGraphInterface::_add_deps(GraphBuilder *builder) noexcept {
    auto &deps = builder->graph_deps();
    auto deps_size = deps.size();
    _device->with_handle([&] {
        for (auto &&dep : deps) {
            auto &src_node = _cuda_graph_nodes[dep.src.value()];
            auto &dst_node = _cuda_graph_nodes[dep.dst.value()];
            LUISA_CHECK_CUDA(cuGraphAddDependencies(_cuda_graph, &dst_node, &src_node, 1));
        }
    });
}

// # update graph instance:
void CUDAGraphInterface::update_graph_instance_node_parms(GraphBuilder *builder) noexcept {
    for (auto &&node : builder->graph_nodes())
        if (builder->node_need_update(node->node_id())) {
            switch (node->type()) {
                case GraphNodeType::Kernel:
                    _update_kernel_node(dynamic_cast<const KernelNode *>(node), builder);
                    break;
                case GraphNodeType::Capture:
                    _update_capture_node(dynamic_cast<const CaptureNodeBase *>(node), builder);
                    break;
                case GraphNodeType::MemoryCopy:
                    _update_memory_node(dynamic_cast<const MemoryNode *>(node), builder);
                    break;
                default: {
                    LUISA_ERROR_WITH_LOCATION("Unspported graph node type!");
                } break;
            }
        }
    builder->clear_need_update_flags();
}

void CUDAGraphInterface::_update_kernel_node(const KernelNode *kernel, GraphBuilder *builder) noexcept {
    auto kernel_id = kernel->kernel_node_id();
    auto encoder = builder->kernel_node_cmd_encoder(kernel_id);
    auto cuda_node = _cuda_graph_nodes[kernel->node_id().value()];
    auto cuda_shader = reinterpret_cast<CUDAShader *>(kernel->shader_resource()->handle());
    cuda_shader->encode_kernel_node_parms(
        [&](auto parms) {
            _device->with_handle([&] {
                LUISA_CHECK_CUDA(cuGraphExecKernelNodeSetParams(_cuda_graph_exec, cuda_node, parms));
            });
        },
        encoder);
}

void CUDAGraphInterface::_update_capture_node(const CaptureNodeBase *capture, GraphBuilder *builder) noexcept {
    auto capture_id = capture->capture_node_id();
    auto child_graph = _cuda_capture_node_graphs[capture_id.value()];
    auto capture_node = _cuda_graph_nodes[capture->node_id().value()];
    auto s = capture_stream();
    _device->with_handle([&] {
        // delete the old graph
        LUISA_CHECK_CUDA(cuGraphDestroy(child_graph));
        // re-capture
        LUISA_CHECK_CUDA_RUNTIME_ERROR(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        capture->capture(builder, reinterpret_cast<uint64_t>(s));
        LUISA_CHECK_CUDA_RUNTIME_ERROR(cudaStreamEndCapture(s, &child_graph));
        LUISA_CHECK_CUDA(cuGraphExecChildGraphNodeSetParams(_cuda_graph_exec, capture_node, child_graph));
    });
    _cuda_capture_node_graphs[capture_id.value()] = child_graph;
}

void CUDAGraphInterface::_update_memory_node(const MemoryNode *memory, GraphBuilder *builder) noexcept {
    auto memory_id = memory->memory_node_id();
    auto cuda_node = _cuda_graph_nodes[memory->node_id().value()];
    auto parms = create_memory_parms(builder, *memory);
    _device->with_handle([&] {
        LUISA_CHECK_CUDA(cuGraphExecMemcpyNodeSetParams(_cuda_graph_exec, cuda_node, &parms, ctx()));
    });
}

CUstream luisa::compute::cuda::graph::CUDAGraphInterface::capture_stream() noexcept {
    if (!_capture_stream) {
        _device->with_handle([&] {
            LUISA_CHECK_CUDA(cuStreamCreate(&_capture_stream, CU_STREAM_NON_BLOCKING));
        });
    }
    return _capture_stream;
}

void CUDAGraphInterface::create_graph_instance(GraphBuilder *builder) noexcept {
    if (!_cuda_graph) {
        _device->with_handle([&] {
            LUISA_CHECK_CUDA(cuGraphCreate(&_cuda_graph, 0));
        });
        build_graph(builder);
    }

    if (!_cuda_graph_exec) {
        _device->with_handle([&] {
            LUISA_CHECK_CUDA(cuGraphInstantiate(&_cuda_graph_exec, _cuda_graph, 0));
        });
    }
}

void CUDAGraphInterface::destroy_graph_instance(GraphBuilder *builder) noexcept {
    LUISA_ASSERT(_cuda_graph_exec, "empty graph instance");
    _device->with_handle([&] {
        LUISA_CHECK_CUDA(cuGraphExecDestroy(_cuda_graph_exec));
    });
    _cuda_graph_exec = nullptr;
}

void CUDAGraphInterface::launch_graph_instance(Stream *stream) noexcept {
    LUISA_ASSERT(_cuda_graph_exec, "empty graph instance");
    _device->with_handle([&] {
        LUISA_CHECK_CUDA(cuGraphLaunch(_cuda_graph_exec, (CUstream)stream->native_handle()));
    });
}

CUDAGraphInterface::CUDAGraphInterface(CUDADevice *device) noexcept : _device{device} {}

CUDAGraphInterface::~CUDAGraphInterface() noexcept {
    for (auto &&g : _cuda_capture_node_graphs) { LUISA_CHECK_CUDA(cuGraphDestroy(g)); }
    if (_cuda_graph) { LUISA_CHECK_CUDA(cuGraphDestroy(_cuda_graph)); }
    if (_cuda_graph_exec) { LUISA_CHECK_CUDA(cuGraphExecDestroy(_cuda_graph_exec)); }
    if (_capture_stream) { LUISA_CHECK_CUDA(cuStreamDestroy(_capture_stream)); }
}

// DETAIL IMPL:
CUDA_MEMCPY3D create_memory_parms(GraphBuilder *builder, const MemoryNode &node) noexcept {
    CUDA_MEMCPY3D parms{};
    auto dir = node.direction();

    {// init parms:

        // void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes);
        // CUdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
        parms.srcXInBytes = 0;
        parms.srcY = 0;
        parms.srcZ = 0;
        parms.srcLOD = 0;
        // parms.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
        parms.srcHost = nullptr;
        parms.srcDevice = 0;
        parms.srcArray = nullptr;
        parms.srcPitch = 0;
        parms.srcHeight = 1;

        parms.dstXInBytes = 0;
        parms.dstY = 0;
        parms.dstZ = 0;
        parms.dstLOD = 0;
        // parms.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
        parms.dstHost = nullptr;
        parms.dstDevice = 0;
        parms.dstArray = nullptr;

        parms.dstPitch = 0;
        parms.dstHeight = 1;
        parms.WidthInBytes = 0;
        parms.Height = 1;
        parms.Depth = 1;
    }

    auto host_ptr = [&](GraphSubVarId id) {
        auto var = builder->sub_var(id);
        auto host_mem_var = var->cast<GraphVar<void *>>();
        return host_mem_var->eval();
    };

    auto device_buffer = [&](GraphSubVarId id) {
        auto var = builder->sub_var(id);
        auto buffer_var = var->cast<GraphBufferVarBase>();
        auto view = buffer_var->eval_buffer_view_base(builder);
        return view;
    };

    auto device_ptr = [&](const GraphBufferVarBase::BufferViewBase &view) {
        auto ptr = (std::byte *)view.native_handle() + view.offset_bytes();
        return reinterpret_cast<CUdeviceptr>(ptr);
    };

    switch (dir) {
        case luisa::compute::graph::MemoryNodeDirection::HostToDevice: {
            parms.srcMemoryType = CU_MEMORYTYPE_HOST;
            parms.srcHost = host_ptr(node.src_var_id());
            parms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            auto view = device_buffer(node.dst_var_id());
            parms.dstDevice = device_ptr(view);
            parms.WidthInBytes = view.size_bytes();
        } break;
        case luisa::compute::graph::MemoryNodeDirection::DeviceToHost: {
            parms.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            auto view = device_buffer(node.src_var_id());
            parms.srcDevice = device_ptr(view);
            parms.dstMemoryType = CU_MEMORYTYPE_HOST;
            parms.dstHost = host_ptr(node.dst_var_id());
            parms.WidthInBytes = view.size_bytes();
        } break;
        case luisa::compute::graph::MemoryNodeDirection::DeviceToDevice: {
            parms.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            auto src_view = device_buffer(node.src_var_id());
            parms.srcDevice = device_ptr(src_view);
            parms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            auto dst_view = device_buffer(node.dst_var_id());
            parms.dstDevice = device_ptr(dst_view);
            parms.WidthInBytes = src_view.size_bytes();
        } break;
        default:
            break;
    }

    auto round_up_to_64 = [](uint64_t x) { return (x + 63) / 64 * 64; };

    parms.dstPitch = parms.srcPitch = parms.WidthInBytes;
    return parms;
}
