#include "cuda_graph_interface.h"
#include <luisa/runtime/graph/graph_builder.h>
#include "../cuda_error.h"
#include "../cuda_shader.h"
#include <luisa/runtime/stream.h>
#include <cuda.h>

using namespace luisa::compute::graph;
using namespace luisa::compute::cuda::graph;
// # build backend graph:
void CUDAGraphInterface::build_graph(GraphBuilder *builder) noexcept {
    _cuda_graph_nodes.resize(builder->graph_nodes().size());

    // set kernel nodes:
    _add_kernel_nodes(builder);
    _add_capture_nodes(builder);
    _add_deps(builder);
    builder->clear_need_update_flags();
}

void CUDAGraphInterface::_add_kernel_nodes(GraphBuilder *builder) noexcept {
    auto &kernels = builder->kernel_nodes();
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
            LUISA_ASSERT(dispatch_args[i].second == luisa::compute::Usage::NONE, "dispatch_arg's usage should be NONE!");
            auto dispatch_arg = builder->graph_var(dispatch_args[i].first)->cast<GraphVar<uint>>();
            grid_size[i] = round_up(dispatch_arg->value(), block_size[i]);
        }

        auto encoder = builder->kernel_node_cmd_encoder(k->kernel_id());

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
        _cuda_graph_nodes[k->node_id()] = node;
    };
}

void CUDAGraphInterface::_add_capture_nodes(GraphBuilder *builder) noexcept {
    auto s = capture_stream();
    auto &capture_nodes = builder->capture_nodes();
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
            size_t numNodes = 0;
            cudaGraphGetNodes(child_graph, nodes, &numNodes);
            _cuda_capture_nodes.emplace_back(node);
            _cuda_graph_nodes[n->node_id()] = node;
        }
    });
}

void CUDAGraphInterface::_add_deps(GraphBuilder *builder) noexcept {
    auto &deps = builder->graph_deps();
    auto deps_size = deps.size();
    _device->with_handle([&] {
        for (auto &&dep : deps) {
            auto &src_node = _cuda_graph_nodes[dep.src];
            auto &dst_node = _cuda_graph_nodes[dep.dst];
            LUISA_CHECK_CUDA(cuGraphAddDependencies(_cuda_graph, &dst_node, &src_node, 1));
        }
    });
}

// # update graph instance:
void CUDAGraphInterface::update_graph_instance_node_parms(GraphBuilder *builder) noexcept {
    for (auto &&node : builder->graph_nodes())
        if (builder->node_need_update(node)) {
            switch (node->type()) {
                case GraphNodeType::Kernel:
                    _update_kernel_node(dynamic_cast<const KernelNode *>(node), builder);
                    break;
                case GraphNodeType::Capture:
                    _update_capture_node(dynamic_cast<const CaptureNodeBase *>(node), builder);
                    break;
                default: {
                    LUISA_ERROR_WITH_LOCATION("Unspported graph node type!");
                } break;
            }
        }
    builder->clear_need_update_flags();
}

void CUDAGraphInterface::_update_kernel_node(const KernelNode *kernel, GraphBuilder *builder) noexcept {
    auto kernel_id = kernel->kernel_id();
    auto encoder = builder->kernel_node_cmd_encoder(kernel_id);
    auto cuda_node = _cuda_graph_nodes[kernel->node_id()];
    auto cuda_shader = reinterpret_cast<CUDAShader *>(kernel->shader_resource()->handle());
    cuda_shader->encode_kernel_node_parms(
        [&](auto parms) {
            _device->with_handle([&] {
                cuGraphExecKernelNodeSetParams(_cuda_graph_exec, cuda_node, parms);
            });
        },
        encoder);
}

void CUDAGraphInterface::_update_capture_node(const CaptureNodeBase *capture, GraphBuilder *builder) noexcept {
    auto capture_id = capture->capture_id();
    auto child_graph = _cuda_capture_node_graphs[capture_id];
    auto capture_node = _cuda_graph_nodes[capture->node_id()];
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
    _cuda_capture_node_graphs[capture_id] = child_graph;
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
            LUISA_CHECK_CUDA(cuGraphInstantiate(&_cuda_graph_exec, _cuda_graph, CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH));
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
