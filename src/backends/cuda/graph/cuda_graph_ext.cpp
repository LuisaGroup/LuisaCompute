#include "cuda_graph_ext.h"
#include "cuda_graph_interface.h"

using namespace luisa::compute::graph;
using namespace luisa::compute::cuda::graph;

GraphInterface *CUDAGraphExt::create_graph_interface() noexcept {
    return new_with_allocator<CUDAGraphInterface>(dynamic_cast<CUDADevice*>(device_interface()));
}

void CUDAGraphExt::destroy_graph_interface(GraphInterface *graph_interface) noexcept {
    luisa::delete_with_allocator(dynamic_cast<CUDAGraphInterface *>(graph_interface));
}