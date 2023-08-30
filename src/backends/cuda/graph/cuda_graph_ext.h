#include <luisa/backends/ext/graph_ext.h>
namespace luisa::compute::cuda::graph {
class CUDAGraphExt : public luisa::compute::graph::GraphExt {
public:
    using GraphExt::GraphExt;
    virtual luisa::compute::graph::GraphInterface *create_graph_interface() noexcept override;
    virtual void destroy_graph_interface(luisa::compute::graph::GraphInterface *graph_interface) noexcept override;
};
}// namespace luisa::compute::cuda::graph