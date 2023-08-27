#pragma once
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/kernel_node.h>
#include <luisa/runtime/graph/memory_node.h>
#include <luisa/runtime/graph/graph_instance.h>

namespace luisa::compute {
class Graph {
public:
    class Guard {
    public:
        Guard(Graph *graph) noexcept { Graph::set_current(graph); }
        ~Guard() noexcept { Graph::set_current(nullptr); }
    };
    static KernelNode *add_node(luisa::compute::detail::ShaderInvoke<1> &&ivk, size_t dispatch_size_x) noexcept {
        return get_current()->add_node_impl(std::move(ivk), dispatch_size_x);
    }
    static KernelNode *add_node(luisa::compute::detail::ShaderInvoke<2> &&ivk, size_t dispatch_size_x, size_t dispatch_size_y) noexcept {
        return get_current()->add_node_impl(std::move(ivk), dispatch_size_x, dispatch_size_y);
    }
    static KernelNode *add_node(luisa::compute::detail::ShaderInvoke<3> &&ivk, size_t dispatch_size_x, size_t dispatch_size_y, size_t dispatch_size_z) noexcept {
        return get_current()->add_node_impl(std::move(ivk), dispatch_size_x, dispatch_size_y, dispatch_size_z);
    }

    static MemoryNode *add_node(luisa::unique_ptr<BufferCopyCommand> &&cpy) noexcept {
        return get_current()->add_node_impl(std::move(cpy));
    }
    static bool is_building() noexcept { return get_current() != nullptr; }

    GraphInstance *instantiate() noexcept { return instantiate_impl(); }
protected:
    // backend-specific implementation
    virtual KernelNode *add_node_impl(luisa::compute::detail::ShaderInvoke<1> ivk, size_t dispatch_size_x) noexcept = 0;
    virtual KernelNode *add_node_impl(luisa::compute::detail::ShaderInvoke<2> ivk, size_t dispatch_size_x, size_t dispatch_size_y) noexcept = 0;
    virtual KernelNode *add_node_impl(luisa::compute::detail::ShaderInvoke<3> ivk, size_t dispatch_size_x, size_t dispatch_size_y, size_t dispatch_size_z) noexcept = 0;

    virtual MemoryNode *add_node_impl(luisa::unique_ptr<BufferCopyCommand> cpy) noexcept = 0;
    virtual GraphInstance *instantiate_impl() noexcept = 0;
private:
    static thread_local Graph *current;
    static void set_current(Graph *graph) noexcept;
    static Graph *get_current() noexcept;
};
}// namespace luisa::compute