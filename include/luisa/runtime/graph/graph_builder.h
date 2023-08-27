#pragma once
#include <luisa/vstl/unique_ptr.h>
#include <luisa/vstl/vector.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/kernel_node.h>

namespace luisa::compute::graph {
class GraphBuilder {
    template<typename... Args>
    friend class GraphDef;

    template<typename T>
    using U = luisa::unique_ptr<T>;
public:
    luisa::vector<U<GraphVarBase>> _vars;
    luisa::vector<U<KernelNode>> _kernel_nodes;

    static GraphBuilder *current() noexcept { return _current().get(); }
    static void set_var_count(size_t size) noexcept { _current()->_vars.resize(size); }
    template<typename T, size_t I>
    static T &define_graph_var() {
        auto var = make_unique<T>(GraphArgId{I});
        auto ptr = var.get();
        _current()->_vars[I] = std::move(var);
        return *ptr;
    }
    template<typename F>
    [[nodiscard]] static U<GraphBuilder> build(F &&fn) {
        _current() = make_unique<GraphBuilder>();
        _current()->_is_building = true;
        fn();
        _current()->_is_building = false;
        return std::move(_current());
    }
    static bool is_building() noexcept {
        return _current() != nullptr && _current()->_is_building;
    }

    static KernelNode *add_kernel_node(span<uint64_t> arg_ids, const Resource* shader_resource) {
        auto node = make_unique<KernelNode>(current(), arg_ids, shader_resource);
        auto ptr = node.get();
        _current()->_kernel_nodes.emplace_back(std::move(node));
        return ptr;
    }
private:
    static U<GraphBuilder> &_current() noexcept {
        static thread_local U<GraphBuilder> _builder = nullptr;
        return _builder;
    }
    bool _is_building = false;
};
}// namespace luisa::compute::graph