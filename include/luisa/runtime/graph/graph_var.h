#pragma once
#include <luisa/runtime/graph/id_with_type.h>
#include <luisa/runtime/graph/graph_var_id.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/basic_traits.h>
#include <luisa/vstl/unique_ptr.h>
#include <luisa/runtime/graph/kernel_node_cmd_encoder.h>

namespace luisa::compute::graph {
class GraphBuilder;
class GraphNode;

enum class GraphResourceTag {
    Basic,
    Buffer,
    HostMemory,
    Max,
};

class LC_RUNTIME_API GraphVarBase {
protected:

    bool _need_update{false};
public:
    template<typename T>
    using U = luisa::unique_ptr<T>;
    friend class GraphBuilder;
    static constexpr string_view graphviz_prefix = "var_";
    // when we analyse a graph, all graph vars are virtual(not a real resource view)
    GraphVarBase(GraphInputVarId id, GraphResourceTag tag) noexcept
        : _is_virtual{id.value() != GraphInputVarId::invalid_id}, _input_var_id{id}, _sub_var_id{id.value()}, _tag{tag} {}

    // create a sub var from another sub var
    GraphVarBase(GraphSubVarId sub_var_id, const GraphVarBase &src, luisa::vector<GraphSubVarId> other_deps = {}) noexcept
        : _is_virtual{src._is_virtual},
          _input_var_id{src._input_var_id},
          _sub_var_id{sub_var_id},
          _tag{src._tag},
          _other_dependent_var_ids{other_deps.begin(), other_deps.end()} {}

    // copy construct
    GraphVarBase(const GraphVarBase &src) noexcept
        : _is_virtual{src._is_virtual},
          _input_var_id{src._input_var_id},
          _sub_var_id{src._sub_var_id},
          _tag{src._tag},
          _other_dependent_var_ids{src._other_dependent_var_ids} {}

    bool is_virtual() const noexcept { return _is_virtual; }
    bool need_update() const noexcept { return _need_update; }

    GraphInputVarId input_var_id() const noexcept { return _input_var_id; }
    GraphSubVarId sub_var_id() const noexcept { return _sub_var_id; }

    GraphResourceTag tag() const noexcept { return _tag; }
    // virtual U<GraphVarBase> clone() const noexcept = 0;
    template<typename T>
    auto cast() noexcept {
        static_assert(std::is_base_of_v<GraphVarBase, T>);
        return dynamic_cast<T *>(this);
    }
    template<typename T>
    auto cast() const noexcept {
        static_assert(std::is_base_of_v<GraphVarBase, T>);
        return dynamic_cast<const T *>(this);
    }
    string_view var_name() const noexcept { return _name; }
    GraphVarBase &set_var_name(string_view name) noexcept;
    bool is_valid() const noexcept { return _input_var_id.value() != GraphInputVarId::invalid_id && _sub_var_id.value() != GraphSubVarId::invalid_id; }
    bool is_sub_var() const noexcept { return _input_var_id.value() < _sub_var_id.value(); }
    bool is_input_var() const noexcept { return _input_var_id.value() == _sub_var_id.value(); }

    virtual void graphviz_def(std::ostream &o) const noexcept;
    virtual void graphviz_id(std::ostream &o) const noexcept { o << graphviz_prefix << sub_var_id(); }
    virtual void graphviz_arg_usage(std::ostream &o) const noexcept;
protected:
    virtual void update_kernel_node_cmd_encoder(
        size_t arg_idx_in_kernel_parms, KernelNodeCmdEncoder *encoder) const noexcept = 0;

    virtual void sub_var_update_check(GraphBuilder *builder) noexcept = 0;

    void clear_need_update_flag() noexcept { _need_update = false; }

    template<typename T, typename... Args>
        requires std::is_base_of_v<GraphVarBase, T>
    [[nodiscard]] T *emplace_sub_var(Args &&...args) const noexcept;
private:
    bool _is_virtual{true};
    GraphInputVarId _input_var_id{GraphInputVarId::invalid_id};
    GraphSubVarId _sub_var_id{GraphSubVarId::invalid_id};
    vector<GraphSubVarId> _other_dependent_var_ids;
    GraphResourceTag _tag;
    string _name;
};

template<typename T>
class GraphVar;

template<typename T>
class GraphSubVar;

template<typename T>
struct is_graph_var : std::false_type {};

template<typename T>
struct is_graph_var<GraphVar<T>> : std::true_type {};

template<typename T>
constexpr bool is_graph_var_v = is_graph_var<T>::value;

namespace detail {
template<typename T>
struct view_to_graph_shader_invocation : std::type_identity<GraphVar<T>> {};

template<typename T>
using view_to_graph_shader_invocation_t = typename view_to_graph_shader_invocation<T>::type;

template<typename T>
struct graph_var_to_view {};

template<typename T>
struct graph_var_to_view<GraphVar<T>> : std::type_identity<T> {};

template<typename T>
using graph_var_to_view_t = typename graph_var_to_view<T>::type;
}// namespace detail
}// namespace luisa::compute::graph

#include <luisa/runtime/graph/graph_builder.h>

namespace luisa::compute::graph {
template<typename T, typename... Args>
    requires std::is_base_of_v<GraphVarBase, T>
[[nodiscard]] T *GraphVarBase::emplace_sub_var(Args &&...args) const noexcept {
    // get a new sub var id
    auto id = GraphBuilder::current()->_sub_vars.size();
    const T &drived = *(this->cast<const T>());
    auto sub_var = make_unique<T>(GraphSubVarId{id}, drived, std::forward<Args>(args)...);
    auto ptr = sub_var.get();
    GraphBuilder::current()->_def_sub_var(std::move(sub_var));
    GraphBuilder::current()->_sub_vars.emplace_back(ptr);
    return ptr;
}
}// namespace luisa::compute::graph