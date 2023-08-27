#pragma once
#include <luisa/runtime/buffer.h>
#include <luisa/core/basic_traits.h>
namespace luisa::compute::graph {

enum class GraphResourceTag {
    Basic,
    Buffer,
};

class GraphArgId {
public:
    explicit GraphArgId(uint64_t value) noexcept : _value{value} {}
    uint64_t value() const noexcept { return _value; }
private:
    uint64_t _value;
};

enum class GraphVarReadWriteTag : uint8_t {
    ReadOnly = 1,
    ReadWrite = 2
};

class GraphVarBase {
public:
    static constexpr uint64_t invalid_id = std::numeric_limits<uint64_t>::max();
    // when we analyse a graph, all graph vars are virtual(not a real resource view)
    GraphVarBase(GraphArgId id, GraphResourceTag tag) noexcept
        : _is_virtual{id.value() != invalid_id}, _arg_id{id}, _tag{tag} {}

    bool is_virtal() const noexcept { return _is_virtual; }
    uint64_t arg_id() const noexcept { return _arg_id.value(); }
    GraphResourceTag tag() const noexcept { return _tag; }
private:
    bool _is_virtual{true};
    GraphArgId _arg_id{invalid_id};
    GraphResourceTag _tag;
protected:
};

template<typename T>
class GraphVar : public GraphVarBase {
public:
    using value_type = T;
    GraphVar(const T &value) noexcept : GraphVarBase{invalid_id}, _value{value} {}
    GraphVar(GraphArgId id) noexcept : GraphVarBase{id, Resource::Tag::Basic} {}
private:
    T _value{};
};

template<typename T>
class GraphVar<BufferView<T>> : public GraphVarBase {
public:
    using value_type = T;
    using GraphVarBase::GraphVarBase;
    GraphVar(const BufferView<T> &view) noexcept : GraphVarBase{invalid_id}, _view{view} {}
    GraphVar(GraphArgId id) noexcept : GraphVarBase{id, GraphResourceTag::Buffer} {}
    // operator BufferView<T>() const noexcept { return _view; }
private:
    BufferView<T> _view{};
};

template<typename T>
using GraphBuffer = GraphVar<BufferView<T>>;

template<typename T>
struct is_graph_var : std::false_type {};

template<typename T>
struct is_graph_var<GraphVar<T>> : std::true_type {};

template<typename T>
constexpr bool is_graph_var_v = is_graph_var<T>::value;

namespace detail {
template<typename T>
struct view_to_graph_shader_invocation {
    using type = graph::GraphVar<T>;
};

template<typename T>
using view_to_graph_shader_invocation_t = typename view_to_graph_shader_invocation<T>::type;
}// namespace detail
}// namespace luisa::compute::graph
