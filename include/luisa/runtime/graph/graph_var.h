#pragma once
#include <luisa/runtime/buffer.h>
#include <luisa/core/basic_traits.h>
#include <luisa/vstl/unique_ptr.h>
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
protected:
    template<typename T>
    using U = luisa::unique_ptr<T>;
    friend class GraphBuilder;
    bool _need_update{false};
public:
    static constexpr uint64_t invalid_id = std::numeric_limits<uint64_t>::max();
    // when we analyse a graph, all graph vars are virtual(not a real resource view)
    GraphVarBase(GraphArgId id, GraphResourceTag tag) noexcept
        : _is_virtual{id.value() != invalid_id}, _arg_id{id}, _tag{tag} {}

    bool is_virtual() const noexcept { return _is_virtual; }
    bool need_update() const noexcept { return _need_update; }

    uint64_t arg_id() const noexcept { return _arg_id.value(); }
    void clear_need_update_flag() noexcept { _need_update = false;}

    GraphResourceTag tag() const noexcept { return _tag; }
    virtual U<GraphVarBase> clone() const noexcept = 0;
    template<typename T>
    T *cast() noexcept {
        static_assert(std::is_base_of_v<GraphVarBase, T>);
        return dynamic_cast<T *>(this);
    }
private:
    bool _is_virtual{true};
    GraphArgId _arg_id{invalid_id};
    GraphResourceTag _tag;
};

template<typename T>
class GraphVar final : public GraphVarBase {
public:
    using value_type = T;
    // GraphVar(const T &value) noexcept : GraphVarBase{invalid_id, GraphResourceTag::Basic}, _value{value} {}
    GraphVar(GraphArgId id) noexcept : GraphVarBase{id, GraphResourceTag::Basic} {}
    U<GraphVarBase> clone() const noexcept override { return make_unique<GraphVar<T>>(*this); }
    void update_check(const T &new_value) noexcept {
        _need_update || = _value != new_value;
        if (_need_update) {
            LUISA_INFO("arg {} need update: new value = {}", this->arg_id(), new_value);
            _value = new_value;
        }
    }
private:
    T _value{};
};

template<typename T>
class GraphVar<BufferView<T>> final : public GraphVarBase {
public:
    using value_type = T;
    using GraphVarBase::GraphVarBase;
    //GraphVar(const BufferView<T> &view) noexcept
    //    : GraphVar(GraphArgId{invalid_id}),
    //      _view{view} {}
    GraphVar(GraphArgId id) noexcept : GraphVarBase{id, GraphResourceTag::Buffer} {}

    U<GraphVarBase> clone() const noexcept override { return make_unique<GraphVar<BufferView<T>>>(*this); }

    void update_check(const BufferView<T> &new_view) noexcept {
        _need_update || = !is_same_view(new_view);
        if (_need_update) {
            LUISA_INFO("arg {} need update: new buffer view handle = {}", this->arg_id(), new_view.handle());
            _view = new_view;
        }
    }
private:
    BufferView<T> _view{};
    bool is_same_view(const BufferView<T> &view) const noexcept {
        return _view.handle() == view.handle() && _view.native_handle() == view.native_handle() && _view.offset() == view.offset() && _view.size() == view.size();
    }
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
