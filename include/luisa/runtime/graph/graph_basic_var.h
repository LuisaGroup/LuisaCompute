#pragma once
#include <luisa/runtime/graph/graph_var.h>

namespace luisa::compute::graph {
class GraphBuilder;

class GraphBasicVarBase : public GraphVarBase {
public:
    using GraphVarBase::GraphVarBase;
    virtual U<GraphBasicVarBase> clone() const noexcept = 0;
protected:
    virtual void sub_var_update_check(GraphBuilder *builder) noexcept override {}// do nothing
};

template<typename T>
class GraphVar final : public GraphBasicVarBase {
    friend class GraphBuilder;
    template<typename... Args>
    friend class Graph;
public:
    using value_type = T;
    // GraphVar(const T &value) noexcept : GraphVarBase{invalid_id, GraphResourceTag::Basic}, _value{value} {}
    GraphVar(GraphInputVarId id) noexcept : GraphBasicVarBase{id, GraphResourceTag::Basic} {}
    //const T &value() const noexcept { return _value; }
    //const T &view() const noexcept { return _value; }
    T eval() const noexcept { return _value; }
    virtual U<GraphBasicVarBase> clone() const noexcept override { return luisa::make_unique<GraphVar<T>>(*this); }
private:
    template<typename T>
    void input_var_update_check(const T &new_value) noexcept {
        LUISA_ASSERT(this->is_input_var(), "this function can only be invoked when it is a sub var");
        _need_update = _need_update || _value != new_value;
        if (_need_update) {
            LUISA_INFO("input_var {} need update: new value = {}", this->input_var_id().value(), new_value);
            _value = new_value;
        }
    }

    virtual void update_kernel_node_cmd_encoder(
        size_t arg_idx_in_kernel_parms, KernelNodeCmdEncoder *encoder) const noexcept override {
        encoder->update_uniform(arg_idx_in_kernel_parms, &_value);
    }
    T _value{};
};

using GraphUInt = GraphVar<uint32_t>;
using GraphInt = GraphVar<int>;
using GraphFloat = GraphVar<float>;
}// namespace luisa::compute::graph