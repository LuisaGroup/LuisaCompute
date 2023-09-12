#pragma once
#include <luisa/runtime/graph/graph_var.h>

namespace luisa::compute::graph {
template<>
class GraphVar<void *> final : public GraphVarBase {
    friend class GraphBuilder;
    template<typename... Args>
    friend class Graph;
public:
    using value_type = void *;
    using GraphVarBase::GraphVarBase;
    GraphVar(GraphInputVarId id) noexcept : GraphVarBase{id, GraphResourceTag::HostMemory} {}
    auto eval() const noexcept { return _value; }

    U<GraphVar<void *>> clone() noexcept { return make_unique<GraphVar<void *>>(*this); };
protected:
    virtual void update_kernel_node_cmd_encoder(
        size_t arg_idx_in_kernel_parms, KernelNodeCmdEncoder *encoder) const noexcept override{};

    virtual void sub_var_update_check(GraphBuilder *builder) noexcept override{};

    void input_var_update_check(void *value) {
        LUISA_ASSERT(this->is_input_var(), "this function can only be invoked when it is a input var");
        _need_update = _need_update || _value != value;
        if (_need_update) {
            LUISA_INFO("input_var {} need update: new host memory pointer = {}", this->input_var_id().value(), value);
            _value = value;
        }
    }
private:
    void *_value = nullptr;
};
}// namespace luisa::compute::graph