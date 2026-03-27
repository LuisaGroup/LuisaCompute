#pragma once
#include "work_graph_node.h"

#include <luisa/core/stl/vector.h>

namespace luisa::compute {

// type-erased WorkGraphNode
struct WorkGraphElaboratedNode {
    luisa::string name;
    luisa::shared_ptr<const detail::FunctionBuilder> builder;
    WorkGraphLaunchType launch_type;
    bool input_record_empty;

    template <typename Record>
    bool input_is() const noexcept {
        if constexpr (std::is_empty_v<Record>) {
            return input_record_empty;
        }
        if (input_record_empty) {
            return false;
        }

        auto *first_arg_ty = builder->arguments().front().type();
        return *first_arg_ty == Type::of<Record>();
    }
};

class WorkGraphDescription {

};

class WorkGraphBuilder {
public:
    using NodeRef = uint32_t;

    template<typename Record, typename T>
    NodeRef add_node(luisa::string_view name, const WorkGraphNode<Record, T>& node) noexcept {

    }

    template<typename Record>
    void add_edge(NodeRef from, NodeRef to) noexcept {
        bool input_record_empty;
    }

    WorkGraphDescription build() noexcept;

private:
    luisa::vector<luisa::vector<NodeRef>> _out_edges;
    luisa::vector<WorkGraphElaboratedNode> _nodes;
};

} // namespace luisa::compute