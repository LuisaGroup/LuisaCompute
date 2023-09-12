#pragma once
#include <luisa/runtime/graph/id_with_type.h>

namespace luisa::compute::graph {
class GraphNodeId : public IdWithType {
public:
    using IdWithType::IdWithType;
};
}// namespace luisa::compute::graph