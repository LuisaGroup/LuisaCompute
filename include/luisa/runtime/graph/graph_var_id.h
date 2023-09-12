#pragma once
#include <luisa/runtime/graph/id_with_type.h>
namespace luisa::compute::graph {
class VarId : public IdWithType {
public:
    using IdWithType::IdWithType;
};
class GraphInputVarId : public VarId {
public:
    using VarId::VarId;
};
class GraphSubVarId : public VarId {
public:
    using VarId::VarId;
};
}// namespace luisa::compute::graph