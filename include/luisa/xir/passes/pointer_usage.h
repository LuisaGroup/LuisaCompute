#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>

namespace luisa::compute::xir {

class Value;
class BasicBlock;

// This pass analyzes the usage of pointers in a function,
// including reference arguments, alloca's, and GEP's.
// It records whether each scalar field of each pointer is
// - Killed: the field is definitely written to;
// - Touched: the field is possibly written to; or
// - Live: the field might be read from in the future.

struct PointerUsage {
    AggregateFieldBitmask kill;
    AggregateFieldBitmask touch;
    AggregateFieldBitmask live;
};

using PointerUsageMap = luisa::unordered_map<Value *, luisa::unique_ptr<PointerUsage>>;

struct BasicBlockPointerUsage {
    PointerUsageMap in;
    PointerUsageMap out;
};

}// namespace luisa::compute::xir
