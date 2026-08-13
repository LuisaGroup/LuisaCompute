#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>

namespace luisa::compute::xir {

class Value;
class BasicBlock;
class FunctionDefinition;
class Module;
class PassReport;

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

    explicit PointerUsage(const Type *type) noexcept
        : kill{type}, touch{type}, live{type} {}
};

using PointerUsageMap = luisa::unordered_map<Value *, luisa::unique_ptr<PointerUsage>>;

struct BasicBlockPointerUsage {
    PointerUsageMap in;
    PointerUsageMap out;
};

struct PointerUsageAnalysisInfo {
    size_t tracked_pointer_count{0u};
    size_t materialized_pointer_count{0u};
    size_t analyzed_block_count{0u};
    size_t conservative_access_count{0u};
    size_t invalid_access_count{0u};
    size_t invalid_function_count{0u};

    [[nodiscard]] bool succeeded() const noexcept {
        return invalid_access_count == 0u && invalid_function_count == 0u;
    }
};

class LUISA_XIR_API PointerUsageAnalysis {
private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    PointerUsageAnalysis() noexcept;
    ~PointerUsageAnalysis() noexcept;
    PointerUsageAnalysis(PointerUsageAnalysis &&) noexcept;
    PointerUsageAnalysis &operator=(PointerUsageAnalysis &&) noexcept;
    PointerUsageAnalysis(const PointerUsageAnalysis &) = delete;
    PointerUsageAnalysis &operator=(const PointerUsageAnalysis &) = delete;

    void clear() noexcept;
    [[nodiscard]] PointerUsageAnalysisInfo analyze(FunctionDefinition *function) noexcept;
    // Solve only the requested pointer-view coordinates. Pointer discovery
    // and access validation still cover the complete function, so this is an
    // exact projection of the full product lattice rather than a local-use
    // approximation. Queries for unrequested pointers return nullptr.
    [[nodiscard]] PointerUsageAnalysisInfo analyze(
        FunctionDefinition *function,
        luisa::span<Value *const> result_pointers) noexcept;
    [[nodiscard]] bool is_current() const noexcept;
    [[nodiscard]] FunctionDefinition *function() const noexcept;
    // Unchecked access for a caller that has established is_current() once
    // and performs no IR mutation across a batch of queries.
    [[nodiscard]] const BasicBlockPointerUsage *
    current_block_usage(BasicBlock *block) const noexcept;
    [[nodiscard]] const BasicBlockPointerUsage *block_usage(BasicBlock *block) const noexcept;
    [[nodiscard]] const PointerUsage *in_usage(BasicBlock *block, Value *pointer) const noexcept;
    [[nodiscard]] const PointerUsage *out_usage(BasicBlock *block, Value *pointer) const noexcept;
};

// Null inputs and malformed bodyless kernels fail with
// invalid_function_count. A bodyless callable is an external declaration and
// produces a successful empty analysis.
[[nodiscard]] LUISA_XIR_API PointerUsageAnalysisInfo pointer_usage_pass_run_on_function(
    FunctionDefinition *function, PointerUsageAnalysis *analysis = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API PointerUsageAnalysisInfo pointer_usage_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
