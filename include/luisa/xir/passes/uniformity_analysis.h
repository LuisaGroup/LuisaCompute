#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/function.h>
#include <luisa/xir/value.h>

namespace luisa::compute::xir {

// Uniformity analysis for SPIR-V descriptor indexing.
//
// Proves whether a value is "dynamically uniform across the invocation group
// (workgroup)" in the SPIR-V/Vulkan sense. The query is a one-way proof:
//   - is_uniform(v) == true   => v is provably workgroup-uniform
//   - is_uniform(v) == false  => unknown/non-uniform; caller must apply NonUniformEXT
//
// This is target-independent but designed specifically as input to the SPIR-V
// backend's NonUniformEXT decoration decision. A false "uniform" classification
// is a miscompile, so propagation is intentionally narrow and conservative.
//
// Intraprocedural for v1: callable arguments and CallInst results are non-uniform.
class LUISA_XIR_API UniformityAnalysis {
public:
    UniformityAnalysis() noexcept = default;
    void analyze(const Function *function) noexcept;
    [[nodiscard]] bool is_uniform(const Value *value) const noexcept;
    void clear() noexcept;

private:
    luisa::unordered_map<const Value *, bool> _uniform;
    const Function *_function{nullptr};
};

}// namespace luisa::compute::xir
