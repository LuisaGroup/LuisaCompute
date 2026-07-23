#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class PassReport;
class FunctionDefinition;
class Module;

enum class AliasResult : uint8_t {
    NoAlias,
    MayAlias,
    MustAlias,
};

struct AliasAnalysisInfo {
    size_t queried_count{0u};
};

// The query is deliberately stateless so operand rewrites cannot leave stale
// cached alias facts. These entry points are retained for pipeline/report API
// compatibility.
[[nodiscard]] LUISA_XIR_API AliasAnalysisInfo alias_analysis_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API AliasAnalysisInfo alias_analysis_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

// Query: do two memory-accessing instructions alias?
[[nodiscard]] LUISA_XIR_API AliasResult alias_analysis_query(Instruction *a, Instruction *b) noexcept;

} // namespace luisa::compute::xir
