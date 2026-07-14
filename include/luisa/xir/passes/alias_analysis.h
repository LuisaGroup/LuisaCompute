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

// Run analysis (pre-computes base alloca for all LOCAL instructions)
[[nodiscard]] LUISA_XIR_API AliasAnalysisInfo alias_analysis_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API AliasAnalysisInfo alias_analysis_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

// Query: do two memory-accessing instructions alias?
[[nodiscard]] LUISA_XIR_API AliasResult alias_analysis_query(Instruction *a, Instruction *b) noexcept;

} // namespace luisa::compute::xir
