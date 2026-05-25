#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class AllocaInst;
class BasicBlock;
class Function;
class FunctionDefinition;
class Instruction;
class Module;

struct CoroutineMarkerInfo {
    Instruction *inst{};
    BasicBlock *block{};
    size_t id{};
};

struct CoroutineContinuationInfo {
    size_t id{};
    BasicBlock *entry_block{};
    Instruction *entry_inst{};
    luisa::vector<BasicBlock *> blocks;
    luisa::vector<size_t> suspend_ids;
};

struct CoroutineTransitionInfo {
    size_t from_continuation{};
    size_t suspend_id{};
    size_t to_continuation{};
    bool exits{};
};

struct CoroutineFrameCandidateInfo {
    AllocaInst *alloca{};
    luisa::vector<size_t> live_across_suspend_ids;
};

struct CoroutineAnalysisInfo {
    bool is_coroutine{};
    luisa::vector<CoroutineMarkerInfo> registers;
    luisa::vector<CoroutineMarkerInfo> suspends;
    luisa::vector<CoroutineContinuationInfo> continuations;
    luisa::vector<CoroutineTransitionInfo> transitions;
    luisa::vector<CoroutineFrameCandidateInfo> frame_candidates;
    luisa::vector<luisa::string> diagnostics;
};

struct CoroutineLowerInfo {
    bool changed{};
    size_t removed_register_count{};
    size_t removed_suspend_count{};
    size_t created_state_alloca_count{};
    size_t created_frame_alloca_count{};
    size_t created_switch_count{};
    luisa::vector<luisa::string> diagnostics;
};

[[nodiscard]] LUISA_XIR_API CoroutineAnalysisInfo coroutine_analysis_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API CoroutineAnalysisInfo coroutine_analysis_run_on_module(Module *module) noexcept;
[[nodiscard]] LUISA_XIR_API CoroutineLowerInfo coroutine_lower_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API CoroutineLowerInfo coroutine_lower_run_on_module(Module *module) noexcept;

}
