#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/instructions/debug_break.h>

namespace llvm {
class Module;
class LLVMContext;
}// namespace llvm

namespace luisa::compute::xir {
class PrintInst;
class Module;
}// namespace luisa::compute::xir

namespace luisa::compute::fallback {

constexpr auto max_thread_frame_size = 4_M;
constexpr auto max_shared_memory_size = 1_M;

struct FallbackCodeGenFeedback {

    using PrintInstMap = luisa::vector<std::pair<
        const xir::PrintInst *,
        luisa::string /* llvm symbol */>>;
    PrintInstMap print_inst_map;

    using DebugCallbackMap = luisa::vector<std::pair<
        xir::DebugBreakInst::Callback,
        luisa::string /* llvm symbol */>>;
    DebugCallbackMap debug_callback_map;

};

[[nodiscard]] FallbackCodeGenFeedback
luisa_fallback_backend_codegen(llvm::LLVMContext &llvm_ctx,
                               llvm::Module *llvm_module,
                               const xir::Module *module) noexcept;

}// namespace luisa::compute::fallback
