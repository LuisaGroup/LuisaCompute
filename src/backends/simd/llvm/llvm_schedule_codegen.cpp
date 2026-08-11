#include "llvm_schedule_codegen.h"
#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd {

LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name,
    bool enable_fast_math) {
    return detail::ScheduleEmitter{
        module, function, specialization_width, entry_name,
        enable_fast_math}
        .run();
}

}// namespace luisa::compute::simd
