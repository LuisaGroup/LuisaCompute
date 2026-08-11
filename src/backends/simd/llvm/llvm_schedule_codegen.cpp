#include "llvm_schedule_codegen.h"
#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd {

LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name) {
    return detail::ScheduleEmitter{
        module, function, specialization_width, entry_name}
        .run();
}

}// namespace luisa::compute::simd
