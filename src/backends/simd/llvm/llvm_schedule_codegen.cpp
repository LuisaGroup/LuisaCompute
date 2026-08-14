#include "llvm_schedule_codegen.h"
#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd {

LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name,
    bool enable_fast_math,
    std::array<uint32_t, 3u> static_block_size,
    bool enable_uniform_buffer_broadcast,
    bool enable_lane_affine_buffer,
    bool enable_paired_leaf_gather) {
    return detail::ScheduleEmitter{
        module, function, specialization_width, entry_name,
        enable_fast_math, static_block_size,
        enable_uniform_buffer_broadcast,
        enable_lane_affine_buffer,
        enable_paired_leaf_gather}
        .run();
}

}// namespace luisa::compute::simd
