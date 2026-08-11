#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace llvm {
class Function;
class Module;
}// namespace llvm

namespace luisa::compute::simd::schedule {
class Function;
}// namespace luisa::compute::simd::schedule

namespace luisa::compute::simd {

// Phase-2 packet ABI:
//   void entry(ptr argument_buffer, ptr return_lanes, i32 active_lane_count)
//
// Value arguments are packed in declaration order with their Luisa ABI size
// and alignment. Scalar returns are written as one contiguous value per lane.
// The ABI is deliberately narrow while resource handles and aggregate memory
// transposition are still being implemented.
struct LLVMScheduleCodegenResult {
    ::llvm::Function *entry{nullptr};
    size_t argument_buffer_size{0u};
    std::string error{};

    [[nodiscard]] bool succeeded() const noexcept {
        return entry != nullptr && error.empty();
    }
};

// Lowers an acyclic Schedule IR function to target-independent LLVM fixed
// vectors. No target ISA or hardware SIMD intrinsic is selected here; the
// LLVM target machine owns legalization, instruction selection, register
// allocation, and scheduling.
[[nodiscard]] LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name = {});

}// namespace luisa::compute::simd
