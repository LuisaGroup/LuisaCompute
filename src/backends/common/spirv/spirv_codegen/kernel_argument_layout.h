#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "../../argument_block_layout.h"

namespace luisa::compute::xir {
class Argument;
class KernelFunction;
}// namespace luisa::compute::xir

namespace lc::spirv {

struct SpirvKernelValueArgumentPlacement {
    const luisa::compute::xir::Argument *argument{nullptr};
    size_t argument_index{std::numeric_limits<size_t>::max()};
    uint32_t byte_offset{0u};
};

struct SpirvKernelArgumentLayoutPlan {
    luisa::vector<SpirvKernelValueArgumentPlacement> value_arguments;
    luisa::string diagnostic;
    size_t buffer_metadata_offset{0u};
    size_t buffer_metadata_count{0u};
    size_t final_size{0u};
    ArgumentBlockLayoutStatus status{
        ArgumentBlockLayoutStatus::SUCCESS};

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArgumentBlockLayoutStatus::SUCCESS;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// The entry-point ABI places every non-resource kernel argument in one byte-
// addressed storage buffer. If direct buffers are present, their metadata
// records follow at the runtime ABI's alignment. The complete block is then
// rounded to a uint32 word boundary. This is the single checked layout plan
// consumed by both dialect validation and emission.
[[nodiscard]] SpirvKernelArgumentLayoutPlan
plan_spirv_kernel_argument_layout(
    const luisa::compute::xir::KernelFunction *kernel) noexcept;

}// namespace lc::spirv
