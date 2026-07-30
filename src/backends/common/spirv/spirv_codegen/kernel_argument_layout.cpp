#include "kernel_argument_layout.h"

#include "../../storage_buffer_metadata.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/function.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] bool is_indirect_dispatch_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() == "LC_IndirectDispatchBuffer";
}

[[nodiscard]] bool is_resource_argument(
    const xir::Argument *argument) noexcept {
    return argument != nullptr &&
           (argument->is_resource() ||
            is_indirect_dispatch_type(argument->type()));
}

}// namespace

SpirvKernelArgumentLayoutPlan
plan_spirv_kernel_argument_layout(
    const xir::KernelFunction *kernel) noexcept {
    SpirvKernelArgumentLayoutPlan plan;
    if (kernel == nullptr) {
        plan.status = ArgumentBlockLayoutStatus::INVALID_ALIGNMENT;
        plan.diagnostic =
            "Native XIR-to-SPIR-V kernel argument layout requires a non-null kernel definition.";
        return plan;
    }

    ArgumentBlockLayout layout{
        std::numeric_limits<uint32_t>::max()};
    auto &arguments = kernel->arguments();
    plan.value_arguments.reserve(arguments.count_size());
    auto argument_index = size_t{0u};
    for (auto *argument : arguments) {
        if (is_resource_argument(argument)) {
            plan.buffer_metadata_count +=
                argument != nullptr && argument->type() != nullptr &&
                argument->type()->is_buffer();
            argument_index++;
            continue;
        }
        auto *type = argument == nullptr ? nullptr : argument->type();
        if (type == nullptr) {
            plan.status = ArgumentBlockLayoutStatus::INVALID_ALIGNMENT;
            plan.diagnostic = luisa::format(
                "Native XIR-to-SPIR-V kernel argument {} has no type and cannot be placed in the argument block.",
                argument_index);
            return plan;
        }
        size_t byte_offset = 0u;
        if (!layout.append(type->size(), type->alignment(),
                           byte_offset)) {
            plan.status = layout.status();
            plan.diagnostic = luisa::format(
                "Native XIR-to-SPIR-V kernel argument {} of type {} (size {}, alignment {}) cannot be placed in the uint32 byte-offset argument block: {}.",
                argument_index, type->description(), type->size(),
                type->alignment(),
                argument_block_layout_status_name(plan.status));
            return plan;
        }
        plan.value_arguments.emplace_back(
            SpirvKernelValueArgumentPlacement{
                .argument = argument,
                .argument_index = argument_index,
                .byte_offset = static_cast<uint32_t>(byte_offset)});
        argument_index++;
    }
    ArgumentBlockTrailerPlacement trailer;
    if (!layout.append_trailers(
            {.metadata_count = plan.buffer_metadata_count,
             .metadata_stride = sizeof(StorageBufferMetadata),
             .metadata_alignment = alignof(StorageBufferMetadata),
             .word_alignment = sizeof(uint32_t)},
            trailer)) {
        plan.status = layout.status();
        plan.diagnostic = plan.buffer_metadata_count == 0u ?
                              luisa::format(
                                  "Native XIR-to-SPIR-V kernel argument block cannot finalize to the runtime uint32 word boundary: {}.",
                                  argument_block_layout_status_name(plan.status)) :
                              luisa::format(
                                  "Native XIR-to-SPIR-V kernel argument block cannot append {} direct-buffer metadata record(s) and finalize within the uint32 byte-offset domain: {}.",
                                  plan.buffer_metadata_count,
                                  argument_block_layout_status_name(plan.status));
        return plan;
    }
    plan.buffer_metadata_offset = trailer.metadata_offset;
    plan.final_size = trailer.final_size;
    return plan;
}

}// namespace lc::spirv
