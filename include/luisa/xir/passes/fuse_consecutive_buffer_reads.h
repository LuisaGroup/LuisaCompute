#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct FuseConsecutiveBufferReadsInfo {
    size_t fused_group_count{0u};
    size_t fused_read_count{0u};
};

[[nodiscard]] LUISA_XIR_API FuseConsecutiveBufferReadsInfo
fuse_consecutive_buffer_reads_pass_run_on_function(Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API FuseConsecutiveBufferReadsInfo
fuse_consecutive_buffer_reads_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
