#pragma once

#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::spirv {

struct SpirvInactivePayloadCleanupInfo {
    size_t cleared_block_count{0u};
    size_t cleared_true_orphan_block_count{0u};
    size_t cleared_disconnected_role_block_count{0u};
    size_t removed_instruction_count{0u};
    size_t removed_phi_incoming_count{0u};
};

// Clears instructions only from blocks outside ordinary all-edge reachability.
// This includes two deliberately distinct categories: true orphans outside the
// emission closure, and dead payload in disconnected raw role blocks whose
// identity remains required by that closure. It never folds conditions, erases
// block identities, or rewrites ordinary-live CFG edges.
[[nodiscard]] SpirvInactivePayloadCleanupInfo
clear_spirv_codegen_inactive_block_payloads(xir::Module *module) noexcept;

// The mandatory SSA recovery boundary immediately after restructure_cfg.
// Every pass in this pipeline must preserve structured terminators and all
// role blocks. In particular, generic DCE/CFG simplification does not belong
// here because it may change a loop's prepare form or erase a structured role
// arm.
[[nodiscard]] xir::PassPipeline
create_spirv_codegen_post_restructure_pipeline() noexcept;

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept -> luisa::unique_ptr<xir::Module>;

}// namespace luisa::compute::spirv
