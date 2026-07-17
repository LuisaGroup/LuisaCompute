#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/fuse_consecutive_buffer_reads.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

// Typed BUFFER_READ/BUFFER_WRITE operations must read/write exactly the element
// type declared by the buffer handle. In particular, replacing consecutive
// reads from buffer<T> with one Tn read changes both the resource operation's
// type and its index stride. The same problem applies to writes, and T3 may
// additionally contain ABI padding.
//
// Keep this pass as an explicit no-op until XIR has a verifier-backed lowering
// that can express a legal byte-addressed vector transaction while preserving
// alignment, bounds, aliasing, and volatile semantics on every backend.
static void run_on_function(FunctionDefinition *) noexcept {}

}// namespace detail

FuseConsecutiveBufferReadsInfo fuse_consecutive_buffer_reads_pass_run_on_function(
    Function *function) noexcept {
    FuseConsecutiveBufferReadsInfo info;
    if (auto def = function->definition()) {
        detail::run_on_function(def);
    }
    return info;
}

FuseConsecutiveBufferReadsInfo fuse_consecutive_buffer_reads_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    FuseConsecutiveBufferReadsInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            detail::run_on_function(def);
        }
    }
    if (report != nullptr) {
        report->set("fused_group_count", 0u);
        report->set("fused_read_count", 0u);
    }
    return info;
}

}// namespace luisa::compute::xir
