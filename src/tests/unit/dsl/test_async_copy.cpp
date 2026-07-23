// Runtime capability probe for CallOp::ASYNC_COPY.
//
// Vulkan backend: async_copy, pipeline_commit, and pipeline_wait_prior are
// implemented via the HLSL-to-SPIR-V fallback path. The HLSL codegen emits
// per-thread copies from the first StructuredBuffer argument to a groupshared
// scratch buffer (_vk_wg_copy_buf). pipeline_wait_prior uses a workgroup
// barrier (GroupMemoryBarrierWithGroupSync) for synchronization.
//
// CUDA backend: async_copy uses lc_pipeline_memcpy_async (cp.async PTX),
// pipeline_commit uses lc_pipeline_commit, and pipeline_wait_prior uses
// lc_pipeline_wait_prior.
//
// Known limitation: the AST API passes uint byte-offsets for dst/src, but
// both CUDA (void*) and SPIR-V (typed pointers) require actual memory
// pointers. Full runtime verification is pending API improvements.
//
// CUDA async copy execution is tested in test_async_copy_cuda.cpp.

#include "ut/ut.hpp"
#include "test_device.h"

#include <vector>

#include <luisa/runtime/device.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    // argv[1] selects the runtime backend; it is not a Boost.UT name filter.
    std::vector<const char *> ut_args;
    ut_args.reserve(static_cast<size_t>(argc - 1));
    ut_args.emplace_back(argv[0]);
    for (auto i = 2; i < argc; i++) {
        ut_args.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        static_cast<int>(ut_args.size()), ut_args.data());
    auto &device = dc->device;
    LUISA_INFO(
        "Skipping ASYNC_COPY device execution on backend '{}': the current AST API models "
        "events as uint32 values and cannot expose a CrossWorkgroup buffer pointer, while "
        "OpUntypedGroupAsyncCopyKHR requires OpTypeEvent and opposite Workgroup/CrossWorkgroup "
        "untyped pointers. The former shared-to-shared, single-invocation test was outside the "
        "SPIR-V operation's semantics.",
        device.backend_name());
    skip / "async_copy_device_execution_requires_event_and_pointer_types"_test = [] {};
}
