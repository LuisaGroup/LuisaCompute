// Runtime capability probe for CallOp::ASYNC_COPY.
// Device execution is intentionally skipped until the AST and XIR type
// systems can represent the event and pointer types required by SPIR-V.

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
