// Test for compute context and backend enumeration.
// This test verifies that the compute context can properly discover and report
// available backends and their associated devices.

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "ut/ut.hpp"
#include "test_device.h"

void _luisa_reg_context() {

    boost::ut::detail::test{"test", "context"} = [] {
        auto argv = boost::ut::detail::cfg::largv;
        const char *exe = (argv && argv[0]) ? argv[0] : luisa::test::safe_argv0();
        // Initialize context with the test executable path
        luisa::compute::Context context{exe};

        // Iterate over all installed backends
        for (auto &&backend : context.installed_backends()) {
            // Get list of device names for this backend
            auto device_names = context.backend_device_names(backend);

            // Verify that at least one device is available
            boost::ut::expect(static_cast<bool>(!device_names.empty()));

            // Log each discovered device
            for (auto &device_name : device_names) {
                LUISA_INFO("Found device '{}' for backend '{}'.",
                           device_name, backend);
            }
        }
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    _luisa_reg_context();
    return 0;
}
