// Test for compute context and backend enumeration.
// This test verifies that the compute context can properly discover and report
// available backends and their associated devices.

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "ut/ut.hpp"
#include "test_device.h"

void _luisa_reg_context(luisa::string backend) {

    boost::ut::detail::test{"test", "context"} = [backend = std::move(backend)] {
        auto argv = boost::ut::detail::cfg::largv;
        const char *exe = (argv && argv[0]) ? argv[0] : luisa::test::safe_argv0();
        luisa::compute::Context context{exe};
        auto installed = context.installed_backends();
        auto found = std::find(installed.begin(), installed.end(), backend) != installed.end();
        boost::ut::expect(found) << "requested backend is not installed";
        if (!found) { return; }
        auto device_names = context.backend_device_names(backend);
        boost::ut::expect(!device_names.empty()) << "requested backend has no devices";
        for (auto &device_name : device_names) {
            LUISA_INFO("Found device '{}' for backend '{}'.",
                       device_name, backend);
        }
    };
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>.", argc > 0 ? argv[0] : "test_context");
        return 1;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    _luisa_reg_context(argv[1]);
    return 0;
}
