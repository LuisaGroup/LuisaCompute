#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "common/config.h"

TEST_CASE("context") {
    luisa::compute::Context context{luisa::test::argv()[0]};
    for (auto &&backend : context.installed_backends()) {
        auto device_names = context.backend_device_names(backend);
        REQUIRE(!device_names.empty());
        for (auto &device_name : device_names) {
            LUISA_INFO("Found device '{}' for backend '{}'.",
                       device_name, backend);
        }
    }
}

