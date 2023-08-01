/**
 * @file: tests/next/runtime/test_context.cpp
 * @author: sailing-innocent
 * @date: 2023-08-01
 * @brief: the first test case, testing if the context is working
*/

#include "common/config.h"
#include <luisa/runtime/context.h>


TEST_SUITE("runtime") {
    TEST_CASE("context") {
        luisa::compute::Context context{luisa::test::argv()[0]};
        // require the testing context are all installed and working
        for (auto &&backend : context.installed_backends()) {
            auto device_names = context.backend_device_names(backend);
            REQUIRE_MESSAGE(!device_names.empty(), "the backend ", backend, " has no device installed");
        }
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string backend_to_test = luisa::test::backends_to_test()[i];
            bool installed = false;
            for (auto &&backend : context.installed_backends()) {
                if (backend == backend_to_test) {
                    installed = true;
                    break;
                }
            }
            CHECK_MESSAGE(installed, "the testing backend ", backend_to_test, " is not installed");
        }
    }
}
