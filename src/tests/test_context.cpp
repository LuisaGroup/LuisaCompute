//
// Created by Mike Smith on 2023/4/5.
//

#include <tests/common/config.h>
#include <runtime/context.h>

TEST_CASE("test_context") {
    REQUIRE_GE(luisa::test::argc(), 1);
    luisa::compute::Context context{luisa::test::argv()[0]};
}
