/**
 * @file: tests/next/example/gallary/render/procedural.cpp
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: the basic pcg example
*/

#include "common/config.h"
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int procedural(Device &device) {
    return 0;
}

}// namespace luisa::test

TEST_SUITE("gallery") {
    using namespace luisa::test;
    LUISA_TEST_CASE_WITH_DEVICE("procedural", procedural(device) == 0);
}
