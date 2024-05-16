/**
 * @file test_soa.cpp
 * @brief The SOA(Struct of Array) test suite
 * @author sailing-innocent
 * @date 2024-05-16
 */

#include "common/config.h"
#include "luisa/luisa-compute.h"

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test::soa {

struct D {
    float3x3 m;
    float2x2 n;
};

struct A {
    float3 a;
    bool2 b;
    bool c;
    D d;
    std::array<std::array<int4, 1>, 1> e;
};

struct LightDistributionTreeNode {
    unsigned int left;
    unsigned int right;
    float left_distribution;
    float right_distribution;
};

struct LightDistributionCell {
    std::array<std::array<LightDistributionTreeNode, 256>, 2> nodes;
};

}// namespace luisa::test::soa

namespace luisa::test {

using namespace luisa::test::soa;

int test_soa(Device &device) {
    // TODO: implement
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    using namespace luisa::test;
    LUISA_TEST_CASE_WITH_DEVICE("soa", test_soa(device) == 0);
}