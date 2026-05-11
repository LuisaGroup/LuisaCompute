#include "src/tests/ut/ut.hpp"
#include <iostream>

using namespace boost::ut;

static inline const auto reg = [] {
    "test_args"_test = [] {
        std::cout << "argc=" << boost::ut::detail::cfg::largc << std::endl;
        for (int i = 0; i < boost::ut::detail::cfg::largc; ++i) {
            std::cout << "argv[" << i << "]=" << (boost::ut::detail::cfg::largv[i] ? boost::ut::detail::cfg::largv[i] : "null") << std::endl;
        }
    };
    return 0;
}();

int main() {}
