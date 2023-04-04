//
// Created by Mike Smith on 2023/4/5.
//

#define TINYOBJLOADER_IMPLEMENTATION
#include <tests/common/tiny_obj_loader.h>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include <core/stl/string.h>
#include <core/stl/vector.h>
#include <tests/common/config.h>

namespace luisa::test {

static luisa::vector<const char *> args;

inline void dt_remove(const char **argv_in) noexcept {
    args.clear();
    for (; *argv_in; ++argv_in) {
        if (!luisa::string_view{*argv_in}.starts_with("--dt-")) {
            args.emplace_back(*argv_in);
        }
    }
    args.emplace_back(nullptr);
}

int argc() noexcept { return static_cast<int>(args.size()); }
const char *const *argv() noexcept { return args.data(); }

}// namespace luisa::test

int main(int argc, const char **argv) {
    doctest::Context context(argc, argv);
    luisa::test::dt_remove(argv);
    auto test_result = context.run();
    if (context.shouldExit()) { return test_result; }
    return test_result;
}
