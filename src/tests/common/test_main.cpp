#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "config.h"

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/logging.h>

// #include <iostream>

namespace luisa::test {

static luisa::vector<const char *> args;

int argc() noexcept { return static_cast<int>(args.size()); }
const char *const *argv() noexcept { return args.data(); }

static luisa::vector<const char *> _backends;
int backends_to_test_count() noexcept { return _backends.size(); }
const char *const *backends_to_test() noexcept {
    return _backends.data();
}

inline void args_filter(const char **argv_in) noexcept {
    args.clear();
    bool default_backend = true;
    for (; *argv_in; ++argv_in) {
        if (!luisa::string_view{*argv_in}.starts_with("--backend-")) {
            args.emplace_back(*argv_in);
        } else {
            // add to backend
            default_backend = false;
            _backends.emplace_back(*argv_in + 10);
        }
    }
    if (default_backend) {
        // default testing case
        _backends.emplace_back("dx");
        _backends.emplace_back("cuda");
    }
    args.emplace_back(nullptr);
}

}// namespace luisa::test

int main(int argc, const char **argv) {
    doctest::Context context(argc, argv);
    luisa::test::args_filter(argv);
    auto test_result = context.run();
    if (context.shouldExit()) { return test_result; }
    return test_result;
}
