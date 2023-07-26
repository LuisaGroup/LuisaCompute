#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "config.h"

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

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

static luisa::vector<const char *> _backends;
int supported_backends_count() noexcept { return _backends.size(); }
const char *const *supported_backends() noexcept {
    return _backends.data();
}

}// namespace luisa::test

int main(int argc, const char **argv) {
    doctest::Context context(argc, argv);
    // TODO: read from config file
    luisa::test::_backends = {"dx", "cuda"};
    luisa::test::dt_remove(argv);
    auto test_result = context.run();
    if (context.shouldExit()) { return test_result; }
    return test_result;
}
