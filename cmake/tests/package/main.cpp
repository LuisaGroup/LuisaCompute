#include <luisa/luisa-compute.h>
#if LUISA_PACKAGE_WITH_RUST
#include <luisa/rust/ir.hpp>
#endif

#include <cstdio>

int main(int argc, char *argv[]) {
    static_assert(LUISA_COMPUTE_VERSION_MAJOR == LUISA_PACKAGE_VERSION_MAJOR);
    static_assert(LUISA_COMPUTE_VERSION_MINOR == LUISA_PACKAGE_VERSION_MINOR);
    static_assert(LUISA_COMPUTE_VERSION_PATCH == LUISA_PACKAGE_VERSION_PATCH);

    luisa::compute::Context context{argc > 0 ? argv[0] : ""};
    auto installed = context.installed_backends();
    if (installed.size() != LUISA_PACKAGE_EXPECTED_BACKEND_COUNT) {
        std::fprintf(stderr,
                     "Expected %d installed backends, found %zu.\n",
                     LUISA_PACKAGE_EXPECTED_BACKEND_COUNT,
                     installed.size());
        return 1;
    }

    constexpr luisa::string_view expected_backend{
        LUISA_PACKAGE_EXPECTED_BACKEND};
    if (!expected_backend.empty()) {
        auto found = false;
        for (auto &&backend : installed) {
            if (luisa::string_view{backend} == expected_backend) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr,
                         "Expected backend '%.*s' was not deployed.\n",
                         static_cast<int>(expected_backend.size()),
                         expected_backend.data());
            return 2;
        }
        static_cast<void>(context.load_backend(expected_backend));
    }

#if LUISA_PACKAGE_EXPECT_DXC
    auto dxc = luisa::DynamicModule::load(context.runtime_directory(),
                                          "dxcompiler");
    if (!dxc || dxc.address("DxcCreateInstance") == nullptr) {
        std::fprintf(stderr, "The deployed DXC runtime could not be loaded.\n");
        return 4;
    }
#endif

#if LUISA_PACKAGE_WITH_RUST
    if (luisa::compute::ir::luisa_compute_ir_new_module_pools() == nullptr) {
        std::fprintf(stderr, "The installed Rust IR API returned a null pool.\n");
        return 3;
    }
#endif

    luisa::log_level_info();
    luisa::log_flush();
    return 0;
}
