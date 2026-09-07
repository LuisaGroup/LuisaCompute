#pragma once

#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>

#include <ut/ut.hpp>

namespace luisa::test::coro_test {

struct Options {
    const char *exe;
    const char *backend;
};

[[nodiscard]] inline luisa::vector<const char *> &ut_arguments() noexcept {
    static luisa::vector<const char *> args;
    return args;
}

inline void prepare_ut_arguments(int argc, char *argv[]) noexcept {
    auto &ut_argv = ut_arguments();
    ut_argv.clear();
    ut_argv.reserve(argc > 0 ? static_cast<size_t>(argc) : 1u);
    ut_argv.emplace_back(argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : safe_argv0());
    for (auto i = 2; i < argc; i++) {
        ut_argv.emplace_back(argv[i]);
    }
}

[[nodiscard]] inline Options parse_options(int argc, char *argv[]) noexcept {
    auto exe = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : safe_argv0();
    if (argc <= 1 || argv[1] == nullptr) {
        LUISA_ERROR("Usage: {} <backend> [<test name|pattern|tags> ...] [Boost.UT options]. <backend>: cuda, dx, fallback, hip, metal, vk", exe);
    }
    prepare_ut_arguments(argc, argv);
    auto &ut_argv = ut_arguments();
    boost::ut::detail::cfg::parse(
        static_cast<int>(ut_argv.size()), ut_argv.data());
    return Options{exe, argv[1]};
}

[[nodiscard]] inline DeviceContext create_device(const Options &options) {
    compute::Context context{options.exe};
    compute::Device device = context.create_device(options.backend);
    return DeviceContext{std::move(context), std::move(device)};
}

[[nodiscard]] inline int run_tests(int argc, char *argv[]) noexcept {
    prepare_ut_arguments(argc, argv);
    auto &ut_argv = ut_arguments();
    auto failed = boost::ut::cfg().run({
        .argc = static_cast<int>(ut_argv.size()),
        .argv = ut_argv.data(),
    });
    return failed ? -1 : 0;
}

}// namespace luisa::test::coro_test
