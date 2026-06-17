#pragma once

#include "test_device.h"

#include <luisa/core/logging.h>

namespace luisa::test::coro_test {

struct Options {
    const char *exe;
    const char *backend;
};

[[nodiscard]] inline Options parse_options(int argc, char *argv[]) noexcept {
    auto exe = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : safe_argv0();
    if (argc <= 1 || argv[1] == nullptr) {
        LUISA_ERROR("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal, vk", exe);
    }
    return Options{exe, argv[1]};
}

[[nodiscard]] inline DeviceContext create_device(const Options &options) {
    compute::Context context{options.exe};
    compute::Device device = context.create_device(options.backend);
    return DeviceContext{std::move(context), std::move(device)};
}

}// namespace luisa::test::coro_test
