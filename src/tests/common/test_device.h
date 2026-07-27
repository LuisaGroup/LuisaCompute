// Common device creation helper for standalone tests.
//
// Usage from main():
//   auto [context, device] = luisa::test::create_device(argc, argv);
//
// Usage from Boost.UT test:
//   auto [context, device] = luisa::test::create_device_from_ut();
//
// Both parse the backend name from argv[1] and print usage on missing arg.

#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace luisa::test {

struct DeviceContext {
    compute::Context context;
    compute::Device device;
};

inline void print_device_usage(const char *exe) noexcept {
    LUISA_INFO(
        "Usage: {} <backend>. <backend> must be an installed runtime backend "
        "(for example: cpu, cuda, dx, fallback, hip, metal, remote, vk).",
        exe);
}

inline void log_test_backend(const char *requested, const compute::Device &device) noexcept {
    auto active = device.backend_name();
    if (active.empty()) {
        // Validation layers may hide the underlying DeviceInterface name. The
        // context still loaded the exact requested backend plugin.
        LUISA_INFO("Test device backend: {}.", requested);
    } else {
        LUISA_INFO("Test device backend: {}.", active);
    }
}

/// Safe executable-path helper for Windows when __argv is unreliable.
inline const char *safe_argv0() noexcept {
#ifdef _WIN32
    static char path[MAX_PATH];
    if (path[0] == '\0') {
        GetModuleFileNameA(nullptr, path, MAX_PATH);
    }
    return path;
#else
    return "";
#endif
}

/// Create a device from main(argc, argv).
/// Exits with code 1 if no backend argument is provided.
[[nodiscard]] inline DeviceContext create_device(int argc, char *argv[]) {
    const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : safe_argv0();
    compute::Context context{exe};
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        print_device_usage(exe);
        exit(1);
    }
    compute::Device device = context.create_device(argv[1]);
    log_test_backend(argv[1], device);
    return {std::move(context), std::move(device)};
}

/// Create a device from Boost.UT's stored argc/argv
/// (available when linking with ut.hpp).
/// Returns std::nullopt if no backend argument is provided.
[[nodiscard]] inline std::optional<DeviceContext> create_device_from_ut() {
    auto argc = boost::ut::detail::cfg::largc;
    auto argv = boost::ut::detail::cfg::largv;
    const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : safe_argv0();
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        print_device_usage(exe);
        return std::nullopt;
    }
    compute::Context context{exe};
    compute::Device device = context.create_device(argv[1]);
    log_test_backend(argv[1], device);
    return DeviceContext{std::move(context), std::move(device)};
}
[[nodiscard]] inline std::optional<DeviceContext> create_device_from_ut(
    int argc, char *argv[],
    const compute::DeviceConfig *config = nullptr,
    bool enable_validation = false) {
    const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : safe_argv0();
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        print_device_usage(exe);
        return std::nullopt;
    }
    compute::Context context{exe};
    compute::Device device = context.create_device(
        argv[1], config, enable_validation);
    log_test_backend(argv[1], device);
    return DeviceContext{std::move(context), std::move(device)};
}

}// namespace luisa::test
