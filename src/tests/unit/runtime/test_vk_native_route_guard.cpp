// Vulkan strict native-route fail-closed runtime test.
// The assertion under test is fatal, so a parent process captures and checks
// the diagnostic emitted by a deliberately failing child process.

#include "ut/ut.hpp"
#include "test_device.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <luisa/backends/ext/vk_config_ext.h>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>

#ifndef LUISA_TEST_VK_HAS_NATIVE_XIR_SPIRV
#error "The Vulkan native-route guard test requires an explicit codegen-route definition."
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr std::string_view strict_native_environment =
    "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV";
constexpr std::string_view child_probe = "--strict-native-route-probe";
constexpr std::string_view child_readback_probe =
    "--strict-native-route-readback-probe";

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

struct ScopedDirectoryCleanup {
    std::filesystem::path path;
    ~ScopedDirectoryCleanup() noexcept {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

[[nodiscard]] std::string read_text_file(
    const std::filesystem::path &path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file},
            std::istreambuf_iterator<char>{}};
}

[[nodiscard]] int run_strict_native_route_probe(
    int argc, char *argv[]) {
    // Keep device initialization outside the contract so the observed fatal
    // diagnostic is tied specifically to the user Function compiled below.
    set_environment_variable(strict_native_environment.data(), nullptr);
    auto dc = luisa::test::create_device(argc, argv);
    set_environment_variable(strict_native_environment.data(), "1");
    Kernel1D kernel = [](BufferUInt output) noexcept {
        output.write(0u, 42u);
    };
    ShaderOption option{.enable_cache = false};
#if LUISA_TEST_VK_HAS_NATIVE_XIR_SPIRV
    option.native_include = R"(
uint lc_strict_native_route_marker(uint value) { return value; }
)";
#endif
    static_cast<void>(dc.device.compile(kernel, option));
    LUISA_WARNING("Vulkan strict native-route probe unexpectedly returned.");
    return 3;
}

struct NativeRouteDxcReadbackProbe final : VulkanDeviceConfigExt {
    bool compiler_seen = false;
    bool library_seen = false;
    bool utils_seen = false;

    void readback_vulkan_device(
        VkInstance instance,
        VkPhysicalDevice physical_device,
        VkDevice,
        VkAllocationCallbacks *,
        VkPipelineCacheHeaderVersionOne const &,
        VkQueue,
        VkQueue,
        VkQueue,
        uint32_t,
        uint32_t,
        uint32_t,
        IDxcCompiler3 *dxc_compiler,
        IDxcLibrary *dxc_library,
        IDxcUtils *dxc_utils) noexcept override {
        compiler_seen = dxc_compiler != nullptr;
        library_seen = dxc_library != nullptr;
        utils_seen = dxc_utils != nullptr;
    }
};

[[nodiscard]] int run_strict_native_route_readback_probe(
    int argc, char *argv[]) {
    set_environment_variable(strict_native_environment.data(), "1");
    DeviceConfig config;
    auto tracker = luisa::make_unique<NativeRouteDxcReadbackProbe>();
    auto tracker_ptr = tracker.get();
    config.extension = std::move(tracker);
    auto dc = luisa::test::create_device_from_ut(
        argc, argv, &config, true);
    LUISA_ASSERT(dc.has_value(), "Failed to create Vulkan test device.");

    Kernel1D kernel = [](BufferUInt output) noexcept {
        output.write(0u, 42u);
    };
    ShaderOption option{.enable_cache = false};
    static_cast<void>(dc->device.compile(kernel, option));
    LUISA_ASSERT(
        !tracker_ptr->compiler_seen &&
            !tracker_ptr->library_seen &&
            !tracker_ptr->utils_seen,
        "Strict native XIR->SPIR-V path should not pass DXC to "
        "readback callback.");
    return 0;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv == nullptr || argv[1] == nullptr ||
        std::string_view{argv[1]} != "vk") {
        LUISA_INFO("Usage: {} vk", argc > 0 ? argv[0] : "test_vk_native_route_guard");
        return 2;
    }
    if (argc >= 3 && argv[2] != nullptr &&
        std::string_view{argv[2]} == child_probe) {
        return run_strict_native_route_probe(argc, argv);
    }
    if (argc >= 3 && argv[2] != nullptr &&
        std::string_view{argv[2]} == child_readback_probe) {
        return run_strict_native_route_readback_probe(argc, argv);
    }

    std::vector<const char *> ut_argv;
    ut_argv.reserve(static_cast<size_t>(argc));
    ut_argv.emplace_back(argv[0]);
    for (auto i = 2; i < argc; i++) { ut_argv.emplace_back(argv[i]); }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        static_cast<int>(ut_argv.size()), ut_argv.data());

    auto executable_path = std::filesystem::absolute(argv[0]).string();
    auto nonce = std::chrono::steady_clock::now()
                     .time_since_epoch()
                     .count();
    auto process_directory = std::filesystem::temp_directory_path() /
                             ("luisa_vk_native_route_guard_" +
                              std::to_string(nonce));
    std::filesystem::create_directories(process_directory);
    ScopedDirectoryCleanup cleanup{process_directory};

    "vk_strict_native_route_fails_closed"_test = [&] {
        auto log_path = process_directory / "strict_native_route.log";
        auto command = luisa::format(
            "\"{}\" vk {} > \"{}\" 2>&1",
            executable_path, child_probe, log_path.string());
        auto status = std::system(command.c_str());
        auto log = read_text_file(log_path);

        expect(status != 0)
            << "strict native mode must terminate the forbidden route";
        expect(log.find(strict_native_environment) != std::string::npos)
            << luisa::format(
                   "strict-route rejection did not identify the environment "
                   "contract; child output:\n{}",
                   log);
#if LUISA_TEST_VK_HAS_NATIVE_XIR_SPIRV
        expect(log.find("requires the HLSL fallback for: native include") !=
               std::string::npos)
            << luisa::format(
                   "native XIR build did not reject the explicit HLSL "
                   "fallback for native include; child output:\n{}",
                   log);
#else
        expect(log.find("built without LUISA_XIR_TO_SPIRV") !=
               std::string::npos)
            << luisa::format(
                   "Vulkan build without native XIR codegen did not reject "
                   "an ordinary user Function; child output:\n{}",
                   log);
#endif
    };
    "vk_strict_native_route_readback_has_no_dxc"_test = [&] {
        auto log_path = process_directory / "strict_native_readback.log";
        auto command = luisa::format(
            "\"{}\" vk {} > \"{}\" 2>&1",
            executable_path, child_readback_probe, log_path.string());
        auto status = std::system(command.c_str());
        auto log = read_text_file(log_path);
        expect(status == 0)
            << luisa::format(
                   "strict native readback probe should return success; "
                   "status={}; child output:\n{}",
                   status, log);
    };
}
