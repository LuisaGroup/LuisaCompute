// Fail-closed rejection test for the elementary special functions on the two
// backends that do not implement them (feature F1).
//
// dx (the HLSL route) and vk (the native XIR-to-SPIR-V route) have no
// erf/erfc/expm1 instruction. Both must *abort with their own diagnostic* while
// generating code instead of emitting an approximation, and ldexp - the one
// member of the family they do support - must still compile on both.
//
// LuisaCompute reports the rejection through LUISA_ERROR (log + abort), so - as
// the test skill's "Fatal checks without exceptions" rule requires - each case
// runs in a child process: this executable re-executes itself with the
// `--erf-probe` / `--ldexp-probe` marker, the parent redirects the child's output
// into a temporary file and asserts both the exit status and the exact
// diagnostic. Each backend asserts *its own* fragment (never `frag_a || frag_b`),
// and the vk case additionally asserts that the HLSL fragment is absent, so a
// silent route change shows up as a different failure instead of a pass.
//
// The ldexp control probe (expected exit status 0) proves the harness can observe
// a *successful* compile, i.e. the erf probe's abort is op-specific.
//
// TODO(backend): the vk HLSL *compatibility* route's rejection (a non-empty
// ShaderOption::native_include selects it) is not probed here: the route selection
// itself is not verified, and the native-route probe below covers the same
// diagnostic class. Add it once user_compute_codegen_route.h has its own test.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

// The build systems define these so a backend that is not compiled in *skips*
// instead of failing on device creation (the LUISA_TEST_VK_HAS_* idiom).
#ifndef LUISA_TEST_HAS_DX
#define LUISA_TEST_HAS_DX 0
#endif
#ifndef LUISA_TEST_HAS_VK
#define LUISA_TEST_HAS_VK 0
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_backend_fatal_rejections_detail {

constexpr std::string_view erf_probe = "--erf-probe";
constexpr std::string_view ldexp_probe = "--ldexp-probe";

/// Compiles `erf(x)`; the backend must reject it while generating code.
[[nodiscard]] int run_erf_probe(int argc, char *argv[]) noexcept {
    auto dc = luisa::test::create_device(argc, argv);
    Kernel1D kernel = [](BufferFloat input, BufferFloat output) noexcept {
        output.write(0u, erf(input.read(0u)));
    };
    static_cast<void>(dc.device.compile(kernel));
    LUISA_WARNING("The erf probe unexpectedly returned: the backend accepted an "
                  "operation it cannot implement.");
    return 3;
}

/// Compiles `ldexp(x, n)`, which both backends do support: the harness control.
[[nodiscard]] int run_ldexp_probe(int argc, char *argv[]) noexcept {
    auto dc = luisa::test::create_device(argc, argv);
    Kernel1D kernel = [](BufferFloat input, BufferInt exponent,
                         BufferFloat output) noexcept {
        output.write(0u, ldexp(input.read(0u), exponent.read(0u)));
    };
    static_cast<void>(dc.device.compile(kernel));
    LUISA_INFO("The ldexp control probe compiled successfully.");
    return 0;
}

struct ChildResult {
    int status{0};
    std::string text;
};

[[nodiscard]] std::string read_text_file(const luisa::filesystem::path &path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file},
            std::istreambuf_iterator<char>{}};
}

/// Runs one probe in a child process and returns its exit status and its
/// redirected output (the in-tree pattern of test_vk_native_route_guard.cpp).
[[nodiscard]] ChildResult run_child(const std::string &executable_path,
                                    std::string_view backend,
                                    std::string_view probe,
                                    const luisa::filesystem::path &log_path) {
    auto command = luisa::format(
        "\"{}\" {} {} > \"{}\" 2>&1",
        executable_path, backend, probe, log_path.string());
#ifdef _WIN32
    // system() goes through cmd /C, which strips the outer quote pair; wrap the
    // whole line so the executable and log paths stay quoted.
    command = luisa::format("\"{}\"", command);
#endif
    auto status = std::system(command.c_str());
    return {status, read_text_file(log_path)};
}

struct ScopedDirectoryCleanup {
    luisa::filesystem::path path;
    ~ScopedDirectoryCleanup() noexcept {
        std::error_code error;
        luisa::filesystem::remove_all(path, error);
    }
};

}// namespace test_backend_fatal_rejections_detail

int main(int argc, char *argv[]) {

    if (argc < 2 || argv[1] == nullptr) {
        LUISA_INFO("Usage: {} <dx|vk>", argc > 0 ? argv[0] : "test_backend_fatal_rejections");
        return 2;
    }
    auto backend = std::string_view{argv[1]};

    // The child dispatch must happen before any Boost.UT registration, so the
    // probe marker never reaches the framework's argument parser.
    if (argc >= 3 && argv[2] != nullptr) {
        if (std::strcmp(argv[2], test_backend_fatal_rejections_detail::erf_probe.data()) == 0) {
            return test_backend_fatal_rejections_detail::run_erf_probe(argc, argv);
        }
        if (std::strcmp(argv[2], test_backend_fatal_rejections_detail::ldexp_probe.data()) == 0) {
            return test_backend_fatal_rejections_detail::run_ldexp_probe(argc, argv);
        }
    }
    if (backend != "dx" && backend != "vk") {
        LUISA_INFO("Only dx and vk reject erf/erfc/expm1; skipping backend '{}'.",
                   backend);
        return 0;
    }
    if (backend == "dx" && !LUISA_TEST_HAS_DX) {
        LUISA_INFO("This build has no dx backend; skipping.");
        return 0;
    }
    if (backend == "vk" && !LUISA_TEST_HAS_VK) {
        LUISA_INFO("This build has no vk backend; skipping.");
        return 0;
    }

    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using namespace test_backend_fatal_rejections_detail;

    auto executable_path = luisa::filesystem::absolute(argv[0]).string();
    auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    auto log_directory = luisa::filesystem::temp_directory_path() /
                         ("luisa_backend_fatal_rejections_" + std::to_string(nonce));
    luisa::filesystem::create_directories(log_directory);
    ScopedDirectoryCleanup cleanup{log_directory};

    // The harness control: an operation both backends support must compile, so a
    // non-zero status below is attributable to erf and not to the probe itself.
    "ldexp_compiles_on_this_backend"_test = [&] {
        auto log_path = log_directory / luisa::format("{}_ldexp_control.log", backend);
        auto result = run_child(executable_path, backend, ldexp_probe, log_path);
        expect(result.status == 0)
            << "ldexp must compile on " << backend
            << " (the erf rejection below must be op-specific); child output:\n"
            << result.text;
    };

    if (backend == "dx") {
        "dx_erf_fails_closed_with_the_hlsl_diagnostic"_test = [&] {
            auto log_path = log_directory / "dx_erf.log";
            auto result = run_child(executable_path, backend, erf_probe, log_path);
            expect(result.status != 0)
                << "erf on dx must abort instead of returning; child output:\n"
                << result.text;
            expect(result.text.find("Unsupported builtin ERF on the HLSL target") !=
                   std::string::npos)
                << "the HLSL route must name the operation and the target; child "
                   "output:\n"
                << result.text;
            expect(result.text.find("ldexp is supported here") != std::string::npos)
                << "the diagnostic must say what still works; child output:\n"
                << result.text;
        };
    } else {
        "vk_erf_fails_closed_on_the_native_spirv_route"_test = [&] {
            auto log_path = log_directory / "vk_erf.log";
            auto result = run_child(executable_path, backend, erf_probe, log_path);
            expect(result.status != 0)
                << "erf on vk must abort instead of returning; child output:\n"
                << result.text;
            expect(result.text.find("GLSL.std.450 has no erf/erfc/expm1 instruction") !=
                   std::string::npos)
                << "the native SPIR-V route must report the dialect boundary; child "
                   "output:\n"
                << result.text;
            // A route change (vk silently taking the HLSL compatibility route
            // instead) must show up as this failure rather than pass through an
            // `||` in the assertion above.
            expect(result.text.find("Unsupported builtin") == std::string::npos)
                << "vk took the HLSL compatibility route instead of the native "
                   "XIR-to-SPIR-V route; child output:\n"
                << result.text;
        };
    }
}
