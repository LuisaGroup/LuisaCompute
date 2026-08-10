#include <luisa/luisa-compute.h>

#include <cstdio>
#include <vector>

#if __cplusplus < 202002L
#error "luisa::compute must export a C++20 compile requirement"
#endif

#if LUISA_PACKAGE_WITH_DSL && !defined(LUISA_ENABLE_DSL)
#error "The installed target lost LUISA_ENABLE_DSL"
#endif
#if !LUISA_PACKAGE_WITH_DSL && defined(LUISA_ENABLE_DSL)
#error "The installed target unexpectedly defines LUISA_ENABLE_DSL"
#endif

#if LUISA_PACKAGE_WITH_GUI && !defined(LUISA_ENABLE_GUI)
#error "The installed target lost LUISA_ENABLE_GUI"
#endif
#if !LUISA_PACKAGE_WITH_GUI && defined(LUISA_ENABLE_GUI)
#error "The installed target unexpectedly defines LUISA_ENABLE_GUI"
#endif

#if LUISA_PACKAGE_WITH_WAYLAND && !defined(LUISA_ENABLE_WAYLAND)
#error "The installed target lost LUISA_ENABLE_WAYLAND"
#endif
#if !LUISA_PACKAGE_WITH_WAYLAND && defined(LUISA_ENABLE_WAYLAND)
#error "The installed target unexpectedly defines LUISA_ENABLE_WAYLAND"
#endif

#if LUISA_PACKAGE_WITH_SAFE_MODE && !defined(LUISA_ENABLE_SAFE_MODE)
#error "The installed target lost LUISA_ENABLE_SAFE_MODE"
#endif
#if !LUISA_PACKAGE_WITH_SAFE_MODE && defined(LUISA_ENABLE_SAFE_MODE)
#error "The installed target unexpectedly defines LUISA_ENABLE_SAFE_MODE"
#endif

#if LUISA_PACKAGE_WITH_CLANG_CXX && !defined(LUISA_ENABLE_CLANGCXX)
#error "The installed target lost LUISA_ENABLE_CLANGCXX"
#endif
#if !LUISA_PACKAGE_WITH_CLANG_CXX && defined(LUISA_ENABLE_CLANGCXX)
#error "The installed target unexpectedly defines LUISA_ENABLE_CLANGCXX"
#endif

#ifndef LUISA_ENABLE_XIR
#error "The installed target lost LUISA_ENABLE_XIR"
#endif

#define LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(expect, definition) \
    static_assert((expect) == (definition),                         \
                  "Exported system-dependency definition mismatch")

#ifdef LUISA_USE_SYSTEM_STL
#define LUISA_PACKAGE_HAS_SYSTEM_STL 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_STL 0
#endif
#ifdef LUISA_USE_SYSTEM_GLFW
#define LUISA_PACKAGE_HAS_SYSTEM_GLFW 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_GLFW 0
#endif
#ifdef LUISA_USE_SYSTEM_LMDB
#define LUISA_PACKAGE_HAS_SYSTEM_LMDB 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_LMDB 0
#endif
#ifdef LUISA_USE_SYSTEM_REPROC
#define LUISA_PACKAGE_HAS_SYSTEM_REPROC 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_REPROC 0
#endif
#ifdef LUISA_USE_SYSTEM_SPDLOG
#define LUISA_PACKAGE_HAS_SYSTEM_SPDLOG 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_SPDLOG 0
#endif
#ifdef LUISA_USE_SYSTEM_XXHASH
#define LUISA_PACKAGE_HAS_SYSTEM_XXHASH 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_XXHASH 0
#endif
#ifdef LUISA_USE_SYSTEM_YYJSON
#define LUISA_PACKAGE_HAS_SYSTEM_YYJSON 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_YYJSON 0
#endif
#ifdef LUISA_USE_SYSTEM_MAGIC_ENUM
#define LUISA_PACKAGE_HAS_SYSTEM_MAGIC_ENUM 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_MAGIC_ENUM 0
#endif
#ifdef LUISA_USE_SYSTEM_MARL
#define LUISA_PACKAGE_HAS_SYSTEM_MARL 1
#else
#define LUISA_PACKAGE_HAS_SYSTEM_MARL 0
#endif

LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_STL, LUISA_PACKAGE_HAS_SYSTEM_STL);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_GLFW, LUISA_PACKAGE_HAS_SYSTEM_GLFW);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_LMDB, LUISA_PACKAGE_HAS_SYSTEM_LMDB);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_REPROC, LUISA_PACKAGE_HAS_SYSTEM_REPROC);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_SPDLOG, LUISA_PACKAGE_HAS_SYSTEM_SPDLOG);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_XXHASH, LUISA_PACKAGE_HAS_SYSTEM_XXHASH);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_YYJSON, LUISA_PACKAGE_HAS_SYSTEM_YYJSON);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_MAGIC_ENUM,
    LUISA_PACKAGE_HAS_SYSTEM_MAGIC_ENUM);
LUISA_PACKAGE_CHECK_SYSTEM_DEFINITION(
    LUISA_PACKAGE_USE_SYSTEM_MARL, LUISA_PACKAGE_HAS_SYSTEM_MARL);

#if defined(_WIN32)
#ifndef LUISA_PLATFORM_WINDOWS
#error "The installed target lost LUISA_PLATFORM_WINDOWS"
#endif
#ifndef _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
#error "The installed target lost its required MSVC ABI definition"
#endif
#else
#ifndef LUISA_PLATFORM_UNIX
#error "The installed target lost LUISA_PLATFORM_UNIX"
#endif
#if defined(__APPLE__) && !defined(LUISA_PLATFORM_APPLE)
#error "The installed target lost LUISA_PLATFORM_APPLE"
#endif
#endif

template<size_t N>
[[nodiscard]] bool check_half_formatters() {
    auto vector_text = luisa::to_string(luisa::Vector<luisa::half, N>{});
    auto matrix_text = luisa::to_string(luisa::Matrix<luisa::half, N>{});
    return !vector_text.empty() && !matrix_text.empty();
}

int main(int argc, char *argv[]) {
    static_assert(LUISA_COMPUTE_VERSION_MAJOR == LUISA_PACKAGE_VERSION_MAJOR);
    static_assert(LUISA_COMPUTE_VERSION_MINOR == LUISA_PACKAGE_VERSION_MINOR);
    static_assert(LUISA_COMPUTE_VERSION_PATCH == LUISA_PACKAGE_VERSION_PATCH);

    // Compile and execute formatters whose behavior depends on the exported
    // spdlog/fmt selection and LUISA_USE_SYSTEM_SPDLOG definition. This caught
    // a real mismatch between packaged fmt releases and Luisa's half types.
    if (!check_half_formatters<2u>() ||
        !check_half_formatters<3u>() ||
        !check_half_formatters<4u>()) {
        std::fprintf(stderr, "The exported half formatters returned no text.\n");
        return 3;
    }
    auto range_text = luisa::format(
        FMT_STRING("{}"), std::vector<size_t>{1u, 2u, 3u});
    if (range_text.empty()) {
        std::fprintf(stderr, "The exported range formatter returned no text.\n");
        return 5;
    }

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

    luisa::log_level_info();
    luisa::log_flush();
    return 0;
}
