#pragma once

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef LUISA_CUSTOM_LOGGER
#include <spdlog/spdlog.h>
#else
#include <luisa/core/stl/memory.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <luisa/core/stl/functional.h>
#include <luisa/core/macro.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/platform.h>

namespace luisa {
#ifndef LUISA_CUSTOM_LOGGER
using spdlog::logger;
using log_level = spdlog::level::level_enum;
#else
enum level_enum : int {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    err = 4,
    critical = 5,
    off = 6,
    n_levels
};
using CustomLoggerCallback = luisa::move_only_function<void(luisa::string &&str, level_enum level)>;
using CustomLoggerFlushCallback = luisa::move_only_function<void()>;
LUISA_CORE_API void set_custom_logger(CustomLoggerCallback &&callback) noexcept;
LUISA_CORE_API void set_custom_logger_flush(CustomLoggerFlushCallback &&callback) noexcept;
#endif

namespace detail {

#ifndef LUISA_CUSTOM_LOGGER
[[nodiscard]] LUISA_CORE_API luisa::logger &default_logger() noexcept;

LUISA_CORE_API void default_logger_set_sink(spdlog::sink_ptr sink) noexcept;
LUISA_CORE_API void default_logger_add_sink(spdlog::sink_ptr sink) noexcept;

LUISA_CORE_API spdlog::sink_ptr create_sink_with_callback(
    luisa::function<void(const char *level,
                         const char *message)>
        callback) noexcept;
#else
LUISA_CORE_API void custom_log(luisa::string &&str, level_enum level) noexcept;
#endif
template<typename T>
luisa::string to_string(T &&t) {
    if constexpr (std::is_same_v<T, luisa::string>) {
        return std::forward<T>(t);
    } else {
        return luisa::string{std::forward<T>(t)};
    }
}
}// namespace detail

template<typename... Args>
void log_verbose(Args &&...args) noexcept {
#ifndef LUISA_CUSTOM_LOGGER
    detail::default_logger().debug(std::forward<Args>(args)...);
#else
    if constexpr (sizeof...(args) == 1) {
        detail::custom_log(luisa::detail::to_string(args...), level_enum::debug);
    } else {
        detail::custom_log(luisa::format(args...), level_enum::debug);
    }
#endif
}

template<typename... Args>
void log_info(Args &&...args) noexcept {
#ifndef LUISA_CUSTOM_LOGGER
    detail::default_logger().info(std::forward<Args>(args)...);
#else
    if constexpr (sizeof...(args) == 1) {
        detail::custom_log(luisa::detail::to_string(args...), level_enum::info);
    } else {
        detail::custom_log(luisa::format(args...), level_enum::info);
    }
#endif
}

template<typename... Args>
void log_warning(Args &&...args) noexcept {
#ifndef LUISA_CUSTOM_LOGGER
    detail::default_logger().warn(std::forward<Args>(args)...);
#else
    if constexpr (sizeof...(args) == 1) {
        detail::custom_log(luisa::detail::to_string(args...), level_enum::warn);
    } else {
        detail::custom_log(luisa::format(args...), level_enum::warn);
    }
#endif
}
LUISA_CORE_API void log_flush() noexcept;

template<typename... Args>
[[noreturn]] LUISA_FORCE_INLINE void log_error(Args &&...args) noexcept {
#ifndef LUISA_CUSTOM_LOGGER
    std::string error_message;
    if constexpr (sizeof...(args) == 1u) {
        error_message = std::string{std::forward<Args>(args)...};
    } else {
        error_message = fmt::format(std::forward<Args>(args)...);
    }
    auto trace = luisa::backtrace();
    for (auto i = 0u; i < trace.size(); i++) {
        auto &&t = trace[i];
        fmt::format_to(std::back_inserter(error_message),
                       FMT_STRING("\n    {:>2} {}"), i, t);
    }
    detail::default_logger().error(error_message);
#else
    if constexpr (sizeof...(args) == 1) {
        detail::custom_log(luisa::detail::to_string(args...), level_enum::err);
    } else {
        detail::custom_log(luisa::format(args...), level_enum::err);
    }
    log_flush();
#endif
    std::abort();
}
/// Set log level as verbose
LUISA_CORE_API void log_level_verbose() noexcept;
/// Set log level as info
LUISA_CORE_API void log_level_info() noexcept;
/// Set log level as warning
LUISA_CORE_API void log_level_warning() noexcept;
/// Set log level as error
LUISA_CORE_API void log_level_error() noexcept;

/// flush the logs

}// namespace luisa

#define LUISA_LOGGING_FORMAT_IMPL(f) f
#define LUISA_LOGGING_FORMAT_IMPL_WITH_FMT(f, ...) \
    ::fmt::format(FMT_STRING(f), __VA_ARGS__)

#define LUISA_LOGGING_FORMAT(f, ...) \
    LUISA_LOGGING_FORMAT_IMPL##__VA_OPT__(_WITH_FMT)(f __VA_OPT__(, ) __VA_ARGS__)

/**
 * @brief Verbose logging
 * 
 * Ex. LUISA_VERBOSE("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_VERBOSE(fmt, ...) \
    ::luisa::log_verbose(LUISA_LOGGING_FORMAT(fmt __VA_OPT__(, ) __VA_ARGS__))

/**
 * @brief Info logging
 * 
 * Ex. LUISA_INFO("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_INFO(fmt, ...) \
    ::luisa::log_info(LUISA_LOGGING_FORMAT(fmt __VA_OPT__(, ) __VA_ARGS__))

/**
 * @brief Warning logging
 * 
 * Ex. LUISA_WARNING("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_WARNING(fmt, ...) \
    ::luisa::log_warning(LUISA_LOGGING_FORMAT(fmt __VA_OPT__(, ) __VA_ARGS__))

/**
 * @brief Error logging
 * 
 * After logging error message, the program will be aborted.
 * Ex. LUISA_ERROR("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_ERROR(fmt, ...) \
    ::luisa::log_error(LUISA_LOGGING_FORMAT(fmt __VA_OPT__(, ) __VA_ARGS__))

/// LUISA_VERBOSE with file and line information
#define LUISA_VERBOSE_WITH_LOCATION(fmt, ...) \
    LUISA_VERBOSE(fmt " [" __FILE__ ":" LUISA_STRINGIFY(__LINE__) "]" __VA_OPT__(, ) __VA_ARGS__)
/// LUISA_INFO with file and line information
#define LUISA_INFO_WITH_LOCATION(fmt, ...) \
    LUISA_INFO(fmt " [" __FILE__ ":" LUISA_STRINGIFY(__LINE__) "]" __VA_OPT__(, ) __VA_ARGS__)
/// LUISA_WARNING with file and line information
#define LUISA_WARNING_WITH_LOCATION(fmt, ...) \
    LUISA_WARNING(fmt " [" __FILE__ ":" LUISA_STRINGIFY(__LINE__) "]" __VA_OPT__(, ) __VA_ARGS__)
/// LUISA_ERROR with file and line information
#define LUISA_ERROR_WITH_LOCATION(fmt, ...) \
    LUISA_ERROR(fmt " [" __FILE__ ":" LUISA_STRINGIFY(__LINE__) "]" __VA_OPT__(, ) __VA_ARGS__)

#define LUISA_NOT_IMPLEMENTED_IMPL() \
    LUISA_ERROR_WITH_LOCATION("Not implemented.")

#define LUISA_NOT_IMPLEMENTED_IMPL_WITH_MESSAGE(fmt, ...) \
    LUISA_ERROR_WITH_LOCATION("Not implemented: " fmt __VA_OPT__(, ) __VA_ARGS__)

#define LUISA_NOT_IMPLEMENTED(...) \
    LUISA_NOT_IMPLEMENTED_IMPL##__VA_OPT__(_WITH_MESSAGE)(__VA_ARGS__)

#define LUISA_ASSERT_FAILED_IMPL(x) \
    LUISA_ERROR_WITH_LOCATION("Assertion '" #x "' failed.")

#define LUISA_ASSERT_FAILED_IMPL_WITH_MESSAGE_IMPL_WITH_FMT(x, fmt, ...) \
    LUISA_ERROR_WITH_LOCATION("Assertion '{}' failed: " fmt, #x, __VA_ARGS__)

#define LUISA_ASSERT_FAILED_IMPL_WITH_MESSAGE_IMPL(x, msg) \
    LUISA_ERROR_WITH_LOCATION("Assertion '" #x "' failed: " msg)

#define LUISA_ASSERT_FAILED_IMPL_WITH_MESSAGE(x, fmt, ...) \
    LUISA_ASSERT_FAILED_IMPL_WITH_MESSAGE_IMPL##__VA_OPT__(_WITH_FMT)(x, fmt __VA_OPT__(, ) __VA_ARGS__)

#define LUISA_ASSERT(x, ...)                                                                   \
    do {                                                                                       \
        if (!(x)) [[unlikely]] {                                                               \
            LUISA_ASSERT_FAILED_IMPL##__VA_OPT__(_WITH_MESSAGE)(x __VA_OPT__(, ) __VA_ARGS__); \
        }                                                                                      \
    } while (false)

#ifndef NDEBUG
#define LUISA_DEBUG_ASSERT(x, ...) LUISA_ASSERT(x __VA_OPT__(, ) __VA_ARGS__)
#else
#define LUISA_DEBUG_ASSERT(x, ...) \
    do {                           \
    } while (false)
#endif
