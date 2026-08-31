#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef LUISA_CUSTOM_LOGGER
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#else
#include <iostream>
#include <mutex>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/magic_enum.h>

namespace luisa {

#ifndef LUISA_CUSTOM_LOGGER
namespace detail {

static std::mutex LOGGER_MUTEX;

template<typename Mt>
class SinkWithCallback : public spdlog::sinks::base_sink<Mt> {

private:
    luisa::function<void(const char *, const char *)> _callback;

public:
    template<class F, std::enable_if_t<!std::is_same_v<std::remove_cvref_t<F>, SinkWithCallback>, int> = 0>
    explicit SinkWithCallback(F &&_callback) noexcept
        : _callback{std::forward<F>(_callback)} {}

protected:
    void sink_it_(const spdlog::details::log_msg &msg) override {
        auto level = msg.level;
        auto level_name = spdlog::level::to_short_c_str(level);
        auto message = fmt::to_string(msg.payload);
        _callback(level_name, message.c_str());
    }
    void flush_() override {}
};

static luisa::logger LOGGER = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    spdlog::logger l{"console", sink};
    l.flush_on(spdlog::level::err);
#ifndef NDEBUG
    spdlog::level::level_enum log_level = spdlog::level::debug;
#else
    spdlog::level::level_enum log_level = spdlog::level::info;
#endif
    if (auto env_level_c_str = getenv("LUISA_LOG_LEVEL")) {
        luisa::string env_level{env_level_c_str};
        for (auto &c : env_level) { c = static_cast<char>(tolower(c)); }
        if (env_level == "verbose") {
            log_level = spdlog::level::debug;
        } else if (env_level == "info") {
            log_level = spdlog::level::info;
        } else if (env_level == "warning") {
            log_level = spdlog::level::warn;
        } else if (env_level == "error") {
            log_level = spdlog::level::err;
        } else {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid log level '{}'. "
                "Please choose from 'verbose', 'info', 'warning' and 'error'. "
                "Fallback to default log level '{}'.",
                env_level, luisa::to_string(log_level));
        }
    }
    l.set_level(log_level);
    return l;
}();

[[nodiscard]] LUISA_CORE_API spdlog::logger &default_logger() noexcept {
    return LOGGER;
}

LUISA_CORE_API void default_logger_set_sink(spdlog::sink_ptr sink) noexcept {
    std::lock_guard _lock{LOGGER_MUTEX};
    LOGGER.sinks().clear();
    if (sink) {
        LOGGER.sinks().emplace_back(std::move(sink));
    }
}

LUISA_CORE_API void default_logger_add_sink(spdlog::sink_ptr sink) noexcept {
    std::lock_guard _lock{LOGGER_MUTEX};
    if (sink) {
        LOGGER.sinks().emplace_back(std::move(sink));
    }
}

LUISA_CORE_API spdlog::sink_ptr create_sink_with_callback(luisa::function<void(const char *level, const char *message)> callback) noexcept {
    return std::make_shared<luisa::detail::SinkWithCallback<std::mutex>>(std::move(callback));
}

}// namespace detail

void log_level_verbose() noexcept { detail::default_logger().set_level(spdlog::level::debug); }
void log_level_info() noexcept { detail::default_logger().set_level(spdlog::level::info); }
void log_level_warning() noexcept { detail::default_logger().set_level(spdlog::level::warn); }
void log_level_error() noexcept { detail::default_logger().set_level(spdlog::level::err); }

void log_flush() noexcept { detail::default_logger().flush(); }
#else
static level_enum default_level =
#ifdef NDEBUG
    level_enum::debug
#else
    level_enum::info
#endif
    ;
static CustomLoggerCallback logger = []() {
    return [](luisa::string &&str, level_enum level) {
        static std::mutex global_log_mtx;
        std::lock_guard lck{global_log_mtx};
        std::cout << str << std::endl;
    };
}();
static CustomLoggerFlushCallback logger_flush = []() {
    return []() {
        std::cout << std::endl;
    };
}();
void set_custom_logger(CustomLoggerCallback &&callback) noexcept {
    logger = std::move(callback);
}
namespace detail {
void custom_log(luisa::string &&str, level_enum level) noexcept {
    if (luisa::to_underlying(level) >= luisa::to_underlying(default_level))
        logger(std::move(str), level);
}
}// namespace detail
void set_custom_logger_flush(CustomLoggerFlushCallback &&callback) noexcept {
    logger_flush = std::move(callback);
}
/// Set log level as verbose
void log_level_verbose() noexcept {
    default_level = level_enum::debug;
}
/// Set log level as info
void log_level_info() noexcept {
    default_level = level_enum::info;
}
/// Set log level as warning
void log_level_warning() noexcept {
    default_level = level_enum::warn;
}
/// Set log level as error
void log_level_error() noexcept {
    default_level = level_enum::err;
}

/// flush the logs
void log_flush() noexcept {
    logger_flush();
}
#endif
}// namespace luisa
