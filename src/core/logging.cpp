//
// Created by Mike Smith on 2021/2/11.
//

#include <spdlog/sinks/stdout_color_sinks.h>
#include <core/logging.h>

namespace luisa {

namespace detail {
static std::mutex LOGGER_MUTEX;
static luisa::logger LOGGER = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    spdlog::logger l{"console", sink};
    l.flush_on(spdlog::level::err);
#ifndef NDEBUG
    l.set_level(spdlog::level::debug);
#else
    l.set_level(spdlog::level::info);
#endif
    return l;
}();
[[nodiscard]] LC_CORE_API spdlog::logger &default_logger() noexcept {
    return LOGGER;
}
LC_CORE_API void set_sink(spdlog::sink_ptr sink) noexcept {
    std::lock_guard _lock{LOGGER_MUTEX};
    LOGGER.sinks().clear();
    LOGGER.sinks().push_back(std::move(sink));
}
}// namespace detail

void log_level_verbose() noexcept { detail::default_logger().set_level(spdlog::level::debug); }
void log_level_info() noexcept { detail::default_logger().set_level(spdlog::level::info); }
void log_level_warning() noexcept { detail::default_logger().set_level(spdlog::level::warn); }
void log_level_error() noexcept { detail::default_logger().set_level(spdlog::level::err); }

void log_flush() noexcept { detail::default_logger().flush(); }

}// namespace luisa
