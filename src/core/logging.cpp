//
// Created by Mike Smith on 2021/2/11.
//

#include <spdlog/sinks/stdout_color_sinks.h>
#include <core/allocator.h>
#include <core/logging.h>

namespace luisa {

namespace detail {
[[nodiscard]] spdlog::logger &default_logger() noexcept {
    static auto logger = [] {
        auto sink = luisa::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        spdlog::logger l{"LuisaCompute", sink};
#ifndef NDEBUG
        l.set_level(spdlog::level::debug);
#else
        l.set_level(spdlog::level::info);
#endif
        return l;
    }();
    return logger;
}
}

void log_level_verbose() noexcept { detail::default_logger().set_level(spdlog::level::debug); }
void log_level_info() noexcept { detail::default_logger().set_level(spdlog::level::info); }
void log_level_warning() noexcept { detail::default_logger().set_level(spdlog::level::warn); }
void log_level_error() noexcept { detail::default_logger().set_level(spdlog::level::err); }

}// namespace luisa
