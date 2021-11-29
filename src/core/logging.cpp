//
// Created by Mike Smith on 2021/2/11.
//

#include <spdlog/sinks/stdout_color_sinks.h>
#include <core/allocator.h>
#include <core/logging.h>

namespace luisa {

namespace detail {

[[nodiscard]] static spdlog::logger *default_logger() noexcept {
    static auto logger = [] {
        auto sink = luisa::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        return spdlog::logger{"LuisaCompute", sink};
    }();
    return &logger;
}

void log_to_default_logger(spdlog::level::level_enum level, std::string_view message) noexcept {
    default_logger()->log(level, "{}",  message);
}

}

void log_level_verbose() noexcept { detail::default_logger()->set_level(spdlog::level::debug); }
void log_level_info() noexcept { detail::default_logger()->set_level(spdlog::level::info); }
void log_level_warning() noexcept { detail::default_logger()->set_level(spdlog::level::warn); }
void log_level_error() noexcept { detail::default_logger()->set_level(spdlog::level::err); }

}// namespace luisa
