//
// Created by Mike Smith on 2021/2/11.
//

#include <core/logging.h>

namespace luisa {

void log_level_verbose() noexcept { spdlog::set_level(spdlog::level::debug); }
void log_level_info() noexcept { spdlog::set_level(spdlog::level::info); }
void log_level_warning() noexcept { spdlog::set_level(spdlog::level::warn); }
void log_level_error() noexcept { spdlog::set_level(spdlog::level::err); }

}// namespace luisa
