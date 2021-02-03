//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <spdlog/spdlog.h>

namespace luisa {

template<typename ...Args>
inline void log_verbose(Args &&...args) noexcept { spdlog::debug(std::forward<Args>(args)...); }

template<typename ...Args>
inline void log_info(Args &&...args) noexcept { spdlog::info(std::forward<Args>(args)...); }

template<typename ...Args>
inline void log_warning(Args &&...args) noexcept { spdlog::warn(std::forward<Args>(args)...); }

template<typename ...Args>
inline void log_error(Args &&...args) noexcept {
    spdlog::error(std::forward<Args>(args)...);
    std::abort();
}

}

#define LUISA_VERBOSE(...) ::luisa::log_verbose(__VA_ARGS__)
#define LUISA_VERBOSE_WITH_LOCATION(fmt, ...) ::luisa::log_verbose(fmt " [{}:{}]", __VA_ARGS__ __VA_OPT__(,) __FILE__, __LINE__)
#define LUISA_INFO(...) ::luisa::log_info(__VA_ARGS__)
#define LUISA_INFO_WITH_LOCATION(fmt, ...) ::luisa::log_info(fmt " [{}:{}]", __VA_ARGS__ __VA_OPT__(,) __FILE__, __LINE__)
#define LUISA_WARNING(...) ::luisa::log_warning(__VA_ARGS__)
#define LUISA_WARNING_WITH_LOCATION(fmt, ...) ::luisa::log_warning(fmt " [{}:{}]", __VA_ARGS__ __VA_OPT__(,) __FILE__, __LINE__)
#define LUISA_ERROR(...) ::luisa::log_error(__VA_ARGS__)
#define LUISA_ERROR_WITH_LOCATION(fmt, ...) ::luisa::log_error(fmt " [{}:{}]", __VA_ARGS__ __VA_OPT__(,) __FILE__, __LINE__)
