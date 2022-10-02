//
// Created by Mike Smith on 2021/10/18.
//

#include <core/logging.h>
#include <api/logging.h>

LUISA_EXPORT_API void luisa_compute_set_log_level_verbose() LUISA_NOEXCEPT {
    luisa::log_level_verbose();
}

LUISA_EXPORT_API void luisa_compute_set_log_level_info() LUISA_NOEXCEPT {
    luisa::log_level_info();
}

LUISA_EXPORT_API void luisa_compute_set_log_level_warning() LUISA_NOEXCEPT {
    luisa::log_level_warning();
}

LUISA_EXPORT_API void luisa_compute_set_log_level_error() LUISA_NOEXCEPT {
    luisa::log_level_error();
}

LUISA_EXPORT_API void luisa_compute_log_verbose(const char *msg) LUISA_NOEXCEPT {
    LUISA_VERBOSE("{}", msg);
}

LUISA_EXPORT_API void luisa_compute_log_info(const char *msg) LUISA_NOEXCEPT {
    LUISA_INFO("{}", msg);
}

LUISA_EXPORT_API void luisa_compute_log_warning(const char *msg) LUISA_NOEXCEPT {
    LUISA_WARNING("{}", msg);
}

void luisa_compute_log_error(const char *msg) LUISA_NOEXCEPT {
    LUISA_ERROR("{}", msg);
}
