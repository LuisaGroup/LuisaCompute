//
// Created by Mike Smith on 2021/10/18.
//

#include <core/logging.h>
#include <api/logging.h>

void luisa_compute_log_level_verbose() LUISA_NOEXCEPT {
    luisa::log_level_verbose();
}

void luisa_compute_log_level_info() LUISA_NOEXCEPT {
    luisa::log_level_info();
}

void luisa_compute_log_level_warning() LUISA_NOEXCEPT {
    luisa::log_level_warning();
}

void luisa_compute_log_level_error() LUISA_NOEXCEPT {
    luisa::log_level_error();
}

void luisa_compute_log_verbose(const char *msg) LUISA_NOEXCEPT {
    LUISA_VERBOSE("{}", msg);
}

void luisa_compute_log_info(const char *msg) LUISA_NOEXCEPT {
    LUISA_INFO("{}", msg);
}

void luisa_compute_log_warning(const char *msg) LUISA_NOEXCEPT {
    LUISA_WARNING("{}", msg);
}

void luisa_compute_log_error(const char *msg) LUISA_NOEXCEPT {
    LUISA_ERROR("{}", msg);
}
