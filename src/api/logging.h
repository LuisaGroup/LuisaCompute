//
// Created by Mike Smith on 2021/10/18.
//

#pragma once

#include <core/platform.h>

LUISA_EXPORT_API void luisa_compute_log_level_verbose() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_level_info() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_level_warning() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_level_error() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_verbose(const char *msg) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_info(const char *msg) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_warning(const char *msg) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_error(const char *msg) LUISA_NOEXCEPT;
