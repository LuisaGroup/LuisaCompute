//
// Created by Mike Smith on 2021/10/18.
//

#pragma once

#include <core/platform.h>
#include <api/language.h>

LUISA_EXPORT_API LCType luisa_compute_type_from_description(const char *desc) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_type_description(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_size(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_alignment(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_dimension(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_member_count(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCType luisa_compute_type_member_types(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCType luisa_compute_type_element_type(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_array(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_scalar(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_vector(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_matrix(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_structure(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_buffer(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_texture(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_heap(const LCType) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_accel(const LCType) LUISA_NOEXCEPT;
