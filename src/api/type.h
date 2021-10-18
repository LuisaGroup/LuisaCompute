//
// Created by Mike Smith on 2021/10/18.
//

#pragma once

#include <core/platform.h>

LUISA_EXPORT_API const void *luisa_compute_type_from_description(const char *desc) LUISA_NOEXCEPT;
LUISA_EXPORT_API const char *luisa_compute_type_description(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_size(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_alignment(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_dimension(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_type_member_count(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_type_member_types(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_type_element_type(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_array(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_scalar(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_vector(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_matrix(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_structure(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_buffer(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_texture(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_heap(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_type_is_accel(const void *t) LUISA_NOEXCEPT;
