//
// Created by Mike Smith on 2021/10/18.
//

#include <api/type.h>

using namespace luisa::compute;

const void *luisa_compute_type_from_description(const char *desc) LUISA_NOEXCEPT {
    return Type::from(desc);
}

size_t luisa_compute_type_size(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->size();
}

size_t luisa_compute_type_alignment(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->alignment();
}

char *luisa_compute_type_description(const void *t) LUISA_NOEXCEPT {
    auto desc = static_cast<const Type *>(t)->description();
    auto s = new char[desc.size() + 1u];
    std::memcpy(s, desc.data(), desc.size());
    s[desc.size()] = '\0';
    return s;
}

size_t luisa_compute_type_dimension(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->dimension();
}

size_t luisa_compute_type_member_count(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->members().size();
}

const void *luisa_compute_type_member_types(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->members().data();
}

const void *luisa_compute_type_element_type(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->element();
}

int luisa_compute_type_is_array(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_array();
}

int luisa_compute_type_is_scalar(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_scalar();
}

int luisa_compute_type_is_vector(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_vector();
}

int luisa_compute_type_is_matrix(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_matrix();
}

int luisa_compute_type_is_structure(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_structure();
}

int luisa_compute_type_is_buffer(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_buffer();
}

int luisa_compute_type_is_texture(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_texture();
}

int luisa_compute_type_is_heap(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_heap();
}

int luisa_compute_type_is_accel(const void *t) LUISA_NOEXCEPT {
    return static_cast<const Type *>(t)->is_accel();
}
