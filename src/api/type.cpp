//
// Created by Mike Smith on 2021/10/18.
//

#include <ast/type.h>
#include <api/type.h>

using namespace luisa::compute;

LCType luisa_compute_type_from_description(const char *desc) LUISA_NOEXCEPT {
    return (LCType)Type::from(desc);
}

size_t luisa_compute_type_size(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->size();
}

size_t luisa_compute_type_alignment(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->alignment();
}

char *luisa_compute_type_description(const LCType t) LUISA_NOEXCEPT {
    auto desc = reinterpret_cast<const Type *>(t)->description();
    auto s = new char[desc.size() + 1u];
    std::memcpy(s, desc.data(), desc.size());
    s[desc.size()] = '\0';
    return s;
}

size_t luisa_compute_type_dimension(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->dimension();
}

size_t luisa_compute_type_member_count(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->members().size();
}

LCType luisa_compute_type_member_types(const LCType t) LUISA_NOEXCEPT {
    return (LCType)reinterpret_cast<const Type *>(t)->members().data();
}

LCType luisa_compute_type_element_type(const LCType t) LUISA_NOEXCEPT {
    return (LCType)reinterpret_cast<const Type *>(t)->element();
}

int luisa_compute_type_is_array(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_array();
}

int luisa_compute_type_is_scalar(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_scalar();
}

int luisa_compute_type_is_vector(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_vector();
}

int luisa_compute_type_is_matrix(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_matrix();
}

int luisa_compute_type_is_structure(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_structure();
}

int luisa_compute_type_is_buffer(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_buffer();
}

int luisa_compute_type_is_texture(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_texture();
}

int luisa_compute_type_is_heap(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_bindless_array();
}

int luisa_compute_type_is_accel(const LCType t) LUISA_NOEXCEPT {
    return reinterpret_cast<const Type *>(t)->is_accel();
}
